"""OpenTSLM-Flamingo for the ARFBench notebook: load, fine-tune and save the model in the
training job, and answer requests on the SageMaker endpoint.

It reads per-series checkpoints such as opentime/tslm-nov26: each series is its own media item,
and missing values (NaN) are masked.

The Flamingo layers in this file (PerceiverResampler, MaskedCrossAttention,
GatedCrossAttentionBlock, FlamingoLayer, FlamingoLMMixin and add_flamingo_layers) are adapted
from OpenFlamingo (https://github.com/mlfoundations/open_flamingo), which adapts
flamingo-pytorch (https://github.com/lucidrains/flamingo-pytorch). Both are under the MIT
License:

    OpenFlamingo: Copyright (c) 2023 Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel,
    Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev,
    Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, Ludwig Schmidt.

    flamingo-pytorch: Copyright (c) 2022 Phil Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    get_wsd_schedule,
)

MEDIA_TOKEN = "<image>"  # tokenizer special token, not a credential  # nosec B105
END_OF_CHUNK_TOKEN = "<|endofchunk|>"  # tokenizer special token, not a credential  # nosec B105
WEIGHTS_FILE = "model.safetensors"
# The other files of a model directory, which save() writes back unchanged.
CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


@dataclass(frozen=True)
class SeriesInput:
    """One time series: its values (NaN where missing), an optional text before its statistics,
    and its name in the prompt (its position, from 1, without one)."""

    values: list[float]
    text: str | None = None
    name: str | None = None


# ---- The model


class PatchTokenizer(nn.Module):
    """Embed patches of one series; each timestep also has a flag that tells if it is observed."""

    def __init__(self, output_dim: int, patch_size: int, max_patches: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.patch_embedding = nn.Conv1d(
            2, output_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, max_patches, output_dim))
        self.input_norm = nn.LayerNorm(output_dim)

    def forward(self, values: Tensor, observed: Tensor) -> Tensor:
        inputs = torch.stack((values.masked_fill(~observed, 0.0), observed.to(values.dtype)), 1)
        patches = self.patch_embedding(inputs).transpose(1, 2)
        return self.input_norm(patches + self.position_embedding[:, : patches.shape[1]])


class PerceiverAttention(nn.Module):
    """Let the latent vectors attend to the media embeddings and to themselves."""

    def __init__(self, dim: int, dim_head: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.media_norm = nn.LayerNorm(dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * heads * 2, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim, bias=False)

    def forward(self, media: Tensor, latents: Tensor, media_mask: Tensor) -> Tensor:
        media = self.media_norm(media)
        normalized_latents = self.latent_norm(latents)
        query = _split_heads(self.to_q(normalized_latents), self.heads)
        key, value = self.to_kv(torch.cat((media, normalized_latents), dim=-2)).chunk(2, dim=-1)
        key, value = _split_heads(key, self.heads), _split_heads(value, self.heads)
        similarity = torch.einsum("bthid,bthjd->bthij", query * self.scale, key)
        latent_mask = torch.ones(*latents.shape[:-1], dtype=torch.bool, device=latents.device)
        key_mask = torch.cat((media_mask, latent_mask), dim=-1)
        similarity = similarity.masked_fill(
            ~key_mask[:, :, None, None, :], torch.finfo(similarity.dtype).min
        )
        output = torch.einsum("bthij,bthjd->bthid", similarity.softmax(dim=-1), value)
        batch, media_count, _, tokens, _ = output.shape
        output = output.permute(0, 1, 3, 2, 4).reshape(batch, media_count, tokens, -1)
        return self.to_out(output)


def _split_heads(values: Tensor, heads: int) -> Tensor:
    batch, media_count, tokens, inner_dim = values.shape
    values = values.reshape(batch, media_count, tokens, heads, inner_dim // heads)
    return values.permute(0, 1, 3, 2, 4)


class PerceiverFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        self.layers = _feedforward(dim, mult)

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


def _feedforward(dim: int, mult: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult, bias=False),
        nn.GELU(),
        nn.Linear(dim * mult, dim, bias=False),
    )


class PerceiverResampler(nn.Module):
    """Compress the patch embeddings of each series into a fixed number of latent vectors."""

    def __init__(self, dim, dim_head, heads, mult, depth, num_latents, max_media) -> None:
        super().__init__()
        self.media_position = nn.Parameter(torch.zeros(max_media, 1, dim))
        self.latents = nn.Parameter(torch.zeros(num_latents, dim))
        self.attention_layers = nn.ModuleList(
            [PerceiverAttention(dim, dim_head, heads) for _ in range(depth)]
        )
        self.feedforward_layers = nn.ModuleList(
            [PerceiverFeedForward(dim, mult) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, media: Tensor, media_mask: Tensor) -> Tensor:
        """Turn media ``[media, patches, dim]`` into latents ``[1, media, latents, dim]``."""
        media_count = media.shape[0]
        media = (media + self.media_position[:media_count]).unsqueeze(0)
        latents = self.latents.expand(1, media_count, -1, -1)
        for attention, feedforward in zip(self.attention_layers, self.feedforward_layers):
            latents = attention(media, latents, media_mask.unsqueeze(0)) + latents
            latents = feedforward(latents) + latents
        return self.norm(latents)


class MaskedCrossAttention(nn.Module):
    """Let each text token attend to the latents of the most recent media token before it."""

    def __init__(self, dim: int, dim_visual: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim_visual, dim_head * heads * 2, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim, bias=False)

    def forward(self, x: Tensor, media: Tensor, media_locations: Tensor) -> Tensor:
        query = self.to_q(self.norm(x)) * self.scale
        key, value = self.to_kv(media).chunk(2, dim=-1)
        text_time = media_locations.cumsum(dim=-1)
        if int(text_time.max()) > media.shape[1]:
            raise ValueError("the text has more media tokens than there are media items")
        # The tokens after one media token read only its latents. Scoring each run of tokens
        # against those latents alone, not against all media, keeps the memory small.
        rows = []
        for index in range(len(x)):
            items, counts = torch.unique_consecutive(text_time[index], return_counts=True)
            parts = []
            for item, segment in zip(items.tolist(), query[index].split(counts.tolist())):
                if item == 0:  # a text token before the first media token attends to nothing
                    parts.append(torch.zeros_like(segment))
                    continue
                q, k, v = (
                    tensor.unflatten(-1, (self.heads, -1)).transpose(0, 1)
                    for tensor in (segment, key[index, item - 1], value[index, item - 1])
                )
                attended = (q @ k.transpose(-1, -2)).softmax(dim=-1) @ v
                parts.append(attended.transpose(0, 1).flatten(1))
            rows.append(torch.cat(parts))
        return self.to_out(torch.stack(rows))


class GatedCrossAttentionBlock(nn.Module):
    """Cross-attention and a feed-forward block, each scaled by a learned tanh gate."""

    def __init__(self, dim: int, dim_visual: int) -> None:
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_visual=dim_visual)
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ff = _feedforward(dim, 4)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: Tensor, media: Tensor, media_locations: Tensor) -> Tensor:
        x = self.attn(x, media, media_locations) * self.attn_gate.tanh() + x
        return self.ff(x) * self.ff_gate.tanh() + x


class FlamingoLayer(nn.Module):
    """A Gemma 3 decoder layer, with a gated cross-attention block before it."""

    def __init__(self, gated_cross_attn_layer: nn.Module | None, decoder_layer: Any) -> None:
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x: Tensor | None = None
        self.media_locations: Tensor | None = None

    @property
    def attention_type(self) -> str:
        # Gemma 3 reads this attribute of each decoder layer to pick its attention mask.
        return self.decoder_layer.attention_type

    def forward(self, lang_x: Tensor, attention_mask: Any = None, **kwargs: Any) -> Any:
        if self.gated_cross_attn_layer is not None:
            lang_x = self.gated_cross_attn_layer(lang_x, self.vis_x, self.media_locations)
        return self.decoder_layer(lang_x, attention_mask=attention_mask, **kwargs)


class FlamingoLMMixin(nn.Module):
    """Gated cross-attention layers for a Gemma 3 vision-language model.

    In a cached generation step, the model gets only the newest token, without the media
    tokens. That token attends to the last media item, as the end of the prompt does.
    """

    def forward(
        self,
        input_ids: Tensor,
        *,
        logits_to_keep: Any = 0,  # generate() passes it only if forward() names it
        cache_position: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        layers = self.language_model.layers
        if cache_position is not None and bool(cache_position[0] > 0):
            media_locations = torch.zeros_like(input_ids)
            media_locations[:, 0] = layers[0].vis_x.shape[1]
        else:
            media_locations = (input_ids == self.media_token_id).long()
        for layer in layers:
            layer.media_locations = media_locations
        return super().forward(
            input_ids=input_ids,
            logits_to_keep=logits_to_keep,
            cache_position=cache_position,
            **kwargs,
        )


def add_flamingo_layers(language_model: Any, media_token_id: int, vis_dim: int, every: int) -> None:
    """Give every ``every``-th decoder layer of a Gemma 3 vision-language model a gated
    cross-attention block, in place."""
    language_model.__class__ = type(
        type(language_model).__name__, (FlamingoLMMixin, type(language_model)), {}
    )
    language_model.media_token_id = media_token_id
    hidden_size = language_model.config.text_config.hidden_size
    decoder_layers = language_model.language_model.layers
    # None stands for a layer without cross-attention; ModuleList keeps it at run time.
    language_model.gated_cross_attn_layers = nn.ModuleList(
        [
            GatedCrossAttentionBlock(hidden_size, vis_dim) if (index + 1) % every == 0 else None
            for index in range(len(decoder_layers))
        ]
    )
    language_model.language_model.layers = nn.ModuleList(
        FlamingoLayer(block, layer)
        for block, layer in zip(language_model.gated_cross_attn_layers, decoder_layers)
    )


class OpenTSLMFlamingo(nn.Module):
    """A Gemma 3 language model that reads time series through cross-attention."""

    def __init__(self, config: dict[str, Any], tokenizer: Any, language_model: Any) -> None:
        super().__init__()
        encoder, resampler = config["encoder_config"], config["resampler_config"]
        self.patch_size = int(encoder["patch_size"])
        self.max_values = int(encoder["max_patches"]) * self.patch_size
        self.max_media = int(resampler["max_media"])
        self.compute_dtype = torch.float32  # from_file sets the dtype it loads in
        self.tokenizer = tokenizer
        self.files: dict[str, bytes] = {}  # the config files, which save() writes back
        language_model.resize_token_embeddings(len(tokenizer))
        add_flamingo_layers(
            language_model,
            media_token_id=tokenizer.convert_tokens_to_ids(MEDIA_TOKEN),
            vis_dim=int(encoder["output_dim"]),
            every=int(config["cross_attention_every_n_layers"]),
        )
        # The weights are named flamingo.vision_encoder, flamingo.perceiver, flamingo.lang_encoder.
        self.flamingo = nn.Module()
        self.flamingo.vision_encoder = PatchTokenizer(
            int(encoder["output_dim"]), self.patch_size, int(encoder["max_patches"])
        )
        self.flamingo.perceiver = PerceiverResampler(
            dim=int(resampler["attention"]["dim"]),
            dim_head=int(resampler["attention"]["dim_head"]),
            heads=int(resampler["attention"]["heads"]),
            mult=int(resampler["feedforward"]["mult"]),
            depth=int(resampler["depth"]),
            num_latents=int(resampler["num_latents"]),
            max_media=self.max_media,
        )
        self.flamingo.lang_encoder = language_model

    @classmethod
    def from_file(cls, path: str | Path, device: str, dtype: torch.dtype) -> OpenTSLMFlamingo:
        """Load a model directory: config.json, generation_config.json, the tokenizer files and
        model.safetensors, as in the model's Hugging Face repository.

        Load in float32, then keep the series encoder, the perceiver, the cross-attention layers and the token embeddings in float32 and store the
        rest of the language model in ``dtype``, as the checkpoint's training and its weights file do.
        With a ``dtype`` other than float32 the forward passes run under autocast to ``dtype``."""
        root = Path(path)
        config = json.loads((root / "config.json").read_text())
        # Local directory.
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)  # nosec B615
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [END_OF_CHUNK_TOKEN, MEDIA_TOKEN]}
        )
        if tokenizer.pad_token_id is None:
            # Tokenizer padding token, not a credential.
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})  # nosec B105
        weights = load_file(root / WEIGHTS_FILE)
        with safe_open(root / WEIGHTS_FILE, framework="pt") as file:
            # save() stores each tied weight once; the metadata names its other names.
            aliases = json.loads((file.metadata() or {}).get("aliases", "{}"))
        weights.update({alias: weights[name] for alias, name in aliases.items()})

        language_config = AutoConfig.for_model(**copy.deepcopy(config["language_model_config"]))
        # transformers builds each part in the dtype that its own config names.
        for part in [
            language_config,
            *map(language_config.__getattribute__, language_config.sub_configs),
        ]:
            if part is not None:
                part.dtype = torch.float32
        with _default_dtype(torch.float32), torch.device(device):
            language_model = AutoModelForCausalLM.from_config(
                language_config,
                attn_implementation=config.get("attention_implementation", "eager"),
            )
            # Local directory.
            language_model.generation_config = GenerationConfig.from_pretrained(  # nosec B615
                root, local_files_only=True
            )
            model = cls(config, tokenizer, language_model)
        model.load_state_dict(weights, strict=True)
        kept = {id(p) for module in model.trained_modules() for p in module.parameters()}
        for parameter in model.parameters():
            if id(parameter) not in kept:
                parameter.data = parameter.data.to(dtype)
        model.compute_dtype = dtype
        model.files = {
            name: (root / name).read_bytes() for name in CONFIG_FILES if (root / name).is_file()
        }
        return model.eval()

    def trained_modules(self) -> list[nn.Module]:
        """Give the modules that the checkpoint's training trains, and keeps in float32."""
        lang = self.flamingo.lang_encoder
        # Gemma 3 ties the output layer to the token embeddings.
        return [
            self.flamingo.vision_encoder,
            self.flamingo.perceiver,
            lang.gated_cross_attn_layers,
            lang.get_input_embeddings(),
        ]

    def save_pretrained(self, directory: str | Path) -> None:
        """Write the model directory; each tied weight is stored once."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name, contents in self.files.items():
            (root / name).write_bytes(contents)
        stored, aliases, seen = {}, {}, {}
        for name, tensor in self.state_dict().items():
            place = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.shape)
            if place in seen:
                aliases[name] = seen[place]
            else:
                seen[place] = name
                stored[name] = tensor.contiguous()
        save_file(stored, root / WEIGHTS_FILE, metadata={"aliases": json.dumps(aliases)})

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        series: list[SeriesInput],
        question: str,
        *,
        context: Sequence[str] = (),
        max_new_tokens: int = 256,
    ) -> str:
        """Answer a question about the series, with greedy decoding. Each ``context`` line goes
        between the series and the question."""
        prompt, values, observed = self.prepare(series, question, context)
        tokens = self.tokenizer([prompt], return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        with self.conditioned_on(values, observed):
            generated = self.flamingo.lang_encoder.generate(
                input_ids,
                attention_mask=tokens["attention_mask"].to(self.device),
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        answer = generated[0, input_ids.shape[1] :]
        return self.tokenizer.decode(answer, skip_special_tokens=True).strip()

    def loss(
        self, series: list[SeriesInput], question: str, answer: str, reduction: str = "mean"
    ) -> Tensor:
        """Give the cross-entropy of the answer tokens and the end-of-sequence token, which
        follow the prompt as in training: their mean, or one per token with ``"none"``."""
        prompt, values, observed = self.prepare(series, question)
        # As the training collator: a space before the answer unless the prompt ends in one.
        target = answer if prompt.endswith((" ", "\n")) else f" {answer}"
        prompt_ids = self.tokenizer(prompt)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        target_ids = [*target_ids, self.tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt_ids + target_ids], device=self.device)
        with self.conditioned_on(values, observed):
            # Only the logits that predict the target tokens: the full vocabulary is large.
            logits = self.flamingo.lang_encoder(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=len(target_ids) + 1,
            ).logits
        labels = torch.tensor(target_ids, device=logits.device)
        return nn.functional.cross_entropy(logits[0, :-1].float(), labels, reduction=reduction)

    def prepare(
        self, series: list[SeriesInput], question: str, context: Sequence[str] = ()
    ) -> tuple[str, Tensor, Tensor]:
        """Give the prompt and the padded series ``[series, time]`` with their observed flags,
        in the layout that the model trained on.

        Raises:
            ValueError: If there are more series than the model takes, a series is longer than
                the encoder takes, or a series has no observed or an infinite value.
        """
        if len(series) > self.max_media:
            raise ValueError(f"the model takes at most {self.max_media} series")
        words, rows = [], []
        for position, item in enumerate(series):
            if len(item.values) > self.max_values:
                raise ValueError(
                    f"series {position} has {len(item.values)} values, "
                    f"but the model takes at most {self.max_values}"
                )
            scaled, observed, mean, std = _scale(item.values)
            sentence = (
                f"This is time series channel {item.name or position + 1}, with mean "
                f"{_statistic(mean)} and std {_statistic(std)}."
            )
            if not observed.all():
                sentence += " Some values of this series are missing."
            words += [
                MEDIA_TOKEN,
                *([item.text] if item.text else []),
                sentence,
                END_OF_CHUNK_TOKEN,
            ]
            rows.append((scaled, observed))
        words += [*context, question]
        length = max(len(scaled) for scaled, _ in rows)
        length += -length % self.patch_size
        values = torch.zeros((len(rows), length), dtype=torch.float32)
        observed = torch.zeros((len(rows), length), dtype=torch.bool)
        for index, (scaled, mask) in enumerate(rows):
            values[index, : len(scaled)] = torch.from_numpy(scaled)
            observed[index, : len(mask)] = torch.from_numpy(mask)
        return " ".join(words), values, observed

    @contextmanager
    def conditioned_on(self, values: Tensor, observed: Tensor):
        """Let the language model attend to the encoded series while the block runs, under
        autocast to the model's compute dtype."""
        dtype = self.compute_dtype
        autocast = torch.autocast(self.device.type, dtype, enabled=dtype != torch.float32)
        values, observed = values.to(self.device, torch.float32), observed.to(self.device)
        layers = self.flamingo.lang_encoder.language_model.layers
        try:
            with autocast:
                encoded = self.flamingo.vision_encoder(values, observed)
                steps = observed.reshape(len(observed), -1, self.patch_size).any(dim=-1)
                latents = self.flamingo.perceiver(encoded, steps)
                for layer in layers:
                    layer.vis_x = latents
                yield
        finally:
            for layer in layers:
                layer.vis_x = layer.media_locations = None


def _scale(values: list[float]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Scale the observed values to a mean of 0 and a standard deviation of 1, 0 where missing,
    as in training. Dividing by the largest magnitude first keeps the squares finite."""
    array = np.asarray(values, dtype=np.float64)
    observed = ~np.isnan(array)
    present = array[observed]
    if not present.size:
        raise ValueError("each time series must have at least one observed value")
    if not np.isfinite(present).all():
        raise ValueError("time series values must not be infinite")
    scaled = np.zeros(array.shape, dtype=np.float64)
    magnitude = float(np.abs(present).max())
    mean = std = 0.0
    if magnitude:
        unit = present / magnitude
        unit_mean, unit_std = float(unit.mean()), float(unit.std())
        scaled[observed] = (unit - unit_mean) / (unit_std + 1e-8 / magnitude)
        mean, std = unit_mean * magnitude, unit_std * magnitude
    return scaled.astype(np.float32), observed, mean, std


def _statistic(value: float) -> str:
    return f"{value:.4e}" if abs(value) >= 1e6 else f"{value:.4f}"


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# ---- Fine-tuning on TimeNet tasks


@dataclass(frozen=True)
class Example:
    """One question with its series, its options in their released order, and its answer."""

    series: list[SeriesInput]
    prompt: str
    options: list[str]
    answer: str
    category: str

    def question(self, rng: random.Random | None = None) -> str:
        """Give the options, shuffled when ``rng`` is given, then the prompt. The checkpoint's
        training shows a list annotation of a task this way: its key, then numbered lines."""
        if not self.options:
            return self.prompt  # the prompt lists the choices
        options = list(self.options)
        if rng is not None:
            rng.shuffle(options)
        lines = "\n".join(f"{number}. {option}" for number, option in enumerate(options, start=1))
        return f"answer_options:\n{lines} {self.prompt}"

    def evaluated_question(self) -> str:
        """Give the question with its options in a fixed order that depends on the question
        only. In ARFBench the answer is often the first released option."""
        return self.question(random.Random(self.prompt))  # shuffling, not security  # nosec B311


SPLIT = {"train": 0.7, "validation": 0.1, "test": 0.2}


def split_by_incident(tasks: list[Any], seed: int = 0, restarts: int = 20) -> dict[str, list]:
    """Split the questions into training, validation and test sets, with ``SPLIT`` of them each.

    All questions of an incident go to one set, so no set sees an incident of another. A question
    about two incidents in different sets is left out, as are questions without an incident.
    Answers cluster by incident, so a random split can put most questions with one answer in one
    set. A local search moves single incidents between the sets while that lowers the sum of the
    gaps in each set's mix of categories and answers, the gaps from its target size, and the share
    of questions left out. It starts from ``restarts`` random splits and keeps the best result.
    """
    from timenet.types import read_value

    incidents = {
        task.id: frozenset(
            str(read_value(item.value))
            for record in task.inputs
            for item in record.annotations
            if item.key == "incident_id"
        )
        for task in tasks
    }
    tasks = sorted((task for task in tasks if incidents[task.id]), key=lambda task: int(task.id))
    labels = _labels(tasks)
    every = [labels[task.id] for task in tasks]
    names = sorted(set().union(*incidents.values()))

    def sets(side: dict[str, str]) -> tuple[dict[str, list], int]:
        parts: dict[str, list] = {name: [] for name in SPLIT}
        left_out = 0
        for task in tasks:
            chosen = {side[incident] for incident in incidents[task.id]}
            if len(chosen) == 1:
                parts[chosen.pop()].append(task)
            else:
                left_out += 1
        return parts, left_out

    def cost(side: dict[str, str]) -> float:
        parts, left_out = sets(side)
        kept = len(tasks) - left_out
        total = left_out / len(tasks)
        for name, part in parts.items():
            if not part:
                return math.inf
            total += _mix_distance([labels[task.id] for task in part], every) * SPLIT[name] ** 0.5
            total += 2 * abs(len(part) / kept - SPLIT[name])
        return total

    rng = random.Random(seed)  # shuffling, not security  # nosec B311
    best: tuple[float, dict[str, str]] = (math.inf, {})
    for _ in range(restarts):
        side = {name: rng.choices(list(SPLIT), weights=list(SPLIT.values()))[0] for name in names}
        current, moved = cost(side), True
        while moved:
            moved = False
            for name in rng.sample(names, len(names)):
                for target in SPLIT:
                    previous, side[name] = side[name], target
                    if (value := cost(side)) < current - 1e-9:
                        current, moved = value, True
                    else:
                        side[name] = previous
        if current < best[0]:
            best = (current, dict(side))
    return sets(best[1])[0]


def save_split(split: dict[str, list], path: str | Path) -> None:
    """Write the task IDs of each set to a JSON file, so every run trains and tests on the same
    questions."""
    ids = {name: sorted(int(task.id) for task in part) for name, part in split.items()}
    lines = ",\n".join(f" {json.dumps(name)}: {json.dumps(part)}" for name, part in ids.items())
    Path(path).write_text("{\n" + lines + "\n}\n")


def load_split(tasks: list[Any], ids: dict[str, list[int]]) -> dict[str, list]:
    """Give the tasks of each set from the task IDs of a ``save_split`` file.

    Raises:
        ValueError: If a task ID is not in ``tasks``, such as for another dataset version.
    """
    by_id = {int(task.id): task for task in tasks}
    missing = sorted(i for part in ids.values() for i in part if i not in by_id)
    if missing:
        raise ValueError(f"{len(missing)} task IDs of the split are not in the dataset")
    return {name: [by_id[i] for i in part] for name, part in ids.items()}


def _labels(tasks: list[Any]) -> dict[Any, tuple[str, str]]:
    """Give each task its category and answer; an answer that fewer than 3 tasks share, such as
    a timestamp, counts as "other"."""
    from timenet.types import read_value

    def raw(task: Any) -> tuple[str, str]:
        notes = {item.key: read_value(item.value) for item in task.annotations}
        return str(notes.get("task_category", "")), str(read_value(task.targets[0]))

    pairs = {task.id: raw(task) for task in tasks}
    counts = Counter(answer for _, answer in pairs.values())
    return {key: (c, a if counts[a] >= 3 else "other") for key, (c, a) in pairs.items()}


def _mix_distance(first: list, second: list) -> float:
    """Give the total variation distance between the label shares of two lists."""
    a, b = Counter(first), Counter(second)
    return sum(abs(a[k] / len(first) - b[k] / len(second)) for k in a | b) / 2


def examples_from_tasks(tasks: list[Any], max_values: int, max_series: int = 32) -> list[Example]:
    """Give one example per TimeNet question task that shows time series only, with at most
    ``max_series`` series, shared evenly by the records of the task."""
    from timenet.types import read_value

    examples = []
    for task in tasks:
        signals = [signal for record in task.inputs for signal in record.signals]
        if not signals or any(signal.spec.modality == "image" for signal in signals):
            continue
        notes = {item.key: read_value(item.value) for item in task.annotations}
        # A classification task offers its label set; an answer task lists its own options.
        label_set = getattr(task, "label_set", None)
        options = (
            [label.name for label in label_set.labels]
            if label_set is not None
            else [str(option) for option in notes.get("answer_options", [])]
        )
        prompt = str(read_value(task.prompt))
        mentioned = "\n".join([prompt, *options])
        limit = max_series // len(task.inputs)
        if not limit:
            continue  # more records than series the model takes
        series = []
        for record in task.inputs:
            series += _record_series(record, max_values, limit, mentioned)
        if not series:
            continue  # no series fits the encoder, or every series is empty
        examples.append(
            Example(
                series=series,
                prompt=prompt,
                options=options,
                answer=str(read_value(task.targets[0])),
                category=str(notes.get("task_category", "")),
            )
        )
    return examples


def _record_series(
    record: Any, max_values: int, max_series: int, mentioned: str
) -> list[SeriesInput]:
    """Give at most ``max_series`` sources of a record, in record order, at the finest resolution
    that fits: first those that ``mentioned`` names, such as the channels in the options of an
    identification question, then those that vary most. As in the checkpoint's training, a series
    has no text besides the sentence that names it and gives its statistics."""
    lengths: dict[str, int] = {}
    for source in record.sources:
        for signal in source.signals:
            lengths[signal.name] = max(lengths.get(signal.name, 0), signal.n_values)
    fitting = [name for name, length in lengths.items() if length <= max_values]
    if not fitting:
        return []
    resolution = max(fitting, key=lambda name: lengths[name])
    groups = []
    for source in record.sources:
        for signal in source.signals:
            if signal.name != resolution:
                continue
            values = signal.to_arrow().to_numpy(zero_copy_only=False).astype(np.float64)
            if not np.isnan(values).all():
                groups.append((float(np.nanstd(values)), source.name, values))

    def named(name: str) -> bool:  # "method:1" is not named by "method:12"
        return re.search(rf"(?<![\w:]){re.escape(name)}(?!\w)", mentioned) is not None

    ranked = sorted(range(len(groups)), key=lambda i: (not named(groups[i][1]), -groups[i][0]))
    return [
        SeriesInput(values=values.tolist(), name=name)
        for _, name, values in (groups[i] for i in sorted(ranked[:max_series]))
    ]


def chosen_option(output: str, options: list[str]) -> str | None:
    """Give the option that the answer names, after the normalization of the checkpoint's
    evaluation. "None of the above" does not name the option "No"."""
    wanted = _normalized(output)
    return next((option for option in options if _normalized(option) == wanted), None)


def _normalized(text: str) -> str:
    """Keep the text after the last "answer:", lowercase it, strip periods and collapse spaces."""
    text = text.strip()
    if "answer:" in text.lower():
        text = text[text.lower().rfind("answer:") + len("answer:") :]
    return " ".join(text.strip().strip(".\n\t ").lower().split())


def accuracy(model: OpenTSLMFlamingo, examples: list[Example]) -> dict[str, Any]:
    """Give the share of questions answered with the right option, overall and per category.

    Per category it also gives the balanced accuracy, the mean share right per correct answer:
    a model that always answers "yes" to presence questions, most of which have an anomaly,
    scores high accuracy but only 50% balanced accuracy."""
    model.eval()
    right: dict[tuple[str, str], list[bool]] = {}
    for example in examples:
        with _autocast(model):
            output = model.generate(example.series, example.evaluated_question(), max_new_tokens=64)
        # A question without options counts the output that states the answer.
        chosen = chosen_option(output, example.options or [example.answer])
        right.setdefault((example.category, example.answer), []).append(chosen == example.answer)
    per_category = {}
    for category in sorted({category for category, _ in right}):
        results = [hits for (c, _), hits in right.items() if c == category]
        per_category[category] = {
            "questions": sum(len(hits) for hits in results),
            "accuracy": sum(map(sum, results)) / sum(len(hits) for hits in results),
            "balanced_accuracy": sum(sum(hits) / len(hits) for hits in results) / len(results),
        }
    total = sum(len(hits) for hits in right.values())
    overall = sum(map(sum, right.values())) / max(total, 1)
    return {"accuracy": overall, "per_category": per_category}


@torch.no_grad()
def mean_loss(model: OpenTSLMFlamingo, examples: list[Example]) -> float:
    model.eval()
    losses = []
    for example in examples:
        with _autocast(model):
            loss = model.loss(example.series, example.evaluated_question(), example.answer)
            losses.append(float(loss))
    return sum(losses) / max(len(losses), 1)


# The checkpoint's ARFBench fine-tuning recipe: a learning rate per parameter group, 3% warmup,
# a flat phase and a linear decay to 0 over the last 20% of the steps, no gradient clipping.
LEARNING_RATES = {"encoder": 4e-4, "adapter": 2e-4, "base": 4e-4}
WARMUP_FRACTION, DECAY_FRACTION = 0.03, 0.2


def train(
    model: OpenTSLMFlamingo,
    train_examples: list[Example],
    val_examples: list[Example],
    *,
    epochs: int,
    accumulation: int,
    seed: int,
    learning_rates: dict[str, float] = LEARNING_RATES,
) -> None:
    """Train the series encoder, the perceiver, the cross-attention layers and the token
    embeddings with AdamW, as the checkpoint was trained; the rest of the language model stays
    frozen. Keep the epoch with the lowest validation loss, or the last one without validation
    examples. Each step averages the loss of the answer tokens of ``accumulation`` questions,
    with their options shuffled."""
    if not train_examples:
        raise ValueError("there are no training examples")
    model.requires_grad_(False)
    encoder, perceiver, cross_attention, embeddings = model.trained_modules()
    groups = {"encoder": [encoder], "adapter": [perceiver, cross_attention], "base": [embeddings]}
    parameter_groups = []
    for name, modules in groups.items():
        for module in modules:
            module.float().requires_grad_(True)  # trained in float32, under bfloat16 autocast
        params = [p for module in modules for p in module.parameters()]
        parameter_groups.append({"params": params, "lr": learning_rates[name]})
    parameters = [p for group in parameter_groups for p in group["params"]]
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=0.01, fused=model.device.type == "cuda"
    )
    steps = max(epochs * math.ceil(len(train_examples) / accumulation), 1)
    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=max(1, int(steps * WARMUP_FRACTION)),
        num_decay_steps=max(1, int(steps * DECAY_FRACTION)),
        num_training_steps=steps,
        decay_type="linear",
    )
    rng = random.Random(seed)  # shuffling, not security  # nosec B311
    best_loss, best_weights = float("inf"), {}
    for epoch in range(1, epochs + 1):
        order = list(train_examples)
        rng.shuffle(order)
        model.train()
        started, total = time.time(), 0.0
        for start in range(0, len(order), accumulation):
            tokens = 0
            for example in order[start : start + accumulation]:
                with _autocast(model):
                    losses = model.loss(
                        example.series, example.question(rng), example.answer, reduction="none"
                    )
                if not torch.isfinite(losses).all():
                    raise ValueError(f"the training loss is not finite: {losses.tolist()}")
                losses.sum().backward()
                tokens += len(losses)
                total += float(losses.detach().mean())
            for p in parameters:  # one mean over the answer tokens of the batch
                if p.grad is not None:
                    p.grad /= tokens
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        val_loss = mean_loss(model, val_examples) if val_examples else None
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": total / len(order),
                    "val_loss": val_loss,
                    "seconds": round(time.time() - started),
                }
            ),
            flush=True,
        )
        # The first epoch always counts, so a run whose loss is NaN still saves a model.
        if val_loss is None or not best_weights or val_loss < best_loss:
            if val_loss is not None:
                best_loss = min(best_loss, val_loss)
            best_weights = {
                name: value.detach().to("cpu", copy=True)
                for name, value in model.named_parameters()
                if value.requires_grad
            }
            print(json.dumps({"best_epoch": epoch}), flush=True)
    model.load_state_dict(best_weights, strict=False)


def _autocast(model: OpenTSLMFlamingo) -> torch.autocast:
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda")


class OpenTSLM:
    """An OpenTSLM-Flamingo model that answers questions about TimeNet time-series tasks."""

    def __init__(self, model: OpenTSLMFlamingo) -> None:
        self.model = model

    @classmethod
    def load(cls, spec: str) -> OpenTSLM:
        """Load a model directory from a path or ``hf://ORG/REPO``, on the GPU when there is one."""
        path = spec
        if spec.startswith("hf://"):
            from huggingface_hub import snapshot_download

            # The caller chooses the checkpoint.
            path = snapshot_download(spec.removeprefix("hf://"))  # nosec B615
        return cls(_load(path))

    def fine_tune(
        self,
        tasks: list[Any],
        validation: list[Any] | None = None,
        *,
        epochs: int = 3,
        accumulation: int = 8,
        seed: int = 0,
    ) -> OpenTSLM:
        """Fine-tune on TimeNet tasks and keep the epoch with the lowest loss on the
        ``validation`` tasks, or the last epoch without them."""
        torch.manual_seed(seed)
        train(
            self.model,
            examples_from_tasks(tasks, self.model.max_values),
            examples_from_tasks(validation or [], self.model.max_values),
            epochs=epochs,
            accumulation=accumulation,
            seed=seed,
        )
        return self

    def eval(self, tasks: list[Any], metrics: list[str] = ("accuracy",)) -> dict[str, Any]:
        """Answer each TimeNet task and give the share that names the right option, overall and,
        under ``per_category``, per category with its balanced accuracy."""
        if set(metrics) != {"accuracy"}:
            raise ValueError("the only metric is accuracy")
        examples = examples_from_tasks(tasks, self.model.max_values)
        return accuracy(self.model, examples)

    def save(self, path: str) -> None:
        """Write the model directory to a local path or to ``s3://BUCKET/PREFIX``."""
        if not path.startswith("s3://"):
            self.model.save_pretrained(path)
            return
        import boto3

        bucket, _, key = path.removeprefix("s3://").rstrip("/").partition("/")
        s3 = boto3.client("s3")
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            for file in sorted(Path(directory).iterdir()):
                s3.upload_file(str(file), bucket, f"{key}/{file.name}")


def _load(path: str | Path) -> OpenTSLMFlamingo:
    """Load in bfloat16 on a GPU, in float32 on a CPU."""
    cuda = torch.cuda.is_available()
    return OpenTSLMFlamingo.from_file(
        path, device="cuda" if cuda else "cpu", dtype=torch.bfloat16 if cuda else torch.float32
    )


# ---- The SageMaker endpoint handler
#
# Request: {"instances": [{"series": [{"values": [...], "text": "...", "name": "..."}],
#                          "context": ["..."], "question": "..."}],
#           "parameters": {"max_new_tokens": 128}}
# A null value is a missing value. Response: {"generated_text": ["...", ...]}.

MAX_INSTANCES = 4
MAX_NEW_TOKENS_LIMIT = 512
MAX_SERIES = 32  # ARFBench questions show up to 32 series
_FLOAT32_MAX = float(torch.finfo(torch.float32).max)

try:
    from sagemaker_inference.errors import GenericInferenceToolkitError

    def _bad_request(message: str) -> Exception:
        # The toolkit turns any other exception into a 500 with a traceback.
        return GenericInferenceToolkitError(400, message)

except ImportError:  # outside the SageMaker container

    def _bad_request(message: str) -> Exception:
        return ValueError(message)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise _bad_request(message)


def model_fn(model_dir: str) -> OpenTSLMFlamingo:
    return _load(model_dir)


def input_fn(request_body: str | bytes, content_type: str = "application/json") -> dict[str, Any]:
    """Parse and check a request; a bad request gets a 400 error."""
    _check(content_type.split(";")[0].strip() == "application/json", "use application/json")
    try:
        payload = json.loads(request_body)
    except ValueError:
        raise _bad_request("the request is not valid JSON") from None
    instances = payload.get("instances") if isinstance(payload, dict) else None
    _check(
        isinstance(instances, list) and 1 <= len(instances) <= MAX_INSTANCES,
        f'the request needs an "instances" list of 1 to {MAX_INSTANCES} items',
    )
    parameters = payload.get("parameters") or {}
    _check(
        isinstance(parameters, dict) and set(parameters) <= {"max_new_tokens"},
        '"parameters" can only hold "max_new_tokens"',
    )
    max_new_tokens = parameters.get("max_new_tokens", 256)
    _check(
        type(max_new_tokens) is int and 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT,
        f'"max_new_tokens" must be an integer from 1 to {MAX_NEW_TOKENS_LIMIT}',
    )
    return {"instances": [_instance(item) for item in instances], "max_new_tokens": max_new_tokens}


def _instance(item: Any) -> dict[str, Any]:
    _check(
        isinstance(item, dict) and isinstance(item.get("question"), str) and item["question"],
        'each instance needs a non-empty "question" string',
    )
    series = item.get("series")
    _check(
        isinstance(series, list) and 1 <= len(series) <= MAX_SERIES,
        f'each instance needs a "series" list of 1 to {MAX_SERIES} items',
    )
    parsed = []
    for one in series:
        values = one.get("values") if isinstance(one, dict) else None
        _check(isinstance(values, list) and values, 'each series needs a non-empty "values" list')
        _check(
            all(isinstance(one.get(key), (str, type(None))) for key in ("text", "name")),
            '"text" and "name" must be strings',
        )
        # A value must stay finite as float32, which is how the model stores it.
        _check(
            all(v is None or (type(v) in (int, float) and abs(v) <= _FLOAT32_MAX) for v in values),
            "series values must be finite numbers or null (missing)",
        )
        floats = [math.nan if value is None else float(value) for value in values]
        parsed.append(SeriesInput(values=floats, text=one.get("text"), name=one.get("name")))
    context = item.get("context", [])
    _check(
        isinstance(context, list) and all(isinstance(line, str) for line in context),
        '"context" must be a list of strings',
    )
    return {"series": parsed, "question": item["question"], "context": context}


def predict_fn(data: dict[str, Any], model: OpenTSLMFlamingo) -> list[str]:
    answers = []
    for index, item in enumerate(data["instances"]):
        try:
            answers.append(
                model.generate(
                    item["series"],
                    item["question"],
                    context=item["context"],
                    max_new_tokens=data["max_new_tokens"],
                )
            )
        except ValueError as error:  # the model rejects a series with ValueError
            raise _bad_request(f"instances[{index}]: {error}") from None
    return answers


def output_fn(prediction: list[str], accept: str = "application/json") -> tuple[str, str]:
    return json.dumps({"generated_text": prediction}), "application/json"
