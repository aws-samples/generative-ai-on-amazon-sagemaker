"""OpenVLA prompt/action helpers, matching the conventions of the released checkpoint.

These mirror `OpenVLAForActionPrediction.predict_action` in the model's remote code
(modeling_prismatic.py on the Hub) and `prismatic/vla/action_tokenizer.py` upstream:

    * 256 uniform bins over [-1, 1]
    * token_id = vocab_size - np.digitize(action, bins)       (vocab_size = 32000)
    * the prompt is followed by token 29871 ("▁") before the 7 action tokens
"""
from typing import List

import numpy as np

ACTION_DIM = 7
N_BINS = 256
SPACE_TOKEN_ID = 29871


def build_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower().strip()}?\nOut:"


class ActionTokenizer:
    def __init__(self, tokenizer, n_bins: int = N_BINS, min_action: float = -1.0, max_action: float = 1.0):
        self.tokenizer = tokenizer
        self.min_action, self.max_action = min_action, max_action
        self.bins = np.linspace(min_action, max_action, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        # The checkpoint's tokenizer has 32000 base tokens + <PAD>; action tokens are the top 256 base ids.
        self.vocab_size = tokenizer.vocab_size
        self.action_token_begin_idx = self.vocab_size - (n_bins + 1)

    def encode(self, action) -> List[int]:
        action = np.clip(np.asarray(action, dtype=np.float64), self.min_action, self.max_action)
        return (self.vocab_size - np.digitize(action, self.bins)).tolist()

    def decode(self, token_ids) -> np.ndarray:
        discretized = self.vocab_size - np.asarray(token_ids)
        discretized = np.clip(discretized - 1, 0, self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized]


def build_training_ids(tokenizer, action_tokenizer, instruction: str, action):
    """Returns (input_ids, labels) for one sample; only action tokens + EOS are supervised."""
    prompt_ids = tokenizer(build_prompt(instruction), add_special_tokens=True).input_ids + [SPACE_TOKEN_ID]
    target_ids = action_tokenizer.encode(action) + [tokenizer.eos_token_id]
    return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids


def build_inference_ids(tokenizer, instruction: str) -> List[int]:
    return tokenizer(build_prompt(instruction), add_special_tokens=True).input_ids + [SPACE_TOKEN_ID]


def predict_action(model, processor, action_tokenizer, image, instruction: str) -> np.ndarray:
    """Greedy-decodes ACTION_DIM action tokens and returns the normalized action in [-1, 1]."""
    import torch

    device = next(model.parameters()).device
    input_ids = torch.tensor([build_inference_ids(processor.tokenizer, instruction)], device=device)
    pixel_values = processor.image_processor.apply_transform(image.convert("RGB"))[None].to(device, torch.bfloat16)
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             pixel_values=pixel_values, max_new_tokens=ACTION_DIM, do_sample=False)
    return action_tokenizer.decode(out[0, -ACTION_DIM:].cpu().numpy())


def load_openvla(path, device):
    """Loads a fine-tuned OpenVLA from `path`: either a merged checkpoint (config.json + weights) or a LoRA adapter
    folder (adapter_config.json), in which case the base model named in the adapter config is downloaded from the
    Hugging Face Hub and the adapter is merged into it on `device`. Returns (model, processor)."""
    import json
    import os

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    kw = dict(torch_dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True, trust_remote_code=True)
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        return AutoModelForVision2Seq.from_pretrained(path, **kw).to(device).eval(), \
            AutoProcessor.from_pretrained(path, trust_remote_code=True)
    from peft import PeftModel

    base = json.load(open(adapter_cfg))["base_model_name_or_path"]
    model = AutoModelForVision2Seq.from_pretrained(base, **kw).to(device)
    model = PeftModel.from_pretrained(model, path).merge_and_unload().eval()
    return model, AutoProcessor.from_pretrained(base, trust_remote_code=True)
