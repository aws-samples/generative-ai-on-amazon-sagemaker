"""Shared helpers: dataset loading, model resolution and batched generation.

A "model source" can be any of:
* a Hugging Face model id, for example ``Qwen/Qwen3-0.6B``
* a local directory that already holds a model (``config.json`` present)
* a local directory that holds ``model.tar.gz``, which is how SageMaker
  delivers a training job's output when it is passed in as an input channel
"""

import glob
import json
import os
import tarfile
from typing import Dict, List

# Must match the SYSTEM_PROMPT used for SFT in 02.01 and in task_05 pipeline_utils.py.
SYSTEM_PROMPT = """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response."""


def load_records(path: str) -> List[Dict]:
    """Load a dataset written by the notebooks.

    Accepts a directory or a file, in either the ``to_json(orient="records")`` array
    format used by 02.01 or JSON Lines. Each record has a ``messages`` list with
    system, user and assistant turns. Optional keys: ``id``, ``reference_answer``.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.jsonl")))
        if not files:
            raise FileNotFoundError(f"No .json or .jsonl files in {path}")
        path = files[0]
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    for i, rec in enumerate(records):
        rec.setdefault("id", str(i))
    return records


def split_record(rec: Dict):
    """Return (prompt_messages, question, reference, reference_answer) for one record."""
    msgs = rec["messages"]
    prompt = [m for m in msgs if m["role"] in ("system", "user")]
    question = next(m["content"] for m in msgs if m["role"] == "user")
    reference = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    reference_answer = rec.get("reference_answer")
    if not reference_answer:
        paragraphs = [p for p in reference.split("\n\n") if p.strip()]
        reference_answer = paragraphs[-1] if paragraphs else reference
    return prompt, question, reference, reference_answer


def resolve_model_dir(source: str, workdir: str = "/tmp/models") -> str:
    """Return a local directory containing HF model files for ``source``."""
    if os.path.isdir(source):
        if os.path.exists(os.path.join(source, "config.json")):
            return source
        tars = glob.glob(os.path.join(source, "**", "*.tar.gz"), recursive=True)
        if tars:
            target = os.path.join(workdir, os.path.basename(os.path.normpath(source)) or "model")
            os.makedirs(target, exist_ok=True)
            print(f"Extracting {tars[0]} to {target}")
            with tarfile.open(tars[0]) as tar:
                tar.extractall(target)
            configs = glob.glob(os.path.join(target, "**", "config.json"), recursive=True)
            if not configs:
                raise FileNotFoundError(f"No config.json inside {tars[0]}")
            return os.path.dirname(configs[0])
        raise FileNotFoundError(f"{source} holds neither a model nor a model.tar.gz")

    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    target = os.path.join(workdir, source.replace("/", "_"))
    print(f"Downloading {source} from the Hugging Face Hub to {target}")
    snapshot_download(repo_id=source, local_dir=target)
    return target


def load_for_generation(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"  # decoder-only models must be left-padded for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map="auto")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompts: List[List[Dict]], num_return_sequences: int = 1,
             max_new_tokens: int = 512, temperature: float = 0.6, top_p: float = 0.9,
             batch_size: int = 8, seed: int = 42) -> List[List[str]]:
    """Generate ``num_return_sequences`` completions for each chat prompt."""
    import torch
    from transformers import set_seed

    set_seed(seed)
    outputs: List[List[str]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        # enable_thinking=False: Qwen3 hybrid models (for example Qwen3-0.6B) otherwise open a <think>
        # block. The SFT data has no thinking section, so generation must match that format.
        # Chat templates without this variable ignore it.
        texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for p in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for i in range(len(batch)):
            outputs.append([d.strip() for d in decoded[i * num_return_sequences:(i + 1) * num_return_sequences]])
        print(f"generated {min(start + batch_size, len(prompts))}/{len(prompts)} prompts")
    return outputs


def release_cuda() -> None:
    """Free cached GPU memory. Call it after the caller has deleted its model references."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
