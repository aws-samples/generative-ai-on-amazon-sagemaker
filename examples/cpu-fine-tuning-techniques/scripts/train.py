#!/usr/bin/env python
"""SFT with LoRA on CPU (SageMaker training job entry point, notebook 01). Training only: the separate evaluation
job (evaluate.py) scores the result.

1. reads the training examples from the `train` channel (JSONL the notebook uploaded to S3; SageMaker copies it to
   /opt/ml/input/data/train/); without a channel it generates the same data
2. trains a LoRA adapter for a few dozen steps, logging the loss (job log, and MLflow when enabled)
3. writes the adapter to /opt/ml/model/adapter and training_metrics.json next to it; SageMaker uploads
   /opt/ml/model to S3 when the job ends
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
from booking_data import make_splits, messages  # noqa: E402
from common import Tracker, chat, load, read_jsonl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--train_dir", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return p.parse_args()


def encode(tok, ex):
    """Token ids of prompt + answer; the loss is computed on the answer tokens only (prompt labels = -100)."""
    prompt = chat(tok, messages(ex["request"]), generation=True)
    ids = tok(prompt + ex["target"] + tok.eos_token, add_special_tokens=False).input_ids
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    return ids, [-100] * n_prompt + ids[n_prompt:]


def main():
    args = parse_args()
    t0 = time.time()
    tracker = Tracker("sft", vars(args))
    if args.train_dir:
        train = read_jsonl(args.train_dir)
        print(f"data from channel: {len(train)} train examples ({args.train_dir})", flush=True)
    else:
        train, _ = make_splits(args.n_train)
        print(f"no data channel: generated {len(train)} train examples", flush=True)

    tok, model = load(args.model_id)
    print(f"loaded {args.model_id} in {time.time() - t0:.0f}s on {os.cpu_count()} CPUs", flush=True)
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    encoded = [encode(tok, ex) for ex in train]
    model.train()
    t_train, losses = time.time(), []
    for step in range(1, args.steps + 1):
        batch = [encoded[(step * args.batch_size + j) % len(encoded)] for j in range(args.batch_size)]
        width = max(len(ids) for ids, _ in batch)
        ids = torch.tensor([i + [tok.pad_token_id] * (width - len(i)) for i, _ in batch])
        labels = torch.tensor([l + [-100] * (width - len(l)) for _, l in batch])
        loss = model(input_ids=ids, attention_mask=(ids != tok.pad_token_id) | (labels != -100), labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(round(loss.item(), 4))
        tracker.log({"loss": loss.item()}, step=step)
        if step % 5 == 0 or step == 1:
            print(f"[step {step}] loss={loss.item():.4f} sec/step={(time.time() - t_train) / step:.2f}", flush=True)

    out = Path(args.model_dir)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {"technique": "sft", "model_id": args.model_id, "steps": args.steps, "train_examples": len(train),
               "trainable_parameters": trainable, "loss": losses,
               "seconds_training": round(time.time() - t_train), "seconds_total": round(time.time() - t0)}
    (out / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done in {metrics['seconds_total']}s (training {metrics['seconds_training']}s)", flush=True)
    tracker.close(out, "training_metrics.json")


if __name__ == "__main__":
    main()
