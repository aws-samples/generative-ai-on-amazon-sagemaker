#!/usr/bin/env python
"""Continued pre-training (CPT) on CPU: teach a small LLM facts from raw text (SageMaker training job entry point).

1. scores the base model on fill-in-the-blank questions about a fictional restaurant (held-out phrasings)
   and its perplexity on held-out sentences
2. trains a LoRA adapter with the plain next-token objective on short raw-text documents (no chat template)
3. scores again and writes the adapter + metrics.json
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
from booking_data import CLOZE, FACTS, TRAIN_SENTENCES, cpt_documents  # noqa: E402
from common import generate, load, save  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return p.parse_args()


def cloze_accuracy(model, tok):
    keys = sorted(CLOZE)
    outs = generate(model, tok, [CLOZE[k] for k in keys], max_new_tokens=16)
    hits = {k: FACTS[k].lower() in o.lower() for k, o in zip(keys, outs)}
    return sum(hits.values()) / len(hits), dict(zip(keys, outs))


@torch.inference_mode()
def perplexity(model, tok, texts):
    model.eval()
    nll, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        nll += model(input_ids=ids, labels=ids).loss.item() * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return math.exp(nll / n)


def main():
    args = parse_args()
    t0 = time.time()
    tok, model = load(args.model_id)
    docs = cpt_documents(phrasings=2)
    heldout = [v[2] for v in TRAIN_SENTENCES.values()]  # third phrasing of each fact: never in training documents

    base_acc, base_out = cloze_accuracy(model, tok)
    base_ppl = perplexity(model, tok, heldout)
    print(f"base      cloze={base_acc:.0%} ppl={base_ppl:.1f}", flush=True)

    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    tok.padding_side = "right"
    model.train()
    t_train = time.time()
    for step in range(1, args.steps + 1):
        batch = [docs[(step * args.batch_size + j) % len(docs)] + tok.eos_token for j in range(args.batch_size)]
        enc = tok(batch, return_tensors="pt", padding=True)
        labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
        loss = model(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 10 == 0 or step == 1:
            print(f"[step {step}] loss={loss.item():.4f} sec/step={(time.time() - t_train) / step:.2f}", flush=True)

    tuned_acc, tuned_out = cloze_accuracy(model, tok)
    tuned_ppl = perplexity(model, tok, heldout)
    print(f"finetuned cloze={tuned_acc:.0%} ppl={tuned_ppl:.1f}", flush=True)
    save(args.model_dir, model, tok, {
        "technique": "cpt", "model_id": args.model_id, "steps": args.steps,
        "base": {"cloze_accuracy": base_acc, "heldout_perplexity": base_ppl},
        "finetuned": {"cloze_accuracy": tuned_acc, "heldout_perplexity": tuned_ppl},
        "seconds_training": round(time.time() - t_train),
        "examples": [{"question": CLOZE[k], "fact": FACTS[k], "base": base_out[k], "finetuned": tuned_out[k]}
                     for k in sorted(CLOZE)[:4]],
    }, t0)


if __name__ == "__main__":
    main()
