#!/usr/bin/env python
"""LoRA fine-tuning + evaluation of a small LLM on CPU, in a few minutes (SageMaker training job entry point).

1. scores the base model on the held-out requests (exact JSON match, per-field accuracy)
2. trains a LoRA adapter for a few dozen steps
3. scores the fine-tuned model on the same requests
4. writes the adapter to /opt/ml/model/adapter and metrics.json next to it
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from booking_data import make_splits, messages  # noqa: E402

FIELDS = ["name", "party_size", "day", "time"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return p.parse_args()


def chat(tok, msgs, generation):
    kw = {"enable_thinking": False} if "qwen3" in tok.name_or_path.lower() else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation, **kw)


def encode(tok, ex):
    prompt = chat(tok, messages(ex["request"]), generation=True)
    full = prompt + ex["target"] + tok.eos_token
    ids = tok(full, add_special_tokens=False).input_ids
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    return ids, [-100] * n_prompt + ids[n_prompt:]


@torch.inference_mode()
def evaluate(model, tok, test, batch_size=16):
    model.eval()
    tok.padding_side = "left"
    exact, field_hits, outputs = 0, {f: 0 for f in FIELDS}, []
    for i in range(0, len(test), batch_size):
        batch = test[i:i + batch_size]
        prompts = [chat(tok, messages(ex["request"]), generation=True) for ex in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        for ex, seq in zip(batch, out[:, enc.input_ids.shape[1]:]):
            text = tok.decode(seq, skip_special_tokens=True).strip()
            outputs.append(text)
            want = json.loads(ex["target"])
            try:
                got = json.loads(text)
            except json.JSONDecodeError:
                got = {}
            exact += got == want
            for f in FIELDS:
                field_hits[f] += isinstance(got, dict) and got.get(f) == want[f]
    n = len(test)
    return {"exact_match": exact / n, **{f"acc_{f}": h / n for f, h in field_hits.items()}}, outputs


def main():
    args = parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    t0 = time.time()
    train, test = make_splits(args.n_train, args.n_test)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    print(f"loaded {args.model_id} in {time.time() - t0:.0f}s on {os.cpu_count()} CPUs", flush=True)

    base_metrics, base_out = evaluate(model, tok, test)
    print("base     ", json.dumps(base_metrics), flush=True)

    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    encoded = [encode(tok, ex) for ex in train]
    model.train()
    t_train = time.time()
    for step in range(1, args.steps + 1):
        batch = [encoded[(step * args.batch_size + j) % len(encoded)] for j in range(args.batch_size)]
        width = max(len(ids) for ids, _ in batch)
        ids = torch.tensor([i + [tok.pad_token_id] * (width - len(i)) for i, _ in batch])
        labels = torch.tensor([l + [-100] * (width - len(l)) for _, l in batch])
        loss = model(input_ids=ids, attention_mask=(ids != tok.pad_token_id) | (labels != -100), labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 5 == 0 or step == 1:
            print(f"[step {step}] loss={loss.item():.4f} sec/step={(time.time() - t_train) / step:.2f}", flush=True)

    tuned_metrics, tuned_out = evaluate(model, tok, test)
    print("finetuned", json.dumps(tuned_metrics), flush=True)

    out = Path(args.model_dir)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    metrics = {"base": base_metrics, "finetuned": tuned_metrics, "steps": args.steps, "model_id": args.model_id,
               "seconds_total": round(time.time() - t0), "seconds_training": round(time.time() - t_train),
               "examples": [{"request": ex["request"], "target": ex["target"], "base": b, "finetuned": f}
                            for ex, b, f in list(zip(test, base_out, tuned_out))[:5]]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done in {metrics['seconds_total']}s (training {metrics['seconds_training']}s)", flush=True)


if __name__ == "__main__":
    main()
