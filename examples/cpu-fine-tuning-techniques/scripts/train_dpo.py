#!/usr/bin/env python
"""Direct Preference Optimization (DPO) on CPU with TRL (SageMaker training job entry point).

Same booking task as SFT, but the model learns from preference pairs instead of targets: for each request, a
preferred answer (exact JSON) and a rejected one (the base model's usual markdown-wrapped answer, or a wrong
value). No reward model and no separate reference model: with a LoRA adapter, TRL uses the base model (adapter
disabled) as the reference.

1. scores the base model (exact JSON match) and how often it already prefers the chosen answer
2. trains a LoRA adapter with DPOTrainer
3. scores again and writes the adapter + metrics.json
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from booking_data import dpo_pairs, make_splits  # noqa: E402
from common import Tracker, booking_prompt, evaluate_booking, load, save  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--loss_type", default="sigmoid+sft", help='"sigmoid" = plain DPO; "sigmoid+sft" adds an SFT term')
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return p.parse_args()


@torch.inference_mode()
def preference_accuracy(model, tok, pairs):
    """Share of pairs where the model gives the chosen answer a higher log-likelihood than the rejected one."""
    model.eval()

    def logp(prompt, answer):
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + tok(answer + tok.eos_token, add_special_tokens=False).input_ids])
        logits = model(input_ids=ids).logits[0, len(p_ids) - 1:-1]
        return torch.log_softmax(logits, -1).gather(1, ids[0, len(p_ids):, None]).sum().item()

    wins = [logp(booking_prompt(tok, p["request"]), p["chosen"]) > logp(booking_prompt(tok, p["request"]), p["rejected"])
            for p in pairs]
    return sum(wins) / len(wins)


def main():
    args = parse_args()
    t0 = time.time()
    tracker = Tracker("dpo", vars(args))
    train, test = make_splits(n_train=400, n_test=30)
    tok, model = load(args.model_id)
    test_pairs = dpo_pairs(test, seed=1)

    base_metrics, base_out = evaluate_booking(model, tok, test)
    base_metrics["preference_accuracy"] = preference_accuracy(model, tok, test_pairs)
    print("base     ", base_metrics, flush=True)
    tracker.log_eval("base", base_metrics)

    # Standard (string) format with the chat template applied once, exactly as at inference time
    ds = Dataset.from_list([{"prompt": booking_prompt(tok, p["request"]), "chosen": p["chosen"] + tok.eos_token,
                             "rejected": p["rejected"] + tok.eos_token} for p in dpo_pairs(train)])
    cfg = DPOConfig(output_dir="/tmp/dpo", max_steps=args.steps, per_device_train_batch_size=args.batch_size,
                    learning_rate=args.learning_rate, beta=args.beta, logging_steps=5, save_strategy="no",
                    # DPO alone only widens the chosen-vs-rejected gap; it can do that by drifting away from both
                    # (here: dropping JSON altogether). Adding an SFT term on the chosen answer keeps it likely.
                    loss_type=args.loss_type.split("+"), loss_weights=[1.0] * len(args.loss_type.split("+")),
                    report_to="none", use_cpu=True, max_length=256, dataloader_num_workers=0,
                    # TRL defaults to bf16 + gradient checkpointing: both slow on CPU, and not needed for a 0.6B LoRA
                    bf16=False, gradient_checkpointing=False)
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    t_train = time.time()
    trainer.train()
    model = trainer.model
    tracker.log_history(trainer.state.log_history)

    tuned_metrics, tuned_out = evaluate_booking(model, tok, test)
    tuned_metrics["preference_accuracy"] = preference_accuracy(model, tok, test_pairs)
    print("finetuned", tuned_metrics, flush=True)
    tracker.log_eval("finetuned", tuned_metrics)
    save(args.model_dir, model, tok, {
        "technique": "dpo", "model_id": args.model_id, "steps": args.steps, "beta": args.beta,
        "loss_type": args.loss_type,
        "base": base_metrics, "finetuned": tuned_metrics, "seconds_training": round(time.time() - t_train),
        "log": [{k: v for k, v in e.items() if k in ("step", "loss", "rewards/accuracies", "rewards/margins")}
                for e in trainer.state.log_history if "loss" in e],
        "examples": [{"request": ex["request"], "target": ex["target"], "base": b, "finetuned": f}
                     for ex, b, f in list(zip(test, base_out, tuned_out))[:3]],
    }, t0)
    tracker.close(args.model_dir)


if __name__ == "__main__":
    main()
