#!/usr/bin/env python
"""Group Relative Policy Optimization (GRPO) on CPU with TRL (SageMaker training job entry point).

Same booking task, but there are no target answers to imitate: for each request the model generates a group of
answers, a verifiable reward scores them (bare JSON with the right fields = 1; partial credit for right fields in
the wrong format), and the policy moves towards the answers that scored above their group's average.

1. scores the base model (exact JSON match, mean reward)
2. trains a LoRA adapter with GRPOTrainer (sampling on CPU, no vLLM, no KL term so no reference model)
3. scores again and writes the adapter + metrics.json
"""
import argparse
import os
import sys
import time
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from booking_data import booking_reward, make_splits  # noqa: E402
from common import booking_prompt, evaluate_booking, load, save  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--prompts_per_step", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return p.parse_args()


def reward(completions, target, **_):
    return [booking_reward(c, t) for c, t in zip(completions, target)]


def mean_reward(outputs, test):
    return sum(booking_reward(o, ex["target"]) for o, ex in zip(outputs, test)) / len(test)


def main():
    args = parse_args()
    t0 = time.time()
    train, test = make_splits(n_train=400, n_test=30)
    tok, model = load(args.model_id)

    base_metrics, base_out = evaluate_booking(model, tok, test)
    base_metrics["mean_reward"] = mean_reward(base_out, test)
    print("base     ", base_metrics, flush=True)

    ds = Dataset.from_list([{"prompt": booking_prompt(tok, ex["request"]), "target": ex["target"]} for ex in train])
    cfg = GRPOConfig(output_dir="/tmp/grpo", max_steps=args.steps, learning_rate=args.learning_rate,
                     per_device_train_batch_size=args.prompts_per_step * args.num_generations,
                     num_generations=args.num_generations, max_completion_length=48, temperature=1.0, beta=0.0,
                     logging_steps=1, save_strategy="no", report_to="none", use_cpu=True, dataloader_num_workers=0,
                    # TRL defaults to bf16 + gradient checkpointing: both slow on CPU, and not needed for a 0.6B LoRA
                    bf16=False, gradient_checkpointing=False)
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=cfg, train_dataset=ds, processing_class=tok,
                          peft_config=LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    t_train = time.time()
    trainer.train()
    model = trainer.model

    tuned_metrics, tuned_out = evaluate_booking(model, tok, test)
    tuned_metrics["mean_reward"] = mean_reward(tuned_out, test)
    print("finetuned", tuned_metrics, flush=True)
    save(args.model_dir, model, tok, {
        "technique": "grpo", "model_id": args.model_id, "steps": args.steps,
        "base": base_metrics, "finetuned": tuned_metrics, "seconds_training": round(time.time() - t_train),
        "reward_curve": [{"step": e.get("step"), "reward": e.get("reward")} for e in trainer.state.log_history
                         if "reward" in e],
        "examples": [{"request": ex["request"], "target": ex["target"], "base": b, "finetuned": f}
                     for ex, b, f in list(zip(test, base_out, tuned_out))[:3]],
    }, t0)


if __name__ == "__main__":
    main()
