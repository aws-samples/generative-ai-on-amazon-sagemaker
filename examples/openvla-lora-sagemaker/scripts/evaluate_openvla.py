#!/usr/bin/env python
"""Closed-form action evaluation of base vs fine-tuned OpenVLA on the validation split (SageMaker job).

Channels:
    data   -> HF DatasetDict from notebook 1 (uses the `validation` split)
    model  -> hf_merged/ prefix written by the training job, or a LoRA adapter folder (base model from the Hub)
Writes /opt/ml/model/eval_results.json and logs to MLflow when MLFLOW_TRACKING_URI is set.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from openvla_utils import ActionTokenizer, load_openvla, predict_action  # noqa: E402

AXES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def score(preds, gts, action_tokenizer):
    err = np.abs(preds - gts)
    same_bin = np.digitize(preds, action_tokenizer.bins) == np.digitize(gts, action_tokenizer.bins)
    out = {"l1_mean": float(err.mean()), "l1_median": float(np.median(err.mean(1))),
           "l2_mean": float(np.sqrt((err ** 2).mean(1)).mean()), "bin_accuracy": float(same_bin.mean())}
    out.update({f"l1_{a}": float(err[:, i].mean()) for i, a in enumerate(AXES)})
    return out


def run_model(path, processor, action_tokenizer, rows):
    model, _ = load_openvla(path, DEVICE)   # merged checkpoint, adapter folder, or a Hub id (the base model)
    preds = np.stack([predict_action(model, processor, action_tokenizer, r["image"], r["instruction"]) for r in rows])
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="openvla/openvla-7b")
    p.add_argument("--data_dir", default=os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data"))
    p.add_argument("--tuned_model_dir", default=os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model"))
    p.add_argument("--output_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--max_samples", type=int, default=0)
    args, _ = p.parse_known_args()

    rows = load_from_disk(args.data_dir)["validation"]
    if args.max_samples and args.max_samples < len(rows):
        # evenly spaced, so every held-out episode/task is represented
        rows = rows.select(np.linspace(0, len(rows) - 1, args.max_samples).round().astype(int).tolist())
    gts = np.array(rows["action"], dtype=np.float64)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    results = {"num_samples": len(rows),
               # trivial reference: always predict the mean action of the evaluated frames (a label prior)
               "mean_action_baseline": score(np.zeros_like(gts) + gts.mean(0), gts, action_tokenizer)}
    for name, path in [("base", args.base_model), ("finetuned", args.tuned_model_dir)]:
        print(f"evaluating {name}: {path}", flush=True)
        results[name] = score(run_model(path, processor, action_tokenizer, rows), gts, action_tokenizer)
        print(name, json.dumps(results[name], indent=1), flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            import mlflow

            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "openvla-lora"))
            with mlflow.start_run(run_name=os.environ.get("TRAINING_JOB_NAME", "openvla-eval")):
                for name in ("base", "finetuned", "mean_action_baseline"):
                    mlflow.log_metrics({f"{name}/{k}": v for k, v in results[name].items()})
        except Exception as e:
            print(f"[mlflow] skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
