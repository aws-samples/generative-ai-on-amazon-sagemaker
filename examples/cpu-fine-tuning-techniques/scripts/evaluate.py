#!/usr/bin/env python
"""Evaluates a LoRA adapter against its base model on held-out data (SageMaker training job entry point, notebook 01).

Inputs (channels): `adapter` (the training job's adapter/ folder in S3) and `test` (the held-out JSONL).
1. loads the base model named in the adapter's config and scores it on the test requests
2. loads the adapter on top and scores again, on the same requests
3. writes eval_results.json (scores + example outputs) to /opt/ml/model, which SageMaker uploads to S3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from common import Tracker, evaluate_booking, load, read_jsonl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir", default=os.environ.get("SM_CHANNEL_ADAPTER"))
    p.add_argument("--test_dir", default=os.environ.get("SM_CHANNEL_TEST"))
    p.add_argument("--adapter_s3_uri", default="", help="for the record only: where the adapter came from")
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    tracker = Tracker("sft-evaluation", vars(args))
    test = read_jsonl(args.test_dir)
    base_id = json.loads((Path(args.adapter_dir) / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"{len(test)} test examples; base model {base_id}", flush=True)

    tok, model = load(base_id)
    base_metrics, base_out = evaluate_booking(model, tok, test)
    print("base     ", json.dumps(base_metrics), flush=True)
    tracker.log_eval("base", base_metrics)

    model = PeftModel.from_pretrained(model, args.adapter_dir)
    tuned_metrics, tuned_out = evaluate_booking(model, tok, test)
    print("finetuned", json.dumps(tuned_metrics), flush=True)
    tracker.log_eval("finetuned", tuned_metrics)

    results = {"base_model": base_id, "adapter": args.adapter_s3_uri, "test_examples": len(test),
               "base": base_metrics, "finetuned": tuned_metrics, "seconds_total": round(time.time() - t0),
               "examples": [{"request": ex["request"], "target": ex["target"], "base": b, "finetuned": f}
                            for ex, b, f in list(zip(test, base_out, tuned_out))[:5]]}
    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"done in {results['seconds_total']}s", flush=True)
    tracker.close(out, "eval_results.json")


if __name__ == "__main__":
    main()
