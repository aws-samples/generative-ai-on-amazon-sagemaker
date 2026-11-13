"""Fine-tune OpenTSLM on ARFBench in a SageMaker training job.

The job evaluates the pretrained model on the test set, fine-tunes it, evaluates it again, and
saves the fine-tuned model and metrics.json (the test results before and after fine-tuning) to
the S3 prefix ``--output``. With ``MLFLOW_TRACKING_ARN`` set, it also logs the parameters, the
loss of each epoch and the test results to that MLflow App.
"""

import argparse
import json
import os
from pathlib import Path

import boto3
from timenet.client import TimeNet

from utils import OpenTSLM, load_split


def flat_metrics(results, prefix):
    """Give the overall and per-category test results as MLflow metric names and values."""
    metrics = {f"{prefix}_accuracy": results["accuracy"]}
    for category, scores in results["per_category"].items():
        name = "".join(c if c.isalnum() else "_" for c in category)
        metrics[f"{prefix}_accuracy_{name}"] = scores["accuracy"]
        metrics[f"{prefix}_balanced_accuracy_{name}"] = scores["balanced_accuracy"]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="hf://ORG/REPO or a model directory")
    parser.add_argument("--output", required=True, help="s3://BUCKET/PREFIX for the model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--split", default="arfbench-split.json")
    args = parser.parse_args()

    tracking_arn = os.environ.get("MLFLOW_TRACKING_ARN")
    mlflow = None
    if tracking_arn:
        import mlflow

        mlflow.set_tracking_uri(tracking_arn)
        mlflow.set_experiment("opentslm-arfbench")
        mlflow.start_run(run_name=os.environ.get("TRAINING_JOB_NAME"))
        mlflow.log_params(
            {"base_model": args.base_model, "epochs": args.epochs, "output": args.output}
        )

    def log_epoch(record):
        if mlflow:
            values = {k: v for k, v in record.items() if k != "epoch" and v is not None}
            mlflow.log_metrics(values, step=record["epoch"])

    try:
        run(args, mlflow, log_epoch)
    except Exception:
        if mlflow:
            mlflow.end_run(status="FAILED")
        raise


def run(args, mlflow, log_epoch):
    dataset = TimeNet().load("datadog/arfbench")  # the hosted TimeNet registry
    split_ids = json.loads((Path(__file__).parent / args.split).read_text())
    split = load_split(dataset.tasks, split_ids)

    model = OpenTSLM.load(args.base_model)
    before = model.eval(split["test"], metrics=["accuracy"])
    model.fine_tune(split["train"], split["validation"], epochs=args.epochs, log=log_epoch)
    after = model.eval(split["test"], metrics=["accuracy"])
    model.save(args.output)  # the files of a Hugging Face model repository

    results = {"before": before, "after": after}
    print(json.dumps(results), flush=True)
    bucket, _, prefix = args.output.removeprefix("s3://").rstrip("/").partition("/")
    boto3.client("s3").put_object(
        Bucket=bucket, Key=f"{prefix}/metrics.json", Body=json.dumps(results, indent=1)
    )
    if mlflow:
        mlflow.log_metrics({**flat_metrics(before, "test_before"), **flat_metrics(after, "test")})
        mlflow.log_dict(results, "metrics.json")
        mlflow.end_run()


if __name__ == "__main__":
    main()
