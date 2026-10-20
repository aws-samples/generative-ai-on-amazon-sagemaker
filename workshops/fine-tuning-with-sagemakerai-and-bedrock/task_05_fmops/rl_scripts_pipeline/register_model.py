"""Register a model that passed the pipeline's quality gate in the SageMaker Model Registry.

The package points at the model.tar.gz from the RL training job, uses the same
DJL LMI container and settings as the 02.01 deployment, and attaches the
re-evaluation report as model quality metrics. It is created with
PendingManualApproval by default so a human approves promotion.
"""

import argparse
import json
import os

import boto3

LMI_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128"
LMI_ENV = {
    "HF_MODEL_ID": "/opt/ml/model",
    "OPTION_TRUST_REMOTE_CODE": "true",
    "OPTION_ROLLING_BATCH": "vllm",
    "OPTION_DTYPE": "bf16",
    "OPTION_QUANTIZE": "fp8",
    "OPTION_TENSOR_PARALLEL_DEGREE": "max",
    "OPTION_MAX_ROLLING_BATCH_SIZE": "32",
    "OPTION_MODEL_LOADING_TIMEOUT": "3600",
    "OPTION_MAX_MODEL_LEN": "4096",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-data-url", required=True)
    p.add_argument("--evaluation-s3-uri", required=True, help="S3 URI of evaluation.json")
    p.add_argument("--group-name", required=True)
    p.add_argument("--model-version", default="v1")
    p.add_argument("--approval-status", default="PendingManualApproval")
    p.add_argument("--gate-file", default="/opt/ml/processing/input/eval/evaluation.json")
    p.add_argument("--guardrail-file", default="/opt/ml/processing/input/gate/gate.json")
    p.add_argument("--output-dir", default="/opt/ml/processing/output/registration")
    return p.parse_args()


def main():
    args = parse_args()
    region = os.environ.get("AWS_REGION") or boto3.session.Session().region_name
    sm = boto3.client("sagemaker", region_name=region)

    try:
        sm.describe_model_package_group(ModelPackageGroupName=args.group_name)
    except sm.exceptions.ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=args.group_name,
            ModelPackageGroupDescription="Qwen3-0.6B medical assistant: SFT followed by an RL pass",
        )

    metadata = {"ModelVersion": args.model_version}
    if os.path.exists(args.gate_file):
        with open(args.gate_file, encoding="utf-8") as f:
            gate = json.load(f)["gate"]
        for key in ("primary_metric", "primary_delta", "primary_p_value", "benchmark_delta"):
            metadata[key] = str(gate[key])
    if os.path.exists(args.guardrail_file):
        with open(args.guardrail_file, encoding="utf-8") as f:
            guard = json.load(f)
        metadata["guardrail_violation_rate"] = str(guard["violation_rate"])
        metadata["guardrail_id"] = guard["guardrail"]["id"]
        metadata["guardrail_version"] = str(guard["guardrail"]["version"])

    response = sm.create_model_package(
        ModelPackageGroupName=args.group_name,
        ModelPackageDescription=f"RL-tuned model {args.model_version}",
        ModelApprovalStatus=args.approval_status,
        InferenceSpecification={
            "Containers": [{
                "Image": LMI_IMAGE.format(region=region),
                "ModelDataUrl": args.model_data_url,
                "Environment": LMI_ENV,
            }],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.g5.2xlarge", "ml.g5.4xlarge", "ml.g6.2xlarge"],
        },
        ModelMetrics={"ModelQuality": {"Statistics": {"ContentType": "application/json", "S3Uri": args.evaluation_s3_uri}}},
        CustomerMetadataProperties=metadata,
    )
    arn = response["ModelPackageArn"]
    print(f"Registered {arn} with status {args.approval_status}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "registration.json"), "w", encoding="utf-8") as f:
        json.dump({"model_package_arn": arn, "metadata": metadata}, f, indent=2)


if __name__ == "__main__":
    main()
