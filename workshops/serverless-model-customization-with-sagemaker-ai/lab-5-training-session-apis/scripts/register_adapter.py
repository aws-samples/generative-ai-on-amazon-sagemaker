"""Register an adapter stored in S3 as a training-session model package in this account.

The adapter prefix must contain checkpoints/hf/adapter_config.json and adapter_model.safetensors,
the layout save_weights_for_sampler writes. Use it to ship a pre-trained adapter and load it with
create_training_client_from_state(...) or create_sampling_client(model_package_arn=...).

  python scripts/register_adapter.py --source s3://bucket/prefix/model/ --dest s3://my-bucket/adapters/x/model/ \
      --group my-group
"""
import argparse
import os
from urllib.parse import urlparse

import boto3

p = argparse.ArgumentParser()
p.add_argument("--source", required=True, help="S3 prefix that contains checkpoints/hf/")
p.add_argument("--dest", default=None, help="copy the adapter here first (same account/region as the package)")
p.add_argument("--group", required=True)
p.add_argument("--base-model", default="huggingface-vlm-qwen3-8-27b")
p.add_argument("--base-version", default="1.3.0")
p.add_argument("--recipe", default=None)
p.add_argument("--description", default="registered adapter")
p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
a = p.parse_args()

s3 = boto3.client("s3", region_name=a.region)
sm = boto3.client("sagemaker", region_name=a.region)

uri = a.source
if a.dest:
    src, dst = urlparse(a.source), urlparse(a.dest)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=src.netloc, Prefix=src.path.lstrip("/")):
        for o in page.get("Contents", []):
            rel = o["Key"][len(src.path.lstrip("/")):]
            s3.copy({"Bucket": src.netloc, "Key": o["Key"]}, dst.netloc, dst.path.lstrip("/") + rel)
    uri = a.dest

try:
    sm.create_model_package_group(ModelPackageGroupName=a.group)
except sm.exceptions.ClientError:
    pass
arn = sm.create_model_package(
    ModelPackageGroupName=a.group,
    ModelPackageDescription=a.description,
    ModelApprovalStatus="Approved",
    InferenceSpecification={"Containers": [{
        "ModelDataSource": {"S3DataSource": {"S3Uri": uri, "S3DataType": "S3Prefix", "CompressionType": "None"}},
        "IsCheckpoint": False,
        "BaseModel": {"HubContentName": a.base_model, "HubContentVersion": a.base_version,
                      **({"RecipeName": a.recipe} if a.recipe else {})},
    }]},
)["ModelPackageArn"]
print(arn)
