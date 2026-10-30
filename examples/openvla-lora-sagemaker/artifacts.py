"""Model artifacts of a training run (notebook 2).

A training run's output prefix holds:
    adapter/            the LoRA adapter
    hf_merged/          adapter merged into the base model (what the endpoint and the eval job load)
`register_model` stores the serving code next to the model (a merged checkpoint, or just a LoRA adapter whose base
model is then downloaded from the Hugging Face Hub) and registers a model package, which is what notebooks 3 and 4
start from.
"""
import boto3

import config

SERVING_FILES = [("serving/inference.py", "inference.py"), ("serving/requirements.txt", "requirements.txt"),
                 ("scripts/openvla_utils.py", "openvla_utils.py")]


def split_s3(uri):
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def register_model(model_s3_uri, *, source, run_mode, region):
    """Uploads the serving handler to <model>/code/ and registers a PendingManualApproval model package.
    model_s3_uri: a merged checkpoint (hf_merged/) or a LoRA adapter folder (adapter_config.json + weights)."""
    s3 = boto3.client("s3", region_name=region)
    sm = boto3.client("sagemaker", region_name=region)
    bucket, key = split_s3(model_s3_uri)
    names = {o["Key"].rsplit("/", 1)[-1] for o in s3.list_objects_v2(Bucket=bucket, Prefix=key).get("Contents", [])}
    if "adapter_config.json" not in names and not {"config.json", "modeling_prismatic.py"} <= names:
        raise RuntimeError(f"{model_s3_uri} holds neither a merged OpenVLA checkpoint nor a LoRA adapter")

    for local, name in SERVING_FILES:
        s3.upload_file(local, bucket, f"{key}code/{name}")

    try:
        sm.describe_model_package_group(ModelPackageGroupName=config.MODEL_PACKAGE_GROUP)
    except sm.exceptions.ClientError:
        sm.create_model_package_group(ModelPackageGroupName=config.MODEL_PACKAGE_GROUP,
                                      ModelPackageGroupDescription=f"OpenVLA-7B LoRA, {config.DATASET} dataset")

    return sm.create_model_package(
        ModelPackageGroupName=config.MODEL_PACKAGE_GROUP,
        ModelPackageDescription=f"{config.BASE_MODEL_ID} + LoRA ({run_mode}) from {source}"[:1000],
        ModelApprovalStatus="PendingManualApproval",
        CustomerMetadataProperties={"source": source[:256], "run_mode": run_mode, "model_s3_uri": model_s3_uri},
        InferenceSpecification={
            "Containers": [{
                "Image": config.INFERENCE_IMAGE.format(region=region),
                "ModelDataSource": {"S3DataSource": {"S3Uri": model_s3_uri, "S3DataType": "S3Prefix",
                                                     "CompressionType": "None"}},
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                    "SAGEMAKER_TS_RESPONSE_TIMEOUT": "900",
                    "SAGEMAKER_MODEL_SERVER_TIMEOUT": "900",
                },
            }],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": config.ENDPOINT_INSTANCE_TYPES,
        },
    )
