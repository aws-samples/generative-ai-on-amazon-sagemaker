# Shared settings for notebooks 1-5.
#
# RUN_MODE "smoke" trains 20 steps on one L40S GPU to prove the pipeline end to end; "full" trains 10 epochs on
# 1-4 L40S GPUs (global batch 32: ~170 optimizer steps for the 300-episode sample).
RUN_MODE = "smoke"

BASE_MODEL_ID = "openvla/openvla-7b"
PROJECT = "openvla-lora"
DATASET_PREFIX = "openvla-bridge"
MAX_EPISODES = 600            # of 8,802 in VyoJ/BridgeData-V2-Scripted-Images (capped by what the source holds)
# Where notebook 1 reads the source episodes from:
#   "sample_data/bridge-300" -> 300-episode Parquet sample shipped with the lab, 19 MB (540 train / 60 val;
#                               CC BY 4.0, see sample_data/README.md)
#   "<folder>"               -> Parquet shards you downloaded yourself (0000.parquet = 4,200 episodes)
#   None           -> download the shards from the Hugging Face Hub (log in with a token if you get HTTP 429)
SOURCE_PARQUET_DIR = "sample_data/bridge-300"
VAL_FRACTION = 0.1

# Training instance types, tried in order when CAPACITY = "fallback" (SageMaker accepts at most 5). All are L40S
# 48 GB, so the recipe holds whichever type SageMaker picks: gradient accumulation adapts to the GPU count, so
# the full run also works on a single GPU (8 accumulation steps instead of 2), just more slowly.
TRAIN = {
    "smoke": {"instance_types": ["ml.g6e.xlarge", "ml.g6e.2xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge"],
              "recipe": "recipes/openvla_lora_smoke.yaml", "max_runtime_h": 2},
    "full": {"instance_types": ["ml.g6e.12xlarge", "ml.g6e.24xlarge", "ml.g6e.xlarge", "ml.g6e.2xlarge",
                                "ml.g6e.4xlarge"],
             "recipe": "recipes/openvla_lora.yaml", "max_runtime_h": 10},
}[RUN_MODE]
EVAL_INSTANCE_TYPES = ["ml.g6e.xlarge", "ml.g6e.2xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge"]

# How training and eval jobs get a GPU:
#   "fallback"  (default) on-demand, first type in the list with capacity. You need a
#               "<type> for training job usage" quota for the types you want SageMaker to use.
#   "on-demand" first type only
#   "spot"      managed Spot on the first type: cheaper, but can queue for hours (5.5 h observed for
#               ml.g6e.xlarge in us-west-2), so use it for self-paced runs, not live sessions.
#               Waits up to SPOT_MAX_WAIT_H; resumes from /opt/ml/checkpoints after an interruption.
CAPACITY = "fallback"
SPOT_MAX_WAIT_H = 24

# Pre-trained artifacts for notebook 2b (skip training): an S3 prefix holding hf_merged/ (and adapter/) from
# a previous run, readable by your role. See the README for how to publish one.
PRETRAINED_S3_URI = ""
# Endpoint instance pools in priority order: SageMaker falls back to the next type when one has no capacity.
# Each needs >= 24 GB GPU memory for the 15 GB bf16 model and an "... for endpoint usage" quota of >= 1.
# CreateEndpointConfig accepts at most 5 pools (priorities 1-5).
ENDPOINT_INSTANCE_TYPES = ["ml.g6.2xlarge", "ml.g5.2xlarge", "ml.g6e.xlarge", "ml.g5.4xlarge", "ml.g6e.2xlarge"]

# PyTorch 2.7 training DLC (as in the blog) and PyTorch inference DLC; {region} is filled in by the notebooks.
TRAINING_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.7-gpu-py312"
INFERENCE_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.6-gpu-py312"

MODEL_PACKAGE_GROUP = f"{PROJECT}-bridge-mpg"
MLFLOW_EXPERIMENT = f"{PROJECT}-bridge"

# MLflow: notebooks 2-3 can override both. With MLFLOW_APP_ARN empty they reuse the App created by the
# serverless-customization workshop stack (same resolver as that lab), or create one if none exists.
USE_MLFLOW = True
MLFLOW_APP_ARN = ""
MLFLOW_ROLE_ARN = ""
MLFLOW_ARTIFACT_STORE_URI = ""
MLFLOW_APP_NAME = "serverless-customization-workshop"
MLFLOW_READY_TIMEOUT_SECONDS = 900


def resolve_mlflow_app(sm_client, *, session=None, arn=None):
    """Uses `arn` (or MLFLOW_APP_ARN) if set; otherwise reuses the workshop's app or creates one."""
    from workshop_mlflow import resolve_app

    return resolve_app(
        sm_client, MLFLOW_APP_ARN if arn is None else arn, session=session, native=True,
        role_arn=MLFLOW_ROLE_ARN, artifact_store_uri=MLFLOW_ARTIFACT_STORE_URI,
        app_name=MLFLOW_APP_NAME, timeout_seconds=MLFLOW_READY_TIMEOUT_SECONDS,
    )


def s3_root(bucket, default_prefix):
    return f"s3://{bucket}/{default_prefix}/{PROJECT}" if default_prefix else f"s3://{bucket}/{PROJECT}"
