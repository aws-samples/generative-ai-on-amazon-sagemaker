# Shared settings for notebooks 1-5.
#
# RUN_MODE "smoke" trains 20 steps on one L40S GPU to prove the pipeline end to end; "full" trains 3 epochs on
# 1-4 L40S GPUs (global batch 32: ~255 optimizer steps on the LIBERO sample; validation stops improving after ~3).
RUN_MODE = "smoke"

BASE_MODEL_ID = "openvla/openvla-7b"
PROJECT = "openvla-lora"
# Training data (notebook 1):
#   "libero" (default): LIBERO simulated tabletop demos, real 7-DoF actions, 40 tasks/instructions. OpenVLA was
#                       not pretrained on LIBERO, so fine-tuning shows a clear gain (libero_data.py).
#   "bridge":           BridgeData V2 images with optical-flow pseudo-actions and one instruction, as in the AWS
#                       blog (bridge_data.py); the model mostly learns the label prior.
DATASET = "libero"
DATASET_PREFIX = f"openvla-{DATASET}"
# Where notebook 1 reads the source data from:
#   the shipped sample (default; no Hugging Face access needed), None to download from the Hub instead (log in
#   with a token if you get HTTP 429), or a folder of Parquet files you prepared yourself.
SOURCE_PARQUET_DIR = {"libero": "sample_data/libero-120", "bridge": "sample_data/bridge-300"}[DATASET]
MAX_EPISODES = 600            # bridge only: episodes to use (capped by what the source holds)
VAL_FRACTION = 0.1            # bridge only; libero holds out one demonstration per task
EVAL_MAX_SAMPLES = 400        # validation frames scored by notebook 3 (evenly spaced over the split)

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
#   "fallback"  (default) on-demand, first type in the list with capacity. SageMaker rejects the job if
#               any listed type has a "<type> for training job usage" quota of 0, so the notebooks drop
#               types without quota before submitting (job_utils.with_quota).
#   "on-demand" first type only
#   "spot"      managed Spot on the first type: cheaper, but can queue for hours (5.5 h observed for
#               ml.g6e.xlarge in us-west-2), so use it for self-paced runs, not live sessions.
#               Waits up to SPOT_MAX_WAIT_H; resumes from /opt/ml/checkpoints after an interruption.
CAPACITY = "fallback"
SPOT_MAX_WAIT_H = 24

# Pre-trained model for notebook 2b (skip training). Default: the LoRA adapter shipped in this repo
# (pretrained/libero-r8/, ~40 MB; the base model comes from the Hugging Face Hub). Alternatively an S3 prefix
# holding hf_merged/ or adapter/ from a previous run, readable by your role.
PRETRAINED_LOCAL_DIR = "pretrained/libero-r8"
PRETRAINED_S3_URI = ""
# Endpoint instance pools in priority order: SageMaker falls back to the next type when one has no capacity.
# Each needs >= 24 GB GPU memory for the 15 GB bf16 model and an "... for endpoint usage" quota of >= 1.
# CreateEndpointConfig accepts at most 5 pools (priorities 1-5).
ENDPOINT_INSTANCE_TYPES = ["ml.g6.2xlarge", "ml.g5.2xlarge", "ml.g6e.xlarge", "ml.g5.4xlarge", "ml.g6e.2xlarge"]

# PyTorch 2.7 training DLC (as in the blog) and PyTorch inference DLC; {region} is filled in by the notebooks.
TRAINING_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.7-gpu-py312"
INFERENCE_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.6-gpu-py312"

MODEL_PACKAGE_GROUP = f"{PROJECT}-{DATASET}-mpg"
MLFLOW_EXPERIMENT = f"{PROJECT}-{DATASET}"

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
