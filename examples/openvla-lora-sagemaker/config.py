# Shared settings for notebooks 1-5.
#
# RUN_MODE "smoke" trains 20 steps on one GPU (ml.g6e.xlarge) to prove the pipeline end to end;
# "full" trains 10 epochs on ml.g6e.12xlarge (4x L40S, global batch 32: ~340 optimizer steps for 600 episodes).
# ml.g6e.48xlarge (8x L40S, the blog's instance) also works with recipes/openvla_lora.yaml if you have quota.
RUN_MODE = "smoke"

BASE_MODEL_ID = "openvla/openvla-7b"
PROJECT = "openvla-lora"
DATASET_PREFIX = "openvla-bridge"
MAX_EPISODES = 600            # of 8,802 in VyoJ/BridgeData-V2-Scripted-Images (capped by what the source holds)
# Where notebook 1 reads the source episodes from:
#   "sample_data/bridge-60" -> 60-episode Parquet sample shipped with the lab, 7 MB (smoke test; CC-BY-4.0,
#                              see sample_data/README.md)
#   "<folder>"              -> Parquet shards you downloaded yourself (0000.parquet = 4,200 episodes)
#   None           -> download the shards from the Hugging Face Hub (log in with a token if you get HTTP 429)
SOURCE_PARQUET_DIR = "sample_data/bridge-60"
VAL_FRACTION = 0.1

TRAIN = {
    "smoke": {"instance_type": "ml.g6e.xlarge", "recipe": "recipes/openvla_lora_smoke.yaml", "max_runtime_h": 2},
    "full": {"instance_type": "ml.g6e.12xlarge", "recipe": "recipes/openvla_lora.yaml", "max_runtime_h": 10},
}[RUN_MODE]
# Managed Spot for training and eval jobs. Jobs wait up to SPOT_MAX_WAIT_H for capacity (queueing counts toward
# it) and resume from /opt/ml/checkpoints after an interruption. Needs the "... for spot training job usage" quota.
USE_SPOT = True
SPOT_MAX_WAIT_H = 24
EVAL_INSTANCE_TYPE = "ml.g6e.xlarge"
# Endpoint instance pools in priority order: SageMaker falls back to the next type when one has no capacity.
# Each needs >= 24 GB GPU memory for the 15 GB bf16 model and an "... for endpoint usage" quota of >= 1.
# CreateEndpointConfig accepts at most 5 pools (priorities 1-5).
ENDPOINT_INSTANCE_TYPES = ["ml.g6.2xlarge", "ml.g5.2xlarge", "ml.g6e.xlarge", "ml.g5.4xlarge", "ml.g6e.2xlarge"]

# PyTorch 2.7 training DLC (as in the blog) and PyTorch inference DLC; {region} is filled in by the notebooks.
TRAINING_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.7-gpu-py312"
INFERENCE_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.6-gpu-py312"

MODEL_PACKAGE_GROUP = f"{PROJECT}-bridge-mpg"
MLFLOW_EXPERIMENT = f"{PROJECT}-bridge"

# MLflow: reuse the App created by the serverless-customization workshop stack (same resolver as that lab).
MLFLOW_APP_ARN = ""
MLFLOW_ROLE_ARN = ""
MLFLOW_ARTIFACT_STORE_URI = ""
MLFLOW_APP_NAME = "serverless-customization-workshop"
MLFLOW_READY_TIMEOUT_SECONDS = 900


def resolve_mlflow_app(sm_client, *, session=None):
    """Reuse or create an app through the shared, session-independent resolver."""
    from workshop_mlflow import resolve_app

    return resolve_app(
        sm_client, MLFLOW_APP_ARN, session=session, native=True,
        role_arn=MLFLOW_ROLE_ARN, artifact_store_uri=MLFLOW_ARTIFACT_STORE_URI,
        app_name=MLFLOW_APP_NAME, timeout_seconds=MLFLOW_READY_TIMEOUT_SECONDS,
    )


def s3_root(bucket, default_prefix):
    return f"s3://{bucket}/{default_prefix}/{PROJECT}" if default_prefix else f"s3://{bucket}/{PROJECT}"
