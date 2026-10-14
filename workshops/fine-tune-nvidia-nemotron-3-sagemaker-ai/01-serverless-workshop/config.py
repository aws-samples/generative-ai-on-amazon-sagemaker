# Base model for this lab.
#
# Must be a JumpStart model that (a) supports serverless customization and
# (b) can be served by the SageMaker LMI/DJL container and imported into
# Amazon Bedrock Custom Model Import.
#
# Verified end-to-end in this lab: huggingface-reasoning-NVIDIA-Nemotron-3-Nano-30B
#
# Known NOT to work end-to-end: huggingface-vlm-NVIDIA-Nemotron-3-Nano-30B3-5-4b (NVIDIA-Nemotron-3-Nano-30B3.5). Training
# succeeds, but no current LMI container recognises model type `NVIDIA-Nemotron-3-Nano-30B3_5`, so
# notebook 4 fails.
BASE_MODEL_ID = "huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16"

# Fixed dataset / resource names used across the notebooks
DATASET_PREFIX = "contractnli-nda-review"

# Optional explicit ARN always wins. Empty first discovers the tagged stack app,
# then the named workshop app, before native SDK discovery/create. Ambiguity fails.
MLFLOW_APP_ARN = ""
# Optional discovery scope and creation overrides; set both for named creation.
# Role scopes discovery first; artifact URI scopes it only when no role is given.
MLFLOW_ROLE_ARN = ""
MLFLOW_ARTIFACT_STORE_URI = ""
MLFLOW_APP_NAME = "nemotron-contractnli-workshop"
MLFLOW_READY_TIMEOUT_SECONDS = 900


def resolve_mlflow_app(sm_client, *, session=None):
    """Reuse or create an app, preserving the evaluation notebook's one-argument call."""
    from pathlib import Path
    import sys

    workshop_root = str(Path(__file__).resolve().parent.parent)
    if workshop_root not in sys.path:
        sys.path.insert(0, workshop_root)
    from workshop_mlflow import resolve_app

    return resolve_app(
        sm_client, MLFLOW_APP_ARN, session=session, native=True,
        role_arn=MLFLOW_ROLE_ARN, artifact_store_uri=MLFLOW_ARTIFACT_STORE_URI,
        app_name=MLFLOW_APP_NAME, timeout_seconds=MLFLOW_READY_TIMEOUT_SECONDS,
    )
