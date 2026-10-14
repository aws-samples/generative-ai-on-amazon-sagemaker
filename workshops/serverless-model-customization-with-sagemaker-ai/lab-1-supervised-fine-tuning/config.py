# Base model for this lab.
#
# Must be a JumpStart model that (a) supports serverless customization and
# (b) can be served by the SageMaker LMI/DJL container and imported into
# Amazon Bedrock Custom Model Import.
#
# Verified end-to-end in this lab: huggingface-reasoning-qwen3-4b
#
# Known NOT to work end-to-end: huggingface-vlm-qwen3-5-4b (Qwen3.5). Training
# succeeds, but no current LMI container recognises model type `qwen3_5`, so
# notebooks 4 and 4a fail.
BASE_MODEL_ID = "huggingface-reasoning-qwen3-4b"

# Fixed dataset / resource names used across the notebooks
DATASET_PREFIX = "contractnli-nda-review"


# Optional explicit ARN wins. Otherwise discover the tagged stack app before
# the named workshop app and native SDK fallback. Ambiguity stops setup.
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
