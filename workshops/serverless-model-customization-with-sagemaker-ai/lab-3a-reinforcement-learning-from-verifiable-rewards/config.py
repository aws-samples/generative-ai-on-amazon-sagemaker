# Workshop experiment tracking configuration.


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
