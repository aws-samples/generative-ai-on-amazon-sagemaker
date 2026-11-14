"""MLflow App setup shared by the two notebook tracks."""

import re
import time


def _wait_ready(sm_client, arn, timeout_seconds, poll_seconds):
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while True:
        app = sm_client.describe_mlflow_app(Arn=arn)
        status = app["Status"]
        if status != previous:
            print(f"MLflow App {arn}: {status}", flush=True)
            previous = status
        if status in ("Created", "Updated"):
            return arn
        if status not in ("Creating", "Updating"):
            raise RuntimeError(f"MLflow App {arn} is {status}; inspect the app in SageMaker before retrying.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"MLflow App {arn} is still {status} after {timeout_seconds}s. Rerun setup to reconnect.")
        time.sleep(min(poll_seconds, remaining))


def _pages(sm_client, operation, **kwargs):
    # Stop rather than create after an incomplete discovery scan.
    for count, page in enumerate(sm_client.get_paginator(operation).paginate(**kwargs), 1):
        yield page
        if count >= 100 and page.get("NextToken"):
            raise RuntimeError("MLflow discovery exceeded 100 pages; set MLFLOW_APP_ARN explicitly.")


def _discover_app(sm_client, app_name, role_arn, artifact_store_uri):
    workshop, named = [], []
    for page in _pages(sm_client, "list_mlflow_apps"):
        for app in page.get("Summaries", []):
            name = app["Name"]
            if name == app_name:
                named.append(app["Arn"])
            if not name.startswith("workshop-mlflow-"):
                continue
            # ListMlflowApps has no tags. Verify the current template's stack tag.
            tags = {tag["Key"]: tag["Value"]
                    for tag_page in _pages(sm_client, "list_tags", ResourceArn=app["Arn"])
                    for tag in tag_page.get("Tags", [])}
            stack = re.fullmatch(
                r"arn:(aws[a-z-]*):cloudformation:([a-z0-9-]+):([0-9]{12}):"
                r"stack/[^/]+/([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})",
                tags.get("WorkshopStackId", ""),
            )
            if not stack or name != f"workshop-mlflow-{stack[4]}":
                continue
            if app["Arn"].split(":")[1:5] != [stack[1], "sagemaker", stack[2], stack[3]]:
                continue
            if role_arn or artifact_store_uri:
                details = sm_client.describe_mlflow_app(Arn=app["Arn"])
                # The notebook's default bucket can differ from the stack bucket.
                # Its execution role is the stronger scope when available.
                if role_arn:
                    if details.get("RoleArn") != role_arn:
                        continue
                elif details.get("ArtifactStoreUri", "").rstrip("/") != artifact_store_uri.rstrip("/"):
                    continue
            workshop.append(app["Arn"])
    matches = workshop or named
    if len(matches) > 1:
        raise RuntimeError("Multiple matching MLflow Apps; set MLFLOW_APP_ARN explicitly.")
    return matches[0] if matches else ""


def resolve_app(sm_client, arn="", *, session=None, native=False, role_arn="",
                artifact_store_uri="", app_name="nemotron-contractnli-workshop",
                timeout_seconds=900, poll_seconds=15):
    """Prefer an explicit ARN, then the tagged stack app, before reuse/create.

    Discovery errors stop setup. Native SDK fallback never requests an upgrade.
    """
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("MLflow timeout and polling interval must be positive.")
    if not callable(getattr(sm_client, "describe_mlflow_app", None)):
        raise RuntimeError("Install the notebook requirements and restart the kernel; this boto3 lacks MLflow App support.")
    arn = arn.strip()
    if arn:
        if not re.fullmatch(r"arn:aws[a-z-]*:sagemaker:[a-z0-9-]+:[0-9]{12}:mlflow-app/[A-Za-z0-9-]+", arn):
            raise ValueError("MLFLOW_APP_ARN must be an MLflow App ARN, not a tracking-server ARN or URL.")
        if arn.split(":")[3] != sm_client.meta.region_name:
            raise ValueError("MLflow App and SageMaker session must use the same Region.")
    else:
        arn = _discover_app(sm_client, app_name, role_arn, artifact_store_uri)
    if not arn and native and not (role_arn or artifact_store_uri):
        from sagemaker.core.helper.session_helper import Session
        from sagemaker.train.common_utils.finetune_utils import _resolve_mlflow_resource_arn

        if session is None:
            session = Session(sagemaker_client=sm_client)
        # No min_mlflow_version: the SDK upgrade branch changes shared defaults.
        arn = _resolve_mlflow_resource_arn(session)
        if not arn:
            raise RuntimeError(
                "The SageMaker SDK could not discover/create a ready MLflow App. "
                "Check its preceding logs for IAM, quota or provisioning errors; "
                "set MLFLOW_APP_ARN to reconnect to a pending app. Training was not started."
            )
    elif not arn:
        if not role_arn or not artifact_store_uri:
            raise ValueError("Creating a workshop MLflow App requires both a role ARN and an artifact S3 URI. Set MLFLOW_ROLE_ARN and MLFLOW_ARTIFACT_STORE_URI or pass notebook defaults.")
        if not re.fullmatch(r"arn:aws[a-z-]*:iam::[0-9]{12}:role/.+", role_arn):
            raise ValueError("MLFLOW_ROLE_ARN must be an IAM role ARN.")
        if not re.fullmatch(r"s3://[^/]+(?:/.*)?", artifact_store_uri):
            raise ValueError("MLFLOW_ARTIFACT_STORE_URI must be an S3 URI for an existing bucket.")
        print(f"Creating MLflow App {app_name}; provisioning can take several minutes.", flush=True)
        arn = sm_client.create_mlflow_app(
            Name=app_name,
            RoleArn=role_arn,
            ArtifactStoreUri=artifact_store_uri,
            AccountDefaultStatus="DISABLED",
        )["Arn"]
    print(f"MLflow App ARN (save in config.py to reuse explicitly): {arn}", flush=True)
    return _wait_ready(sm_client, arn, timeout_seconds, poll_seconds)
