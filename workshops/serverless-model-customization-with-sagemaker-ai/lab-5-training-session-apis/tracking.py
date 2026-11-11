"""MLflow tracking for the Lab 5 notebooks.

The notebooks log their settings, the metrics of each training step and the evaluation results to the workshop's
SageMaker AI MLflow App. If no MLflow App can be found or reached, tracking is turned off with a printed message and
the notebook runs as before; every function here then does nothing.

The MLflow App is found with workshop_mlflow.resolve_app, the same helper the other labs use: an explicit ARN from
the environment variable MLFLOW_APP_ARN, then the app that the workshop stack created, then the SageMaker SDK's
default app.
"""

from __future__ import annotations

import os
import re

_active = False


def start(experiment: str, run_name: str, params: dict | None = None) -> bool:
    """Connect to the MLflow App, start a run in the experiment, and log params.

    Args:
        experiment: MLflow experiment name, for example "lab5-gsm-hard-grpo".
        run_name: Name of this run, for example the training session name.
        params: Settings to log as run parameters (values are converted to strings).

    Returns:
        True if tracking is on, False if it was turned off.
    """
    global _active
    try:
        import boto3
        import mlflow

        from workshop_mlflow import resolve_app

        sm = boto3.client("sagemaker")
        # timeout_seconds: wait at most 2 minutes for an app that is still being created.
        arn = resolve_app(sm, os.environ.get("MLFLOW_APP_ARN", ""), native=True,
                          app_name="serverless-customization-workshop", timeout_seconds=120)
        # sagemaker-mlflow lets MLflow use the app ARN as the tracking URI and signs requests with AWS credentials.
        mlflow.set_tracking_uri(arn)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        _active = True
        print(f"MLflow: logging to experiment '{experiment}', run '{run_name}'")
    except Exception as exc:  # noqa: BLE001
        _active = False
        print(f"MLflow tracking is off ({type(exc).__name__}: {str(exc)[:200]}). The notebook runs without it.")
    return _active


def log(metrics: dict, step: int | None = None) -> None:
    """Log numeric metrics, optionally at a training step. Non-numeric values are skipped."""
    if not _active:
        return
    try:
        import mlflow

        # MLflow metric names allow letters, digits, "_", "-", ".", " " and "/"; replace other characters.
        clean = {re.sub(r"[^\w\-. /]", "_", k): float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
        mlflow.log_metrics(clean, step=step)
    except Exception as exc:  # noqa: BLE001
        print(f"MLflow: could not log metrics ({str(exc)[:200]})")


def end() -> None:
    """End the run. Does nothing if tracking is off."""
    global _active
    if not _active:
        return
    try:
        import mlflow

        mlflow.end_run()
    finally:
        _active = False
