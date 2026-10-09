"""Optional MLflow tracking for the job scripts.

Tracking is on when MLFLOW_TRACKING_URI is set, for example to a SageMaker AI MLflow tracking server ARN
(the sagemaker-mlflow plugin in requirements.txt resolves ARNs). It reads these environment variables:

  MLFLOW_TRACKING_URI      tracking server ARN or URI; tracking is off when empty
  MLFLOW_EXPERIMENT_NAME   experiment to log to
  MLFLOW_RUN_NAME          name of a new run
  MLFLOW_RUN_ID            resume this run instead of starting a new one (used by dpo_train.py to let
                           its evaluation subprocess log into the same run)
  MLFLOW_TAG_<NAME>        added to a new run as tag <name>, for example the pipeline execution id

A tracking failure never fails the job: setup errors switch tracking off, and logging errors print a warning.
"""

import contextlib
import os
import re


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


@contextlib.contextmanager
def run(default_name: str):
    """Yield the mlflow module inside an active run, or None when tracking is off or unavailable."""
    if not enabled():
        yield None
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        if not os.environ.get("MLFLOW_RUN_ID"):
            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME") or "rl-alignment")
        tags = {k[len("MLFLOW_TAG_"):].lower(): v for k, v in os.environ.items() if k.startswith("MLFLOW_TAG_") and v}
        if os.environ.get("MLFLOW_RUN_ID"):
            active = mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"])
        else:
            active = mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME") or default_name, tags=tags)
    except Exception as err:  # tracking must never stop the job
        print(f"[mlflow] tracking disabled: {type(err).__name__}: {err}")
        yield None
        return
    print(f"[mlflow] logging to run {active.info.run_id} in experiment {active.info.experiment_id}")
    with active:
        yield mlflow


def _key(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-. /]", "_", name)[:250]


def log_params(mlflow, params: dict, prefix: str = "") -> None:
    if mlflow is None:
        return
    try:
        mlflow.log_params({_key(prefix + k): str(v)[:500] for k, v in params.items()})
    except Exception as err:
        print(f"[mlflow] log_params failed: {err}")


def log_metrics(mlflow, metrics: dict, prefix: str = "", step=None) -> None:
    """Log every numeric value of a (possibly nested) dict, with keys joined by '.'."""
    if mlflow is None:
        return
    flat = {}

    def walk(d, path):
        for k, v in d.items():
            key = f"{path}{k}"
            if isinstance(v, dict):
                walk(v, key + ".")
            elif isinstance(v, bool):
                flat[_key(key)] = float(v)
            elif isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
                flat[_key(key)] = float(v)

    walk(metrics, prefix)
    try:
        if flat:
            mlflow.log_metrics(flat, step=step)
    except Exception as err:
        print(f"[mlflow] log_metrics failed: {err}")


def log_artifacts(mlflow, paths, artifact_path: str = "") -> None:
    if mlflow is None:
        return
    for p in paths:
        if os.path.exists(p):
            try:
                mlflow.log_artifact(p, artifact_path or None)
            except Exception as err:
                print(f"[mlflow] log_artifact {p} failed: {err}")


def active_run_id(mlflow):
    if mlflow is None:
        return None
    run = mlflow.active_run()
    return run.info.run_id if run else None
