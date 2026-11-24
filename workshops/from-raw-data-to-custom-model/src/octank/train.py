"""Serverless model customization helpers, following the aws-ai-ml skill templates
(sft.py, rlvr.py, continuous_customization.md).

Kept thin on purpose: the trainer calls live in `workflow.py` and are shown in the notebooks; these
helpers only remove boilerplate (model package group, dataset registration, safe submission).
"""
from __future__ import annotations

import re

from botocore.exceptions import ClientError


def session():
    import boto3
    from sagemaker.core.helper.session_helper import Session
    return Session(sagemaker_client=boto3.Session().client("sagemaker"))


def ensure_model_package_group(name: str, description: str = ""):
    from sagemaker.core.resources import ModelPackageGroup
    try:
        return ModelPackageGroup.create(model_package_group_name=name,
                                        model_package_group_description=description)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ResourceInUse", "ValidationException"):
            return ModelPackageGroup.get(model_package_group_name=name)
        raise


def register_dataset(name: str, s3_uri: str) -> str:
    from sagemaker.ai_registry.dataset import DataSet
    return DataSet.create(name=safe_name(name, 63), source=s3_uri, wait=True).arn


def safe_name(s: str, max_len: int) -> str:
    s = re.sub(r"[^a-z0-9-]", "-", s.lower()).strip("-")
    return re.sub(r"-+", "-", s)[:max_len].rstrip("-")


def model_package_of(training_job_name: str) -> str:
    from sagemaker.core.resources import TrainingJob
    return TrainingJob.get(training_job_name=training_job_name).output_model_package_arn


_LAST_SUBMIT = [0.0]


def submit(trainer, gap_s: float = 5.0, retries: int = 5):
    """trainer.train(wait=False) with a pause between submissions.

    Serverless job names are `<model>-<technique>-<YYYYMMDDHHMMSS>` (1-second resolution), so two
    jobs of the same technique submitted in the same second collide (ResourceInUse). We keep at
    least `gap_s` seconds between submissions from this process and retry after a pause if another
    process grabbed the same name.
    """
    import time
    for attempt in range(retries):
        wait = _LAST_SUBMIT[0] + gap_s - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            job = trainer.train(wait=False)
            _LAST_SUBMIT[0] = time.time()
            return job
        except Exception as e:  # noqa: BLE001
            _LAST_SUBMIT[0] = time.time()
            if "ResourceInUse" not in str(e) or attempt == retries - 1:
                raise
            time.sleep(gap_s * (attempt + 1))
