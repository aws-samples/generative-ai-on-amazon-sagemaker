"""Training and evaluation steps shared by the attendee notebooks (through `arms`) and the instructor
script (`scripts/train_eval.py`), so both run exactly the same code; only the data size differs.

  sft_trainer(...)   -> SFTTrainer from a starting model (base id or model package), config.SFT_HP
  rlvr_trainer(...)  -> RLVRTrainer with the hybrid reward Lambda, config.RLVR_HP
  wait_job(name)     -> TrainingJob once it finishes (raises if it failed)
  timing(name)       -> exact phase durations from SageMaker (queue, download, training, upload)
  launch_eval(...)   -> managed generation pipelines for the 40- and 80-candidate suites
  finish_eval(...)   -> stop after inference, score in code + fast judge -> per-case DataFrame
"""
from __future__ import annotations

import time

import boto3
import pandas as pd

from . import benchmarks, config, fasteval, train


def sft_trainer(S, start_from: str, dataset_arn: str, model_package_group, sess):
    from sagemaker.train.common import TrainingType
    from sagemaker.train.sft_trainer import SFTTrainer
    t = SFTTrainer(model=start_from, training_type=TrainingType.LORA, model_package_group=model_package_group,
                   training_dataset=dataset_arn, s3_output_path=S.s3("training-output") + "/",
                   sagemaker_session=sess, role=S.role_arn)
    for k, v in config.SFT_HP.items():
        setattr(t.hyperparameters, k, v)
    return t


def rlvr_trainer(S, start_from: str, dataset_uri: str, model_package_group: str, sess, reward_lambda_arn: str,
                 dataset_name: str):
    """RLVR from `start_from` (the SFT model package) with the hybrid reward Lambda.
    The reward function is verified on 3 dataset rows before training. The SDK's verifier needs the
    dataset's S3 URI (it can't read a dataset ARN) and train() re-verifies with the ARN, so we verify
    with the URI here and skip the second check."""
    from sagemaker.train.rlvr_trainer import RLVRTrainer
    t = RLVRTrainer(model=start_from, model_package_group=model_package_group,
                    training_dataset=train.register_dataset(dataset_name, dataset_uri),
                    s3_output_path=S.s3("training-output") + "/", sagemaker_session=sess,
                    role=S.role_arn, custom_reward_function=reward_lambda_arn)
    for k, v in config.RLVR_HP.items():
        setattr(t.hyperparameters, k, v)
    t._verify_reward_function(sample_count=3, training_dataset=dataset_uri)     # raises on failure
    t.skip_reward_validation = True
    return t


def wait_job(name: str, poll: int = 60, on_status=None):
    from sagemaker.core.resources import TrainingJob
    while True:
        j = TrainingJob.get(training_job_name=name)
        if on_status:
            on_status(j)
        if j.training_job_status in ("Completed", "Failed", "Stopped"):
            if j.training_job_status != "Completed":
                raise RuntimeError(f"{name} {j.training_job_status}: {j.failure_reason}")
            return j
        time.sleep(poll)


def timing(name: str, region: str) -> dict:
    """Exact durations from SageMaker: per secondary-status phase, queue, total, billable seconds."""
    d = boto3.client("sagemaker", region_name=region).describe_training_job(TrainingJobName=name)
    phases = {}
    for t in d.get("SecondaryStatusTransitions", []):
        if t.get("EndTime"):
            phases[t["Status"]] = phases.get(t["Status"], 0) + (t["EndTime"] - t["StartTime"]).total_seconds()
    end = d.get("TrainingEndTime") or d["LastModifiedTime"]
    start = d.get("TrainingStartTime")
    return {"job": name, "created": d["CreationTime"].isoformat(),
            "training_start": start.isoformat() if start else None, "end": end.isoformat(),
            "wall_s": round((end - d["CreationTime"]).total_seconds()),
            "queue_s": round((start - d["CreationTime"]).total_seconds()) if start else None,
            "training_time_s": d.get("TrainingTimeInSeconds"), "billable_s": d.get("BillableTimeInSeconds"),
            "phases_s": {k: round(v) for k, v in phases.items()}}


def launch_eval(S, models: dict[str, str], tag: str, base_with: str | None = None, log=print) -> dict:
    """Start one generation pipeline per model per suite. Returns {"40": execs, "80": execs}."""
    return {"40": fasteval.launch(S, models, benchmarks.cases_40(), f"{tag}40", log=log, base_with=base_with),
            "80": fasteval.launch(S, models, benchmarks.cases_80(), f"{tag}80", log=log,
                                  max_new_tokens=config.EVAL_MAX_NEW_TOKENS_80, base_with=base_with)}


def finish_eval(S, execs: dict, log=print) -> pd.DataFrame:
    """Wait for answer generation, stop the pipelines, score everything (rules, tokens, judge)."""
    fasteval.stop_after_inference(S, {**{f"{k}@40": v for k, v in execs["40"].items()},
                                      **{f"{k}@80": v for k, v in execs["80"].items()}}, log=log)
    return pd.concat([fasteval.score(S, execs["40"], benchmarks.cases_40(), log=log, skip_invalid=True),
                      fasteval.score(S, execs["80"], benchmarks.cases_80(), log=log, skip_invalid=True)],
                     ignore_index=True)


def by_problem_type(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy (rule pass), tokens and cut-off rate per model and problem type."""
    rows = []
    for name, tracks in benchmarks.PROBLEM_TYPES:
        sub = df[df.track.isin(tracks)]
        for m, g in sub.groupby("model"):
            rows.append({"problem_type": name, "model": m, "n": len(g), "rule_pass": g.hard_pass.mean(),
                         "tokens": g.tokens.mean(), "cut_off": g.cut_off.mean()})
    return pd.DataFrame(rows)
