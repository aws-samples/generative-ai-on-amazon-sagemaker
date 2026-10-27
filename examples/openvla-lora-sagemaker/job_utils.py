"""Helpers for following SageMaker training jobs from the notebooks (used by notebooks 2 and 3).

A job runs in SageMaker independently of the notebook: closing the browser, a dropped connection or a kernel
restart does not stop it. `latest_job` finds it again and `wait_for_job` picks the monitoring back up.
"""
import time
from datetime import datetime

import boto3

TERMINAL = ("Completed", "Failed", "Stopped")


def latest_job(name_contains, region=None):
    """Returns the name of the most recently created training job whose name contains `name_contains`."""
    sm = boto3.client("sagemaker", region_name=region)
    jobs = sm.list_training_jobs(NameContains=name_contains, SortBy="CreationTime", SortOrder="Descending",
                                 MaxResults=1)["TrainingJobSummaries"]
    if not jobs:
        raise RuntimeError(f"no training job with '{name_contains}' in its name in {sm.meta.region_name}")
    return jobs[0]["TrainingJobName"]


def job_status(job_name, region=None):
    """One-line status: state, current stage, elapsed time, and the latest `[step N]` log line."""
    sm = boto3.client("sagemaker", region_name=region)
    d = sm.describe_training_job(TrainingJobName=job_name)
    end = d.get("TrainingEndTime") or datetime.now(d["CreationTime"].tzinfo)
    elapsed = int((end - d["CreationTime"]).total_seconds() // 60)
    return d, f"{d['TrainingJobStatus']}/{d['SecondaryStatus']} ({elapsed} min) {_last_step_line(job_name, region)}"


def wait_for_job(job_name, region=None, poll=60):
    """Prints a status line whenever it changes, instead of streaming every log line (which can exceed
    Jupyter's output rate limit). Safe to interrupt and re-run: it only reads the job's state."""
    region = region or boto3.session.Session().region_name
    print(f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
    last = None
    while True:
        d, line = job_status(job_name, region)
        if line != last:
            print(time.strftime("%H:%M:%S"), line, flush=True)
            last = line
        if d["TrainingJobStatus"] in TERMINAL:
            if d["TrainingJobStatus"] != "Completed":
                raise RuntimeError(d.get("FailureReason", d["TrainingJobStatus"]))
            return d
        time.sleep(poll)


def print_phases(job):
    """Minutes spent in each stage (waiting for capacity, downloading the image, training, uploading)."""
    transitions = job.get("SecondaryStatusTransitions", [])
    for t in transitions:
        end = t.get("EndTime") or job.get("TrainingEndTime") or t["StartTime"]
        print(f"  {t['Status']:<12} {(end - t['StartTime']).total_seconds() / 60:5.1f} min  {t.get('StatusMessage', '')[:80]}")
    total = (job.get("TrainingEndTime", job["CreationTime"]) - job["CreationTime"]).total_seconds() / 60
    print(f"  {'total':<12} {total:5.1f} min, billed {job.get('BillableTimeInSeconds', 0) / 60:.1f} min")


def _last_step_line(job_name, region):
    logs = boto3.client("logs", region_name=region)
    last = ""
    try:
        # filter_log_events often returns empty pages with a nextToken, so read every page
        for page in logs.get_paginator("filter_log_events").paginate(
                logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=job_name,
                filterPattern='"[step"', startTime=int((time.time() - 900) * 1000)):
            if page["events"]:
                last = page["events"][-1]["message"].strip()[:160]
    except logs.exceptions.ResourceNotFoundException:
        pass
    return last


def capacity_config(instance_types, capacity, *, volume_size_in_gb, max_runtime_s, spot_max_wait_h):
    """Compute + StoppingCondition for a job, for config.CAPACITY:

    "fallback"  on-demand; SageMaker starts the job on the first type in `instance_types` with capacity
    "on-demand" on-demand, instance_types[0] only
    "spot"      managed Spot on instance_types[0]; may queue for hours, resumes from checkpoints if interrupted
    """
    from sagemaker.core.shapes import InstancePreference
    from sagemaker.core.training.configs import Compute, StoppingCondition

    if capacity == "fallback":
        if not 1 <= len(instance_types) <= 5:
            raise ValueError(f"instance_preferences takes 1-5 instance types, got {len(instance_types)}")
        # Mutually exclusive with instance_type and Spot
        compute = Compute(instance_preferences=[InstancePreference(instance_type=t) for t in instance_types],
                          instance_count=1, volume_size_in_gb=volume_size_in_gb)
    elif capacity in ("on-demand", "spot"):
        compute = Compute(instance_type=instance_types[0], instance_count=1, volume_size_in_gb=volume_size_in_gb,
                          enable_managed_spot_training=capacity == "spot")
    else:
        raise ValueError(f"CAPACITY must be 'fallback', 'on-demand' or 'spot', not {capacity!r}")
    stop = StoppingCondition(
        max_runtime_in_seconds=max_runtime_s,
        # Spot only: total time allowed, including waiting for capacity and restarts after interruptions
        max_wait_time_in_seconds=max(max_runtime_s, spot_max_wait_h * 3600) if capacity == "spot" else None,
    )
    return compute, stop


def selected_instance(job):
    """The instance type a job actually ran on (with "fallback", SageMaker picks it from the list)."""
    rc = job.get("ResourceConfig", {})
    return rc.get("SelectedInstanceType") or rc.get("InstanceType")
