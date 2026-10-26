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
