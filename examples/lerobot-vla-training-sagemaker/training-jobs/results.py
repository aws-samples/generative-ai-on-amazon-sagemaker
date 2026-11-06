#!/usr/bin/env python3
"""Summarise LeRobot SageMaker training jobs: status, timings, peak VRAM, step_s from CloudWatch logs.
Usage: python results.py <job-name-prefix> [--region us-east-2]"""
import argparse, re, sys, boto3, datetime as dt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--region", default="us-east-2")
    a = ap.parse_args(); sm = boto3.client("sagemaker", region_name=a.region); logs = boto3.client("logs", region_name=a.region)
    jobs = []; tok = None
    while True:
        kw = dict(NameContains=a.prefix, SortBy="CreationTime", SortOrder="Descending", MaxResults=100)
        if tok: kw["NextToken"] = tok
        r = sm.list_training_jobs(**kw); jobs += r["TrainingJobSummaries"]; tok = r.get("NextToken")
        if not tok: break
    for j in sorted(jobs, key=lambda x: x["CreationTime"]):
        d = sm.describe_training_job(TrainingJobName=j["TrainingJobName"])
        rc = d["ResourceConfig"]; name = j["TrainingJobName"]
        bill = d.get("BillableTimeInSeconds"); tr = d.get("TrainingTimeInSeconds")
        env = d.get("Environment", {})
        res = {"vram": {}, "wall": None, "step_s": [], "fail": d.get("FailureReason", "")[:300], "err": []}
        try:
            streams = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=name)["logStreams"]
            for st in streams:
                events = logs.get_log_events(logGroupName="/aws/sagemaker/TrainingJobs", logStreamName=st["logStreamName"], limit=10000, startFromHead=False)["events"]
                for e in events:
                    m = e["message"]
                    if "=== PEAK_VRAM" in m:
                        g = re.search(r"gpu=(\d+) used_mib=(\d+)", m); res["vram"][g.group(1)] = int(g.group(2))
                    elif "=== RESULT" in m:
                        w = re.search(r"train_wall_s=(\d+)", m); res["wall"] = int(w.group(1)) if w else None
                    elif "step_s:" in m or "update_s:" in m:
                        ss = re.search(r"(?:step_s|update_s):([\d.]+)", m); res["step_s"].append(float(ss.group(1)))
                    elif re.search(r"(OutOfMemoryError|CUDA out of memory|Traceback|Error:|ERROR)", m) and len(res["err"]) < 3:
                        res["err"].append(m.strip()[:200])
        except Exception as ex:
            res["err"].append(f"log read failed: {ex}")
        step = f"{sum(res['step_s'][-20:])/len(res['step_s'][-20:]):.3f}s" if res["step_s"] else "-"
        vram = ",".join(f"{v/1024:.1f}G" for _, v in sorted(res["vram"].items())) or "-"
        print(f"{name:70s} {d['TrainingJobStatus']:10s} {rc.get('SelectedInstanceCount', rc.get('InstanceCount'))}x{rc.get('SelectedInstanceType') or rc.get('InstanceType') or 'pending':16s} "
              f"model={env.get('JOB_NAME','?').split('-')[1] if '-' in env.get('JOB_NAME','') else '?':5s} peft={env.get('PEFT','0')} bs={env.get('BATCH_SIZE','preset'):6s} steps={env.get('STEPS','?'):5s} "
              f"train={tr or 0:5d}s bill={bill or 0:5d}s wall={res['wall'] or '-'} step={step} vram={vram}")
        if res["fail"]: print(f"    FAIL: {res['fail']}")
        for e in res["err"]: print(f"    log: {e}")

if __name__ == "__main__": main()
