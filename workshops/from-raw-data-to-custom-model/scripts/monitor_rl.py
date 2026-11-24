"""Monitor an RL (RLVR) training job step by step: reward, response length, cut-off rate, entropy, KL.

    nohup ~/.venvs/gift_concierge/bin/python scripts/monitor_rl.py <job-name> > work/instructor/logs/monitor.out 2>&1 &

Writes work/instructor/logs/rlvr_monitor.csv (or $MONITOR_CSV; one row per step) and a verdict line per new step.
Flags (never stops the job): length > 1.2x step 1 for 3 steps in a row, cut-off rate > 10%,
reward below step 1 for 3 steps in a row.
"""
import csv
import re
import sys
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
JOB = sys.argv[1]
REGION = "us-west-2"
GROUP = "/aws/sagemaker/TrainingJobs"
import os  # noqa: E402
OUT = Path(os.environ.get("MONITOR_CSV", ROOT / "work" / "instructor" / "logs" / "rlvr_monitor.csv"))
KEYS = {"score": r"critic/score/mean'?:\s*(?:np\.float64\()?([0-9.e-]+)",
        "length": r"response_length/mean'?:\s*(?:np\.float64\()?([0-9.e-]+)",
        "clip": r"response_length/clip_ratio'?:\s*(?:np\.float64\()?([0-9.e-]+)",
        "entropy": r"actor/entropy'?:\s*(?:np\.float64\()?([0-9.e-]+)",
        "kl": r"actor/kl_loss'?:\s*(?:np\.float64\()?([0-9.e-]+)"}
STEP = re.compile(r"(?:^|[\s,'(])step:\s*(\d+)")
logs = boto3.client("logs", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)


def stream():
    r = logs.describe_log_streams(logGroupName=GROUP, logStreamNamePrefix=JOB)
    names = [s["logStreamName"] for s in r.get("logStreams", []) if "concise" not in s["logStreamName"]]
    return names[0] if names else None


def steps(name):
    rows, kw = {}, {"logGroupName": GROUP, "logStreamNames": [name], "filterPattern": '"critic/score/mean"'}
    while True:
        r = logs.filter_log_events(**kw)
        for e in r["events"]:
            m = e["message"]
            s = STEP.search(m)
            if not s:
                continue
            row = {"step": int(s.group(1))}
            for k, pat in KEYS.items():
                v = re.findall(pat, m)
                row[k] = round(float(v[-1] if k in ("length", "clip") else v[0]), 4) if v else None
            rows[row["step"]] = row
        if not r.get("nextToken"):
            return [rows[k] for k in sorted(rows)]
        kw["nextToken"] = r["nextToken"]


def verdict(rows):
    b, flags = rows[0], []
    last3 = rows[-3:]
    if len(last3) == 3 and all(r["length"] > 1.2 * b["length"] for r in last3):
        flags.append("LENGTH DRIFT")
    if rows[-1]["clip"] and rows[-1]["clip"] > 0.10:
        flags.append("CUT-OFFS > 10%")
    if len(last3) == 3 and all(r["score"] < b["score"] for r in last3):
        flags.append("REWARD BELOW START")
    return ", ".join(flags) or "ok"


seen = 0
while True:
    st = sm.describe_training_job(TrainingJobName=JOB)
    status = st["TrainingJobStatus"]
    name = stream()
    rows = steps(name) if name else []
    if len(rows) > seen:
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", *KEYS])
            w.writeheader(); w.writerows(rows)
        for r in rows[seen:]:
            print(time.strftime("%H:%M:%S"), f"step {r['step']:>2} reward {r['score']:.3f} len {r['length']:.0f} "
                  f"clip {r['clip']:.3f} entropy {r['entropy']} kl {r['kl']} | {verdict(rows[:rows.index(r) + 1])}",
                  flush=True)
        seen = len(rows)
    if status in ("Completed", "Failed", "Stopped"):
        print(time.strftime("%H:%M:%S"), "job", status, flush=True)
        break
    time.sleep(120)
