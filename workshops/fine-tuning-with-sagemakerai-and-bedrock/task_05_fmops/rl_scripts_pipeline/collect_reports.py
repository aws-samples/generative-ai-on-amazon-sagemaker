"""Publish the RL training job's evaluation report so the pipeline's later steps can read it.

RLAlignment runs dpo_train.py, the same script as Lab 02.03. It writes eval/evaluation.json, the per-sample
answers (per_sample_<model>.jsonl) and, in the pipeline, the RL model's red-team answers (redteam_rl.jsonl)
into the job's output.tar.gz. A ConditionStep can read JSON values only from a ProcessingStep's property file,
and the GuardrailGate step needs the answers as plain files, so this small CPU step unpacks the archive.

Input:  /opt/ml/processing/input/rl_job_output    output.tar.gz of the RLAlignment training job
Output: /opt/ml/processing/output/evaluation      evaluation.json, summary.json, per_sample_<model>.jsonl,
                                                  redteam_rl.jsonl
"""

import argparse
import glob
import os
import shutil
import tarfile


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rl-output", default="/opt/ml/processing/input/rl_job_output")
    p.add_argument("--evaluation-dir", default="/opt/ml/processing/output/evaluation")
    p.add_argument("--work-dir", default="/tmp/rl_output")
    return p.parse_args()


def main():
    args = parse_args()
    archives = glob.glob(os.path.join(args.rl_output, "**", "*.tar.gz"), recursive=True)
    if not archives:
        raise FileNotFoundError(f"No output archive found under {args.rl_output}")
    os.makedirs(args.work_dir, exist_ok=True)
    with tarfile.open(archives[0]) as tar:
        tar.extractall(args.work_dir)
    src = os.path.join(args.work_dir, "eval")
    if not os.path.exists(os.path.join(src, "evaluation.json")):
        raise FileNotFoundError(f"evaluation.json is missing from {src}")
    os.makedirs(args.evaluation_dir, exist_ok=True)
    for path in glob.glob(os.path.join(src, "*.json*")):
        shutil.copy(path, args.evaluation_dir)
    print(f"{args.evaluation_dir}: {sorted(os.listdir(args.evaluation_dir))}")


if __name__ == "__main__":
    main()
