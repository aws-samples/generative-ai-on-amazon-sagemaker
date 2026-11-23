"""Evaluate a use case's held-out tasks with the weights of a model package (no training).

python scripts/eval_package.py --usecase returns --model-package-arn ARN --n 128 --out runs/x
"""
import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.runtime import RunConfig, TrainingRun, summarize_eval  # noqa: E402
from sagemaker.train.training_session import ServiceClient  # noqa: E402
from usecases import USECASES  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--usecase", required=True)
p.add_argument("--model-package-arn", default=None, help="omit for the base model")
p.add_argument("--n", type=int, default=128)
p.add_argument("--max-new-tokens", type=int, default=1536)
p.add_argument("--max-turns", type=int, default=10)
p.add_argument("--out", required=True)
p.add_argument("--reasoning", default="off", help="low|medium|high|xhigh, or off (enable_thinking=False)")
p.add_argument("--sampler", default="session", choices=["session", "refresh", "package"])
a = p.parse_args()


async def main():
    """Open a session, evaluate the use case's test tasks, and write eval.jsonl and rollouts.jsonl under --out."""
    uc = importlib.import_module(USECASES[a.usecase])
    tasks = uc.load("test", a.n)
    cfg = RunConfig(region=os.environ["AWS_REGION"], base_model_arn=os.environ["BASE_MODEL_ARN"],
                    role_arn=os.environ["ROLE_ARN"], model_package_group_arn=os.environ["MODEL_PACKAGE_GROUP_ARN"],
                    s3_output_path=os.environ["S3_OUTPUT_PATH"], max_new_tokens=a.max_new_tokens,
                    max_turns=a.max_turns, rollout_timeout=300,
                    chat_template_kwargs={"enable_thinking": False} if a.reasoning == "off" else {"reasoning_effort": a.reasoning})
    run = TrainingRun(cfg, a.out)
    t0 = time.monotonic()
    run.open(init_from_model_package=a.model_package_arn)
    try:
        if a.model_package_arn and a.sampler == "refresh":
            await run.refresh()
        elif a.model_package_arn and a.sampler == "package":
            svc = ServiceClient(region=cfg.region, role_arn=cfg.role_arn,
                                model_package_group_arn=cfg.model_package_group_arn, s3_output_path=cfg.s3_output_path)
            run.backend.set_sampling_client(svc.create_sampling_client(model_package_arn=a.model_package_arn))
        opened = time.monotonic() - t0
        rs = await run.evaluate(uc.run_agent, tasks)
        s = {"model_package_arn": a.model_package_arn, "sampler": a.sampler, "open_seconds": opened, **summarize_eval(rs)}
        run._log("eval.jsonl", s)
        for r in rs:
            run._log("rollouts.jsonl", {"id": r.payload.get("id"), "reward": r.reward, "error": r.error,
                                        "result_error": (r.result or {}).get("error"),
                                        "transcript": (r.result or {}).get("transcript")})
        print("EVAL", json.dumps(s))
    finally:
        run.close()


logging.basicConfig(level=logging.WARNING)
asyncio.run(main())
