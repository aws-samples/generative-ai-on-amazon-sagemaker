"""Baseline, train and evaluate one use case on a training session.

Example:
  python scripts/run.py --usecase math --steps 10 --groups 8 --k 8 --eval-n 64 --out runs/math1
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.runtime import RunConfig, TrainingRun, config_dict, summarize_eval  # noqa: E402
from usecases import USECASES  # noqa: E402



def parse():
    """Parse and return the command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--usecase", required=True, choices=USECASES)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--train-n", type=int, default=512)
    p.add_argument("--eval-n", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--loss", default="ppo")
    p.add_argument("--refresh", default="sync", choices=["sync", "overlap"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--rollout-timeout", type=float, default=600.0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-context", type=int, default=16384)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--reasoning", default="off")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    p.add_argument("--base-model-arn", default=os.environ.get("BASE_MODEL_ARN"))
    p.add_argument("--role-arn", default=os.environ.get("ROLE_ARN"))
    p.add_argument("--mpg-arn", default=os.environ.get("MODEL_PACKAGE_GROUP_ARN"))
    p.add_argument("--s3-output", default=os.environ.get("S3_OUTPUT_PATH"))
    p.add_argument("--out", required=True)
    p.add_argument("--save", action="store_true", help="save inference weights at the end")
    p.add_argument("--extra", default="{}", help="JSON kwargs for the use case loader")
    p.add_argument("--filter", default=None, help="pass-rates JSON; train only on tasks with mixed outcomes")
    return p.parse_args()


async def main(a):
    """Evaluate the base model, train for a.steps steps with periodic evaluations, evaluate again, and optionally save.

    Writes config.json and the run logs under a.out. SIGTERM cancels the run; the session is closed in all cases.
    """
    uc = importlib.import_module(USECASES[a.usecase])
    extra = json.loads(a.extra)
    train = uc.load("train", a.train_n, **extra)
    if a.filter:
        rates = json.loads(Path(a.filter).read_text())
        train = [t for t in train if t["id"] in rates and len(set(rates[t["id"]])) > 1]
        logging.getLogger("run").info("filtered train tasks: %d", len(train))
    test = uc.load("test", a.eval_n, **extra)
    cfg = RunConfig(region=a.region, base_model_arn=a.base_model_arn, role_arn=a.role_arn,
                    model_package_group_arn=a.mpg_arn, s3_output_path=a.s3_output, learning_rate=a.lr,
                    loss=a.loss, groups_per_step=a.groups, group_size=a.k, max_new_tokens=a.max_new_tokens,
                    max_context_tokens=a.max_context, max_turns=a.max_turns, concurrency=a.concurrency,
                    refresh=a.refresh, temperature=a.temperature, rollout_timeout=a.rollout_timeout,
                    chat_template_kwargs=({"enable_thinking": False} if a.reasoning == "off"
                                          else {"reasoning_effort": a.reasoning} if a.reasoning else {}))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**config_dict(cfg), **vars(a)}, indent=1, default=str))
    run = TrainingRun(cfg, out)
    t_open = time.monotonic()
    run.open()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    try:
        log = logging.getLogger("run")
        log.info("session %s open in %.0fs", run.session_name, time.monotonic() - t_open)

        async def do_eval(tag):
            """Evaluate the current weights on the test tasks and log the summary and rollouts under `tag`."""
            await run.wait_refresh()
            t0 = time.monotonic()
            rs = await run.evaluate(uc.run_agent, test)
            s = {"step": run.step, "tag": tag, **summarize_eval(rs), "time": time.monotonic() - t0}
            run._log("eval.jsonl", s)
            run.log_rollouts("eval_rollouts.jsonl", rs, step=run.step)
            log.info("EVAL %s", json.dumps(s))

        await do_eval("base")
        rng = random.Random(0)
        order = list(range(len(train)))
        rng.shuffle(order)
        for s in range(a.steps):
            idx = [order[(s * a.groups + i) % len(order)] for i in range(a.groups)]
            m, groups = await run.train_step(uc.run_agent, [train[i] for i in idx])
            keep = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                    if not k.startswith("train/") or k in ("train/actor/pg_loss", "train/actor/ppo_kl", "train/actor/grad_norm")}
            log.info("STEP %s", json.dumps(keep))
            run.log_rollouts("train_rollouts.jsonl", [g[0] for g in groups], step=m["step"])
            if a.eval_every and run.step % a.eval_every == 0 and run.step != a.steps:
                await do_eval(f"step{run.step}")
        await do_eval("final")
        if a.save:
            arn = run.save(inference=True)
            run._log("saved.jsonl", {"step": run.step, "model_package_arn": arn})
            log.info("SAVED %s", arn)
    finally:
        run.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "botocore", "urllib3", "strands", "openai", "sagemaker"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        asyncio.run(main(parse()))
    except asyncio.CancelledError:
        logging.warning("cancelled")
