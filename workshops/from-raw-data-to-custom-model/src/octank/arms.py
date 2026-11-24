"""Run each training stage as one background unit: train -> evaluate as soon as training finishes.

Used by the attendee notebooks. Each arm runs in a background thread so the notebook stays
interactive; progress is persisted in work/<run>/state.json under "arms", so a kernel restart picks
it up (re-run `start`: finished steps are skipped, running jobs are re-attached).

    arms.start(S, "sft", make_sft)                        # base -> SFT
    arms.start(S, "sft_rlvr", make_rlvr, after="sft")      # SFT model -> RLVR

`make_*` take the starting model (base id or a model package ARN) and return a configured trainer
(see `octank.workflow`). Evaluation is opt-in (`evaluate=True`): in the 2-hour session the demo and
the comparison use the instructor's models, and an attendee evaluation would finish after the
session. It is the fast path (`workflow.launch_eval` / `finish_eval`).
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

import pandas as pd

from . import config, state, train, workflow

_EVAL_LOCK = threading.Lock()     # evaluation pipelines must be constructed one at a time
_THREADS: dict[str, threading.Thread] = {}


def _arm(S, name: str) -> dict:
    return state.load(S).get("arms", {}).get(name, {})


def _set(S, name: str, **kv) -> None:
    state.update_arm(S, name, **kv, updated=time.strftime("%H:%M:%S"))


def model_package(S, name: str) -> str | None:
    return _arm(S, name).get("model_package")


def eval_csv(S, name: str):
    return S.local(f"eval_{name}.csv")


def _run(S, name, make_trainer, after, evaluate):
    try:
        a = _arm(S, name)
        if a.get("phase") == "done":
            return
        if "job" not in a:
            start = config.BASE_MODEL_ID
            if after:
                _set(S, name, phase=f"waiting for {after}", heartbeat=time.time())
                while not model_package(S, after):
                    if _arm(S, after).get("phase") == "FAILED":
                        raise RuntimeError(f"upstream arm {after} failed")
                    time.sleep(60)
                    _set(S, name, heartbeat=time.time())       # lets another kernel see we're alive
                start = model_package(S, after)
            if "job" in _arm(S, name):                          # another kernel submitted it meanwhile
                return _run(S, name, make_trainer, after, evaluate)
            job = train.submit(make_trainer(start))
            _set(S, name, job=job.training_job_name, start_from=start, phase="training")
        a = _arm(S, name)
        if "model_package" not in a:
            j = workflow.wait_job(a["job"], on_status=lambda j: _set(
                S, name, job_status=j.training_job_status, secondary=j.secondary_status))
            _set(S, name, model_package=j.output_model_package_arn, timing=workflow.timing(a["job"], S.region))
        if not evaluate:
            _set(S, name, phase="trained")
            return
        if "eval_execs" not in _arm(S, name):
            with _EVAL_LOCK:
                execs = workflow.launch_eval(S, {name: model_package(S, name)}, f"eval_{name}_")
            _set(S, name, phase="evaluating", eval_execs=execs)
        df = workflow.finish_eval(S, _arm(S, name)["eval_execs"])
        df.to_csv(eval_csv(S, name), index=False)
        _set(S, name, phase="done", eval_csv=str(eval_csv(S, name)))
    except Exception as e:  # noqa: BLE001
        _set(S, name, phase="FAILED", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])


def start(S, name: str, make_trainer: Callable[[str], object], after: str | None = None,
          evaluate: bool = False):
    """Train `name` in the background (then evaluate it if `evaluate`). Safe to re-run (resumes)."""
    if name in _THREADS and _THREADS[name].is_alive():
        return _THREADS[name]
    _set(S, name, after=after, evaluate=evaluate)
    t = threading.Thread(target=_run, args=(S, name, make_trainer, after, evaluate), daemon=True,
                         name=f"arm-{name}")
    t.start()
    _THREADS[name] = t
    return t


HEARTBEAT_STALE_S = 180      # a waiting thread refreshes its heartbeat every 60 s


def orphaned(S, name: str = "sft_rlvr", after: str = "sft") -> bool:
    """True if `name` should have started but nothing is driving it: `after` has a model package,
    `name` has no training job, and no kernel has refreshed its wait heartbeat recently (e.g. the
    Lab 3 kernel was shut down while it waited for SFT)."""
    if not model_package(S, after):
        return False
    a = _arm(S, name)
    if "job" in a or a.get("phase") in ("done", "trained", "FAILED"):
        return False
    if name in _THREADS and _THREADS[name].is_alive():
        return False
    return time.time() - float(a.get("heartbeat") or 0) > HEARTBEAT_STALE_S


def check_handoff(S) -> bool:
    """Print what to do if SFT -> RLVR never launched because its waiting kernel stopped."""
    if not orphaned(S):
        return False
    print("Your SFT model is ready, but SFT -> RLVR was never launched (the Lab 3 kernel that was\n"
          "waiting for SFT stopped). Resume it from here with:\n\n"
          "    arms.resume_rlvr(S)\n")
    return True


def resume_rlvr(S, mpg: str = "octank-concierge-attendee", dataset_name: str = "octank-attendee-rlvr",
                evaluate: bool = False, force: bool = False):
    """Launch SFT -> RLVR from this kernel with the same trainer as Lab 3 (`workflow.rlvr_trainer`,
    the reward Lambda and RLVR dataset recorded in state). Refuses if another kernel is still
    waiting for SFT (fresh heartbeat), unless `force`."""
    if not force and not orphaned(S):
        a = _arm(S, "sft_rlvr")
        raise RuntimeError(f"sft_rlvr doesn't need resuming (phase={a.get('phase')!r}, job={a.get('job')!r}); "
                           "pass force=True to override")
    st = state.load(S)
    rlvr_uri, lam = st["datasets"]["rlvr"], st["reward_lambda"]
    sess = train.session()

    def make_rlvr(start_from):
        return workflow.rlvr_trainer(S, start_from, rlvr_uri, mpg, sess, lam, dataset_name)

    return start(S, "sft_rlvr", make_rlvr, after="sft", evaluate=evaluate)


def status(S) -> pd.DataFrame:
    arms = state.load(S).get("arms", {})
    cols = ["phase", "job", "job_status", "secondary", "error", "updated"]
    rows = []
    for n, a in arms.items():
        t = a.get("timing") or {}
        rows.append({"arm": n, **{c: a.get(c) for c in cols},
                     "train_min": round(t["wall_s"] / 60, 1) if t.get("wall_s") else None})
    return pd.DataFrame(rows).set_index("arm") if rows else pd.DataFrame()


def results(S) -> pd.DataFrame:
    """Per-case evaluation results of every finished arm (model column = arm name)."""
    parts = [pd.read_csv(eval_csv(S, n)) for n, a in state.load(S).get("arms", {}).items()
             if a.get("phase") == "done" and eval_csv(S, n).exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def wait(S, names, poll: int = 60):
    """Block until all named arms are done (or failed)."""
    while True:
        arms = state.load(S).get("arms", {})
        if all(arms.get(n, {}).get("phase") in ("done", "trained", "FAILED") for n in names):
            return status(S)
        time.sleep(poll)
