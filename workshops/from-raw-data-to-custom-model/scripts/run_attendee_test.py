"""Headless end-to-end test of the attendee notebooks on 1,000 raw samples.

    nohup python3 scripts/run_attendee_test.py > work/attendee_test/run.out 2>&1 &    (system python: nbclient; kernel "gift_concierge")
    ... run_attendee_test.py 01 02      # only Labs 1 and 2
    ... run_attendee_test.py --dry-run  # Labs 1-3 + 5, training replaced by trainer dry runs

Executes the real notebooks with the `gift_concierge` kernel (pinned SDKs), with these substitutions:
  - Lab 0 is not run (its cells are %pip install + endpoint creation)
  - Lab 2 review widget -> scripted decisions (approve, or reject for unclear intent)
  - Lab 4 stops before deployment
  - --dry-run: Lab 3's launch cell validates both trainers with `train(dry_run=True)` instead of
    submitting jobs, and Labs 4-5 are skipped
Per-cell wall times and outputs go to work/attendee_test/ (timings.csv + executed notebooks).
"""
import csv
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB, OUT = ROOT / "notebooks", ROOT / "work" / "attendee_test"
OUT.mkdir(parents=True, exist_ok=True)
TIMINGS = OUT / "timings.csv"
DRY = "--dry-run" in sys.argv

SCRIPTED_REVIEW = """
import datetime as dt
queue = labeled[labeled["review_reason"] != ""].sort_values("review_reason")
q = review.ReviewQueue(queue.head(15), save_path=run.path("review_decisions.parquet"))
q._rows = [{"request_id": r.request_id,
            "decision": "reject" if r.review_reason == "unclear_intent" else "approve",
            "occasion": r.lbl_occasion, "budget_total": r.lbl_budget_total,
            "relationship": r.relationship, "reason": r.review_reason,
            "reviewer": "headless-test", "ts": dt.datetime.utcnow().isoformat()}
           for r in queue.head(15).itertuples()]
q._save()
"""

DRY_LAUNCH = """
# headless dry run: validate both trainers without submitting jobs
for name, t in (("sft", make_sft(config.BASE_MODEL_ID)), ("sft_rlvr", make_rlvr(config.BASE_MODEL_ID))):
    t.train(dry_run=True)
    print(name, "dry run OK")
"""


def prepare(name: str) -> nbformat.NotebookNode:
    nb = nbformat.read(NB / name, as_version=4)
    if name.startswith("02"):
        for c in nb.cells:
            if c.cell_type == "code" and "q.show()" in c.source:
                c.source = SCRIPTED_REVIEW
    if name.startswith("03") and DRY:
        for c in nb.cells:
            if c.cell_type == "code" and "arms.start(" in c.source:
                c.source = DRY_LAUNCH
            elif c.cell_type == "code" and c.source.strip().startswith("# Re-run any time"):
                c.source = "print('dry run: no arms launched')"
    if name.startswith("04"):
        stop = next(i for i, c in enumerate(nb.cells)
                    if c.cell_type == "markdown" and c.source.startswith("## 1. Deploy"))
        nb.cells = nb.cells[:stop]
    return nb


def run(name: str, timeout: int):
    nb = prepare(name)
    client = NotebookClient(nb, timeout=timeout, kernel_name="gift_concierge",
                            resources={"metadata": {"path": str(NB)}})
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            t0 = time.time()
            err = ""
            try:
                client.execute_cell(cell, i)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[-300:]}"
            dt_s = round(time.time() - t0, 1)
            first = cell.source.strip().splitlines()[0][:70] if cell.source.strip() else ""
            with open(TIMINGS, "a", newline="") as f:
                csv.writer(f).writerow([name, i, dt_s, first, err])
            print(f"{time.strftime('%H:%M:%S')} {name} cell {i:2d} {dt_s:7.1f}s  {first}"
                  + (f"  !! {err}" if err else ""), flush=True)
            if err:
                nbformat.write(nb, OUT / name)
                raise SystemExit(f"FAILED in {name} cell {i}")
    nbformat.write(nb, OUT / name)


if __name__ == "__main__":
    t_all = time.time()
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not TIMINGS.exists() or not only:
        with open(TIMINGS, "w", newline="") as f:
            csv.writer(f).writerow(["notebook", "cell", "seconds", "first_line", "error"])
    labs = [("01_generate_profile.ipynb", 3600), ("02_curate.ipynb", 3600),
            ("03_prepare_train.ipynb", 3600), ("04_evaluate_deploy.ipynb", 6 * 3600),
            ("05_compare.ipynb", 1800)]
    if DRY:
        labs = labs[:3] + labs[4:]          # no endpoint in a dry run; Lab 5 runs on the reference data
    for name, timeout in labs:
        if not only or any(name.startswith(o) for o in only):
            run(name, timeout)
    print(f"DONE total {round((time.time() - t_all) / 60, 1)} min", flush=True)
