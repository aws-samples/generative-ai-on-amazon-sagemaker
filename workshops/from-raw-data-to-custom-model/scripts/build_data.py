"""Build a full dataset end to end (resumable). The instructor runs it with the instructor profile
before the event; the attendee profile is the same pipeline the notebooks run, at 1,000 raw samples.

    nohup ~/.venvs/gift_concierge/bin/python scripts/build_data.py instructor > work/instructor/logs/build.out 2>&1 &
    nohup ~/.venvs/gift_concierge/bin/python scripts/build_data.py attendee > work/attendee/logs/build.out 2>&1 &

Stages: catalog (data/catalog_raw.jsonl, generated once) -> raw requests -> dedup -> safety ->
label -> resolve -> finalize -> gap-fill -> teacher (rejection-sampled against every rule,
reasoning <= ~1,100 tokens) -> SFT / RLVR / eval files -> S3.
The instructor run's LLM caches and state are version-controlled in instructor/ (compacted at the
end), so a rebuild only calls Bedrock for inputs that changed. Derived files (parquet, datasets,
logs) go to work/instructor/ and rebuild from the caches. The published benchmarks in data/ are
never overwritten; their requests are always placed in the holdout split.
Status: work/<run>/logs/status.json.
"""
import json
import logging
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from octank import catalog, config, generate, profile, state  # noqa: E402
from octank.cache import JsonlCache  # noqa: E402
from octank.pipeline import ATTENDEE, INSTRUCTOR, Run  # noqa: E402

PROFILES = {"instructor": INSTRUCTOR, "attendee": ATTENDEE}
RUN = sys.argv[1] if len(sys.argv) > 1 else "instructor"
cfg = PROFILES[RUN]
s = config.get_settings(RUN)
run = Run(s, cfg)
log_dir = s.local("logs", "x").parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(log_dir / "build.log"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("build")
STATUS = log_dir / "status.json"
T0 = time.time()


def status(stage, **kv):
    st = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    st.update(stage=stage, elapsed_min=round((time.time() - T0) / 60, 1),
              updated=time.strftime("%Y-%m-%d %H:%M:%S"), **kv)
    STATUS.write_text(json.dumps(st, indent=2, default=str))
    log.info("STATUS %s %s", stage, json.dumps(kv, default=str)[:400])


def main():
    if not catalog.RAW_PATH.exists():
        status("catalog")
        catalog.save_raw(catalog.generate())
    cat = catalog.load_clean()
    status("catalog_done", catalog_items=len(cat))

    raw_path = run.path("raw_requests.parquet")
    raw = pd.read_parquet(raw_path) if raw_path.exists() else run.generate_raw()
    status("raw_done", raw_rows=len(raw), **generate.stats(raw[raw["_flaw"] == "none"]))

    cur_path = run.path("curated.parquet")
    if cur_path.exists():
        cur = pd.read_parquet(cur_path)
    else:
        d, _ = run.dedup(raw)
        status("curate:safety", after_dedup=len(d))
        d = run.safety(d)
        d = d[d["safety_decision"] != "drop"]
        status("curate:label", after_safety=len(d))
        d = run.label(d)
        cur, _ = run.resolve(d)
        cur = run.finalize(cur, cat)
        cur.to_parquet(cur_path, index=False)
        run.report(raw, cur).to_csv(run.path("curation_report.csv"), index=False)
    status("curate_done", curated=len(cur))

    final_path = run.path("final.parquet")
    if final_path.exists():
        final = pd.read_parquet(final_path)
    else:
        final, _ = run.gapfill(cur, cat)
        final.to_parquet(final_path, index=False)
    status("gapfill_done", final_rows=len(final), **generate.stats(final),
           occasion_skew=profile.skew_report(final))

    sp = run.split(final)
    pools = run.pools(sp, cat)
    taught = run.teach(sp[sp["split"].isin(["sft", "holdout"])], pools)
    status("teach_done", taught=len(taught), ok_rate=round(float(taught["ok"].mean()), 3))

    leftovers = set(sp.loc[sp["_flaw"].isin(["toxic", "ambiguous"]), "request_id"])
    files = run.write_datasets(sp, pools, taught, exclude_ids=leftovers)
    status("quality_filter", **run.filter_stats)
    uris = run.upload(files)
    if s.tracked_dir is not None:              # keep the version-controlled caches small and stable
        sizes = {p.stem: JsonlCache(p).compact() for p in sorted(run.cache_dir.glob("*.jsonl"))}
        status("caches_compacted", rows=sizes)
    counts = {k: sum(1 for _ in open(v)) for k, v in files.items()}
    state.update(s, datasets=uris, counts=counts, rl_prompts=run.filter_stats.get("rl_prompts"))
    status("done", datasets=uris, counts=counts)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        status("FAILED", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2500:])
        raise
