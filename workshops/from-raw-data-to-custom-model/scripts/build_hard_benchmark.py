"""Build the two hard benchmark tracks (resumable; cached LLM calls).

  A. hard_known  : hard-by-construction requests, candidates from the training catalog
  B. hard_novel  : the same kind of requests, candidates ONLY from ~300 new products the models
                   never saw in training (data/catalog_novel.jsonl) -> tests attribute inference

Hard by construction: every request has a hidden need (usually with a conflicting interest),
2-3 items, and is kept only if the difficulty grader rates it Hard and a valid answer exists.
Teacher (gpt-oss-120b) reference answers are attached where it finds one; its pass rate is a
difficulty indicator. Output: data/hard_{known,novel}_cases.jsonl + work/hard_bench/stats.json

    nohup ~/.venvs/gift_concierge/bin/python scripts/build_hard_benchmark.py > work/hard_bench/build.out 2>&1 &
"""
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402

from octank import catalog, config, difficulty, generate, llm, pools, teacher, vocab  # noqa: E402
from octank.cache import JsonlCache  # noqa: E402
from octank.formats import write_jsonl  # noqa: E402
from octank.vocab import ATTRIBUTES, CATEGORIES  # noqa: E402

N = {"known": int(os.environ.get("HARD_N_KNOWN", "250")), "novel": int(os.environ.get("HARD_N_NOVEL", "300"))}
WORK = ROOT / "work" / "hard_bench"
WORK.mkdir(parents=True, exist_ok=True)
NOVEL_PATH = catalog.NOVEL_PATH
T0 = time.time()


def log(stage, **kv):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:.1f} min]", stage,
          json.dumps(kv, default=str)[:400], flush=True)


# ---- 1. novel catalog -----------------------------------------------------------------------------
def novel_catalog(per_category=26, batch=13) -> pd.DataFrame:
    if NOVEL_PATH.exists():
        return pd.read_json(NOVEL_PATH, lines=True)
    known = {n.lower() for n in catalog.load_clean()["name"]}
    rng = random.Random(9001)
    jobs, k = [], 0
    for cat in CATEGORIES:
        for b in range(per_category // batch):
            focus = [vocab.INTERESTS[(k * 7 + j) % len(vocab.INTERESTS)] for j in range(5)]
            k += 1
            theme = rng.choice(catalog._THEMES) + "; NEW products, different from typical best sellers"
            jobs.append((cat, theme, batch, 9001 + k, list(ATTRIBUTES), focus))
    items, seen = [], set(known)
    for _, res, _err in llm.pmap(catalog._gen_batch, jobs, workers=24, desc="novel catalog"):
        for it in res or []:
            if it["name"] and it["name"].lower() not in seen and it["description"]:
                seen.add(it["name"].lower())
                items.append(it)
    items.sort(key=lambda d: (d["category"], d["name"]))
    for i, it in enumerate(items):
        it["id"] = f"OCT-{5000 + i}"                       # disjoint from the training catalog ids
    df = pd.DataFrame(items)[["id", "name", "category", "price", "interests", "attributes", "description"]]
    df.to_json(NOVEL_PATH, orient="records", lines=True)
    return df


# ---- 2. hard specs (graded before any message is written) ------------------------------------------
def hard_specs(track: str, cat: pd.DataFrame, n: int) -> list[dict]:
    rng = random.Random({"known": 101, "novel": 202}[track])
    out, tried = [], 0
    while len(out) < n and tried < n * 60:
        tried += 1
        s = generate.sample_spec(rng, tried)
        if not s["hidden_avoid"] or s["budget_total"] < 30:
            continue
        if s["num_items"] == 1:
            s["num_items"] = rng.choice([2, 3]) if s["budget_total"] >= 45 else 2
        s["request_id"] = f"h{track[0]}{tried:05d}"
        pool = pools.build_pool(s, cat)
        g = difficulty.grade(s, pool)
        if g["label"] == "Hard" and g["feasible"]:
            out.append({"spec": s, "pool": pool, "difficulty": g["score"]})
    log("specs", track=track, kept=len(out), tried=tried)
    return out


def main():
    cats = {"known": catalog.load_clean(), "novel": novel_catalog()}
    log("catalogs", known=len(cats["known"]), novel=len(cats["novel"]),
        novel_per_category=cats["novel"].category.value_counts().to_dict())
    stats = {}
    for track in ("known", "novel"):
        items = hard_specs(track, cats[track], N[track])
        specs = [x["spec"] for x in items]
        msgs = generate.write_messages(specs,
                                       cache=JsonlCache(WORK / f"messages_{track}.jsonl"), workers=24)
        cases = []
        for x in items:
            s, m = x["spec"], msgs.get(x["spec"]["request_id"])
            if not m:
                continue
            low = m.lower()
            if any(h.replace("_", " ") in low for h in s["hidden_avoid"]):     # message leaked the need
                continue
            req = {**{k: v for k, v in s.items() if k != "style"}, "style": s["style"], "message": m,
                   "source": f"hard_{track}"}
            cases.append({"request": req, "candidates": x["pool"], "difficulty": x["difficulty"]})
        log("messages", track=track, cases=len(cases), leaked_or_missing=len(items) - len(cases))
        taught = teacher.teach_many([(c["request"], c["candidates"]) for c in cases],
                                    cache=JsonlCache(WORK / f"teacher_{track}.jsonl"), workers=32)
        by_id = {t["request_id"]: t for t in taught}
        for c in cases:
            t = by_id.get(c["request"]["request_id"]) or {}
            c["reference"] = json.loads(t["answer"]) if t.get("ok") else None
        write_jsonl(cases, config.DATA_DIR / f"hard_{track}_cases.jsonl")
        ok = [c["reference"] is not None for c in cases]
        stats[track] = {"cases": len(cases), "teacher_ok_rate": round(sum(ok) / max(len(ok), 1), 3),
                        "difficulty_mean": round(sum(c["difficulty"] for c in cases) / max(len(cases), 1), 1),
                        "items": pd.Series([c["request"]["num_items"] for c in cases]).value_counts().to_dict(),
                        "hidden_needs": pd.Series([",".join(c["request"]["hidden_avoid"]) for c in cases])
                        .value_counts().head(8).to_dict()}
        log("track_done", track=track, **stats[track])
    (WORK / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    log("DONE", **{k: v["cases"] for k, v in stats.items()})


if __name__ == "__main__":
    main()
