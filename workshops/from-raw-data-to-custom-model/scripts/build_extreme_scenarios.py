"""Extreme demo scenarios: requests built to max every difficulty factor.

Hard benchmark requests max the hidden-need and item-count factors but leave budget tightness
(~5/20), scarcity of good options (~0/10) and explicit avoids (~1/6) on the table. Here:
  - hidden need with a conflicting interest, 3 items, 1-2 explicit avoids
  - budget set just above the cheapest relevant, allowed 3-item set in the whole catalog
    (1.00-1.12x, rounded up to $5), which also makes good options scarce in the pool
  - kept only if a relevant set fits the budget and the grade is >= 70 with BOTH 40- and
    80-candidate pools; messages that name the hidden need are dropped
Also writes data/scenario_grades.json: difficulty of every app scenario (app presets,
benchmark, extreme) at 40 and 80 candidates, so the app's random buttons follow the pool size.

    ~/.venvs/gift_concierge/bin/python scripts/build_extreme_scenarios.py
"""
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from octank import catalog, config, difficulty, generate, pools  # noqa: E402
from octank.cache import JsonlCache  # noqa: E402
from octank.formats import write_jsonl  # noqa: E402
from octank.util import as_list  # noqa: E402

N = int(os.environ.get("N_EXTREME", "60"))
MIN_SCORE = 70
WORK = ROOT / "work" / "extreme"
WORK.mkdir(parents=True, exist_ok=True)
OUT = config.DATA_DIR / "extreme_cases.jsonl"
GRADES = config.DATA_DIR / "scenario_grades.json"
cat = catalog.load_clean()


def extreme_specs(n):
    rng, out, tried = random.Random(707), [], 0
    while len(out) < n and tried < n * 400:
        tried += 1
        s = generate.extreme_spec(rng, tried, cat)
        if s:
            s["request_id"] = f"hx{tried:05d}"
            d = s.pop("_difficulty")
            out.append((s, d["40"], d["80"]))
    print(f"specs: kept {len(out)} of {tried} tried", flush=True)
    return out


def main():
    specs = extreme_specs(int(N * 1.4))                   # spare for messages dropped as leaks
    msgs = generate.write_messages([s for s, _, _ in specs],
                                   cache=JsonlCache(WORK / "messages.jsonl"), workers=16)
    cases = []
    for s, d40, d80 in specs:
        m = msgs.get(s["request_id"])
        if not m or any(h.replace("_", " ") in m.lower() for h in s["hidden_avoid"]):
            continue
        req = {**s, "message": m, "source": "extreme"}
        cases.append({"request": req, "candidates": pools.build_pool(req, cat), "difficulty": {"40": d40, "80": d80}})
        if len(cases) >= N:
            break
    write_jsonl(cases, OUT)
    print(f"wrote {len(cases)} extreme scenarios -> {OUT}", flush=True)

    # difficulty of every scenario the app can draw, at both pool sizes
    grades = {}
    for f in ("app_scenarios.jsonl", "benchmark_cases.jsonl", "extreme_cases.jsonl"):
        for c in map(json.loads, open(config.DATA_DIR / f)):
            r = c["request"]
            if r["request_id"] in grades:
                continue
            grades[r["request_id"]] = {str(k): difficulty.grade(r, pool)["score"] for k, pool in
                                       ((40, pools.build_pool(r, cat)), (80, pools.build_pool_80(r, cat)))}
    GRADES.write_text(json.dumps(grades, indent=0))
    tiers = {}
    for g in grades.values():
        for k, v in g.items():
            tiers.setdefault(k, {}).setdefault(difficulty.tier(v), 0)
            tiers[k][difficulty.tier(v)] += 1
    print(f"graded {len(grades)} scenarios -> {GRADES}; tiers by pool size: {tiers}", flush=True)


if __name__ == "__main__":
    main()
