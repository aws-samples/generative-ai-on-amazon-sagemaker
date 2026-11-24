"""Benchmark suites: the held-out evaluation cases every model is scored on.

| Track          | Cases | Candidates | What it tests                                              |
|----------------|-------|------------|------------------------------------------------------------|
| typical        | 261   | 40         | the natural request mix (held out from training)           |
| hard_known     | 201   | 40         | hard-by-construction requests, training-catalog products   |
| hard_novel     | 231   | 40         | the same kind of requests, 311 products never in training  |
| hard_known_80  | 201   | 80         | hard_known with twice the traps                            |
| hard_novel_80  | 230   | 80         | hard_novel with twice the traps                            |
| extreme_80     | 60    | 80         | the Extreme demo scenarios (difficulty >= 70)              |

`cases_40()` and `cases_80()` return the case dicts used by `fasteval` (generation budgets differ:
`config.EVAL_MAX_NEW_TOKENS` / `EVAL_MAX_NEW_TOKENS_80`).
"""
from __future__ import annotations

import json

import pandas as pd

from . import catalog, difficulty, pools
from .config import DATA_DIR

TRACKS = {"typical": "Typical, 40 candidates", "hard_known": "Hard, known products, 40",
          "hard_novel": "Hard, new products, 40", "hard_known_80": "Hard, known products, 80",
          "hard_novel_80": "Hard, new products, 80", "extreme_80": "Extreme scenarios, 80"}
HARD = ["hard_known", "hard_novel", "hard_known_80", "hard_novel_80"]
# Problem types for reporting (the pooled average hides where RL helps).
PROBLEM_TYPES = [("Typical", ["typical"]), ("Hard, 40 candidates", HARD[:2]), ("Hard, 80 candidates", HARD[2:]),
                 ("Extreme, 80 candidates", ["extreme_80"]), ("All hard + Extreme", HARD + ["extreme_80"]),
                 ("Overall", list(TRACKS))]


def _load(path, track: str) -> list[dict]:
    out = []
    for c in map(json.loads, open(path)):
        d = c.get("difficulty")
        if d is None:
            d = difficulty.grade(c["request"], c["candidates"])["score"]
        out.append({**c, "track": track, "difficulty": d})
    return out


def cases_40() -> list[dict]:
    return (_load(DATA_DIR / "benchmark_cases.jsonl", "typical")
            + _load(DATA_DIR / "hard_known_cases.jsonl", "hard_known")
            + _load(DATA_DIR / "hard_novel_cases.jsonl", "hard_novel"))


def cases_80() -> list[dict]:
    """Hard tracks rebuilt with 80-candidate pools (the novel track draws from the raw novel
    catalog, as published), plus the Extreme scenarios."""
    cats = {"known": catalog.load_clean(), "novel": pd.read_json(catalog.NOVEL_PATH, lines=True)}
    out = []
    for t in ("known", "novel"):
        for c in map(json.loads, open(DATA_DIR / f"hard_{t}_cases.jsonl")):
            pool = pools.build_pool_80(c["request"], cats[t])
            g = difficulty.grade(c["request"], pool)
            if g["feasible"]:
                out.append({"request": c["request"], "candidates": pool, "reference": None,
                            "track": f"hard_{t}_80", "difficulty": g["score"]})
    for c in map(json.loads, open(DATA_DIR / "extreme_cases.jsonl")):
        pool = pools.build_pool_80(c["request"], cats["known"])
        out.append({"request": c["request"], "candidates": pool, "reference": None, "track": "extreme_80",
                    "difficulty": difficulty.grade(c["request"], pool)["score"]})
    return out
