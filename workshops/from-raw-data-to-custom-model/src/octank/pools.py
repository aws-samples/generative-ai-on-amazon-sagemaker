"""Candidate pools: the CANDIDATES list each request is shown, with deliberate traps.

40 candidates per request (`build_pool`): ~6 real matches (on-interest, allowed, affordable) hidden
among near-budget items that break a multi-item total, over-budget items, strong interest matches
that carry an avoided or situation-implied attribute, near-duplicates of the top match, generic
items and filler. `build_pool_80` doubles every trap group (same ~6 real matches) as a stress test.
Deterministic per request_id.
"""
from __future__ import annotations

import hashlib
import random

import pandas as pd

from .util import as_list

POOL_SIZE = 40
POOL_SIZE_80 = 80
_KEEP = ["id", "name", "category", "price", "interests", "attributes", "description"]


def _rng(request_id: str) -> random.Random:
    # Deterministic per request: the same id always gets the same pool. Changing the salt or the hash
    # changes every pool, and everything built from them (benchmarks, datasets, difficulty grades).
    return random.Random(int(hashlib.sha3_256(("v2" + request_id).encode()).hexdigest()[:8], 16))


def _avoid(req: dict) -> set:
    return set(as_list(req.get("avoid"))) | set(as_list(req.get("hidden_avoid")))


def _frames(req: dict, catalog: pd.DataFrame):
    budget = float(req["budget_total"])
    per_item = budget / int(req.get("num_items") or 1)
    interests, avoid = set(as_list(req.get("interests"))), _avoid(req)
    c = catalog.copy()
    c["_ov"] = c["interests"].apply(lambda t: len(interests & set(as_list(t))))
    c["_av"] = c["attributes"].apply(lambda t: bool(avoid & set(as_list(t))))
    return c, {
        "good": c[(c._ov > 0) & ~c._av & (c.price <= per_item)],
        "near_budget": c[(c._ov > 0) & ~c._av & (c.price > per_item) & (c.price <= per_item * 1.25)],
        "over": c[(c._ov > 0) & ~c._av & (c.price > budget)],
        "traps": c[(c._ov > 0) & c._av].sort_values("_ov", ascending=False),
        "generic": c[(c._ov == 0) & ~c._av & (c.price <= per_item)],
    }


def _assemble(c: pd.DataFrame, chosen: list[str], rng: random.Random) -> list[dict]:
    rng.shuffle(chosen)
    recs = c.set_index("id").loc[chosen].reset_index()[_KEEP].to_dict(orient="records")
    for r in recs:
        r["interests"], r["attributes"] = as_list(r["interests"]), as_list(r["attributes"])
        r["price"] = float(r["price"])
    return recs


def _pool(req: dict, catalog: pd.DataFrame, size: int, rng: random.Random, plan) -> list[dict]:
    c, f = _frames(req, catalog)
    chosen: list[str] = []

    def add(frame, k):
        ids = list(frame["id"])
        rng.shuffle(ids)
        for i in ids:
            if len(chosen) >= size or k <= 0:
                break
            if i not in chosen:
                chosen.append(i)
                k -= 1

    for group, k in plan:
        if group == "near_duplicates":
            if chosen:                                    # same category as the top match
                add(c[c.category == c.set_index("id").loc[chosen[0], "category"]], k)
        elif group == "filler":
            add(c, size)
        elif group == "good":
            add(f["good"].sort_values("_ov", ascending=False).head(15), k)
        elif group == "traps":
            add(f["traps"].head(k * 5 // 2), k)           # strongest interest matches first
        else:
            add(f[group], k)
    return _assemble(c, chosen, rng)


def build_pool(req: dict, catalog: pd.DataFrame, size: int = POOL_SIZE) -> list[dict]:
    """40 candidates: 6 real matches, 6 near-budget, 4 over budget, 8 traps (interest match carrying
    an avoided / implied attribute), 4 near-duplicates, 6 generic, filler."""
    plan = [("good", 6), ("near_budget", 6), ("over", 4), ("traps", 8), ("near_duplicates", 4),
            ("generic", 6), ("filler", 0)]
    return _pool(req, catalog, size, _rng(str(req["request_id"])), plan)


def build_pool_80(req: dict, catalog: pd.DataFrame, size: int = POOL_SIZE_80) -> list[dict]:
    """80 candidates: the same 6 real matches, every trap group doubled (stress test)."""
    plan = [("good", 6), ("near_budget", 12), ("over", 8), ("traps", 16), ("near_duplicates", 8),
            ("generic", 12), ("filler", 0)]
    return _pool(req, catalog, size, _rng(str(req["request_id"]) + "-80"), plan)


def has_feasible_answer(req: dict, pool: list[dict]) -> bool:
    """Is there at least one valid pick set? (cheapest allowed items fit the budget)"""
    n = int(req.get("num_items") or 1)
    ok = sorted(p["price"] for p in pool if not _avoid(req) & set(as_list(p["attributes"])))
    return len(ok) >= n and sum(ok[:n]) <= float(req["budget_total"])


def interest_match_rate(req: dict, pool: list[dict], pick_ids: list[str]) -> float:
    """Fraction of picks sharing at least one interest with the request."""
    by = {c["id"]: c for c in pool}
    ints = set(as_list(req.get("interests")))
    hits = [bool(ints & set(as_list(by[i]["interests"]))) for i in pick_ids if i in by]
    return sum(hits) / len(hits) if hits else 0.0


def has_relevant_option(req: dict, pool: list[dict]) -> bool:
    """At least one affordable candidate shares an interest and has no explicitly avoided attribute.
    Requests without one can only be answered off-topic - poor training signal.
    (Explicit avoid only, as used to build the published RL prompt sets.)"""
    ints = set(as_list(req.get("interests")))
    avoid = set(as_list(req.get("avoid")))
    budget = float(req["budget_total"])
    return any(ints & set(as_list(c["interests"])) and not avoid & set(as_list(c["attributes"]))
               and c["price"] <= budget for c in pool)


def cheapest_relevant_set(spec: dict, catalog: pd.DataFrame, n: int) -> float | None:
    """Price of the cheapest n relevant (interest match), allowed items in the whole catalog."""
    interests = set(as_list(spec.get("interests")))
    ok = catalog[catalog.interests.apply(lambda t: bool(interests & set(as_list(t))))
                 & ~catalog.attributes.apply(lambda t: bool(_avoid(spec) & set(as_list(t))))]
    prices = sorted(ok.price.astype(float))
    return sum(prices[:n]) if len(prices) >= n else None
