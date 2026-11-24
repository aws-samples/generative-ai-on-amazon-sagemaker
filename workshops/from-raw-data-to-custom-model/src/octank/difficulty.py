"""Deterministic difficulty grade for a gift request + its candidate pool.

Uses the authoritative tags (which the model does NOT see) to measure what makes a request
hard: a need implied only by the shopper's situation, several items under one budget, a tight
budget for the relevant items, few good options and many traps (strong interest matches that carry
an avoided attribute, near-budget items).
"""
from __future__ import annotations

from .util import as_list

# points per factor (max 100)
W_HIDDEN, W_ITEMS, W_BUDGET, W_OPTIONS, W_TRAPS, W_AVOID = 30, 24, 20, 10, 10, 6
LABELS = [(30, "Easy"), (55, "Medium"), (101, "Hard")]


TIERS = [(30, "Easy"), (55, "Medium"), (70, "Hard"), (101, "Extreme")]


def tier(score: float) -> str:
    """Display tier for the app: grade()'s labels plus "Extreme" (>= 70). grade() itself keeps
    Easy/Medium/Hard so earlier benchmark selections (Hard >= 55) stay reproducible."""
    return next(lbl for cut, lbl in TIERS if score < cut)


def grade(req: dict, pool: list[dict]) -> dict:
    n = int(req.get("num_items") or 1)
    budget = float(req["budget_total"])
    interests = set(as_list(req.get("interests")))
    explicit = set(as_list(req.get("avoid")))
    hidden = set(as_list(req.get("hidden_avoid")))
    avoid = explicit | hidden

    def bad(c):
        return bool(avoid & set(as_list(c.get("attributes"))))

    relevant = [c for c in pool if interests & set(as_list(c.get("interests"))) and not bad(c)]
    affordable = sorted(float(c["price"]) for c in relevant if float(c["price"]) <= budget)
    traps = [c for c in pool if interests & set(as_list(c.get("interests"))) and bad(c)]
    hidden_traps = [c for c in traps if hidden & set(as_list(c.get("attributes")))]

    feasible = len(affordable) >= n and sum(affordable[:n]) <= budget   # the cheapest n must fit together
    cheapest = sum(affordable[:n]) if feasible else None
    slack = (1 - cheapest / budget) if feasible else 0.0      # 0 = budget exactly covers the cheapest set

    pts = {
        "Need implied by the situation": W_HIDDEN if hidden else 0,
        "Items to fit in one budget": W_ITEMS * (n - 1) / 2,
        "Budget tightness": W_BUDGET * (1 - min(slack / 0.5, 1.0)),
        "Few good options": W_OPTIONS * max(0, 6 - len(affordable)) / 6,
        "Trap items": W_TRAPS * min(len(traps), 8) / 8,
        "Explicit avoid list": W_AVOID * min(len(explicit), 3) / 3,
    }
    score = round(sum(pts.values()))
    label = next(lbl for cut, lbl in LABELS if score < cut)
    notes = [
        f"hidden need: avoid {', '.join(sorted(hidden))} ({len(hidden_traps)} tempting candidates carry it)"
        if hidden else "no hidden need",
        f"{n} item(s) for ${budget:.0f}; cheapest relevant set ${cheapest:.0f} ({slack:.0%} slack)"
        if feasible else f"{n} item(s) for ${budget:.0f}: no relevant set fits the budget",
        f"{len(affordable)} relevant, allowed, affordable candidates of {len(pool)}",
        f"{len(traps)} interest matches carry an avoided attribute",
    ]
    return {"score": score, "label": label, "points": {k: round(v, 1) for k, v in pts.items()},
            "notes": notes, "feasible": feasible, "hidden_need": bool(hidden)}
