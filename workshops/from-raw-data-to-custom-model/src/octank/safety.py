"""Toxicity filter: gpt-oss-safeguard-20b classifies each request against a written policy.

The policy is plain markdown (policy/toxicity.md). Edit it and re-run - no retraining needed.
"""
from __future__ import annotations

import pandas as pd

from . import llm
from .cache import content_hash
from .config import POLICY_DIR, SAFETY_MODEL

DEFAULT_POLICY = POLICY_DIR / "toxicity.md"


def load_policy(path=DEFAULT_POLICY) -> str:
    return open(path).read()


def classify_one(request_id: str, message: str, policy: str, region: str | None = None) -> dict:
    data, _ = llm.converse_json(SAFETY_MODEL, f"Request: \"\"\"{message}\"\"\"", system=policy,
                                max_tokens=800, temperature=0.0, region=region)
    data = data if isinstance(data, dict) else {}
    try:
        violation = int(data.get("violation", 0))
    except (TypeError, ValueError):
        violation = 0
    return {"request_id": request_id, "violation": violation,
            "category": str(data.get("category", "none")),
            "confidence": str(data.get("confidence", "low")),
            "rationale": str(data.get("rationale", ""))[:300]}


def classify_df(df: pd.DataFrame, policy: str | None = None, cache=None, workers: int = 48,
                region: str | None = None) -> pd.DataFrame:
    policy = policy or load_policy()
    # Cache hits require the same message text (ids alone can be re-used by a resumed run).
    todo = [(r.request_id, r.message) for r in df.itertuples()
            if cache is None or not cache.fresh(r.request_id, r.message)]

    def run(item):
        return classify_one(item[0], item[1], policy, region=region)

    def persist(item, res):
        if cache is not None and isinstance(res, dict):
            cache.put({**res, "input_hash": content_hash(item[1])})

    results = llm.pmap(run, todo, workers=workers, desc="safety", on_result=persist)
    recs = {r["request_id"]: r for _, r, e in results if r}
    if cache is not None:
        recs = {**{r.request_id: cache.get(r.request_id) for r in df.itertuples()
                   if cache.fresh(r.request_id, r.message)}, **recs}
    out = pd.DataFrame([recs[i] for i in df["request_id"] if i in recs])
    return out.add_prefix("safety_").rename(columns={"safety_request_id": "request_id"})


def decide(row) -> str:
    """drop | review | keep"""
    if row.get("safety_violation") == 1:
        return "review" if row.get("safety_confidence") == "low" else "drop"
    return "keep"
