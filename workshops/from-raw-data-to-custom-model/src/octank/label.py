"""Auto-labeling with confidence (gpt-oss-20b) + gold-set calibration.

The free-text message is the source of truth; the structured form fields are often missing or
wrong. The labeler extracts structured labels from the message and returns a confidence score.
Disagreements with the form and low-confidence labels are routed to human review.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import llm
from .cache import content_hash
from .config import GENERATOR_MODEL, LABELER_MODEL
from .vocab import AGE_BANDS, INTERESTS, LABEL_ATTRIBUTES, OCCASIONS, RELATIONSHIPS

LABEL_FIELDS = ["occasion", "relationship", "age_band", "budget_total", "num_items",
                "interests", "avoid", "intent_clear"]

_PROMPT = """Extract structured labels from this gift-shop chat message.

MESSAGE: \"\"\"{message}\"\"\"

Allowed values:
- occasion: one of {occasions}, or "unknown"
- relationship (recipient's relationship to the shopper): one of {relationships}, or "other"
- age_band (recipient): one of {ages}, or "unknown"
- budget_total: total budget in USD as a number, or null if not stated
- num_items: number of gifts wanted (integer), or null if not stated (assume 1 if they say "a gift")
- interests: list, ONLY from {interests}
- avoid: list of things to avoid, ONLY from {attributes}
- intent_clear: false if ANY of these hold, else true:
  * the constraints contradict each other (e.g. "luxurious" items with a tiny budget, an
    alcohol gift for a sober friend)
  * key facts are unknown to the shopper (recipient's age for a child, what they like, which
    occasion it is)
  * the shopper's intent toward the recipient is unclear or possibly unwelcome (romantic gifts
    for a coworker/boss/married person, "not sure where we stand" with an ex, secret gifts to
    a minor)
- confidence: your confidence (0.0-1.0) that ALL labels above are correct. Use <= 0.6 when
  you had to guess any field.

Return ONLY JSON with keys: occasion, relationship, age_band, budget_total, num_items,
interests, avoid, intent_clear, confidence."""


def _prompt(message: str) -> str:
    return _PROMPT.format(message=message, occasions=OCCASIONS, relationships=RELATIONSHIPS,
                          ages=AGE_BANDS, interests=INTERESTS, attributes=LABEL_ATTRIBUTES)


def _normalize(d: dict) -> dict:
    def num(v):
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None
    occ = d.get("occasion")
    rel = d.get("relationship")
    age = d.get("age_band")
    n_items = num(d.get("num_items"))
    return {
        "occasion": occ if occ in OCCASIONS else None,
        "relationship": rel if rel in RELATIONSHIPS else None,
        "age_band": age if age in AGE_BANDS else None,
        "budget_total": num(d.get("budget_total")),
        "num_items": int(n_items) if n_items else None,
        "interests": [t for t in (d.get("interests") or []) if t in INTERESTS],
        "avoid": [t for t in (d.get("avoid") or []) if t in LABEL_ATTRIBUTES],
        "intent_clear": bool(d.get("intent_clear", True)),
        "confidence": float(np.clip(num(d.get("confidence")) or 0.0, 0.0, 1.0)),
    }


def label_one(request_id: str, message: str, model: str = LABELER_MODEL,
              region: str | None = None) -> dict:
    data, _ = llm.converse_json(model, _prompt(message), max_tokens=1500, temperature=0.0,
                                reasoning_effort="low", region=region)
    return {"request_id": request_id, **_normalize(data if isinstance(data, dict) else {})}


def label_df(df: pd.DataFrame, cache=None, model: str = LABELER_MODEL, workers: int = 48,
             region: str | None = None) -> pd.DataFrame:
    """Label every row (resumable via cache). Returns labels with `lbl_` prefix joined on id."""
    # Cache hits require the same message text (ids alone can be re-used by a resumed run).
    todo = [(r.request_id, r.message) for r in df.itertuples()
            if cache is None or not cache.fresh(r.request_id, r.message)]

    def run(item):
        return label_one(item[0], item[1], model=model, region=region)

    def persist(item, res):
        if cache is not None and isinstance(res, dict):
            cache.put({**res, "input_hash": content_hash(item[1])})

    results = llm.pmap(run, todo, workers=workers, desc=f"label[{model.split('.')[1][:12]}]",
                       on_result=persist)
    recs = {r["request_id"]: r for _, r, e in results if r}
    if cache is not None:
        recs = {**{r.request_id: cache.get(r.request_id) for r in df.itertuples()
                   if cache.fresh(r.request_id, r.message)}, **recs}
    lab = pd.DataFrame([recs[i] for i in df["request_id"] if i in recs])
    return lab.add_prefix("lbl_").rename(columns={"lbl_request_id": "request_id"})


def compare_with_form(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where labels disagree with the structured form fields."""
    out = df.copy()
    out["dis_occasion"] = out["occasion"].notna() & out["lbl_occasion"].notna() & \
        (out["occasion"] != out["lbl_occasion"])
    out["dis_budget"] = out["budget_total"].notna() & out["lbl_budget_total"].notna() & \
        ((out["budget_total"] - out["lbl_budget_total"]).abs() > 0.1 * out["budget_total"])
    out["dis_relationship"] = out["lbl_relationship"].notna() & \
        (out["relationship"] != out["lbl_relationship"])
    out["any_disagreement"] = out["dis_occasion"] | out["dis_budget"]
    return out


def route_for_review(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Reason each row needs human review ('' = no review needed)."""
    def reason(r):
        if not r.get("lbl_intent_clear", True):
            return "unclear_intent"
        if r.get("any_disagreement"):
            return "form_disagrees_with_text"
        if pd.isna(r.get("budget_total")) and pd.isna(r.get("lbl_budget_total")):
            return "no_budget_anywhere"
        if r.get("lbl_occasion") is None:
            return "occasion_not_in_text"
        if r.get("lbl_confidence", 1.0) < threshold:
            return "low_confidence"
        return ""
    return df.apply(reason, axis=1)


# --- Gold set: calibrate the labeler against reference labels -----------------------------------
def field_accuracy(pred: pd.DataFrame, gold: pd.DataFrame,
                   fields=("occasion", "relationship", "budget_total", "intent_clear")) -> dict:
    m = pred.merge(gold, on="request_id", suffixes=("_pred", "_gold"))
    acc = {}
    for f in fields:
        a, b = m[f"{f}_pred"], m[f"{f}_gold"]
        if f == "budget_total":
            ok = (a.isna() & b.isna()) | ((a - b).abs() <= 0.05 * b.abs().clip(lower=1))
        else:
            ok = (a == b) | (a.isna() & b.isna())
        acc[f] = round(100 * ok.mean(), 1)
    return acc


def calibration_curve(pred: pd.DataFrame, gold: pd.DataFrame,
                      thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    """For each threshold: % rows auto-accepted and their accuracy (occasion+relationship+budget)."""
    m = pred.merge(gold, on="request_id", suffixes=("_pred", "_gold"))
    budget_ok = (m["budget_total_pred"].isna() & m["budget_total_gold"].isna()) | \
        ((m["budget_total_pred"] - m["budget_total_gold"]).abs()
         <= 0.05 * m["budget_total_gold"].abs().clip(lower=1))
    correct = (m["occasion_pred"] == m["occasion_gold"]) & \
        (m["relationship_pred"] == m["relationship_gold"]) & budget_ok
    rows = []
    for t in thresholds:
        auto = m["confidence_pred"] >= t
        rows.append({"threshold": t, "auto_accept_pct": round(100 * auto.mean(), 1),
                     "review_pct": round(100 * (1 - auto.mean()), 1),
                     "auto_accuracy_pct": round(100 * correct[auto].mean(), 1) if auto.any()
                     else None})
    return pd.DataFrame(rows)


def draft_gold_labels(df: pd.DataFrame, cache=None, workers: int = 24,
                      region: str | None = None) -> pd.DataFrame:
    """Gold-set draft labels from the stronger gpt-oss-120b (humans correct them afterwards)."""
    lab = label_df(df, cache=cache, model=GENERATOR_MODEL, workers=workers, region=region)
    lab.columns = [c.replace("lbl_", "") for c in lab.columns]
    lab["label_source"] = "model_draft:" + GENERATOR_MODEL
    return lab


def to_json_lists(df: pd.DataFrame, cols=("interests", "avoid")) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = out[c].apply(lambda v: json.dumps(list(v) if v is not None else []))
    return out
