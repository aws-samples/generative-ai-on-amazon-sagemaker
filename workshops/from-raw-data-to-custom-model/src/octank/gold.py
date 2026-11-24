"""Gold set (~150) + human annotation slice (50).

Sampled from the instructor raw data, stratified by planted-flaw type so it covers the hard cases.
Draft labels come from gpt-oss-120b (model_draft); humans correct them (human_corrected).
"""
from __future__ import annotations

import json

import pandas as pd

from . import label, safety
from .config import DATA_DIR

GOLD_PATH = DATA_DIR / "gold_labels.jsonl"
HUMAN_SLICE_PATH = DATA_DIR / "human_annotation_slice.csv"

GOLD_MIX = {"none": 85, "toxic": 15, "ambiguous": 20, "contradiction": 15, "missing_budget": 8,
            "missing_occasion": 7}
HUMAN_MIX = {"none": 15, "toxic": 10, "ambiguous": 12, "contradiction": 8, "missing_budget": 3,
             "missing_occasion": 2}


def sample_gold(raw: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    parts = []
    for flaw, k in GOLD_MIX.items():
        g = raw[raw["_flaw"] == flaw]
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    return pd.concat(parts).drop_duplicates("message").reset_index(drop=True)


def build_gold(raw: pd.DataFrame, run, workers: int = 24) -> pd.DataFrame:
    g = sample_gold(raw)
    drafts = label.draft_gold_labels(g[["request_id", "message"]], cache=run.cache("gold_labels"),
                                     workers=workers, region=run.s.region)
    saf = safety.classify_df(g[["request_id", "message"]], cache=run.cache("gold_safety"),
                             workers=workers, region=run.s.region)
    out = g[["request_id", "message", "_flaw"]].merge(drafts, on="request_id") \
        .merge(saf[["request_id", "safety_violation", "safety_category"]], on="request_id")
    out = out.rename(columns={"safety_violation": "violation", "safety_category": "violation_category"})
    out["reviewer"] = ""
    return out


def save_gold(gold: pd.DataFrame) -> None:
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD_PATH, "w") as f:
        for r in json.loads(gold.to_json(orient="records")):
            f.write(json.dumps(r) + "\n")


def load_gold() -> pd.DataFrame:
    return pd.read_json(GOLD_PATH, lines=True)


def human_slice(gold: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    """50 rows for human annotators: message + model draft + empty human columns."""
    parts = []
    for flaw, k in HUMAN_MIX.items():
        g = gold[gold["_flaw"] == flaw]
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    s = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cols = ["occasion", "relationship", "age_band", "budget_total", "num_items", "interests",
            "avoid", "intent_clear", "violation"]
    out = s[["request_id", "message"]].copy()
    for c in cols:
        out[f"draft_{c}"] = s[c].apply(lambda v: json.dumps(v) if isinstance(v, list) else v)
        out[f"human_{c}"] = ""
    out["human_notes"] = ""
    out["annotator"] = ""
    return out


def merge_human(gold: pd.DataFrame, annotated_csv) -> pd.DataFrame:
    """Overwrite gold labels with completed human annotations (non-empty human_* cells)."""
    h = pd.read_csv(annotated_csv).fillna("")
    g = gold.set_index("request_id").copy()
    for r in h.itertuples():
        changed = False
        for c in ["occasion", "relationship", "age_band", "budget_total", "num_items",
                  "interests", "avoid", "intent_clear", "violation"]:
            v = getattr(r, f"human_{c}")
            if str(v).strip() == "":
                continue
            if c in ("interests", "avoid"):
                v = json.loads(v) if str(v).startswith("[") else [x.strip() for x in str(v).split(",")]
            elif c in ("budget_total",):
                v = float(v)
            elif c in ("num_items", "violation"):
                v = int(float(v))
            elif c == "intent_clear":
                v = str(v).lower() in ("true", "1", "yes")
            g.at[r.request_id, c] = v
            changed = True
        if changed:
            g.at[r.request_id, "label_source"] = "human_corrected"
            g.at[r.request_id, "reviewer"] = r.annotator
    return g.reset_index()
