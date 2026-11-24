"""Profiling: completeness, distributions, skew and coverage gaps."""
from __future__ import annotations

import math

import pandas as pd

from .vocab import OCCASIONS, budget_band

FIELDS = ["relationship", "age_band", "occasion", "budget_total", "num_items", "interests",
          "avoid", "notes", "message"]


def completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value rate per field (empty strings/lists count as missing for required fields)."""
    rows = []
    for c in FIELDS:
        if c not in df:
            continue
        s = df[c]
        if c in ("interests",):
            miss = s.apply(lambda v: v is None or (hasattr(v, "__len__") and len(v) == 0))
        elif c in ("avoid", "notes"):
            miss = s.isna()                       # empty avoid/notes is legitimate
        else:
            miss = s.isna() | (s.astype(str).str.strip() == "")
        rows.append({"field": c, "missing": int(miss.sum()), "missing_pct": 100 * miss.mean()})
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)


def distribution(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col]
    if col == "budget_total":
        s = s.map(budget_band)
    vc = s.value_counts(dropna=False)
    return pd.DataFrame({col: vc.index.astype(str), "count": vc.values,
                         "pct": 100 * vc.values / max(len(df), 1)})


def skew_report(df: pd.DataFrame, col: str = "occasion") -> dict:
    """Top-2 share and normalized entropy (1.0 = perfectly balanced)."""
    vc = df[col].dropna().value_counts(normalize=True)
    ent = -sum(p * math.log(p) for p in vc if p > 0)
    norm = ent / math.log(len(vc)) if len(vc) > 1 else 0.0
    return {"top2_share_pct": round(100 * vc.head(2).sum(), 1),
            "normalized_entropy": round(norm, 3), "n_values": int(len(vc))}


def coverage_gaps(df: pd.DataFrame, min_share: float = 0.02, min_count: int = 5) -> pd.DataFrame:
    """Occasions and budget bands below a minimum share -> synthetic gap-fill targets."""
    n = max(len(df), 1)
    target = max(min_count, int(math.ceil(min_share * n)))
    rows = []
    occ = df["occasion"].value_counts()
    for o in OCCASIONS:
        have = int(occ.get(o, 0))
        if have < target:
            rows.append({"dimension": "occasion", "value": o, "have": have, "target": target,
                         "need": target - have})
    bands = df["budget_total"].map(budget_band).value_counts()
    for b in ["0-25", "25-50", "50-100", "100-200", "200-500"]:
        have = int(bands.get(b, 0))
        if have < target:
            rows.append({"dimension": "budget_band", "value": b, "have": have, "target": target,
                         "need": target - have})
    return pd.DataFrame(rows)


def text_stats(df: pd.DataFrame) -> dict:
    lens = df["message"].fillna("").str.split().apply(len)
    return {"words_mean": round(lens.mean(), 1), "words_p95": int(lens.quantile(0.95)),
            "words_max": int(lens.max()), "empty": int((lens == 0).sum())}


def plot_overview(df: pd.DataFrame, title: str = ""):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    occ = df["occasion"].fillna("MISSING").value_counts()
    occ.plot.barh(ax=axes[0], color=["#c0392b" if v == "MISSING" else "#2e86de"
                                     for v in occ.index])
    axes[0].invert_yaxis()
    axes[0].set_title("occasion")
    df["budget_total"].map(budget_band).fillna("MISSING").value_counts().sort_index() \
        .plot.bar(ax=axes[1], color="#10ac84")
    axes[1].set_title("budget band")
    df["relationship"].value_counts().plot.bar(ax=axes[2], color="#8e44ad")
    axes[2].set_title("relationship")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
