"""Small helpers for values that round-trip through pandas/parquet (numpy arrays, NaN)."""
from __future__ import annotations

import math


def as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, float) and math.isnan(v):
        return []
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, str):
        return [v] if v else []
    return list(v)


def val(v):
    """None for NaN/None, else the value (numpy scalars -> python)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item") and not hasattr(v, "__len__"):
        return v.item()
    return v


def clean_record(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if k in ("interests", "avoid", "hidden_avoid"):
            out[k] = as_list(v)
        elif hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
            out[k] = v.tolist()
        else:
            out[k] = val(v)
    return out
