"""Synthetic gap-fill: generate new requests for under-covered segments, same technique."""
from __future__ import annotations

import random

import pandas as pd

from .generate import sample_spec, write_messages
from .vocab import BUDGET_BANDS

_BAND_IDX = {f"{lo}-{hi}": i for i, (lo, hi) in enumerate(BUDGET_BANDS)}


def gapfill_specs(gaps: pd.DataFrame, id_offset: int, seed: int = 99,
                  max_per_gap: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    specs, i = [], id_offset
    for g in gaps.itertuples():
        need = g.need if max_per_gap is None else min(g.need, max_per_gap)
        for _ in range(need):
            if g.dimension == "occasion":
                s = sample_spec(rng, i, occasion=g.value)
            else:
                s = sample_spec(rng, i, budget_band_idx=_BAND_IDX[g.value])
            s["request_id"] = f"g{i:06d}"
            specs.append(s)
            i += 1
    return specs


def generate_gapfill(gaps: pd.DataFrame, id_offset: int = 0, seed: int = 99, workers: int = 32,
                     cache=None, max_per_gap: int | None = None,
                     region: str | None = None) -> pd.DataFrame:
    specs = gapfill_specs(gaps, id_offset, seed, max_per_gap)
    if not specs:
        return pd.DataFrame()
    msgs = write_messages(specs, workers=workers, cache=cache, region=region)
    rows = [dict(s, message=msgs[s["request_id"]], source="gapfill", _flaw="none", _dup_of=None,
                 _orig_occasion=s["occasion"], _orig_budget=s["budget_total"])
            for s in specs if s["request_id"] in msgs]
    return pd.DataFrame(rows)
