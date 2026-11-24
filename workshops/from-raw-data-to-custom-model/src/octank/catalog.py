"""Octank product catalog: generate once (instructor), plant defects, profile, clean.

`data/catalog_raw.jsonl` (~1,190 items, 12 categories x ~100, with planted defects) was generated
once with `generate()` and is committed as seed data. Attendees never regenerate it - they load,
profile and clean it with `clean()`, which is deterministic, so everyone gets the same catalog.
`data/catalog_novel.jsonl` (311 products never used in training) feeds one hard benchmark track.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from . import llm
from .config import DATA_DIR, GENERATOR_MODEL
from .vocab import ATTRIBUTES, CATEGORIES, CATEGORY_ALIASES, INTERESTS

RAW_PATH = DATA_DIR / "catalog_raw.jsonl"
NOVEL_PATH = DATA_DIR / "catalog_novel.jsonl"

PRICE_RANGES = {
    "books": (12, 60), "music_vinyl": (18, 140), "outdoor_fishing": (15, 320),
    "kitchen_coffee": (12, 260), "home_decor": (15, 240), "self_care": (10, 150),
    "games_puzzles": (12, 120), "kids_toys": (10, 130), "tech_gadgets": (20, 400),
    "experiences_giftcards": (25, 300), "stationery": (8, 160), "pets": (8, 140),
}

_CATALOG_PROMPT = """You are building a product catalog for Octank, an online gift shop.
Write {n} distinct, realistic products for the category "{category}".
Batch theme (to keep products varied across batches): {theme}

Rules:
- Price in USD between {lo} and {hi}; vary prices across the range.
- "interests": 1-3 values chosen ONLY from this list: {interests}
- "attributes": 0-3 values chosen ONLY from this list: {attributes}
  (e.g. "gadget" for electronics, "alcohol" for wine/spirits, "food" for edibles,
  "caffeine" for coffee/tea, "experience" for classes/tickets, "consumable" for things used up).
- "description": one or two sentences, concrete and specific, no marketing fluff.
- Names must be specific (brand-like, not generic) and unique.

Return ONLY a JSON list of objects with keys: name, price, interests, attributes, description."""

_THEMES = ["entry-level picks", "premium picks", "quirky / unique finds", "classic best sellers",
           "eco-friendly options", "small budget-friendly items", "items for beginners",
           "items for enthusiasts"]


def _gen_batch(args):
    category, theme, n, seed, attributes, focus = args
    lo, hi = PRICE_RANGES[category]
    if focus:
        theme = f"{theme}; make most items relevant to these interests: {', '.join(focus)}"
    prompt = _CATALOG_PROMPT.format(n=n, category=category, theme=theme, lo=lo, hi=hi,
                                    interests=", ".join(INTERESTS),
                                    attributes=", ".join(attributes))
    data, _ = llm.converse_json(GENERATOR_MODEL, prompt, max_tokens=6000, temperature=0.9,
                                reasoning_effort="low")
    items = []
    for d in data if isinstance(data, list) else []:
        try:
            price = round(float(d["price"]), 2)
        except (KeyError, TypeError, ValueError):
            continue
        items.append({
            "name": str(d.get("name", "")).strip(),
            "category": category,
            "price": min(max(price, lo), hi),
            "interests": [t for t in d.get("interests", []) if t in INTERESTS][:3],
            "attributes": [t for t in d.get("attributes", []) if t in attributes][:3],
            "description": str(d.get("description", "")).strip(),
        })
    return items


def generate(per_category: int = 100, batch_size: int = 20, seed: int = 17,
             workers: int = 24, attributes=ATTRIBUTES, interest_focus: bool = True) -> pd.DataFrame:
    """Generate the clean catalog with gpt-oss-120b, then plant defects.
    interest_focus: each batch is steered toward a rotating slice of INTERESTS (even coverage)."""
    rng = random.Random(seed)
    jobs, k = [], 0
    for cat in CATEGORIES:
        for b in range(max(1, per_category // batch_size)):
            focus = [INTERESTS[(k * 5 + j) % len(INTERESTS)] for j in range(5)] if interest_focus else None
            k += 1
            jobs.append((cat, _THEMES[(b + rng.randrange(8)) % len(_THEMES)], batch_size, seed + b,
                         list(attributes), focus))
    results = llm.pmap(_gen_batch, jobs, workers=workers, desc="catalog")
    items, seen = [], set()
    for _, res, err in results:
        for it in res or []:
            key = it["name"].lower()
            if it["name"] and key not in seen:
                seen.add(key)
                items.append(it)
    items.sort(key=lambda d: (d["category"], d["name"]))
    for i, it in enumerate(items):
        it["id"] = f"OCT-{1000 + i}"
    df = pd.DataFrame(items)[["id", "name", "category", "price", "interests", "attributes",
                              "description"]]
    return plant_defects(df, seed=seed)


def plant_defects(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Planted catalog defects: inconsistent category names, missing tags, duplicate SKUs."""
    rng = random.Random(seed)
    df = df.copy()
    df["interests"] = df["interests"].apply(list)
    inv = {}
    for alias, canon in CATEGORY_ALIASES.items():
        inv.setdefault(canon, []).append(alias)
    n = len(df)
    for i in rng.sample(range(n), k=int(n * 0.10)):              # 10% messy category names
        df.at[i, "category"] = rng.choice(inv[df.at[i, "category"]])
    for i in rng.sample(range(n), k=int(n * 0.05)):              # 5% missing interest tags
        df.at[i, "interests"] = []
    dups = df.iloc[rng.sample(range(n), k=max(3, int(n * 0.02)))].copy()  # 2% duplicate SKUs
    dups["name"] = dups["name"] + " "                             # trailing-space variants
    return pd.concat([df, dups], ignore_index=True)


def save_raw(df: pd.DataFrame, path: Path = RAW_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec) + "\n")
    return path


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:  # noqa: D103
    return pd.read_json(path, lines=True)


def profile(df: pd.DataFrame) -> dict:
    """Catalog data-quality report."""
    return {
        "items": len(df),
        "duplicate_skus": int(df["id"].duplicated().sum()),
        "non_canonical_categories": int((~df["category"].isin(CATEGORIES)).sum()),
        "category_spellings": sorted(set(df["category"]) - set(CATEGORIES)),
        "missing_interest_tags": int(df["interests"].apply(lambda x: len(x) == 0).sum()),
        "price_min": float(df["price"].min()), "price_max": float(df["price"].max()),
        "items_per_category": (df["category"].map(lambda c: CATEGORY_ALIASES.get(c, c))
                               .value_counts().to_dict()),
    }


def _infer_interests(row) -> list[str]:
    text = f"{row['name']} {row['description']}".lower()
    return [t for t in INTERESTS if t in text][:3]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cleaning: canonical categories, dedupe SKUs, backfill missing interest tags
    from name/description keywords (rows still untagged are dropped)."""
    out = df.copy()
    out["name"] = out["name"].str.strip()
    out["category"] = out["category"].map(lambda c: CATEGORY_ALIASES.get(c, c))
    out = out[out["category"].isin(CATEGORIES)]
    out = out.drop_duplicates(subset=["id"], keep="first")
    missing = out["interests"].apply(len) == 0
    out.loc[missing, "interests"] = out[missing].apply(_infer_interests, axis=1)
    out = out[out["interests"].apply(len) > 0]
    return out.sort_values("id").reset_index(drop=True)


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
