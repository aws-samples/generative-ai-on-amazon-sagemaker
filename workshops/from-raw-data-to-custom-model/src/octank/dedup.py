"""De-duplication: exact (normalized hash) + near-duplicate (MinHash + LSH on word shingles).

MinHash estimates Jaccard similarity between the sets of word 3-grams of two messages; LSH
buckets similar signatures so we only compare likely pairs. Lightweight - no embedding model.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd
from datasketch import MinHash, MinHashLSH

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s$]")


def normalize(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", (text or "").lower())).strip()


def exact_duplicates(df: pd.DataFrame, col: str = "message") -> pd.Series:
    """True for every row whose normalized text already appeared earlier (keep first)."""
    h = df[col].map(lambda t: hashlib.sha3_256(normalize(t).encode()).hexdigest())
    return h.duplicated(keep="first")


def shingles(text: str, k: int = 3) -> set[str]:
    toks = normalize(text).split()
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash(text: str, num_perm: int = 128, k: int = 3) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=1)
    for s in shingles(text, k):
        m.update(s.encode())
    return m


def near_duplicates(df: pd.DataFrame, col: str = "message", threshold: float = 0.5,
                    num_perm: int = 128, k: int = 3) -> pd.DataFrame:
    """Return pairs (keep_id, dup_id, jaccard_est) for near-duplicate messages.

    Rows are processed in order; a row that matches an earlier kept row is marked a duplicate.
    """
    if df["request_id"].duplicated().any():
        raise ValueError("request_id must be unique before near-duplicate detection")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs: dict[str, MinHash] = {}
    pairs = []
    for rid, text in zip(df["request_id"], df[col]):
        m = minhash(text, num_perm, k)
        hits = lsh.query(m)
        if hits:
            best = max(hits, key=lambda h: sigs[h].jaccard(m))
            pairs.append({"keep_id": best, "dup_id": rid, "jaccard": round(sigs[best].jaccard(m), 3)})
            continue
        lsh.insert(rid, m)
        sigs[rid] = m
    return pd.DataFrame(pairs, columns=["keep_id", "dup_id", "jaccard"])


def jaccard(a: str, b: str, k: int = 3) -> float:
    sa, sb = shingles(a, k), shingles(b, k)
    return len(sa & sb) / max(len(sa | sb), 1)
