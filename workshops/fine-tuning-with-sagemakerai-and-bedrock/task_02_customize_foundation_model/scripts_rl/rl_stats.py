"""Paired significance tests for model comparisons. Needs only numpy.

Every model is evaluated on the same prompts, so the comparison is paired: for
each prompt we take the score difference (candidate minus baseline) and ask
whether the mean difference is distinguishable from zero.

* ``paired_bootstrap_ci``: percentile bootstrap confidence interval of the mean difference.
* ``paired_permutation_pvalue``: two-sided sign-flip permutation test. It makes no
  normality assumption, which matters for bounded and skewed scores such as ROUGE.
* ``holm``: Holm-Bonferroni correction, because several metrics are tested at once.
"""

from typing import Dict, List, Sequence

import numpy as np


def _paired(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples need equal length, got {a.shape} and {b.shape}")
    keep = ~(np.isnan(a) | np.isnan(b))
    return b[keep] - a[keep]


def paired_bootstrap_ci(baseline, candidate, n_resamples: int = 10000, confidence: float = 0.95, seed: int = 0):
    diff = _paired(baseline, candidate)
    if diff.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_permutation_pvalue(baseline, candidate, n_resamples: int = 10000, seed: int = 0) -> float:
    diff = _paired(baseline, candidate)
    if diff.size == 0 or np.allclose(diff, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    # The +1 terms count the observed statistic itself, so p is never exactly zero.
    return float((np.sum(null >= observed - 1e-12) + 1) / (n_resamples + 1))


def cohens_dz(baseline, candidate) -> float:
    """Standardized paired effect size: mean difference over the std of the differences."""
    diff = _paired(baseline, candidate)
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    # Identical differences on every prompt have no spread; report 0 rather than infinity
    # so the result stays valid JSON for the pipeline's JsonGet.
    return float(diff.mean() / sd) if sd > 0 else 0.0


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def compare_models(baseline_rows: List[Dict], candidate_rows: List[Dict], metrics: List[str],
                   alpha: float = 0.05, n_resamples: int = 10000, seed: int = 0) -> Dict[str, Dict]:
    """Compare two per-sample result lists joined on ``id``.

    Returns, per metric: baseline and candidate means, mean delta, bootstrap CI,
    raw and Holm-adjusted p-values, effect size, n, and a ``significant`` flag.
    """
    base = {str(r["id"]): r for r in baseline_rows}
    cand = {str(r["id"]): r for r in candidate_rows}
    ids = sorted(set(base) & set(cand))
    out, raw_p = {}, {}
    for metric in metrics:
        a = np.array([np.nan if base[i].get(metric) is None else base[i][metric] for i in ids], dtype=float)
        b = np.array([np.nan if cand[i].get(metric) is None else cand[i][metric] for i in ids], dtype=float)
        keep = ~(np.isnan(a) | np.isnan(b))
        lo, hi = paired_bootstrap_ci(a[keep], b[keep], n_resamples=n_resamples, seed=seed)
        p = paired_permutation_pvalue(a[keep], b[keep], n_resamples=n_resamples, seed=seed)
        raw_p[metric] = p
        out[metric] = {
            "n": int(keep.sum()),
            "baseline_mean": float(a[keep].mean()) if keep.any() else float("nan"),
            "candidate_mean": float(b[keep].mean()) if keep.any() else float("nan"),
            "delta": float((b[keep] - a[keep]).mean()) if keep.any() else float("nan"),
            "ci_low": lo,
            "ci_high": hi,
            "p_value": p,
            "effect_size_dz": cohens_dz(a[keep], b[keep]),
        }
    for metric, p_adj in holm(raw_p).items():
        out[metric]["p_value_holm"] = p_adj
        out[metric]["significant"] = bool(p_adj <= alpha and out[metric]["ci_low"] > 0)
    return out
