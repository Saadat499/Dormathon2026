"""Exact binomial intervals, sample-size search and drift (PSI). Real calculations, no lookups."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.special import betaincinv


def clopper_pearson(x: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval for x successes out of n."""
    if n <= 0:
        return 0.0, 1.0
    a = (1.0 - confidence) / 2.0
    lo = 0.0 if x == 0 else float(betaincinv(x, n - x + 1, a))
    hi = 1.0 if x == n else float(betaincinv(x + 1, n - x, 1.0 - a))
    return lo, hi


def sample_size(target: float, expected: float, confidence: float = 0.95, prevalence: float = 0.2, max_n: int = 20000) -> dict:
    """Smallest number of urgent scans for which the exact lower bound would meet `target`,
    assuming the true sensitivity equals `expected`. Rounding the count of detections makes the
    answer slightly jumpy, so treat it as a planning estimate."""
    if not (0 < target < 1 and 0 < expected < 1):
        raise ValueError("Sensitivities must be between 0 and 1.")
    if not (0.5 < confidence < 1):
        raise ValueError("Confidence must be between 0.5 and 1.")
    if not (0 < prevalence <= 1):
        raise ValueError("The share of urgent scans must be above 0 and at most 1.")
    if expected <= target:
        return {
            "positives_needed": None, "scans_needed": None, "method": "clopper-pearson",
            "note": "Expected sensitivity must be higher than the target, otherwise no sample size can confirm it.",
        }
    n = np.arange(1, max_n + 1)
    x = np.rint(expected * n).astype(int)
    a = (1.0 - confidence) / 2.0
    lower = np.where(x == 0, 0.0, betaincinv(np.maximum(x, 1), n - x + 1, a))
    ok = np.nonzero(lower >= target)[0]
    if ok.size == 0:
        return {
            "positives_needed": None, "scans_needed": None, "method": "clopper-pearson",
            "note": f"More than {max_n:,} urgent scans would be needed.",
        }
    pos = int(n[ok[0]])
    return {
        "positives_needed": pos,
        "scans_needed": math.ceil(pos / prevalence),
        "method": "clopper-pearson",
        "note": "Exact (Clopper-Pearson) lower bound, assuming the true sensitivity equals the expected value. A planning estimate, not a guarantee.",
    }


def psi(reference: Sequence[float], target: Sequence[float], bins: int = 10, eps: float = 1e-4) -> Optional[float]:
    """Population Stability Index of two score samples (quantile bins from the reference).
    Same definition Member B uses in site_readiness.py. None if either sample is empty."""
    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if ref.size == 0 or tgt.size == 0:
        return None
    edges = np.percentile(ref, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)  # tied scores can collapse bins
    rc, _ = np.histogram(ref, bins=edges)
    tc, _ = np.histogram(tgt, bins=edges)
    k = len(rc)
    rp = (rc + eps) / (rc.sum() + eps * k)
    tp = (tc + eps) / (tc.sum() + eps * k)
    return float(np.sum((tp - rp) * np.log(tp / rp)))
