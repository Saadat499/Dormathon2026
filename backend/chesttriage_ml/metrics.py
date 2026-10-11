"""Metrics with masked (NaN) labels, cluster bootstrap CIs and operating points."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def _valid(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~np.isnan(y)
    return y[m], s[m]


def auroc(y, s) -> float:
    y, s = _valid(y, s)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def auprc(y, s) -> float:
    y, s = _valid(y, s)
    return float("nan") if len(np.unique(y)) < 2 else float(average_precision_score(y, s))


def bootstrap_ci(fn, y, s, groups=None, n_boot: int = 200, seed: int = 0):
    """95% CI. If groups (e.g. patient ids) are given, whole groups are resampled."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        idxs = None
    else:
        g = np.asarray(groups)
        order = np.argsort(g, kind="stable")
        gs = g[order]
        cuts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        idxs = np.split(order, cuts[1:])
    vals = []
    for _ in range(n_boot):
        if idxs is None:
            i = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(idxs), len(idxs))
            i = np.concatenate([idxs[j] for j in pick])
        v = fn(y[i], s[i])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def threshold_for_sensitivity(y, s, target: float) -> float:
    """Largest threshold whose sensitivity on (y, s) is >= target."""
    y, s = _valid(y, s)
    pos = s[y == 1]
    if len(pos) == 0:
        return float("nan")
    return float(np.quantile(pos, 1.0 - target, method="lower"))


def operating_point(y, s, thr: float) -> dict:
    y, s = _valid(y, s)
    pred = s >= thr
    tp = int(((y == 1) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    return {"threshold": thr, "n": len(y), "n_pos": tp + fn, "sensitivity": div(tp, tp + fn),
            "specificity": div(tn, tn + fp), "false_alarm_rate": div(fp, fp + tn),
            "ppv": div(tp, tp + fp), "flag_rate": float(pred.mean()) if len(pred) else float("nan")}


def composite_label(Y: np.ndarray, idx) -> np.ndarray:
    """'Any urgent finding': 1 if any positive, NaN if none positive but some masked, else 0."""
    sub = Y[:, idx]
    anypos = (sub == 1).any(axis=1)
    anynan = np.isnan(sub).any(axis=1)
    return np.where(anypos, 1.0, np.where(anynan, np.nan, 0.0))


def composite_score(P: np.ndarray, idx) -> np.ndarray:
    return P[:, idx].max(axis=1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
