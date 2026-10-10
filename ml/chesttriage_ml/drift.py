"""Site drift score for Site Readiness: how easily a simple classifier tells a site sample apart
from the training data, using backbone features.

    score = cross-validated AUROC of logistic regression, reference features vs site features
    ~0.5  indistinguishable from the training data
    ~1.0  completely different (new scanner, processing, population or file format)

This complements the OOD flag, which only catches gross mistakes (rotated, inverted, non-chest)
and flags just a few percent of a genuinely different hospital's images. It is a WARNING signal:
it says the images differ, not whether accuracy dropped; only labelled site data shows that.
Caveat: it also picks up harmless differences (resizing, JPEG vs PNG), so read it against
reference values (e.g. pseudo-sites drawn from the training data) rather than as an absolute scale.
See reports/drift_controls.md (scripts/04b_drift_controls.py) for the JPEG / resize controls.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def domain_auroc(X_ref: np.ndarray, X_site: np.ndarray, n_splits: int = 5, C: float = 0.01,
                 max_ref: int = 2000, seed: int = 0) -> float:
    """Cross-validated AUROC separating X_site from a size-matched sample of X_ref."""
    rng = np.random.default_rng(seed)
    n = len(X_site)
    if n < 2 * n_splits:
        return float("nan")
    ref = X_ref[rng.choice(len(X_ref), size=min(len(X_ref), max_ref, max(n, 50)), replace=False)]
    X = np.vstack([ref, X_site]).astype(np.float64)
    y = np.r_[np.zeros(len(ref)), np.ones(n)]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        s[te] = LogisticRegression(C=C, max_iter=1000).fit(X[tr], y[tr]).decision_function(X[te])
    return float(roc_auc_score(y, s))


MIN_SITE_SAMPLE = 20


def drift_summary(X_ref: np.ndarray, X_site: np.ndarray, ood=None) -> dict:
    if len(X_site) < MIN_SITE_SAMPLE:
        raise ValueError(f"site sample has {len(X_site)} images; need at least {MIN_SITE_SAMPLE} "
                         "(aim for 50+) for a drift score")
    out = {"n_site": int(len(X_site)), "domain_auroc": domain_auroc(X_ref, X_site)}
    if ood is not None:
        o = ood.predict(X_site)
        out.update(ood_flag_rate=float(o["flag"].mean()), median_ood_score=float(np.median(o["maha"])))
    return out
