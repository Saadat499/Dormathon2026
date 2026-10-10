"""Linear multi-label head on frozen features (one logistic regression per finding).

Why linear: it is fast, its probabilities are meaningful (no class re-weighting), it
saves as plain numpy (easy to serve), and the saliency map has an exact closed form
(see saliency.py). Missing labels (NaN) are skipped per finding.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from .metrics import auroc


class LinearHead:
    def __init__(self, findings):
        self.findings = list(findings)
        self.W = self.b = self.mean = self.std = None
        self.best_C = {}
        self.val_auroc = {}

    # ---- training
    def fit(self, Xtr, Ytr, Xva, Yva, Cs=(0.001, 0.01, 0.1, 1.0), n_jobs: int = -1):
        self.mean = Xtr.mean(0).astype(np.float32)
        self.std = (Xtr.std(0) + 1e-6).astype(np.float32)
        Ztr, Zva = (Xtr - self.mean) / self.std, (Xva - self.mean) / self.std
        K, D = len(self.findings), Xtr.shape[1]

        def one(k):
            m = ~np.isnan(Ytr[:, k])
            mv = ~np.isnan(Yva[:, k])
            if len(np.unique(Ytr[m, k])) < 2:
                return np.zeros(D, np.float32), 0.0, None, float("nan")
            best = None
            for C in Cs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LogisticRegression(C=C, max_iter=300, tol=1e-3)
                    clf.fit(Ztr[m], Ytr[m, k])
                a = auroc(Yva[:, k], clf.decision_function(Zva))
                cand = (-np.inf if np.isnan(a) else a, C, clf.coef_[0].astype(np.float32), float(clf.intercept_[0]))
                if best is None or cand[0] > best[0]:
                    best = cand
            if not np.isfinite(best[0]):          # validation had no positives: AUROC undefined
                best = (float("nan"),) + best[1:]
            return best[2], best[3], best[1], best[0]

        res = Parallel(n_jobs=n_jobs)(delayed(one)(k) for k in range(K))
        self.W = np.stack([r[0] for r in res]).astype(np.float32)
        self.b = np.array([r[1] for r in res], dtype=np.float32)
        self.best_C = {f: r[2] for f, r in zip(self.findings, res)}
        self.val_auroc = {f: r[3] for f, r in zip(self.findings, res)}
        return self

    # ---- inference
    def logits(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std) @ self.W.T + self.b

    # ---- persistence
    def save(self, path) -> None:
        np.savez(path, W=self.W, b=self.b, mean=self.mean, std=self.std,
                 findings=np.array(self.findings))

    @classmethod
    def load(cls, path) -> "LinearHead":
        z = np.load(path, allow_pickle=False)
        h = cls([str(f) for f in z["findings"]])
        h.W, h.b, h.mean, h.std = z["W"], z["b"], z["mean"], z["std"]
        return h
