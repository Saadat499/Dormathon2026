"""Out-of-distribution / 'is this a usable chest X-ray?' detector on backbone features.

Two signals, either one raises ood_flag:
  1. Mahalanobis distance (PCA-whitened) from the training-feature distribution.
  2. A small probe trained to separate real chest X-rays from realistic mistakes
     (rotated, flipped, inverted, noisy, blurred, low-contrast) and synthetic non-X-ray images.
It is imperfect by design: subtle scanner shifts are mostly NOT caught. Report misses honestly.
Note: a threshold fitted on NIH will also flag many CheXpert images, since the two datasets
look different. That is a drift signal for Site Readiness, not a bug.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import IMG_SIZE
from .preprocess import to_gray


class OODDetector:
    def __init__(self, n_components: int = 128, percentile: float = 99.5, probe_threshold: float = 0.5):
        self.n_components, self.percentile, self.probe_threshold = n_components, percentile, probe_threshold
        self.scaler = self.pca = self.probe = None
        self.threshold = np.inf

    def fit(self, X_in: np.ndarray, X_neg: np.ndarray | None = None, seed: int = 0):
        self.scaler = StandardScaler().fit(X_in)
        Z = self.scaler.transform(X_in)
        k = int(min(self.n_components, Z.shape[1], Z.shape[0] - 1))
        self.pca = PCA(n_components=k, whiten=True, random_state=seed).fit(Z)
        if X_neg is not None and len(X_neg) > 0:
            rng = np.random.default_rng(seed)
            take = rng.choice(len(X_in), size=min(len(X_in), 5000), replace=False)
            X = np.vstack([X_in[take], X_neg])
            y = np.r_[np.zeros(len(take)), np.ones(len(X_neg))]
            self.probe = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
            self.probe.fit(self.scaler.transform(X), y)
        return self

    def set_threshold(self, X_heldout_in: np.ndarray):
        """Choose the Mahalanobis cut-off from held-out in-distribution data."""
        self.threshold = float(np.percentile(self.maha(X_heldout_in), self.percentile))
        return self

    def maha(self, X: np.ndarray) -> np.ndarray:
        return (self.pca.transform(self.scaler.transform(X)) ** 2).sum(axis=1)

    def probe_prob(self, X: np.ndarray) -> np.ndarray:
        if self.probe is None:
            return np.zeros(len(X))
        return self.probe.predict_proba(self.scaler.transform(X))[:, 1]

    def predict(self, X: np.ndarray) -> dict:
        m, p = self.maha(X), self.probe_prob(X)
        return {"flag": (m > self.threshold) | (p > self.probe_threshold), "maha": m, "probe": p,
                "flag_maha": m > self.threshold, "flag_probe": p > self.probe_threshold}


# ----------------------------------------------------------------- test images
def make_variants(img: Image.Image, seed: int = 0) -> dict:
    """Realistic mistakes derived from a real chest X-ray."""
    g = to_gray(img).resize((IMG_SIZE, IMG_SIZE))
    a = np.asarray(g, np.float32)
    rng = np.random.default_rng(seed)
    return {
        "rotated_90": g.rotate(90), "rotated_180": g.rotate(180), "rotated_270": g.rotate(270),
        "flipped_vertical": ImageOps.flip(g), "inverted": ImageOps.invert(g),
        "heavy_noise": Image.fromarray(np.clip(a + rng.normal(0, 60, a.shape), 0, 255).astype(np.uint8)),
        "heavy_blur": g.filter(ImageFilter.GaussianBlur(10)),
        "low_contrast": Image.fromarray((128 + (a - 128) * 0.15).astype(np.uint8)),
    }


def synthetic_non_chest(n: int, seed: int = 0) -> list:
    """(name, image) pairs that are clearly not chest X-rays."""
    rng = np.random.default_rng(seed)
    out, S = [], IMG_SIZE
    xx, yy = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
    for i in range(n):
        kind = i % 5
        if kind == 0:
            a, name = rng.integers(0, 256, (S, S)), "white_noise"
        elif kind == 1:
            a, name = np.full((S, S), rng.integers(0, 256)), "blank"
        elif kind == 2:
            ang = rng.uniform(0, 6.28)
            a, name = 255 * (np.cos(ang) * xx + np.sin(ang) * yy), "gradient"
        elif kind == 3:
            k = int(rng.integers(4, 32))
            a, name = 255 * (((xx * k).astype(int) + (yy * k).astype(int)) % 2), "checkerboard"
        else:
            a = np.zeros((S, S))
            for _ in range(8):
                cx, cy, r = rng.uniform(0, 1, 2).tolist() + [rng.uniform(0.05, 0.3)]
                a += 255 * (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2)
            a, name = np.clip(a, 0, 255), "blobs"
        out.append((name, Image.fromarray(np.asarray(a, np.uint8))))
    return out
