"""Saliency maps (Grad-CAM) for the linear head, computed exactly in numpy.

For logit_k = sum_c (W_kc / std_c) * mean_hw f_c + const, the Grad-CAM channel weights are
W_kc / (std_c * h * w), so cam_k = ReLU( sum_c (W_kc / std_c) * f_c(h, w) ), i.e. the classic
CAM. No autograd needed. Call these 'saliency maps': they show where the model's evidence
came from, and with a frozen ImageNet backbone they are coarse (7x7 grid). Audit them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import IMG_SIZE
from .preprocess import to_gray


def cam_for(head, fmap: np.ndarray, k: int, out_size: int = IMG_SIZE) -> np.ndarray:
    """fmap: (C,h,w) for one image -> (out_size,out_size) map in [0,1]."""
    c = np.tensordot(head.W[k] / head.std, fmap, axes=(0, 0))
    c = np.maximum(c, 0)
    if c.max() > 0:
        c = c / c.max()
    im = Image.fromarray((c * 255).astype(np.uint8)).resize((out_size, out_size), Image.Resampling.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.6) -> Image.Image:
    import matplotlib
    base = np.asarray(to_gray(img).resize((IMG_SIZE, IMG_SIZE)), np.float32) / 255.0
    base3 = np.stack([base] * 3, axis=-1)
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    w = (alpha * cam)[..., None]
    out = base3 * (1 - w) + heat * w
    return Image.fromarray((out * 255).astype(np.uint8))


def save_panel(img: Image.Image, cam: np.ndarray, path) -> None:
    """Original | saliency overlay, side by side."""
    left = to_gray(img).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    right = overlay(img, cam)
    panel = Image.new("RGB", (IMG_SIZE * 2 + 8, IMG_SIZE), (255, 255, 255))
    panel.paste(left, (0, 0))
    panel.paste(right, (IMG_SIZE + 8, 0))
    panel.save(path)
