"""Image preprocessing shared by training, feature extraction and the live engine.

Pure numpy/PIL on purpose: only the backbone forward pass needs torch.
Images are stretched to a 224x224 square (NIH-224 is already square; CheXpert is not).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


def to_gray(img: Image.Image) -> Image.Image:
    """Any PIL image -> 8-bit grayscale ('L')."""
    if img.mode in ("I;16", "I", "F"):
        a = np.asarray(img, dtype=np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-6) * 255.0
        img = Image.fromarray(a.astype(np.uint8))
    return img.convert("L")


def to_array(img: Image.Image) -> np.ndarray:
    """PIL -> normalised float32 array (3, 224, 224)."""
    g = to_gray(img).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.float32) / 255.0
    a = np.repeat(a[None, :, :], 3, axis=0)
    return (a - _MEAN) / _STD


def load_image(path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img
