"""Out-of-distribution check on an image.

HONEST STATUS: this is a small set of hand-written image checks (colour, size, shape, contrast,
upright symmetry). It is a PLACEHOLDER for Member A's learned detector, which is not in the repo.
It has not been validated against labelled out-of-distribution data. It can miss things, so a
"not flagged" result never means the image is in-distribution. A flag always makes the scan
`unsure` (moved up the queue), never `normal`.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
METHOD = "heuristic image checks (placeholder, not the learned detector)"


class ImageError(ValueError):
    """The bytes are not an image we can read (message is safe to show the user)."""


@dataclass
class OodResult:
    ood_flag: bool
    score: float
    reason: str
    reasons: list = field(default_factory=list)
    method: str = METHOD


def load_image(data: bytes) -> Image.Image:
    if len(data) > 132 and data[128:132] == b"DICM":
        raise ImageError("DICOM files are not supported by this backend yet. Export the image as PNG or JPG.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ("PNG", "JPEG"):
            raise ImageError("Only PNG and JPG images are supported.")
        img.load()
    except ImageError:
        raise
    except Exception:  # Pillow raises many types, including decompression-bomb errors
        raise ImageError("This file could not be read as a PNG or JPG image.") from None
    return img


def to_display(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """8-bit grayscale or RGB copy, downsampled. 16-bit images are rescaled, not clipped."""
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        a = np.asarray(img, dtype=np.float64)
        lo, hi = np.percentile(a, [0.5, 99.5])
        img = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8), "L")
    elif img.mode in ("LA", "PA", "RGBA", "P", "CMYK"):
        img = img.convert("RGB")
    elif img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    out = img.copy()
    out.thumbnail((max_side, max_side))
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def check_image(img: Image.Image) -> OodResult:
    w, h = img.size
    small = to_display(img, 128)
    rgb = np.asarray(small.convert("RGB"), dtype=np.float64) / 255.0
    gray = np.asarray(small.convert("L"), dtype=np.float64) / 255.0

    strengths: list[tuple[float, str]] = []
    sat = float(np.mean(rgb.max(axis=2) - rgb.min(axis=2)))
    if sat > 0.06:
        strengths.append((min(1.0, 0.5 + (sat - 0.06) * 5), "not a chest X-ray (colour image)"))
    if min(w, h) < 128:
        strengths.append((0.9, "image is too small"))
    ratio = w / h
    if ratio > 1.6 or ratio < 0.6:
        strengths.append((0.8, "unusual shape for a chest X-ray (very wide or very tall)"))
    if float(gray.std()) < 0.05:
        strengths.append((0.9, "almost blank or very low contrast"))
    g = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((64, 64)), dtype=np.float64)
    lr, ud = _corr(g, g[:, ::-1]), _corr(g, g[::-1, :])
    if ud > lr + 0.05 and ud > 0.3:
        strengths.append((min(1.0, 0.55 + (ud - lr)), "rotated (upright symmetry not found)"))
    if not strengths:
        return OodResult(False, round(max(0.0, min(0.45, 0.45 - lr * 0.4)), 3), "no out-of-distribution signal found by the placeholder checks")
    score = max(s for s, _ in strengths)
    reasons = [r for _, r in strengths]
    return OodResult(score >= 0.5, round(float(score), 3), "; ".join(reasons), reasons)
