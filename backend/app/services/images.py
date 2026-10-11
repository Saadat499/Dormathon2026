"""Reading, validating and storing uploaded images, plus a basic image sanity check.

`basic_check` is a handful of hand-written tests (colour, size, shape, contrast, upright symmetry).
It is NOT Member A's trained out-of-distribution detector and has not been validated. It is used
only when that detector is not loaded on the host, and every result says so. A flag from it can
only move a scan up (to `unsure`); "no flag" never means the image is in-distribution.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .. import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
BASIC_METHOD = "basic image checks (placeholder: the trained detector is not loaded on this host)"


class ImageError(ValueError):
    """The bytes are not an image this service accepts. The message is safe to show the user."""


def load_image(data: bytes) -> Image.Image:
    """PNG or JPEG only. A DICOM file must be converted to PNG or JPEG first."""
    if not data:
        raise ImageError("The uploaded file is empty.")
    if len(data) > 132 and data[128:132] == b"DICM":
        raise ImageError("DICOM files cannot be read by this server. Export the image as PNG or JPG and upload that.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        if fmt not in ("PNG", "JPEG"):
            raise ImageError("Only PNG and JPG images are supported.")
        if img.width * img.height > config.MAX_IMAGE_PIXELS:
            raise ImageError("The image has too many pixels.")
        img.load()
    except ImageError:
        raise
    except Exception:  # Pillow raises many types, including decompression-bomb errors
        raise ImageError("This file could not be read as a PNG or JPG image. It may be corrupt.") from None
    if img.width < 8 or img.height < 8:
        raise ImageError("The image is too small to be a scan.")
    return img


def to_display(img: Image.Image, max_side: int = config.MAX_IMAGE_SIDE) -> Image.Image:
    """8-bit grayscale or RGB copy, at most `max_side` pixels on its longer side.
    16-bit images are rescaled to 8 bits, not clipped."""
    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        a = np.asarray(img, dtype=np.float64)
        lo, hi = np.percentile(a, [0.5, 99.5])
        img = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8))
    elif img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    out = img.copy()
    out.thumbnail((max_side, max_side))
    return out


@dataclass
class BasicCheck:
    flag: bool
    score: float
    reason: str
    reasons: list = field(default_factory=list)
    method: str = BASIC_METHOD


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def basic_check(img: Image.Image) -> BasicCheck:
    w, h = img.size
    small = to_display(img, 128)
    rgb = np.asarray(small.convert("RGB"), dtype=np.float64) / 255.0
    gray = np.asarray(small.convert("L"), dtype=np.float64) / 255.0

    found: list[tuple[float, str]] = []
    sat = float(np.mean(rgb.max(axis=2) - rgb.min(axis=2)))
    if sat > 0.06:
        found.append((min(1.0, 0.5 + (sat - 0.06) * 5), "not a chest X-ray (colour image)"))
    if min(w, h) < 128:
        found.append((0.9, "image is too small"))
    ratio = w / h
    if ratio > 1.6 or ratio < 0.6:
        found.append((0.8, "unusual shape for a chest X-ray"))
    if float(gray.std()) < 0.05:
        found.append((0.9, "almost blank or very low contrast"))
    g = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((64, 64)), dtype=np.float64)
    lr, ud = _corr(g, g[:, ::-1]), _corr(g, g[::-1, :])
    if ud > lr + 0.05 and ud > 0.3:
        found.append((min(1.0, 0.55 + (ud - lr)), "rotated (upright symmetry not found)"))
    if not found:
        return BasicCheck(False, round(max(0.0, min(0.45, 0.45 - lr * 0.4)), 3),
                          "no problem found by the basic image checks (this does not show the image is in-distribution)")
    score = max(s for s, _ in found)
    reasons = [r for _, r in found]
    return BasicCheck(score >= 0.5, round(float(score), 3), "; ".join(reasons), reasons)
