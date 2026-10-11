"""Turns a saliency map into the transparent overlay the Study Review page draws over the image.

The map itself comes from Member A's `saliency.cam_for` (the exact class-activation map of the
linear head on the frozen backbone: a coarse 7x7 grid, upsampled). This module only colours it.
A saliency map shows where the model's evidence came from. It is not proof that the model looked
at the right thing, and A's own audit found maps sitting on image borders and text markers.

matplotlib is deliberately not used: it would add tens of megabytes to a 512 MB host.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def jet(x: np.ndarray) -> np.ndarray:
    """The classic 'jet' colour map (blue -> cyan -> yellow -> red) for values in [0, 1]. Returns uint8 RGB."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def overlay_rgba(cam: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """`cam`: (h, w) values in [0, 1] for the image stretched to a square (how the model saw it).
    `size`: (width, height) of the stored display image. Low activation is transparent."""
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    rgb = jet(cam)
    alpha = (np.clip(cam, 0.0, 1.0) ** 0.8 * 255.0).astype(np.uint8)
    img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    return img.resize(size, Image.Resampling.BILINEAR)


def side_by_side(gray: Image.Image, cam: np.ndarray, size: int = 224, alpha: float = 0.6) -> Image.Image:
    """Original | overlay panel (same layout as Member A's `saliency.save_panel`), for the failure gallery."""
    base = gray.convert("L").resize((size, size))
    b = np.asarray(base, dtype=np.float32) / 255.0
    c = np.asarray(Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR),
                   dtype=np.float32) / 255.0
    heat = jet(c).astype(np.float32) / 255.0
    w = (alpha * c)[..., None]
    mixed = np.stack([b] * 3, axis=-1) * (1 - w) + heat * w
    panel = Image.new("RGB", (size * 2 + 8, size), (255, 255, 255))
    panel.paste(base.convert("RGB"), (0, 0))
    panel.paste(Image.fromarray((mixed * 255).astype(np.uint8)), (size + 8, 0))
    return panel
