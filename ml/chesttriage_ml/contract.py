"""Builds the prediction record that Member C's API and Member D's UI consume.

Required keys (the hour-one data contract):
  study_id, findings{name: prob}, priority, confidence_band, ood_flag, heatmap_path
Extra keys (safe to ignore): composite_score, top_urgent_finding, flagged_findings,
  ood_score, model_version (and image_path in the demo bundle).
For out-of-distribution images 'findings' is {} (the image is not scored).

model_version is a hash of the backbone, head and policy files, so it changes whenever the
model, the calibration or an approved threshold changes. Log it with every decision.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import __version__

VERSION_FILES = ("backbone.json", "head.npz", "policy.json")


def model_version(art_dir) -> str:
    h = hashlib.sha256()
    for name in VERSION_FILES:
        f = Path(art_dir) / name
        h.update(name.encode())
        h.update(f.read_bytes() if f.exists() else b"missing")
    return f"partA-{__version__}+{h.hexdigest()[:10]}"


def make_record(study_id, policy, p_row, priority, band, composite, ood_flag=False,
                ood_score=None, heatmap_path=None, version=None) -> dict:
    ood_flag = bool(ood_flag)
    urg = policy.urgent_idx()
    top = None
    flagged = []
    if not ood_flag:
        top = policy.findings[urg[int(np.argmax(p_row[urg]))]]
        cut = policy.t_raise if priority == "Raised" else policy.t_unsure
        if priority in ("Raised", "AI unsure"):
            flagged = [policy.findings[i] for i in urg if p_row[i] >= cut]
    return {
        "study_id": study_id,
        "findings": {} if ood_flag else {f: round(float(p), 4) for f, p in zip(policy.findings, p_row)},
        "priority": str(priority),
        "confidence_band": str(band),
        "ood_flag": ood_flag,
        "heatmap_path": None if heatmap_path is None else str(heatmap_path),
        "composite_score": None if ood_flag else round(float(composite), 4),
        "top_urgent_finding": top,
        "flagged_findings": flagged,
        "ood_score": None if ood_score is None else round(float(ood_score), 3),
        "model_version": version or f"partA-{__version__}",
    }
