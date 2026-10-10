"""Shared constants and path helpers.

The urgency mapping below is a PROVISIONAL ASSUMPTION. It must be reviewed by a
clinician (Member B owns this). After agreement, edit artifacts/policy.json
(the 'urgent' list) instead of changing code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Findings present in BOTH NIH ChestX-ray14 and CheXpert (canonical names).
FINDINGS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "effusion", "pneumonia", "pneumothorax",
]
DISPLAY = {
    "atelectasis": "Atelectasis", "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation", "edema": "Edema",
    "effusion": "Pleural effusion", "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
}
NIH_NAME = {f: f.capitalize() for f in FINDINGS}           # NIH calls it "Effusion"
CHEXPERT_NAME = {f: f.capitalize() for f in FINDINGS}
CHEXPERT_NAME["effusion"] = "Pleural Effusion"

# PROVISIONAL: which findings count as time-sensitive. Needs clinician review.
DEFAULT_URGENT = ["pneumothorax", "effusion", "edema", "consolidation", "pneumonia"]

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def artifacts_dir() -> Path:
    p = Path(os.environ.get("CHESTTRIAGE_ARTIFACTS", "artifacts")).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def art(*parts: str) -> Path:
    """Path inside the artifacts dir. Folders (or the parent of a file) are created."""
    p = artifacts_dir().joinpath(*parts)
    (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
    return p


def save_dataset_paths(**roots) -> None:
    f = art("paths.json")
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur.update({k: str(v) for k, v in roots.items() if v})
    f.write_text(json.dumps(cur, indent=2))


def load_dataset_paths() -> dict:
    f = art("paths.json")
    return json.loads(f.read_text()) if f.exists() else {}
