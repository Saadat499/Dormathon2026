"""GET /evidence: what the model has been shown to do, with provenance for every block.

Main keys (the Evidence page renders these) hold REAL results only:
  per_finding, cross_dataset   Member A's reports for the frozen model (NIH test, CheXpert external).
  calibration                  Member B's calibration code run on A's real predictions (fit on NIH
                               validation, measured on NIH test). Temperature scaling is NOT applied to
                               the scores the app serves, and nothing here says the model is well calibrated.
  shortcut_audit               B's AP/PA view audit run on A's real NIH test predictions, plus A's
                               saliency-attention proxies. The chest-tube audit needs annotations nobody has.
  failures                     A's "where it fails" examples (public NIH images only).
All of that is precomputed by scripts/build_model_artifacts.py into model_artifacts/evidence.json.

`extras.member_b_simulated_cohorts` passes through B's locked_evidence.json. B's exporter builds its
cohorts with np.random, so those numbers are a protocol demonstration on SYNTHETIC data, not model
performance. They are kept out of the main keys so the page never shows them as results.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ..config import APP_DIR

log = logging.getLogger("voxelsort.evidence")
EVIDENCE_FILE = "evidence.json"
B_SYNTHETIC = APP_DIR / "static_evidence" / "locked_evidence.json"


def load(artifacts_dir: Path) -> Optional[dict]:
    """Returns the stored evidence, or None when there is none (the endpoint then answers 404)."""
    path = artifacts_dir / EVIDENCE_FILE
    ev: Optional[dict] = None
    if path.is_file():
        try:
            ev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.exception("evidence.json could not be read")
    if ev is None:
        return None
    ev.setdefault("extras", {})
    if B_SYNTHETIC.is_file():
        try:
            raw = json.loads(B_SYNTHETIC.read_text(encoding="utf-8"))
            ev["extras"]["member_b_simulated_cohorts"] = {
                "data": "SYNTHETIC",
                "statement": ("Member B's locked_evidence.json. Its cohorts are generated with random numbers by "
                              "evidence_exporter.py: a demonstration of B's evaluation protocol, NOT model performance. "
                              "Example: its Queue Race headline (STAT breaches 52% to 0%) and its calibration figures "
                              "(e.g. ECE 0.443 to 0.432) describe simulated data."),
                "content": raw,
            }
        except Exception:  # noqa: BLE001
            log.exception("B's locked_evidence.json could not be read")
    return ev


def public(ev: dict, base_url: str) -> dict:
    """Copy with absolute image links (the frontend is served from another origin)."""
    base = base_url.rstrip("/")
    out = dict(ev)
    out["failures"] = [
        {**f, "image_url": (base + f["image_path"]) if f.get("image_path") else None}
        for f in ev.get("failures", [])
    ]
    return out
