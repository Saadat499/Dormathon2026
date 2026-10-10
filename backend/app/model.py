"""Where model output comes from. Three honest states:

  none    No model on this host. predict() returns None, so the scan is `unsure` (never `normal`).
  replay  Member A's saved predictions are looked up by sha256 or by file name = study id.
          The model is NOT run on the upload. A scan with no saved match is treated as `none`.
  (live)  Running A's network on the host is not wired: its files are not in the repo. To add it,
          implement `predict` below and return a Prediction; nothing else needs to change.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .cohorts import PredictionSet


@dataclass
class Prediction:
    findings: list
    confidence_band: Optional[str]
    source: str  # "replay" | "model"
    ood_flag: bool = False


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NoModel:
    mode = "none"
    description = "No model attached: uploads are placed in the queue as 'unsure'."

    def predict(self, data: bytes, filename: str) -> Optional[Prediction]:
        return None


class ReplayModel:
    mode = "replay"

    def __init__(self, ps: PredictionSet):
        self.by_hash = {i.sha256: i for i in ps.items if i.sha256}
        self.by_id = {i.study_id.lower(): i for i in ps.items}
        self.description = "Replay of Member A's saved predictions: uploads are matched to saved results, the model is not run."

    def predict(self, data: bytes, filename: str) -> Optional[Prediction]:
        hit = self.by_hash.get(sha256(data)) or self.by_id.get(Path(filename or "").stem.lower())
        if hit is None or not hit.findings:
            return None
        return Prediction(hit.findings, hit.confidence_band, "replay", hit.ood_flag)


def make_provider(ps: PredictionSet):
    return ReplayModel(ps) if ps.is_real else NoModel()
