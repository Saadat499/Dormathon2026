"""Saved predictions: Member A's real model outputs with true labels.

File: <artifacts>/saved_predictions_nih.csv.gz, built by scripts/build_model_artifacts.py from A's
`predictions/nih_full.csv` (NIH ChestX-ray14, patient-level splits made by A):

  split "val"   11,341 images. The REFERENCE set: the threshold scale, the default threshold and the
                sensitivity floor are all computed here. Nothing is tuned on the test split.
  split "test"  21,845 images, never used for training, thresholds or model choice. Reported next to
                the validation numbers and used to draw cases for the Queue Race.

There is no synthetic fallback. If the file is missing the endpoints that need it answer 503 and
`/health` says why. A file whose SOURCE.json says `"data": "synthetic_test_fixture"` (the unit
tests make one) is labelled as such in every response that uses it.

Threshold scale: the threshold `value` (0..1) is a PERCENTILE of the validation scores, not a raw
probability: value 0.41 means "raise every scan whose score is above the 41st percentile of the
reference set". The raw score cut-off is returned alongside as `score_cutoff`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import delete, func, insert, select

from ..db import Database
from ..evalkit.calibration import select_threshold_at_sensitivity
from ..evalkit.protocol import compute_binary_metrics
from ..models import SavedPrediction, Setting
from .queue import Lanes

log = logging.getLogger("voxelsort.predictions")
PREDICTIONS_FILE = "saved_predictions_nih.csv.gz"
SOURCE_FILE = "SOURCE.json"
DEFAULT_FINDINGS = ["atelectasis", "cardiomegaly", "consolidation", "edema", "effusion", "pneumonia", "pneumothorax"]
DEFAULT_URGENT = ["pneumothorax", "effusion", "edema", "consolidation", "pneumonia"]


@dataclass
class Split:
    study_ids: np.ndarray
    scores: np.ndarray            # composite score (max urgent probability)
    labels: np.ndarray            # 1 = any urgent finding, 0 = none
    ood: np.ndarray               # bool
    probs: np.ndarray             # (n, n_findings)

    def __len__(self) -> int:
        return int(len(self.scores))

    @property
    def effective_scores(self) -> np.ndarray:
        """An out-of-distribution scan is forced to `unsure`, so it can never count as raised."""
        return np.where(self.ood, 0.0, self.scores)


def _empty_split(k: int) -> Split:
    return Split(np.array([], dtype=object), np.array([], dtype=float), np.array([], dtype=int),
                 np.array([], dtype=bool), np.zeros((0, k), dtype=np.float32))


@dataclass
class PredictionStore:
    source: str = "none"                      # real | synthetic_test_fixture | none
    description: str = "No saved predictions are loaded."
    problems: list = field(default_factory=list)
    findings: list = field(default_factory=lambda: list(DEFAULT_FINDINGS))
    urgent: list = field(default_factory=lambda: list(DEFAULT_URGENT))
    model_version: Optional[str] = None
    provenance: dict = field(default_factory=dict)
    val: Split = field(default_factory=lambda: _empty_split(len(DEFAULT_FINDINGS)))
    test: Split = field(default_factory=lambda: _empty_split(len(DEFAULT_FINDINGS)))

    @property
    def available(self) -> bool:
        return len(self.val) > 0 and int((self.val.labels == 1).sum()) > 0 and int((self.val.labels == 0).sum()) > 0

    @property
    def is_real(self) -> bool:
        return self.source == "real"

    # --- threshold scale -------------------------------------------------------------------
    def cutoff(self, value: float) -> float:
        """Raw score cut-off for a percentile threshold."""
        return float(np.quantile(self.val.scores, min(1.0, max(0.0, float(value)))))

    def lanes(self, value: float, unsure_budget: float) -> Lanes:
        if not self.available:
            return Lanes(None, None)
        t_raise = self.cutoff(value)
        t_unsure = min(t_raise, self.cutoff(max(0.0, value - unsure_budget)))
        return Lanes(t_raise, t_unsure)

    def percentile_of(self, score_cutoff: float) -> float:
        return float((self.val.scores < score_cutoff).mean())

    # --- metrics ---------------------------------------------------------------------------
    def _metrics(self, split: Split, lanes: Lanes) -> Optional[dict]:
        if len(split) == 0:
            return None
        m = compute_binary_metrics(split.labels, split.effective_scores, threshold=lanes.t_raise)   # Member B's function
        n_pos, n_neg = m["tp"] + m["fn"], m["tn"] + m["fp"]
        unsure = split.ood | ((split.scores >= lanes.t_unsure) & (split.scores < lanes.t_raise))
        return {
            "abstention_rate": float(unsure.mean()),
            "sensitivity": (m["tp"] / n_pos) if n_pos else None,
            "false_alarm_rate": (m["fp"] / n_neg) if n_neg else None,
            "n": int(len(split)),
            "n_positive": int(n_pos),
            "missed_urgent_not_raised": int(m["fn"]),
            "missed_urgent_left_routine": int(((split.labels == 1) & ~unsure & (split.effective_scores < lanes.t_raise)).sum()),
            "wrong_flags": int(m["fp"]),
            "raised_share": float((split.effective_scores >= lanes.t_raise).mean()),
        }

    def operating_point(self, value: float, unsure_budget: float) -> dict:
        """Numbers for /threshold/preview. Top-level figures are from the validation (reference) split,
        the same data the sensitivity floor is checked on; `held_out_test` reports the clean test split."""
        lanes = self.lanes(value, unsure_budget)
        ref = self._metrics(self.val, lanes) or {}
        out = {"value": float(value), **{k: ref.get(k) for k in ("abstention_rate", "sensitivity", "false_alarm_rate", "n", "n_positive")}}
        out.update(
            score_cutoff=lanes.t_raise, unsure_score_cutoff=lanes.t_unsure, raised_share=ref.get("raised_share"),
            missed_urgent_not_raised=ref.get("missed_urgent_not_raised"),
            missed_urgent_left_routine=ref.get("missed_urgent_left_routine"), wrong_flags=ref.get("wrong_flags"),
            computed_on="NIH validation split (reference set; the sensitivity floor is checked here)",
            held_out_test=self._metrics(self.test, lanes),
        )
        return out

    def sensitivity_at(self, value: float) -> Optional[float]:
        lanes = self.lanes(value, 0.0)
        m = self._metrics(self.val, lanes)
        return None if m is None else m["sensitivity"]

    def highest_value_meeting(self, target: float) -> Optional[float]:
        """Highest threshold on a 0.01 grid whose validation sensitivity is at least `target`
        (fewest false alarms at that target). The raw cut-off comes from Member B's
        `select_threshold_at_sensitivity`; it is then snapped to the percentile grid and re-checked."""
        if not self.available:
            return None
        cached = (self.provenance.get("default_threshold") or {})
        if cached and abs(float(cached.get("target", -1)) - target) < 1e-9:
            v = round(float(cached["value"]), 2)
        else:
            b = select_threshold_at_sensitivity(self.val.labels, self.val.effective_scores, target_sensitivity=target)
            v = round(int(self.percentile_of(float(b.get("selected_threshold", b.get("threshold", 0.0)))) * 100) / 100, 2)
        v = min(0.99, max(0.01, v))
        ok = lambda x: (self.sensitivity_at(x) or 0.0) >= target  # noqa: E731
        while v > 0.01 and not ok(v):
            v = round(v - 0.01, 2)
        while v < 0.99 and ok(round(v + 0.01, 2)):
            v = round(v + 0.01, 2)
        return v if ok(v) else None

    def summary(self) -> dict:
        return {
            "source": self.source, "real": self.is_real, "available": self.available, "description": self.description,
            "model_version": self.model_version, "n_reference_val": len(self.val), "n_held_out_test": len(self.test),
            "problems": self.problems,
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_store(artifacts_dir: Path, db: Optional[Database] = None) -> PredictionStore:
    """Read the saved-predictions file, mirror it into the `saved_predictions` table and return the
    in-memory arrays. Never raises: a problem is reported in `store.problems` and on /health."""
    store = PredictionStore()
    path = artifacts_dir / PREDICTIONS_FILE
    src = artifacts_dir / SOURCE_FILE
    if src.is_file():
        try:
            store.provenance = json.loads(src.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            store.problems.append(f"{SOURCE_FILE} could not be read: {e}")
    policy = artifacts_dir / "policy.json"
    if policy.is_file():
        try:
            p = json.loads(policy.read_text(encoding="utf-8"))
            store.findings, store.urgent = list(p["findings"]), list(p["urgent"])
        except Exception as e:  # noqa: BLE001
            store.problems.append(f"policy.json could not be read: {e}")
    elif store.provenance.get("findings"):
        store.findings = list(store.provenance["findings"])
        store.urgent = list(store.provenance.get("urgent", DEFAULT_URGENT))
    if not path.is_file():
        store.problems.append(f"{PREDICTIONS_FILE} is missing from {artifacts_dir}. Threshold preview, the sensitivity floor "
                              "and the Queue Race are unavailable until it is restored (scripts/build_model_artifacts.py).")
        return store
    try:
        import pandas as pd
        df = pd.read_csv(path)
        need = {"study_id", "split", "label_any_urgent", "composite_score", "ood_flag"} | {f"p_{f}" for f in store.findings}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        df = df[df["label_any_urgent"].notna() & df["composite_score"].notna()].copy()
        df["ood_flag"] = df["ood_flag"].astype(str).str.lower().isin(("true", "1"))
        df["label_any_urgent"] = df["label_any_urgent"].astype(int)

        def split(name: str) -> Split:
            d = df[df["split"] == name]
            return Split(d["study_id"].to_numpy(dtype=object), d["composite_score"].to_numpy(dtype=float),
                         d["label_any_urgent"].to_numpy(dtype=int), d["ood_flag"].to_numpy(dtype=bool),
                         d[[f"p_{f}" for f in store.findings]].to_numpy(dtype=np.float32))

        store.val, store.test = split("val"), split("test")
        store.source = "real" if store.provenance.get("data") == "real" else str(store.provenance.get("data") or "unverified")
        store.model_version = store.provenance.get("model_version")
        store.description = (f"{len(store.val)} validation and {len(store.test)} held-out test predictions "
                             f"({store.provenance.get('dataset', 'dataset not stated')}, model {store.model_version or 'version not stated'})")
        if store.source == "unverified":
            store.problems.append(f"{SOURCE_FILE} does not state where the predictions came from; they are not treated as real.")
        if db is not None:
            _mirror_to_db(db, df, store, _sha256(path))
    except Exception as e:  # noqa: BLE001
        log.exception("saved predictions could not be loaded")
        store.val = store.test = _empty_split(len(store.findings))
        store.source = "none"
        store.problems.append(f"{PREDICTIONS_FILE} could not be loaded: {e}")
    return store


def _mirror_to_db(db: Database, df, store: PredictionStore, sha: str) -> None:
    """Keep the `saved_predictions` table equal to the file (re-loaded only when the file changes)."""
    with db.session() as s:
        have = s.get(Setting, "saved_predictions_sha256")
        n = s.scalar(select(func.count()).select_from(SavedPrediction))
        if have is not None and have.value == sha and n == len(df):
            return
        s.execute(delete(SavedPrediction))
        cols = [f"p_{f}" for f in store.findings]
        probs = df[cols].to_numpy(dtype=float)
        rows = [
            {"study_id": sid, "site": "nih", "split": sp, "true_label_any_urgent": int(lab), "score": float(sc),
             "probs": {f: round(float(p), 5) for f, p in zip(store.findings, pr)}, "ood_flag": bool(o), "source": store.source}
            for sid, sp, lab, sc, o, pr in zip(df["study_id"], df["split"], df["label_any_urgent"], df["composite_score"], df["ood_flag"], probs)
        ]
        for i in range(0, len(rows), 5000):
            s.execute(insert(SavedPrediction), rows[i:i + 5000])
        if have is None:
            s.add(Setting(key="saved_predictions_sha256", value=sha))
        else:
            have.value = sha
    log.info("saved_predictions table loaded: %d rows", len(df))
