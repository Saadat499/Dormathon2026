"""Saved predictions: the scores and true labels that Safety vs Speed and Queue Race are computed from.

Two sources, always labelled:
  replay          Member A's saved predictions from backend/model_artifacts/ (real model output).
  synthetic_demo  A seeded, made-up cohort used ONLY when A's files are missing or unreadable.
                  It is computed on the fly (nothing hard-coded) but it is NOT model output.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .ranking import norm_name, urgency_score

SYNTH_FINDINGS = ["pneumothorax", "pleural effusion", "consolidation", "pulmonary edema", "pneumonia"]
RESERVED = {"study_id", "id", "label", "score", "ood_flag", "confidence_band", "sha256", "split", "priority", "heatmap_path", "filename"}


@dataclass
class Saved:
    study_id: str
    score: Optional[float]
    label: Optional[int]
    findings: list
    ood_flag: bool = False
    confidence_band: Optional[str] = None
    sha256: Optional[str] = None


@dataclass
class PredictionSet:
    source: str                      # "replay" | "synthetic_demo"
    items: list = field(default_factory=list)
    description: str = ""
    problems: list = field(default_factory=list)

    @property
    def labelled(self) -> list:
        return [i for i in self.items if i.label is not None and i.score is not None]

    @property
    def positives(self) -> list:
        return [i for i in self.labelled if i.label == 1]

    @property
    def negatives(self) -> list:
        return [i for i in self.labelled if i.label == 0]

    @property
    def is_real(self) -> bool:
        return self.source == "replay"


def _as_findings(raw: Any) -> list:
    if isinstance(raw, dict):
        raw = [{"name": k, "probability": v} for k, v in raw.items()]
    out = []
    for f in raw or []:
        p = f.get("probability", f.get("prob"))
        if p is None or not math.isfinite(float(p)):
            continue
        out.append({"name": norm_name(f["name"]), "probability": min(1.0, max(0.0, float(p)))})
    return sorted(out, key=lambda f: -f["probability"])


def _label(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ("1", "true", "yes", "urgent", "positive"):
            return 1
        if v in ("0", "false", "no", "routine", "negative"):
            return 0
        return None
    return 1 if bool(v) else 0


def _from_record(r: dict) -> Saved:
    findings = _as_findings(r.get("findings"))
    score = urgency_score(findings) if findings else (float(r["score"]) if r.get("score") not in (None, "") else None)
    label = _label(r.get("label", r.get("urgent")))
    return Saved(
        study_id=str(r.get("study_id", r.get("id"))), score=score, label=label, findings=findings,
        ood_flag=bool(_label(r.get("ood_flag")) or 0), confidence_band=r.get("confidence_band") or None,
        sha256=r.get("sha256"),
    )


def load_artifacts(directory: Path) -> Optional[PredictionSet]:
    """Read predictions.json or predictions.csv. Returns None when nothing is there; a PredictionSet
    with `problems` filled in is never returned half-parsed (caller falls back and reports it)."""
    problems: list[str] = []
    for name in ("predictions.json", "predictions.csv"):
        p = directory / name
        if not p.is_file():
            continue
        try:
            if p.suffix == ".json":
                data = json.loads(p.read_text())
                records = data.get("predictions", data) if isinstance(data, dict) else data
            else:
                with p.open(newline="") as fh:
                    records = []
                    for row in csv.DictReader(fh):
                        findings = {c: row[c] for c in row if c not in RESERVED and row[c] not in (None, "")}
                        records.append({**{k: row.get(k) for k in RESERVED if k in row},
                                        "findings": [{"name": k, "probability": float(v)} for k, v in findings.items()] or None})
            items = [_from_record(r) for r in records]
            items = [i for i in items if i.study_id and i.score is not None]
            if not items:
                raise ValueError("no usable rows (need study_id plus findings or a score)")
            ps = PredictionSet("replay", items, f"{p.name}: {len(items)} saved predictions, {len([i for i in items if i.label is not None])} with labels")
            return ps
        except Exception as e:  # noqa: BLE001 - report any parse failure instead of crashing the API
            problems.append(f"{p.name}: {e}")
    if problems:
        return PredictionSet("synthetic_demo", [], "", problems)
    return None


def _clamp(x: float) -> float:
    return min(0.999, max(0.001, x))


def synthetic_cohort(n: int = 600, seed: int = 7, prevalence: float = 0.2) -> PredictionSet:
    """Made-up scores with known labels (two overlapping bell curves). NOT model output."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        pos = rng.random() < prevalence
        score = _clamp(rng.gauss(0.72 if pos else 0.22, 0.18 if pos else 0.16))
        lead = rng.choice(SYNTH_FINDINGS)
        others = rng.sample([f for f in SYNTH_FINDINGS if f != lead], 2)
        findings = _as_findings([{"name": lead, "probability": score}] + [{"name": o, "probability": rng.random() * 0.3 * score} for o in others])
        items.append(Saved(f"SYN-{i + 1:04d}", urgency_score(findings), int(pos), findings))
    return PredictionSet("synthetic_demo", items, f"seeded synthetic cohort (seed {seed}, n={n}); not model output")


def load_prediction_set(directory: Path) -> PredictionSet:
    ps = load_artifacts(directory)
    if ps is not None and ps.is_real:
        return ps
    synth = synthetic_cohort()
    if ps is not None:
        synth.problems = ps.problems
    return synth


# Threshold maths shared by /threshold, /threshold/preview and the Queue Race ----------------------
def operating_point(ps: PredictionSet, value: float, band: float) -> dict:
    lab = ps.labelled
    pos = [i for i in lab if i.label == 1]
    neg = [i for i in lab if i.label == 0]
    tp = sum(1 for i in pos if i.score >= value)
    fp = sum(1 for i in neg if i.score >= value)
    unsure = sum(1 for i in lab if (i.score < value and i.score >= value - band) or i.ood_flag and i.score < value)
    return {
        "value": value,
        "abstention_rate": unsure / len(lab) if lab else 0.0,
        "sensitivity": tp / len(pos) if pos else None,
        "false_alarm_rate": fp / len(neg) if neg else 0.0,
        "n": len(lab),
        "n_positive": len(pos),
    }


def highest_threshold_meeting(ps: PredictionSet, target: float, step: float = 0.01) -> Optional[float]:
    """Highest threshold on a 0.01 grid whose sensitivity is at least `target` (fewest false alarms)."""
    pos = [i.score for i in ps.positives]
    if not pos:
        return None
    for k in range(int(round(0.99 / step)), 0, -1):
        v = round(k * step, 2)
        if sum(1 for s in pos if s >= v) / len(pos) >= target:
            return v
    return None
