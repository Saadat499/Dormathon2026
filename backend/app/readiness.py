"""Site Readiness: is there enough evidence to use VoxelSort at a new site?

Real calculations on what the user uploads:
  * sensitivity with an exact (Clopper-Pearson) interval, from labels plus model scores
  * drift (PSI) of the site's scores against the saved predictions, or, when no scores exist,
    the share of images the placeholder OOD check flags
  * a verdict from fixed rules (below). The verdict rules are an ASSUMPTION written by C because
    Member B's site_readiness.py defines a different GO/WARNING/NO-GO scheme.

Scores come from (1) a `score` column in the labels CSV, or (2) Member A's saved predictions when the
file name / `filename` column matches a saved study id. This host does not run the model, so with
neither source there is no sensitivity and the verdict is `not_enough_evidence`.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Optional

from . import config
from .cohorts import PredictionSet
from .stats import clopper_pearson, psi

NOTE_DRIFT = "Warning signal only, not a proven predictor of performance at a new site."
ID_COLUMNS = ("filename", "file", "image", "study_id", "id")


def parse_labels(text: str) -> list[dict]:
    """CSV with a `label` column (1 = urgent). Optional `score` and a file / id column."""
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    header = [c.strip().lower() for c in rows[0]]
    if "label" not in header:
        raise ValueError("The labels CSV needs a header row with a 'label' column (1 = urgent finding, 0 = not).")
    li = header.index("label")
    si = header.index("score") if "score" in header else None
    ii = next((header.index(c) for c in ID_COLUMNS if c in header), None)
    out = []
    for n, r in enumerate(rows[1:], start=2):
        if len(r) <= li:
            continue
        lab = r[li].strip().lower()
        if lab not in ("0", "1"):
            raise ValueError(f"Row {n}: label must be 0 or 1, got '{r[li].strip()[:20]}'.")
        score = None
        if si is not None and len(r) > si and r[si].strip():
            try:
                score = float(r[si])
            except ValueError:
                raise ValueError(f"Row {n}: score must be a number.") from None
            if not 0.0 <= score <= 1.0 or score != score:
                raise ValueError(f"Row {n}: score must be between 0 and 1.")
        key = r[ii].strip() if ii is not None and len(r) > ii else None
        out.append({"label": int(lab), "score": score, "key": key})
    return out


def evaluate(*, image_checks: list[dict], labels_text: Optional[str], ps: PredictionSet, threshold: float,
             target: float = config.MIN_SENSITIVITY_TARGET) -> dict:
    notes: list[str] = []
    rows = parse_labels(labels_text) if labels_text and labels_text.strip() else []

    by_id = {i.study_id.lower(): i for i in ps.items} if ps.is_real else {}
    scored = []
    for r in rows:
        s = r["score"]
        if s is None and r["key"] and by_id:
            hit = by_id.get(Path(r["key"]).stem.lower())
            s = hit.score if hit else None
        scored.append({**r, "score": s})

    # Drift --------------------------------------------------------------------------------------
    site_scores = [r["score"] for r in scored if r["score"] is not None]
    n_img = len(image_checks)
    n_ood = sum(1 for c in image_checks if c["ood_flag"])
    ref = [i.score for i in ps.items if i.score is not None]
    p = psi(ref, site_scores) if len(site_scores) >= 20 and ref else None
    if p is not None:
        drift = {"score": round(p, 4), "warning": p >= config.DRIFT_WARNING_PSI, "kind": "psi_of_model_scores", "note": NOTE_DRIFT}
        if not ps.is_real:
            drift["note"] += " The reference scores are synthetic because Member A's saved predictions are not available."
    elif n_img:
        share = n_ood / n_img
        drift = {
            "score": round(share, 4), "warning": share >= config.DRIFT_WARNING_PSI, "kind": "ood_flag_share",
            "note": (f"No model scores were available, so this score is the share of images ({n_ood} of {n_img}) "
                     f"flagged by the placeholder OOD check, not a score drift. " + NOTE_DRIFT),
        }
    else:
        drift = {"score": None, "warning": False, "kind": "none", "note": "No images and no scores were supplied, so no drift check could run. " + NOTE_DRIFT}

    # Sensitivity --------------------------------------------------------------------------------
    positives = [r for r in scored if r["label"] == 1]
    with_score = [r for r in positives if r["score"] is not None]
    sensitivity = None
    verdict = "not_enough_evidence"
    if not rows:
        notes.append("No labels CSV was supplied, so sensitivity could not be measured.")
    elif not positives:
        notes.append("The labels CSV has no urgent (label = 1) scans, so sensitivity could not be measured.")
    elif not with_score:
        notes.append("No model scores are available for the labelled scans. This server does not run the model: add a 'score' column to the CSV "
                     "(model output, 0 to 1) or use file names that match Member A's saved study ids.")
    else:
        if len(with_score) < len(positives):
            notes.append(f"{len(positives) - len(with_score)} of {len(positives)} urgent scans had no score and were left out.")
        hit = sum(1 for r in with_score if r["score"] >= threshold)
        n = len(with_score)
        lo, hi = clopper_pearson(hit, n, 0.95)
        sensitivity = {"point": hit / n, "ci_low": lo, "ci_high": hi, "width": hi - lo, "method": "clopper-pearson", "n_positive": n}
        if hi < target:
            verdict = "not_ready"
        elif n < config.MIN_POSITIVES_FOR_VERDICT or lo < target:
            verdict = "not_enough_evidence"
            if n < config.MIN_POSITIVES_FOR_VERDICT:
                notes.append(f"Only {n} urgent scans; at least {config.MIN_POSITIVES_FOR_VERDICT} are needed before a verdict other than 'not enough evidence'.")
        else:
            verdict = "use_with_caution" if drift["warning"] else "ready"
        notes.append(f"Sensitivity counts an urgent scan as caught when its score is at or above the current threshold ({threshold:.2f}).")
        notes.append("Sensitivity uses the labels CSV only; uploaded images feed the drift check, they are not scored here.")

    return {
        "status": "done",
        "drift": drift,
        "sensitivity": sensitivity,
        "target": target,
        "verdict": verdict,
        "note": " ".join(notes) if notes else None,
        "n_images": n_img,
        "n_images_flagged_ood": n_ood,
        "n_label_rows": len(rows),
    }
