"""Adapter: Member B's locked_evidence.json -> the shapes the Evidence page and the contract expect.

Everything shown comes from B's file. Nothing is invented here. Two facts are surfaced instead of
hidden:
  * B's exporter builds its cohorts with np.random (evidence_exporter.py), so the numbers are from a
    SIMULATION, not from a trained model run on real scans. `provenance` and a `data` column say so.
  * Calibration barely changed expected calibration error (ECE), so nothing here claims the
    probabilities are calibrated.
Items only A can supply (cross-dataset, failures, the uncalibrated curve) are left out, and the
page then says "Not available yet" instead of a made-up number.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

SYNTHETIC = "synthetic simulation (B's exporter), not real scans"


def build_evidence(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    metrics = raw.get("locked_finding_metrics", {})
    cal = raw.get("calibration_reports", {})

    per_finding = [
        {
            "finding": name,
            "target_sensitivity": m["target_sensitivity"],
            "sensitivity": m["achieved_sensitivity"],
            "specificity": m["achieved_specificity"],
            "auroc": m["auroc"],
            "n_test": m["n_test_samples"],
            "n_positive": m["n_positives"],
            "data": SYNTHETIC,
        }
        for name, m in metrics.items()
    ]

    out: dict = {
        "provenance": {
            "source": "eval_safety/voxelsort_eval_safety/locked_evidence.json (Member B)",
            "data": "synthetic",
            "statement": "These results come from B's simulated cohorts, not from a trained model on real scans. Do not present them as model performance.",
            "calibration_statement": "Temperature scaling changed ECE very little (see calibration). Make no claim that these probabilities are calibrated.",
        },
        "per_finding": per_finding,
        "failures": [],
    }

    # Calibration: B saved the calibrated reliability diagram and both ECE values, not the
    # uncalibrated curve. "before" stays empty rather than being reconstructed.
    first = "Pneumothorax" if "Pneumothorax" in cal else next(iter(cal), None)
    if first:
        c = cal[first]
        out["calibration"] = {
            "finding": first,
            "before": [],
            "after": [
                {"confidence": b["mean_confidence"], "accuracy": b["empirical_accuracy"], "count": b["count"]}
                for b in c["reliability_diagram"] if b["count"] > 0
            ],
            "ece_before": c["uncalibrated_ece"],
            "ece_after": c["calibrated_ece"],
            "per_finding_ece": {k: {"before": v["uncalibrated_ece"], "after": v["calibrated_ece"], "ece_reduction_pct": v["ece_reduction_pct"]} for k, v in cal.items()},
            "data": "synthetic",
        }

    sc = raw.get("shortcut_audits", {})
    ct, vp = sc.get("chest_tube_pneumothorax"), sc.get("view_position")
    if ct and vp:
        out["shortcut_audit"] = {
            "data_source": SYNTHETIC,
            "chest_tube_with_tube_sensitivity": ct["with_chest_tube_sensitivity"],
            "chest_tube_without_tube_sensitivity": ct["without_chest_tube_sensitivity"],
            "chest_tube_gap": ct["sensitivity_disparity_gap"],
            "chest_tube_result": ct["clinical_interpretation"],
            "ap_pa_auroc_gap": vp["auroc_disparity_gap"],
            "ap_pa_result": vp["interpretation"],
        }

    # Extra material from B that the contract does not (yet) name. Passed through untouched.
    out["extras"] = {
        "sample_size_and_power": raw.get("sample_size_and_power"),
        "queue_race_summary": raw.get("queue_race_summary"),
        "site_readiness_reports": raw.get("site_readiness_reports"),
        "ood_detector_audit": raw.get("ood_detector_audit"),
        "note": "queue_race_summary and the OOD audit are simulations with B's own tiers (STAT/URGENT/ROUTINE); they are not the live Queue Race.",
    }
    return out
