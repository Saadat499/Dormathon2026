"""Build backend/model_artifacts/ from Member A's full ML delivery.

    python scripts/build_model_artifacts.py --delivery "C:\\path\\to\\chesttriage_ml"

`--delivery` is the folder that contains artifacts_v2_imagenet_fullnih/ WITH its predictions/,
demo_images/ and reports/ sub-folders (they are not in the git repo). Nothing is retrained and no
dataset is downloaded: this script only selects, verifies, copies and summarises.

What it writes
  committed (small, derived, public NIH data only)
    MANIFEST.json                    Member A's manifest for the selected model, unchanged
    SOURCE.json                      provenance: what was built from what, checksums, verification results
    saved_predictions_nih.csv.gz     NIH validation + test predictions with labels (scores, no images)
    evidence.json                    real evaluation results for GET /evidence
    demo/demo_studies.json           saved model outputs for the demo images, keyed by image SHA-256
    demo/images/*.png, demo/heatmaps/*.png, demo/failures/*.jpg
  not committed (git-ignored; fetched by scripts/fetch_model.py or copied here)
    backbone.json, backbone_densenet121-onnx.onnx, head.npz, policy.json, ood.joblib, drift_reference.npy

CheXpert images, labels and per-image predictions are NOT copied: the Stanford research-use
agreement does not allow redistribution. CheXpert appears only through A's aggregate report tables.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.evalkit.calibration import ModelCalibrator, select_threshold_at_sensitivity  # noqa: E402
from app.evalkit.protocol import compute_binary_metrics  # noqa: E402
from app.evalkit.shortcut_audit import ShortcutAuditor  # noqa: E402
from app.services import heatmap  # noqa: E402
from app.services.predictor import DRIFT_FILE, RUNTIME_FILES, RealModel, file_sha256, ood_reason  # noqa: E402

SELECTED = "artifacts_v2_imagenet_fullnih"
RAW_BASE = "https://raw.githubusercontent.com/Saadat499/Dormathon2026/5ff63982734a4051c6d349868799f820ea56405f/ml/"
TARGET = 0.9
# Demo queue: 12 studies in arrival order (newest first), chosen so every lane is visible at the default
# threshold. N = normal, U = unsure, R = raised, O = out-of-distribution demo image. Position 11 has waited
# past the 120-minute cap when the server starts; position 12 is the already-reviewed one.
SEED_PATTERN = ["R", "N", "U", "O", "R", "N", "R", "U", "N", "R", "N", "R"]


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), n)


def rows(path: Path, keep: list[str] | None = None, digits: int = 3) -> list[dict]:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row.pop("artifacts_folder", None)
            row.pop("model_version", None)
            rec = {}
            for k, v in row.items():
                if keep and k not in keep:
                    continue
                try:
                    f = float(v)
                    rec[k] = int(f) if f.is_integer() and "." not in v else round(f, digits)
                except (TypeError, ValueError):
                    rec[k] = v
            out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", required=True, help="Member A's project folder (contains the artifacts_* folders)")
    ap.add_argument("--out", default=str(BACKEND / "model_artifacts"))
    args = ap.parse_args()
    delivery, out = Path(args.delivery), Path(args.out)
    art = delivery / SELECTED
    if not (art / "predictions" / "nih_full.csv").is_file():
        print(f"{art} does not contain predictions/nih_full.csv. Point --delivery at the full ML delivery.")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- 1. select and verify the model
    manifest = json.loads((art / "MANIFEST.json").read_text())
    version = manifest["model_version"]
    checks = {}
    for name, want in manifest["deploy_files"].items():
        got = file_sha256(art / name)
        checks[name] = got == want
        if got != want:
            print(f"CHECKSUM MISMATCH {name}: the delivery does not match its own MANIFEST.json")
            return 3
    from chesttriage_ml.contract import model_version
    assert model_version(art) == version, "model_version recomputed from backbone.json + head.npz + policy.json differs from MANIFEST"
    policy = json.loads((art / "policy.json").read_text())
    assert manifest["policy"] == policy, "policy.json differs from the policy recorded in MANIFEST.json"
    head_sha = manifest["deploy_files"]["head.npz"]
    assert head_sha.startswith(policy["fitted"]["head"]), "policy.json was fitted for a different head.npz"
    backbone = json.loads((art / "backbone.json").read_text())
    assert backbone["name"] == manifest["backbone"] == "densenet121-onnx"
    repo_copy = BACKEND.parent / "ml" / SELECTED
    same_as_repo = {n: (repo_copy / n).is_file() and file_sha256(repo_copy / n) == file_sha256(art / n)
                    for n in list(RUNTIME_FILES) + [DRIFT_FILE]} if repo_copy.is_dir() else {}

    shutil.copy2(art / "MANIFEST.json", out / "MANIFEST.json")
    runtime_sha = {}
    for name in list(RUNTIME_FILES) + [DRIFT_FILE]:
        shutil.copy2(art / name, out / name)
        runtime_sha[name] = file_sha256(out / name)

    # ------------------------------------------------------------- 2. saved predictions (NIH val + test)
    full = pd.read_csv(art / "predictions" / "nih_full.csv")
    assert set(full["model_version"].unique()) == {version}, "predictions were exported by a different model version"
    findings = policy["findings"]
    cols = ["study_id", "split", "label_any_urgent", "composite_score"] + [f"p_{f}" for f in findings] + ["ood_flag"]
    keep = full[full["split"].isin(["val", "test"])][cols].copy()
    for c in ["composite_score"] + [f"p_{f}" for f in findings]:
        keep[c] = keep[c].round(5)
    keep["label_any_urgent"] = keep["label_any_urgent"].astype(int)
    with gzip.GzipFile(out / "saved_predictions_nih.csv.gz", "wb", mtime=0) as gz:      # mtime=0: reproducible bytes
        gz.write(keep.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    val, test = keep[keep["split"] == "val"], keep[keep["split"] == "test"]
    comp_check = float(np.abs(full[[f"p_{f}" for f in policy["urgent"]]].max(axis=1) - full["composite_score"]).max())
    assert comp_check < 1e-9, "composite_score is not the max urgent probability"

    # default threshold: Member B's rule on the VALIDATION split, snapped to the percentile grid
    vs, vy = val["composite_score"].to_numpy(), val["label_any_urgent"].to_numpy()
    eff = np.where(val["ood_flag"].to_numpy(dtype=bool), 0.0, vs)
    b_sel = select_threshold_at_sensitivity(vy, eff, target_sensitivity=TARGET)
    value = int((vs < b_sel["selected_threshold"]).mean() * 100) / 100

    def sens(v):
        return compute_binary_metrics(vy, eff, threshold=float(np.quantile(vs, v)))["sensitivity"]
    while sens(value) < TARGET:
        value = round(value - 0.01, 2)
    while sens(round(value + 0.01, 2)) >= TARGET:
        value = round(value + 0.01, 2)

    # ------------------------------------------------------------- 3. demo studies: verify live == saved
    model = RealModel(out, threads=2)
    assert model.version == version
    contract = {json.loads(l)["study_id"]: json.loads(l) for l in (art / "predictions" / "demo_contract.jsonl").read_text().splitlines() if l.strip()}
    demo_manifest = pd.read_csv(art / "predictions" / "demo_manifest.csv")
    demo_manifest = demo_manifest[demo_manifest["dataset"] != "chexpert"]
    score_only = full.set_index("study_id")["composite_score"].to_dict()
    contract_label = {sid: (0 if pd.isna(v) else int(v)) for sid, v in zip(demo_manifest["study_id"], demo_manifest["label_any_urgent"])}
    for sub in ("images", "heatmaps", "failures"):
        d = out / "demo" / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    seed_order, studies, max_diff, flag_mismatch = {}, [], 0.0, 0
    t_raise, t_unsure = float(np.quantile(vs, value)), float(np.quantile(vs, max(0.0, value - policy["unsure_budget"])))
    q = demo_manifest[demo_manifest["category"] == "queue_nih"]["study_id"].tolist()
    lane = lambda sid: "R" if score_only[sid] >= t_raise else ("U" if score_only[sid] >= t_unsure else "N")  # noqa: E731
    pools = {k: [sid for sid in q if lane(sid) == k] for k in "RUN"}
    pools["R"].sort(key=lambda sid: -contract_label[sid])                 # truly urgent demo scans first
    pools["O"] = [sid for sid in demo_manifest[demo_manifest["category"] == "ood_demo"]["study_id"] if "rotated" in sid]
    for pos, k in enumerate(SEED_PATTERN, start=1):
        if pools[k]:
            seed_order[pools[k].pop(0)] = pos
    for _, m in demo_manifest.iterrows():
        sid = m["study_id"]
        rec = contract[sid]
        src = art / rec["image_path"]
        data = src.read_bytes()
        pil = Image.open(src)
        pil.load()
        live = model.predict(pil)
        if live.ood_flag != bool(rec["ood_flag"]):
            flag_mismatch += 1
        if rec["findings"] and live.findings:
            max_diff = max(max_diff, max(abs(rec["findings"][f["name"]] - f["probability"]) for f in live.findings))
        shutil.copy2(src, out / "demo" / "images" / src.name)
        heat = None
        if live.cam is not None:
            heat = f"demo/heatmaps/{sid}.png"
            heatmap.overlay_rgba(live.cam, pil.size).save(out / heat, "PNG", optimize=True)
        studies.append({
            "study_id": sid, "category": m["category"], "dataset": m["dataset"],
            "label_any_urgent": None if pd.isna(m["label_any_urgent"]) else int(m["label_any_urgent"]),
            "true_findings": None if pd.isna(m["true_findings"]) else str(m["true_findings"]),
            "findings": rec["findings"], "composite_score": rec["composite_score"], "top_urgent_finding": rec["top_urgent_finding"],
            "ood_flag": bool(rec["ood_flag"]), "ood_score": rec["ood_score"],
            "ood_reason": live.ood.reason if (live.ood and live.ood.ood_flag) else None,
            "sha256": hashlib.sha256(data).hexdigest(), "image": f"demo/images/{src.name}", "heatmap": heat,
            "model_version": rec["model_version"], "seed_order": seed_order.get(sid),
        })
    assert flag_mismatch == 0, "live OOD flags differ from the saved demo outputs"
    assert max_diff < 2e-4, f"live probabilities differ from the saved demo outputs by {max_diff}"
    (out / "demo" / "demo_studies.json").write_text(json.dumps({
        "model_version": version,
        "note": "Saved outputs of Member A's frozen model for public NIH ChestX-ray14 demo images (and A's synthetic non-chest test images). "
                "Replay mode returns one of these only for a byte-identical upload (sha256).",
        "studies": studies}, indent=1), encoding="utf-8")

    # ------------------------------------------------------------- 4. evidence (real results only)
    rep = art / "reports"
    internal = {x["finding"]: x for x in rows(rep / "internal_metrics.csv", digits=4)}
    cross_raw = rows(rep / "cross_dataset.csv", digits=4)
    cross = {x["finding"]: x for x in cross_raw}
    per_finding = [{
        "finding": f, "auroc": r(x["auroc"], 3), "auroc_ci_low": r(x["auroc_lo"], 3), "auroc_ci_high": r(x["auroc_hi"], 3),
        "auprc": r(x["auprc"], 3), "n_positive": int(x["n_pos_test"]), "prevalence": r(x["prevalence"], 3),
        "sensitivity_at_val90_threshold": r(cross[f]["sens_nih"], 3), "false_alarm_rate_there": r(cross[f]["false_alarm_nih"], 3),
        "data": "NIH test split, held out",
    } for f, x in internal.items()]
    cross_dataset = [{k: (r(x[k], 3) if isinstance(x[k], float) else x[k]) for k in
                      ("finding", "auroc_nih", "auroc_chexpert", "auroc_drop", "sens_nih", "sens_chexpert",
                       "false_alarm_nih", "false_alarm_chexpert", "n_pos_chexpert")} for x in cross_raw]

    vfull, tfull = full[full["split"] == "val"], full[full["split"] == "test"]
    cal_per = {}
    curves = {}
    for f in findings:
        cal = ModelCalibrator(method="temperature").fit(vfull[f"p_{f}"].to_numpy(), vfull[f"y_{f}"].to_numpy())
        y, p = tfull[f"y_{f}"].to_numpy(), tfull[f"p_{f}"].to_numpy()
        before = ModelCalibrator.compute_reliability_curve(y, p)
        after = ModelCalibrator.compute_reliability_curve(y, cal.transform(p))
        cal_per[f] = {"ece_before": before["ece"], "ece_after": after["ece"], "temperature": r(cal.temp_scaler.temperature, 3),
                      "n_test": int(len(y)), "n_positive_test": int(y.sum())}
        curves[f] = (before, after)
    cx_path = art / "predictions" / "chexpert_full.csv"
    if cx_path.is_file():                                                 # aggregates only: no CheXpert row is copied
        cx = pd.read_csv(cx_path)
        cx = cx[cx["split"] == "external"]
        for f in findings:
            m = cx[f"y_{f}"].notna()
            c = ModelCalibrator.compute_reliability_curve(cx.loc[m, f"y_{f}"].to_numpy(), cx.loc[m, f"p_{f}"].to_numpy())
            cal_per[f]["ece_chexpert_external"] = c["ece"]
            cal_per[f]["n_chexpert_external"] = int(m.sum())
    # The calibration gap between the two datasets belongs next to the AUROC drop, where the page shows it.
    for row in cross_dataset:
        c = cal_per.get(row["finding"], {})
        row["ece_nih_test"] = c.get("ece_before")
        row["ece_chexpert"] = c.get("ece_chexpert_external")
    for row in per_finding:
        row["ece_nih_test"] = cal_per.get(row["finding"], {}).get("ece_before")
    lead = "effusion"                                                     # the finding with the most positive test scans
    pts = lambda c: [{"confidence": b["mean_confidence"], "accuracy": b["empirical_accuracy"], "count": b["count"]}  # noqa: E731
                     for b in c["binned_curve"] if b["count"] > 0]
    calibration = {
        "finding": lead, "before": pts(curves[lead][0]), "after": pts(curves[lead][1]),
        "ece_before": cal_per[lead]["ece_before"], "ece_after": cal_per[lead]["ece_after"],
        "temperature": cal_per[lead]["temperature"], "per_finding": cal_per,
        "data": "real: fitted on NIH validation, measured on NIH test (held out)",
        "method": "Member B's ModelCalibrator (temperature scaling) and compute_reliability_curve (10 equal-width bins)",
        "note": ("'Before' is the model as served. 'After' applies a temperature fitted on the validation split. The temperature is NOT "
                 "applied to the scores the app serves (Member A's policy.json has no temperatures), so 'before' describes the live "
                 "probabilities. On NIH test (same source as the training data) the expected calibration error is small because most "
                 "scans have low probabilities; on CheXpert, a different hospital's dataset, it is much larger (per_finding."
                 "ece_chexpert_external). Report these numbers; do not describe the model as well calibrated."),
    }

    t_cut = float(np.quantile(vs, value))
    view = ShortcutAuditor.audit_view_position_confounder(
        tfull.rename(columns={"view": "view_position"}), finding_col="label_any_urgent", prob_col="composite_score", threshold=t_cut)
    vm = view["view_metrics"]
    att_md = (rep / "heatmap_attention.md").read_text(encoding="utf-8")
    att = {}
    for line in att_md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[1].replace(".", "").isdigit():
            att[cells[0]] = [float(c) for c in cells[1:]]
    shortcut = {
        "data_source": "real: NIH test split (held out) and Member A's saliency audit",
        "ap_view_any_urgent_auroc": r(vm["AP"]["auroc"], 3), "pa_view_any_urgent_auroc": r(vm["PA"]["auroc"], 3),
        "ap_pa_auroc_gap": view["auroc_disparity_gap"],
        "ap_view_sensitivity_at_current_cutoff": r(vm["AP"]["sensitivity"], 3), "pa_view_sensitivity_at_current_cutoff": r(vm["PA"]["sensitivity"], 3),
        "ap_view_false_alarm_rate": r(1 - vm["AP"]["specificity"], 3), "pa_view_false_alarm_rate": r(1 - vm["PA"]["specificity"], 3),
        "ap_view_n": int(vm["AP"]["n_samples"]), "pa_view_n": int(vm["PA"]["n_samples"]),
        "ap_pa_result_by_b_rule": view["interpretation"],
        "ap_pa_reading": (f"B's rule only checks the AUROC gap (limit 0.10); the gap here is {view['auroc_disparity_gap']:.3f}. At the current "
                          f"cut-off AP (bedside) scans are raised much more often than PA scans: false-alarm rate "
                          f"{1 - vm['AP']['specificity']:.3f} against {1 - vm['PA']['specificity']:.3f}. View position may act as a shortcut."),
        "ap_pa_method": "Member B's ShortcutAuditor.audit_view_position_confounder on the 'any urgent finding' score",
        "saliency_top_band_share_nih_missed_urgent": att.get("nih_missed_urgent", [None] * 3)[2],
        "saliency_top_band_share_nih_confident_hit": att.get("nih_confident_hit", [None] * 3)[2],
        "saliency_top_band_share_if_uniform": att.get("uniform map (= area share)", [None] * 3)[2],
        "saliency_border_share_chexpert_missed_urgent": att.get("chexpert_missed_urgent", [None] * 3)[0],
        "saliency_border_share_if_uniform": att.get("uniform map (= area share)", [None] * 3)[0],
        "saliency_reading": "Member A's audit: maps for missed urgent scans sit on the top band (neck, clavicles) far more than a uniform map would; several CheXpert maps sit on edges, tubes and one text marker. Saliency is not evidence that the model is right.",
        "chest_tube_audit": "not available: neither dataset copy carries a chest-tube annotation, so B's chest-tube audit cannot be run on real data",
    }

    fails = pd.read_csv(rep / "where_it_fails.csv")
    fails = fails[fails["category"].isin(["nih_missed_urgent", "nih_false_alarm"])]
    failures = []
    for _, f in fails.iterrows():
        sid = f["study_id"]
        img = out / "demo" / "images" / f"{sid}.png"
        if not img.is_file():
            continue
        pil = Image.open(img)
        pil.load()
        cam, cam_finding = model.engine.cam(pil)
        heatmap.side_by_side(pil, cam).save(out / "demo" / "failures" / f"{sid}.jpg", "JPEG", quality=85)
        if f["category"] == "nih_missed_urgent":
            cap = (f"Missed urgent scan. Labelled {f['true_findings']}, but the model's urgent score was only {f['composite_score']:.2f} "
                   f"(its top urgent finding: {f['top_urgent_finding']}). Right: saliency for {cam_finding}.")
        else:
            cap = (f"False alarm. No urgent finding in the label, but the model scored {f['composite_score']:.2f} for "
                   f"{f['top_urgent_finding']}. Right: saliency for {cam_finding}.")
        failures.append({"study_id": sid, "kind": f["category"].replace("nih_", ""), "caption": cap,
                         "image_path": f"/evidence/failures/{sid}.jpg", "dataset": "NIH ChestX-ray14 (public)"})

    comparison = rows(rep / "model_comparison.csv", digits=3)
    evidence = {
        "provenance": {
            "data": "real",
            "model_version": version, "artifacts_folder": SELECTED,
            "statement": ("Results of Member A's frozen model on public datasets. Trained on NIH ChestX-ray14 train patients; "
                          "thresholds and calibration fitted on the NIH validation split; reported on the NIH test split "
                          "(21,845 images, never used for training, thresholds or model choice) and on CheXpert (a different hospital's dataset)."),
            "sources": ["reports/internal_metrics.csv", "reports/cross_dataset.csv", "reports/where_it_fails.csv",
                        "reports/heatmap_attention.md", "reports/ood_report.csv", "reports/drift_controls.csv",
                        "reports/budget_table.csv", "reports/model_comparison.csv", "predictions/nih_full.csv"],
            "calibration_statement": "No claim is made that the probabilities are well calibrated. See the calibration block for the measured curves.",
            "caveats": [
                "Public data is enriched: 'any urgent finding' is 21% of NIH and 72% of CheXpert, far above a typical worklist. PPV must be recomputed for a hospital's own prevalence.",
                "NIH labels are text-mined from reports and CheXpert labels are extracted by a labeller: both are noisy.",
                "The urgent list (pneumothorax, effusion, edema, consolidation, pneumonia) is a provisional assumption that needs clinician review.",
                "The model is modest: AUROC 0.81 for 'any urgent finding' on NIH test. Pneumonia is weak (0.69) and is not a headline finding.",
                "Performance dropped on a different hospital's dataset (0.811 to 0.755). Labels, prevalence and image format all contribute.",
                "At the 90% sensitivity target about half of non-urgent scans are also raised.",
            ],
        },
        "per_finding": per_finding,
        "cross_dataset": cross_dataset,
        "calibration": calibration,
        "shortcut_audit": shortcut,
        "failures": failures,
        "extras": {
            "ood_report": rows(rep / "ood_report.csv", digits=4),
            "ood_statement": "Member A's detector flagged 99-100% of the gross mistakes in A's tests and 0.6% of NIH test images. The same mistake types were used to train it, so real-world catch rates will be lower. It flags only 6.7% of CheXpert: it is not a site-shift detector.",
            "drift_controls": rows(rep / "drift_controls.csv", digits=3),
            "budget_table": [x for x in rows(rep / "budget_table.csv", digits=3) if x.get("mode") == "queue_rank"],
            "model_comparison_v1_vs_v2": comparison,
        },
    }
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1), encoding="utf-8")

    # ------------------------------------------------------------- 5. provenance
    any_urgent = next(x for x in comparison if x["finding"] == "ANY_URGENT")
    source = {
        "data": "real",
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "built_by": "backend/scripts/build_model_artifacts.py",
        "model_version": version, "artifacts_folder": SELECTED, "frozen_at": manifest["frozen_at"],
        "dataset": "NIH ChestX-ray14 (public), patient-level splits by Member A",
        "findings": findings, "urgent": policy["urgent"],
        "selection": {
            "chosen": SELECTED,
            "why": ("Member A's README names it the frozen model. Its MANIFEST checksums match the files, its model_version is "
                    "reproduced from backbone.json + head.npz + policy.json, policy.json was fitted for this head.npz, and every "
                    "prediction row carries this version. On the same 21,845 held-out NIH test images it beats v1 on every finding."),
            "rejected": {"artifacts_v1_imagenet_frozen": "same backbone, head trained on a 30,000-image subset; lower AUROC on every finding",
                         "artifacts": "intermediate working copy, identical to v1 (same model_version); not used by any report"},
            "any_urgent_auroc_v1": any_urgent["auroc_A"], "any_urgent_auroc_v2": any_urgent["auroc_B"],
            "paired_difference": [any_urgent["diff_B_minus_A"], any_urgent["diff_lo"], any_urgent["diff_hi"]],
        },
        "verification": {
            "deploy_file_checksums_match_manifest": checks,
            "runtime_files_identical_to_repo_ml_folder": same_as_repo,
            "policy_matches_manifest": True, "policy_fitted_for_this_head": True,
            "composite_is_max_urgent_probability": True,
            "demo_images_rescored": len(studies),
            "max_abs_difference_live_vs_saved_probability": max_diff,
            "ood_flags_match_saved": True,
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        },
        "splits": {"reference": "val", "report": "test", "val_rows": int(len(val)), "test_rows": int(len(test)),
                   "val_urgent": int(val["label_any_urgent"].sum()), "test_urgent": int(test["label_any_urgent"].sum()),
                   "rule": "thresholds and the sensitivity floor use the validation split only; the test split is reported, never tuned on"},
        "default_threshold": {"target": TARGET, "value": value, "score_cutoff": float(np.quantile(vs, value)),
                              "b_selected_score_threshold": b_sel["selected_threshold"],
                              "method": "Member B's select_threshold_at_sensitivity on the validation split, snapped down to the 0.01 percentile grid"},
        "demo": {"studies": len(studies), "seeded": len(seed_order), "chexpert_included": False,
                 "seed_selection": "12 demo images chosen so that raised, unsure, normal and out-of-distribution scans are all visible at the default threshold; not a performance sample"},
        "runtime_files": runtime_sha,
        "runtime_base_url": RAW_BASE + SELECTED + "/",
        "committed_files": {n: file_sha256(out / n) for n in ("MANIFEST.json", "saved_predictions_nih.csv.gz", "evidence.json", "demo/demo_studies.json")},
        "licence_note": "NIH ChestX-ray14 images and labels are public (NIH Clinical Center; cite Wang et al. 2017). CheXpert rows and images are excluded: its research-use agreement forbids redistribution.",
    }
    (out / "SOURCE.json").write_text(json.dumps(source, indent=1), encoding="utf-8")
    print(json.dumps({k: source[k] for k in ("model_version", "splits", "default_threshold", "demo")}, indent=1))
    print("max |live - saved| probability on demo images:", max_diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
