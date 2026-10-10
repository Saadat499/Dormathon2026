"""Dry-run the full Part A pipeline on synthetic data with the numpy 'fake' backbone (no torch).
Usage: python tests/test_pipeline.py
Checks that every stage runs, outputs have the right shape, the live engine reproduces the
exported predictions, the demo bundle obeys the safety rules and deploys as is, and the freeze
step catches stale outputs. It does NOT check model quality (the images are synthetic)."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from make_synthetic_data import build  # noqa: E402

work = Path(tempfile.gettempdir()) / "chesttriage_test"
shutil.rmtree(work, ignore_errors=True)
build(work / "data")
env = {**os.environ, "CHESTTRIAGE_ARTIFACTS": str(work / "artifacts")}


def run(*args, expect_fail=False):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, capture_output=True, text=True)
    print(f"$ {' '.join(map(str, args[:1]))} {' '.join(map(str, args[1:3]))} ... exit={r.returncode}")
    if expect_fail:
        assert r.returncode, f"{args[0]} should have failed"
        return r.stdout + r.stderr
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(1)
    return r.stdout


S = "scripts/"
run(S + "00_check_setup.py", "--nih_root", work / "data/nih", "--chexpert_root", work / "data/chexpert",
    "--nih_n", 400, "--chexpert_n", 250, "--backbone", "fake")
run(S + "01_extract_features.py", "--workers", 2, "--batch_size", 32)
run(S + "02_train_head.py", "--n_boot", 30)
run(S + "03_cross_dataset_eval.py", "--n_boot", 30)
run(S + "04_build_ood.py", "--n_neg_images", 60, "--n_eval_images", 30, "--n_synth", 40)
run(S + "04b_drift_controls.py", "--n", 20, "--repeats", 2, "--n_eval", 40)
run(S + "05_export_predictions.py")
run(S + "06_heatmaps_and_failures.py", "--n", 2, "--n_queue", 8, "--n_external", 4)
run(S + "07_freeze.py")
run(S + "08_queue_race.py", "--reps", 3, "--utilisation", 0.9, 1.0)

import numpy as np, pandas as pd
os.environ["CHESTTRIAGE_ARTIFACTS"] = str(work / "artifacts")
A = work / "artifacts"
for f in ["head.npz", "policy.json", "ood.joblib", "backbone.json", "reports/cross_dataset.csv",
          "reports/ood_report.csv", "reports/where_it_fails.csv", "predictions/nih_contract.jsonl",
          "predictions/chexpert_full.csv", "predictions/demo_contract.jsonl", "reports/label_overlap.csv",
          "predictions/demo_manifest.csv", "MANIFEST.json", "reports/budget_table.csv", "reports/drift_report.csv",
          "drift_reference.npy", "reports/heatmap_sheet.png", "reports/queue_race.csv",
          "reports/drift_controls.md", "reports/format_effect_on_predictions.csv"]:
    assert (A / f).exists(), f"missing {f}"
recs = [json.loads(l) for l in open(A / "predictions/nih_contract.jsonl")]
req = {"study_id", "findings", "priority", "confidence_band", "ood_flag", "heatmap_path"}
assert all(req <= set(r) for r in recs), "contract keys missing"
assert {r["priority"] for r in recs} <= {"Raised", "Normal", "AI unsure"}
cx = pd.read_csv(A / "subsets/chexpert.csv")
assert (cx.split.isin(["external", "gold_valid"])).all() and len(cx) > 0
assert not cx.path.str.contains("lateral").any(), "lateral views must be filtered"
assert cx[[c for c in cx.columns if c.startswith("y_")]].isna().any().any(), "uncertain labels should be masked (NaN)"
assert cx.study_id.is_unique and pd.read_csv(A / "subsets/nih.csv").study_id.is_unique, "duplicate studies"

from chesttriage_ml.engine import ChestTriageEngine
eng = ChestTriageEngine(A)
nih = pd.read_csv(A / "subsets/nih.csv")
by_id = {r["study_id"]: r for r in recs}
bad = 0
sample = nih.sample(12, random_state=1)
out = eng.predict_batch(sample.path.tolist(), sample.study_id.tolist(), heatmap_dir=A / "heatmaps_test")
for rec in out:
    ref = by_id[rec["study_id"]]
    same = rec["priority"] == ref["priority"] and rec["ood_flag"] == ref["ood_flag"] and all(
        abs(rec["findings"].get(k, 0) - v) < 2e-3 for k, v in ref["findings"].items())
    bad += not same
print(f"engine vs export mismatches: {bad} of {len(out)}")
assert bad == 0, "engine must reproduce exported predictions"
assert any(r["heatmap_path"] and Path(r["heatmap_path"]).exists() for r in out), "no heatmaps written"
assert {r["model_version"] for r in out} == {recs[0]["model_version"]}, "engine and export disagree on model_version"

# ---- demo bundle: OOD rule applied, deployable relative paths, no duplicates
demo = [json.loads(l) for l in open(A / "predictions/demo_contract.jsonl")]
assert len({r["study_id"] for r in demo}) == len(demo), "duplicate demo studies"
full = pd.concat([pd.read_csv(A / "predictions/nih_full.csv"), pd.read_csv(A / "predictions/chexpert_full.csv")])
ood_ids = set(full.study_id[full.ood_flag])
for r in demo:
    assert r["image_path"] and not Path(r["image_path"]).is_absolute() and (A / r["image_path"]).exists(), r
    assert r["heatmap_path"] is None or (not Path(r["heatmap_path"]).is_absolute()
                                         and (A / r["heatmap_path"]).exists()), r
    if r["study_id"] in ood_ids or r["ood_flag"]:
        assert r["ood_flag"] and r["priority"] == "AI unsure" and r["findings"] == {} and r["heatmap_path"] is None, r
assert any(r["study_id"].startswith("synthetic_") and r["ood_flag"] for r in demo), "no OOD demo flagged"
wf = pd.read_csv(A / "reports/where_it_fails.csv")
cxf = wf[wf.dataset == "chexpert"]
assert (cxf.ood_flag == cxf.study_id.isin(ood_ids)).all(), "where_it_fails must report the real OOD flag"

# ---- capacity-budget policy: Raised ~ top 20% of NIH validation, AI unsure the next 10%
pj = json.load(open(A / "policy.json"))
assert pj["mode"] == "budget" and pj["t_unsure"] < pj["t_raise"], pj
assert abs(pj["fitted"]["raised_share"] - 0.20) < 0.05 and abs(pj["fitted"]["raised_or_unsure_share"] - 0.30) < 0.06, pj["fitted"]
bt = pd.read_csv(A / "reports/budget_table.csv")
assert set(bt.dataset) == {"nih_test", "chexpert_external"} and set(bt["mode"]) == {"fixed_cutoff", "queue_rank"}
assert sorted(bt.budget.unique()) == [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
assert (bt.ceiling_sensitivity <= 1).all() and "false_alarm_rate" in bt and "ppv_at_dataset_prevalence" in bt
assert (bt.artifacts_folder == "artifacts").all() and bt.model_version.str.startswith("partA-").all()
qrk = bt[bt["mode"] == "queue_rank"]
assert (abs(qrk.flag_rate - qrk.budget) < 0.03).all(), "rank mode must flag the budget share at any site"
assert "recomputed" in (A / "reports/budget_table.md").read_text()
from chesttriage_ml.policy import Policy
lr = Policy.load(A / "policy.json").queue_rank([0.9, 0.1, 0.5, 0.2, 0.8, 0.3, 0.4, 0.6, 0.7, 0.05], [0] * 9 + [1])
assert list(lr).count("Raised") == 2 and lr[0] == "Raised" and lr[9] == "AI unsure", lr
qr = pd.read_csv(A / "reports/queue_race.csv")
assert set(qr.policy) == {"fcfs", "lanes_fixed", "lanes_rank", "score"} and (qr.max_wait_any >= 0).all()
assert set(qr.assumed_prevalence) == {0.10, 0.20} and "arrivals_per_hour" in qr
assert "ASSUMPTIONS" in (A / "reports/queue_race.md").read_text()
fc = qr[qr.policy == "fcfs"].nonurgent_mean_wait
assert (fc > 0).all(), "queue must form"
assert (A / "reports/heatmap_attention.csv").exists()

# ---- Site Readiness interface: scores even for OOD images, same values as *_full.csv
cxs = pd.read_csv(A / "subsets/chexpert.csv").head(6)
rows = eng.analyse_batch(cxs.path.tolist(), cxs.study_id.tolist())
fi = full.set_index("study_id")
assert all(r["composite_score"] is not None and "p_effusion" in r for r in rows)
assert all(abs(r["composite_score"] - fi.composite_score[r["study_id"]]) < 2e-3 for r in rows)
assert all(r["ood_flag"] == bool(fi.ood_flag[r["study_id"]]) for r in rows)
dr = eng.site_drift(pd.read_csv(A / "subsets/chexpert.csv").path.head(40).tolist())
assert 0 <= dr["domain_auroc"] <= 1 and "ood_flag_rate" in dr, dr
try:
    eng.site_drift(pd.read_csv(A / "subsets/chexpert.csv").path.head(10).tolist())
    raise AssertionError("site samples under 20 images must be refused")
except ValueError:
    pass
cam, f = eng.cam(sample.path.iloc[0])
assert cam.shape == (224, 224) and 0 <= cam.min() and cam.max() <= 1 and f in eng.policy.urgent
print("\nALL CHECKS PASSED")

# ---- Member B's calibration hook: temperatures must flow into policy, exports and the engine
(A / "calibration.json").write_text(json.dumps({"temperature": {"effusion": 2.0, "pneumothorax": 1.5}}))
run(S + "05_export_predictions.py")
pol = json.load(open(A / "policy.json"))
assert pol["temperature"] == {"effusion": 2.0, "pneumothorax": 1.5}, "calibration not applied"
eng2 = ChestTriageEngine(A)
r0 = eng.predict(sample.path.iloc[0], sample.study_id.iloc[0])
r1 = eng2.predict(sample.path.iloc[0], sample.study_id.iloc[0])
ref = {json.loads(l)["study_id"]: json.loads(l) for l in open(A / "predictions/nih_contract.jsonl")}[sample.study_id.iloc[0]]
assert abs(r1["findings"]["effusion"] - ref["findings"]["effusion"]) < 2e-3, "engine ignores calibration"
assert abs(r1["findings"]["effusion"] - r0["findings"]["effusion"]) > 1e-4 or r0["findings"]["effusion"] in (0.0, 1.0)
assert r1["model_version"] != r0["model_version"] == recs[0]["model_version"], "model_version must track calibration"
eng.reload()
assert eng.version == r1["model_version"], "reload() must pick up the new policy"
print("calibration hook OK")
msg = run(S + "07_freeze.py", expect_fail=True)        # demo bundle is now stale -> freeze refuses
assert "different model/policy" in msg
print("freeze staleness check OK")

# ---- image lookup falls back to extension-less match (png vs jpg copies of a dataset)
from chesttriage_ml.data import _lookup
s = _lookup(pd.Series(["a/b.png", "c.png", "zzz.png"]), {"a/b": "/x/a/b.jpg", "c.png": "/x/c.png"})
assert s.tolist()[:2] == ["/x/a/b.jpg", "/x/c.png"] and pd.isna(s.iloc[2])
print("lookup fallback OK")
print("\nALL EXTRA CHECKS PASSED")
