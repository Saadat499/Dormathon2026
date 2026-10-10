"""Compare two model folders on the SAME held-out NIH test patients (e.g. v1 vs v2, or v2 vs a fine-tuned v3).

  python scripts/09_compare_models.py --a artifacts_v1_imagenet_frozen --b artifacts_v2_imagenet_fullnih

Test set = folder B's NIH 'test' patients minus any patient that EITHER model used for training or
validation (so neither model has seen them, and neither chose settings on them). The script refuses
to run if that leaves nothing. Both heads are scored on identical images:
  - per-finding AUROC for A and B with patient-level bootstrap 95% CIs
  - the paired difference B - A with a paired bootstrap 95% CI (same resampled patients for both),
    which is the right test for "is B better"; overlapping marginal CIs are not.
Features: when both folders use the identical backbone file, B's feature cache is used for both heads.
Otherwise each folder's backbone is run on the test images (slower).
Writes reports/model_comparison.{csv,md} into folder B.
"""
import _common  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chesttriage_ml.backbone import load_backbone_from_artifacts
from chesttriage_ml.config import DEFAULT_URGENT, FINDINGS
from chesttriage_ml.contract import model_version
from chesttriage_ml.data import LABEL_COLS
from chesttriage_ml.features import embed_pils, load_features
from chesttriage_ml.headmodel import LinearHead
from chesttriage_ml.metrics import auroc, composite_label, sigmoid
from chesttriage_ml.preprocess import load_image
from _common import md_table


def bb_sha(folder: Path) -> str:
    info = json.loads((folder / "backbone.json").read_text())
    return hashlib.sha256((folder / info["weights"]).read_bytes()).hexdigest() if info.get("weights") else info["name"]


def tag(folder: Path) -> str:
    info = json.loads((folder / "backbone.json").read_text())
    return info.get("tag", info["name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    A, B = Path(a.a).resolve(), Path(a.b).resolve()
    sa, sb = pd.read_csv(A / "subsets/nih.csv"), pd.read_csv(B / "subsets/nih.csv")
    seen = set()
    for s in (sa, sb):
        seen |= set(s.patient_id[s.split.isin(["train", "val"])])
    test = sb[(sb.split == "test") & ~sb.patient_id.isin(seen)]
    n_drop = int(((sb.split == "test") & sb.patient_id.isin(seen)).sum())
    if test.empty:
        raise SystemExit("No NIH test patients are unseen by both models; cannot compare fairly.")
    print(f"common clean test: {len(test)} images, {test.patient_id.nunique()} patients "
          f"({n_drop} B-test images dropped because a model trained/validated on those patients)")

    if bb_sha(A) == bb_sha(B):
        Xb, db = load_features(B / "features" / f"nih_{tag(B)}", sb)
        keep = db.study_id.isin(set(test.study_id)).values
        XA = XB = Xb[keep]; d = db[keep].reset_index(drop=True)
        print("identical backbone file in both folders -> one feature set for both heads")
    else:
        pils = [load_image(p) for p in test.path]
        XA = embed_pils(load_backbone_from_artifacts(A), pils)
        XB = embed_pils(load_backbone_from_artifacts(B), pils)
        d = test.reset_index(drop=True)
    Y = d[LABEL_COLS].values.astype(float)
    PA = sigmoid(LinearHead.load(A / "head.npz").logits(XA))
    PB = sigmoid(LinearHead.load(B / "head.npz").logits(XB))
    ui = [FINDINGS.index(f) for f in DEFAULT_URGENT]
    scopes = [(f, Y[:, k], PA[:, k], PB[:, k]) for k, f in enumerate(FINDINGS)]
    scopes.append(("ANY_URGENT", composite_label(Y, ui), PA[:, ui].max(1), PB[:, ui].max(1)))

    # patient-level paired bootstrap: identical resampled patients for A and B
    g = d.patient_id.values
    order = np.argsort(g, kind="stable"); gs = g[order]
    groups = np.split(order, np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])[1:])
    rng = np.random.default_rng(a.seed)
    boots = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(a.n_boot)]
    rows = []
    for name, y, pa, pb in scopes:
        va, vb = auroc(y, pa), auroc(y, pb)
        ba, bb_ = np.array([auroc(y[i], pa[i]) for i in boots]), np.array([auroc(y[i], pb[i]) for i in boots])
        dd = bb_ - ba
        lo_d, hi_d = np.nanpercentile(dd, [2.5, 97.5])
        rows.append({"finding": name, "n_pos": int(np.nansum(y == 1)), "auroc_A": va,
                     "A_lo": np.nanpercentile(ba, 2.5), "A_hi": np.nanpercentile(ba, 97.5), "auroc_B": vb,
                     "B_lo": np.nanpercentile(bb_, 2.5), "B_hi": np.nanpercentile(bb_, 97.5),
                     "diff_B_minus_A": vb - va, "diff_lo": lo_d, "diff_hi": hi_d,
                     "verdict": "B better" if lo_d > 0 else ("A better" if hi_d < 0 else "no clear difference")})
    df = pd.DataFrame(rows)
    df.insert(0, "B_folder", B.name); df.insert(0, "A_folder", A.name)
    df.to_csv(B / "reports" / "model_comparison.csv", index=False)
    md = ["# Model comparison on common clean NIH test patients", "",
          f"A = `{A.name}` ({model_version(A)}), B = `{B.name}` ({model_version(B)})",
          f"Test: {len(d)} images, {len(groups)} patients, unseen by both models in training and validation. "
          f"Patient-level bootstrap, {a.n_boot} resamples; diff CI is paired.", "",
          md_table(df.drop(columns=["A_folder", "B_folder"]))]
    (B / "reports" / "model_comparison.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
