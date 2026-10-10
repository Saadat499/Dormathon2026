"""Phase 0 go/no-go: find the data, build subsets, label-overlap table, backbone smoke test.

Examples
  python scripts/00_check_setup.py --download                       # uses kagglehub
  python scripts/00_check_setup.py --nih_root /data/nih --chexpert_root /data/cx
  python scripts/00_check_setup.py --backbone fake --nih_root ... --chexpert_root ...   # dry run, no torch
"""
import _common  # noqa: F401  (sets sys.path)
import argparse
import time

import numpy as np

from chesttriage_ml.backbone import get_backbone, save_backbone_info
from chesttriage_ml.config import DEFAULT_URGENT, FINDINGS, art, load_dataset_paths, save_dataset_paths
from chesttriage_ml.data import LABEL_COLS, label_overlap_table, load_chexpert, load_nih
from chesttriage_ml.metrics import composite_label
from chesttriage_ml.preprocess import load_image, to_array
from _common import md_table, stamp, subset_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download both datasets with kagglehub")
    ap.add_argument("--nih_root"); ap.add_argument("--chexpert_root")
    ap.add_argument("--nih_n", type=int, default=30000, help="images in the NIH subset")
    ap.add_argument("--chexpert_n", type=int, default=15000, help="images in the CheXpert subset")
    ap.add_argument("--backbone", default="densenet121", choices=["densenet121", "resnet50", "resnet18", "densenet121-onnx", "fake"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blank_as", type=float, default=0.0, help="CheXpert blank label -> this value")
    a = ap.parse_args()

    if a.download:
        import kagglehub
        nih_root = kagglehub.dataset_download("khanfashee/nih-chest-x-ray-14-224x224-resized")
        cx_root = kagglehub.dataset_download("ashery/chexpert")
        save_dataset_paths(nih_root=nih_root, chexpert_root=cx_root)
    save_dataset_paths(nih_root=a.nih_root, chexpert_root=a.chexpert_root)
    P = load_dataset_paths()
    if "nih_root" not in P or "chexpert_root" not in P:
        raise SystemExit("Give --download or both --nih_root and --chexpert_root")
    print("NIH root     :", P["nih_root"]); print("CheXpert root:", P["chexpert_root"])

    try:
        nih = load_nih(P["nih_root"], a.nih_n, a.seed)
        cx = load_chexpert(P["chexpert_root"], a.chexpert_n, a.seed, a.blank_as)
    except Exception as e:
        raise SystemExit(f"GATE: FAIL - data not usable ({e}). Fix within the hour or switch to AquaTriage.")
    nih.to_csv(subset_csv("nih"), index=False); cx.to_csv(subset_csv("chexpert"), index=False)

    lines = ["# Setup report", "", f"Artifacts folder: `{art().name}`", ""]
    for name, df in (("NIH", nih), ("CheXpert", cx)):
        sz = []
        for p in df["path"].sample(min(10, len(df)), random_state=0):
            im = load_image(p); sz.append(f"{im.size[0]}x{im.size[1]} {im.mode}")
        lines += [f"## {name}: {len(df)} images, {df['patient_id'].nunique()} patients",
                  f"splits: {df['split'].value_counts().to_dict()}",
                  f"sample image sizes: {sorted(set(sz))}", ""]
    ov = label_overlap_table(nih, cx, DEFAULT_URGENT)
    stamp(ov).to_csv(art("reports", "label_overlap.csv"), index=False)
    ui = [FINDINGS.index(f) for f in DEFAULT_URGENT]
    lines += ["## Label overlap (shared findings)", md_table(ov), "",
              f"ANY_URGENT prevalence (provisional urgent set {DEFAULT_URGENT}): "
              f"NIH {np.nanmean(composite_label(nih[LABEL_COLS].values, ui)):.3f}, "
              f"CheXpert(external) {np.nanmean(composite_label(cx[cx['split']=='external'][LABEL_COLS].values, ui)):.3f}",
              "",
              "Notes: public-data prevalence is NOT hospital prevalence. The queue simulation must use an "
              "assumed prevalence as a parameter. CheXpert 'uncertain' labels are masked (ignored); blanks "
              f"are treated as {a.blank_as}. NIH labels are text-mined and noisy.", ""]

    print("Loading backbone:", a.backbone)
    try:
        bb = get_backbone(a.backbone)
        x = np.stack([to_array(load_image(p)) for p in nih["path"].iloc[:16]])
        t = time.time(); e = bb.embed_np(x); dt = time.time() - t
    except Exception as e:
        raise SystemExit(f"GATE: FAIL - backbone does not run ({e}). Try --backbone resnet18, else switch to AquaTriage.")
    ok = e.shape == (len(x), bb.dim) and np.isfinite(e).all()
    lines += ["## Backbone smoke test", f"{a.backbone}: embedding {e.shape}, finite={bool(np.isfinite(e).all())}, "
              f"{len(x) / dt:.1f} img/s (first call includes warm-up)", ""]
    save_backbone_info(art(), bb)
    (art("reports") / "setup_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("GATE:", "PASS - data reachable, backbone runs" if ok else "FAIL - check backbone output")
    print("Reminder: the Kaggle mirrors are for the hackathon prototype only. Check the NIH and CheXpert "
          "(Stanford AIMI research-use agreement) terms before showing or sharing anything built on them.")


if __name__ == "__main__":
    main()
