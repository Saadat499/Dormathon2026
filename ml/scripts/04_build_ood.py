"""Fit the OOD detector and measure what it catches and what it misses."""
import _common  # noqa: F401
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from chesttriage_ml.backbone import load_backbone_from_artifacts
from chesttriage_ml.config import art
from chesttriage_ml.drift import drift_summary
from chesttriage_ml.features import embed_pils
from chesttriage_ml.ood import OODDetector, make_variants, synthetic_non_chest
from chesttriage_ml.preprocess import load_image
from _common import backbone_name, load_xy, md_table, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_neg_images", type=int, default=400, help="train images turned into mistake variants")
    ap.add_argument("--n_eval_images", type=int, default=150, help="held-out images for variant evaluation")
    ap.add_argument("--n_synth", type=int, default=150)
    ap.add_argument("--percentile", type=float, default=99.5)
    ap.add_argument("--nonchest_dir", help="optional folder of real non-chest images (e.g. hand X-ray, photos)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    bb = load_backbone_from_artifacts(art())
    Xn, _, dn = load_xy("nih", backbone_name())
    Xc, _, dc = load_xy("chexpert", backbone_name())
    sp = dn.split.values
    tr, va, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "val"), np.flatnonzero(sp == "test")
    rng = np.random.default_rng(a.seed)

    def variants_for(idx):
        out = {}
        for i in idx:
            for k, im in make_variants(load_image(dn.path[i]), seed=int(i)).items():
                out.setdefault(k, []).append(im)
        return out

    print("building negatives from training images ...")
    vt = variants_for(rng.choice(tr, min(a.n_neg_images, len(tr)), replace=False))
    neg = [embed_pils(bb, v) for v in vt.values()]
    neg.append(embed_pils(bb, [im for _, im in synthetic_non_chest(a.n_synth, a.seed)]))
    user_test = []
    if a.nonchest_dir:
        files = sorted(p for p in Path(a.nonchest_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        ims = [load_image(p) for p in files]
        half = len(ims) // 2
        if half:
            neg.append(embed_pils(bb, ims[:half]))
            user_test = ims[half:]
    det = OODDetector(percentile=a.percentile).fit(Xn[tr], np.vstack(neg), a.seed).set_threshold(Xn[va])

    def row(cat, role, X):
        o = det.predict(X)
        return {"category": cat, "role": role, "n": len(X), "flagged_rate": float(o["flag"].mean()),
                "flag_maha": float(o["flag_maha"].mean()), "flag_probe": float(o["flag_probe"].mean()),
                "median_maha": float(np.median(o["maha"]))}

    rows = [row("NIH test (real chest X-rays)", "should NOT be flagged", Xn[te]),
            row("CheXpert external (new dataset)", "drift signal, not an error", Xc[(dc.split == "external").values])]
    ev = variants_for(rng.choice(va, min(a.n_eval_images, len(va)), replace=False))
    for k, v in ev.items():
        rows.append(row(f"held-out {k}", "should be flagged", embed_pils(bb, v)))
    rows.append(row("synthetic non-chest (held-out)", "should be flagged",
                    embed_pils(bb, [im for _, im in synthetic_non_chest(100, a.seed + 1)])))
    if user_test:
        rows.append(row("your non-chest images (held-out half)", "should be flagged", embed_pils(bb, user_test)))
    rep = pd.DataFrame(rows)
    stamp(rep).to_csv(art("reports", "ood_report.csv"), index=False)
    joblib.dump(det, art("ood.joblib"))
    print(md_table(rep))
    miss = rep[(rep.role == "should be flagged") & (rep.flagged_rate < 0.5)]
    print("\nCaught (>=90% flagged):", ", ".join(rep[(rep.role == 'should be flagged') & (rep.flagged_rate >= 0.9)].category) or "none")
    print("Misses (<50% flagged):", ", ".join(miss.category) or "none", " <- show one of these honestly in the demo")
    print("saved ood.joblib and reports/ood_report.csv")

    # ---- site drift score (Site Readiness): reference = NIH validation features, shipped with the model
    ref = Xn[rng.choice(va, min(2000, len(va)), replace=False)]
    np.save(art("drift_reference.npy"), ref.astype(np.float16))
    ex = np.flatnonzero((dc.split == "external").values)
    view = dn.view.astype(str).values
    samples = [("NIH test (same source)", "reference value, expect ~0.5", Xn[te]),
               ("NIH test, AP views only (pseudo-site)", "same source, different mix", Xn[te[view[te] == "AP"]]),
               ("NIH test, PA views only (pseudo-site)", "same source, different mix", Xn[te[view[te] == "PA"]]),
               ("CheXpert external (other hospital)", "should be high", Xc[ex])]
    drows = []
    for cat, role, X in samples:
        for n in (50, 200):
            if len(X) >= n:
                pick = X[rng.choice(len(X), n, replace=False)]
                d = drift_summary(ref.astype(np.float32), pick, det)
                drows.append({"sample": cat, "role": role, **d})
    drep = pd.DataFrame(drows)
    stamp(drep).to_csv(art("reports", "drift_report.csv"), index=False)
    print("\nSite drift score (domain AUROC vs NIH validation; 0.5 = indistinguishable):")
    print(md_table(drep))
    print("saved drift_reference.npy and reports/drift_report.csv")


if __name__ == "__main__":
    main()
