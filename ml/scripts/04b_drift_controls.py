"""Controls for the site drift score: is it measuring the site, or just the file format?

Every comparison uses DIFFERENT NIH test images in the two groups (no pairing), so the classifier
cannot win by matching an image to its own re-encoded copy:
  baseline     NIH originals vs other NIH originals                    -> should be ~0.5
  jpeg         NIH originals vs other NIH images re-saved as JPEG q75
  resize_jpeg  NIH originals vs other NIH images resized to 320x320 then JPEG q75 (CheXpert-small-like storage)
  chexpert     NIH originals vs CheXpert external images                 (the real site difference)
If jpeg / resize_jpeg score well above the baseline, the drift score reacts to file processing alone,
and a high site score cannot be read as "a different patient population or scanner".
Writes reports/drift_controls.{csv,md}.
"""
import _common  # noqa: F401
import argparse
import io

import numpy as np
import pandas as pd
from PIL import Image

from chesttriage_ml.backbone import load_backbone_from_artifacts
from chesttriage_ml.config import art
from chesttriage_ml.data import LABEL_COLS
from chesttriage_ml.drift import domain_auroc
from chesttriage_ml.headmodel import LinearHead
from chesttriage_ml.metrics import auroc, composite_label
from chesttriage_ml.features import embed_pils
from chesttriage_ml.preprocess import load_image, to_gray
from _common import get_policy, get_subset, md_table, stamp, stamp_line


def jpeg(img, size=None, q=75):
    g = to_gray(img)
    if size:
        g = g.resize((size, size), Image.Resampling.BILINEAR)
    buf = io.BytesIO(); g.save(buf, "JPEG", quality=q); buf.seek(0)
    out = Image.open(buf); out.load()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="images per group")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=3000, help="labelled NIH test images for the prediction-effect check")
    a = ap.parse_args()
    bb = load_backbone_from_artifacts(art())
    nih, cx = get_subset("nih"), get_subset("chexpert")
    test = nih[nih.split == "test"].path.values
    cxe = cx[cx.split == "external"].path.values
    rng = np.random.default_rng(a.seed)
    rows = []
    for r in range(a.repeats):
        pick = rng.choice(len(test), 4 * a.n, replace=False)            # four disjoint groups of NIH images
        g = [test[pick[i * a.n:(i + 1) * a.n]] for i in range(4)]
        ref = embed_pils(bb, [load_image(p) for p in g[0]])
        arms = {"baseline": [load_image(p) for p in g[1]],
                "jpeg": [jpeg(load_image(p)) for p in g[2]],
                "resize_jpeg": [jpeg(load_image(p), 320) for p in g[3]],
                "chexpert": [load_image(p) for p in cxe[rng.choice(len(cxe), a.n, replace=False)]]}
        for arm, ims in arms.items():
            rows.append({"repeat": r, "comparison": arm, "n_per_group": a.n,
                         "domain_auroc": domain_auroc(ref, embed_pils(bb, ims), seed=r)})
        print(f"repeat {r + 1}/{a.repeats}: " + ", ".join(f"{x['comparison']} {x['domain_auroc']:.3f}" for x in rows[-4:]))
    df = pd.DataFrame(rows)
    stamp(df).to_csv(art("reports", "drift_controls.csv"), index=False)
    s = df.groupby("comparison", sort=False).domain_auroc.agg(["mean", "std", "min", "max"]).reset_index()
    base, fmt = s.set_index("comparison")["mean"]["baseline"], s.set_index("comparison")["mean"][["jpeg", "resize_jpeg"]].max()
    if fmt >= max(0.7, base + 0.15):
        verdict = (f"The drift score DOES react to file processing alone: re-encoding NIH images (no change of patients or "
                   f"scanner) scores up to {fmt:.2f} vs {base:.2f} for NIH vs NIH. A high site score therefore cannot be "
                   "read as a different population or scanner; it may only mean different image processing. Use it as "
                   "'these images differ from training, check them', compared against these control values.")
    else:
        verdict = (f"Re-encoding alone barely moves the drift score ({fmt:.2f} vs {base:.2f} baseline), so a high site "
                   "score reflects more than file format. It still says nothing about accuracy at that site.")
    # does the format change the PREDICTIONS? (separates "format confounds drift" from "format hurts accuracy")
    head, pol = LinearHead.load(art("head.npz")), get_policy()
    ev = nih[nih.split == "test"].sample(min(a.n_eval, int((nih.split == "test").sum())), random_state=a.seed)
    yu = composite_label(ev[LABEL_COLS].values.astype(float), pol.urgent_idx())
    eff = []
    for arm, f in (("original", lambda im: im), ("jpeg", jpeg), ("resize_jpeg", lambda im: jpeg(im, 320))):
        sc = pol.score(pol.probs(head.logits(embed_pils(bb, [f(load_image(p)) for p in ev.path]))))
        eff.append({"images": arm, "n": len(ev), "any_urgent_auroc": auroc(yu, sc), "mean_score": float(sc.mean())})
    eff = pd.DataFrame(eff)
    stamp(eff).to_csv(art("reports", "format_effect_on_predictions.csv"), index=False)
    md = ["# Drift score controls", "", stamp_line(), "",
          f"Domain AUROC (cross-validated logistic regression on backbone features), {a.n} vs {a.n} images, "
          f"{a.repeats} repeats, different images in each group.", "", md_table(s, 3), "", "**Verdict:** " + verdict, "",
          "## Effect of the same re-encoding on the model's predictions (NIH test, any urgent finding)", "",
          md_table(eff, 3), "",
          "A drift score near 1.0 for re-encoded images, with only a small AUROC change, means the classifier detects "
          "compression artefacts easily even when they barely affect predictions. Part of the NIH -> CheXpert drop "
          "may still come from format (CheXpert is JPEG), so attribute that drop to 'a different hospital's dataset', "
          "not to scanners or patients alone."]
    (art("reports") / "drift_controls.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
