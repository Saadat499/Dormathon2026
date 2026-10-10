"""Generate saliency heatmaps, pick 'Where it fails' examples and build the demo bundle.

Uses the live engine, so this also confirms the engine reproduces the exported predictions.

Two different outputs, on purpose:
  reports/where_it_fails.csv   failure ANALYSIS: decisions from the score only (OOD ignored), so a
                               miss is a model miss. The real ood_flag is reported alongside.
  predictions/demo_contract.jsonl  what the APP shows: the real pipeline, OOD rule applied. It holds a
                               realistic queue sample (NIH test + some CheXpert), the failure examples
                               and OOD demo images. heatmap_path and image_path are relative to the
                               artifacts folder, and the images are copied there, so it deploys as is.
Also: artifacts/heatmaps/*.png, artifacts/demo_images/*, predictions/demo_manifest.csv (labels).
"""
import _common  # noqa: F401
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from chesttriage_ml.config import FINDINGS, IMG_SIZE, art
from chesttriage_ml.engine import ChestTriageEngine
from chesttriage_ml.ood import make_variants, synthetic_non_chest
from chesttriage_ml.preprocess import load_image
from _common import get_subset, md_table, stamp, stamp_line


def pick(df, cond, asc, n):
    return df[cond].sort_values("composite_score", ascending=asc).head(n)


def rel(p):
    return None if p is None else Path(p).resolve().relative_to(art()).as_posix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="examples per failure category")
    ap.add_argument("--n_queue", type=int, default=30, help="random NIH test studies for the demo queue")
    ap.add_argument("--n_external", type=int, default=10, help="random CheXpert studies for the demo queue")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    eng = ChestTriageEngine(art())
    nih_s, cx_s = get_subset("nih"), get_subset("chexpert")
    paths = {**dict(zip(nih_s.study_id, nih_s.path)), **dict(zip(cx_s.study_id, cx_s.path))}
    nih = pd.read_csv(art("predictions", "nih_full.csv"))
    cx = pd.read_csv(art("predictions", "chexpert_full.csv"))
    nt = nih[(nih.split == "test") & nih.label_any_urgent.notna()]
    cxe = cx[(cx.split == "external") & cx.label_any_urgent.notna()]
    heat = art("heatmaps")

    # ---- failure analysis (score only)
    sel = []
    for tag, df in (("nih", nt), ("chexpert", cxe)):
        y, pr = df.label_any_urgent, df.priority_score_only
        sel += [(f"{tag}_confident_hit", r) for _, r in pick(df, (y == 1) & (pr == "Raised"), False, a.n).iterrows()]
        sel += [(f"{tag}_missed_urgent", r) for _, r in pick(df, (y == 1) & (pr == "Normal"), True, a.n).iterrows()]
        sel += [(f"{tag}_false_alarm", r) for _, r in pick(df, (y == 0) & (pr == "Raised"), False, a.n).iterrows()]
    ids = [r.study_id for _, r in sel]
    recs = eng.predict_batch([paths[i] for i in ids], ids, heatmap_dir=heat, ignore_ood=True)
    rows = []
    for (cat, r), rec in zip(sel, recs):
        rows.append({"category": cat, "study_id": r.study_id, "dataset": r.dataset, "label_any_urgent": r.label_any_urgent,
                     "true_findings": ",".join(f for f in FINDINGS if r[f"y_{f}"] == 1),
                     "priority_score_only": rec["priority"], "composite_score": rec["composite_score"],
                     "top_urgent_finding": rec["top_urgent_finding"], "ood_flag": bool(r.ood_flag),
                     "heatmap_path": rel(rec["heatmap_path"]), "image_path": paths[r.study_id]})

    # ---- OOD demo images: realistic mistakes + synthetic non-X-rays
    demo_img = art("demo_images")
    ood_items = []
    for sid in nt.study_id.iloc[:3]:
        v = make_variants(load_image(paths[sid]))
        ood_items += [(f"{sid}__{k}", v[k]) for k in ("rotated_90", "inverted", "heavy_noise", "heavy_blur")]
    ood_items += [(f"synthetic_{n}_{i}", im) for i, (n, im) in enumerate(synthetic_non_chest(5, 123))]
    for name, im in ood_items:
        im.save(demo_img / f"{name}.png")

    # ---- demo bundle: realistic queue + failure examples + OOD demos, real pipeline (OOD rule applied)
    rng = np.random.default_rng(a.seed)
    queue = [("queue_nih", s) for s in rng.permutation(nt.study_id.values)[:a.n_queue]]
    queue += [("queue_chexpert", s) for s in rng.permutation(cxe.study_id.values)[:a.n_external]]
    demo, seen = [], set()
    for cat, sid in queue + [(c, r.study_id) for c, r in sel]:
        if sid not in seen:
            seen.add(sid)
            src = Path(paths[sid])
            dst = demo_img / f"{sid}{src.suffix.lower()}"
            shutil.copyfile(src, dst)
            demo.append((cat, sid, dst))
    demo += [("ood_demo", n, demo_img / f"{n}.png") for n, _ in ood_items]
    drecs = eng.predict_batch([p for _, _, p in demo], [s for _, s, _ in demo], heatmap_dir=heat)
    lab = pd.concat([nih, cx]).set_index("study_id")
    man = []
    with open(art("predictions", "demo_contract.jsonl"), "w") as fh:
        for (cat, sid, p), rec in zip(demo, drecs):
            rec["heatmap_path"], rec["image_path"] = rel(rec["heatmap_path"]), rel(p)
            fh.write(json.dumps(rec) + "\n")
            known = sid in lab.index
            man.append({"study_id": sid, "category": cat, "dataset": lab.dataset[sid] if known else "demo",
                        "label_any_urgent": lab.label_any_urgent[sid] if known else np.nan,
                        "true_findings": ",".join(f for f in FINDINGS if known and lab[f"y_{f}"][sid] == 1),
                        "priority": rec["priority"], "ood_flag": rec["ood_flag"]})
    man = pd.DataFrame(man)
    man.to_csv(art("predictions", "demo_manifest.csv"), index=False)

    od = man[man.category == "ood_demo"]
    rows += [{"category": "ood_demo", "study_id": r.study_id, "dataset": "demo", "label_any_urgent": np.nan,
              "true_findings": "", "priority_score_only": None, "composite_score": None, "top_urgent_finding": None,
              "ood_flag": bool(r.ood_flag), "heatmap_path": None, "image_path": f"demo_images/{r.study_id}.png"}
             for r in od.itertuples()]
    out = pd.DataFrame(rows)
    stamp(out).to_csv(art("reports", "where_it_fails.csv"), index=False)
    heatmap_sheet(out[out.heatmap_path.notna()])
    attention_report(eng, out[out.heatmap_path.notna()])

    print(out.groupby("category").size())
    print(f"\nDemo bundle: {len(man)} studies, priorities {man.priority.value_counts().to_dict()}")
    print(f"OOD demo: {int(od.ood_flag.sum())} of {len(od)} flagged. Misses (use one honestly in the demo):")
    miss = od[~od.ood_flag.astype(bool)]
    print(miss[["study_id", "priority"]].to_string(index=False) if len(miss) else "none")
    print("\nPick ONE missed_urgent and ONE false_alarm for the 'Where it fails' slide.")


def region_masks(n: int = IMG_SIZE) -> dict:
    """Regions where saliency usually means a shortcut (labels, markers, collimation edges), not anatomy."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    edge = np.minimum.reduce([yy, xx, 1 - yy, 1 - xx])
    c = 0.15
    return {"border_10pct": edge < 0.10,
            "corners_15pct": ((yy < c) | (yy > 1 - c)) & ((xx < c) | (xx > 1 - c)),
            "top_band_12pct": yy < 0.12}


def attention_report(eng, df):
    """Share of each saliency map's mass in border / corner / top-band regions, against the share a
    uniform map would put there (= region area). These are PROXIES: a map can avoid the border and still
    sit on a tube or the heart. They flag maps to look at; they do not show the maps are right."""
    masks = region_masks()
    rows = []
    for r in df.itertuples():
        cam, f = eng.cam(r.image_path, r.top_urgent_finding)
        tot = cam.sum() + 1e-9
        row = {"category": r.category, "study_id": r.study_id, "finding": f}
        for k, m in masks.items():
            row[k] = float(cam[m].sum() / tot)
        rows.append(row)
    att = pd.DataFrame(rows)
    stamp(att).to_csv(art("reports", "heatmap_attention.csv"), index=False)
    area = {k: float(m.mean()) for k, m in masks.items()}
    summ = att.drop(columns=["study_id", "finding"]).groupby("category").mean()
    summ.loc["uniform map (= area share)"] = area
    md = ["# Heatmap attention proxies", "", stamp_line(), "",
          "Mean share of saliency mass in each region. Above the area share = the map is drawn to that region. "
          "Proxies only: check reports/heatmap_sheet.png by eye.", "", md_table(summ.reset_index(), 3)]
    (art("reports") / "heatmap_attention.md").write_text("\n".join(md))
    print("\n".join(md))


def heatmap_sheet(df, cols: int = 4):
    """One sheet of the failure-analysis heatmaps for a by-eye audit: is the hot area on the lungs,
    or on text, borders, tubes or the shoulder? Saliency shows where evidence came from, not that
    the model 'looked at the right thing'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    if df.empty:
        return
    rows = (len(df) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, r in zip(np.atleast_1d(axes).ravel(), df.itertuples()):
        ax.imshow(Image.open(art() / r.heatmap_path))
        ax.set_title(f"{r.category}\nlabel: {r.true_findings or 'no urgent finding'} | map: {r.top_urgent_finding} "
                     f"{r.composite_score:.2f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(art("reports", "heatmap_sheet.png"), dpi=110)
    plt.close(fig)
    print("saved reports/heatmap_sheet.png (check by eye: lungs vs text/borders/devices)")


if __name__ == "__main__":
    main()
