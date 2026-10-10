"""Cross-dataset test: train on NIH, test on CheXpert with thresholds fixed on NIH validation data.

This is the evidence that a model does not automatically transfer to a new site. It also
motivates the Site Readiness check. CheXpert 'uncertain' labels are masked in all metrics.
"""
import _common  # noqa: F401
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chesttriage_ml.config import FINDINGS, art
from chesttriage_ml.headmodel import LinearHead
from chesttriage_ml.metrics import (auroc, bootstrap_ci, composite_label, composite_score,
                                    operating_point, threshold_for_sensitivity)
from _common import backbone_name, get_policy, load_xy, md_table, stamp, stamp_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_sens", type=float, default=0.90)
    ap.add_argument("--n_boot", type=int, default=200)
    a = ap.parse_args()

    bb, pol = backbone_name(), get_policy()
    head = LinearHead.load(art("head.npz"))
    Xn, Yn, dn = load_xy("nih", bb)
    Xc, Yc, dc = load_xy("chexpert", bb)
    Pn, Pc = pol.probs(head.logits(Xn)), pol.probs(head.logits(Xc))
    va, te = (dn.split == "val").values, (dn.split == "test").values
    ex, gv = (dc.split == "external").values, (dc.split == "gold_valid").values
    ui = pol.urgent_idx()

    scopes = [(f, Yn[:, k], Pn[:, k], Yc[:, k], Pc[:, k]) for k, f in enumerate(FINDINGS)]
    scopes.append(("ANY_URGENT", composite_label(Yn, ui), composite_score(Pn, ui),
                   composite_label(Yc, ui), composite_score(Pc, ui)))
    rows = []
    for name, yn, pn, yc, pc in scopes:
        thr = threshold_for_sensitivity(yn[va], pn[va], a.target_sens)
        lo_n, hi_n = bootstrap_ci(auroc, yn[te], pn[te], dn.patient_id.values[te], a.n_boot)
        lo_c, hi_c = bootstrap_ci(auroc, yc[ex], pc[ex], dc.patient_id.values[ex], a.n_boot)
        on, oc = operating_point(yn[te], pn[te], thr), operating_point(yc[ex], pc[ex], thr)
        rows.append({
            "finding": name, "auroc_nih": auroc(yn[te], pn[te]), "nih_lo": lo_n, "nih_hi": hi_n,
            "auroc_chexpert": auroc(yc[ex], pc[ex]), "cx_lo": lo_c, "cx_hi": hi_c,
            "auroc_drop": auroc(yc[ex], pc[ex]) - auroc(yn[te], pn[te]),
            "thr_at_target": thr, "sens_nih": on["sensitivity"], "sens_chexpert": oc["sensitivity"],
            "sens_drop": oc["sensitivity"] - on["sensitivity"], "false_alarm_nih": on["false_alarm_rate"],
            "false_alarm_chexpert": oc["false_alarm_rate"], "ppv_nih": on["ppv"], "ppv_chexpert": oc["ppv"],
            "n_pos_chexpert": oc["n_pos"], "auroc_chexpert_gold": auroc(yc[gv], pc[gv]) if gv.sum() else np.nan,
            "n_gold": int(gv.sum()),
        })
    df = pd.DataFrame(rows)
    stamp(df).to_csv(art("reports", "cross_dataset.csv"), index=False)
    md = ["# Cross-dataset results (train NIH -> test CheXpert)", "", stamp_line(), "",
          f"Thresholds fixed on NIH validation at {a.target_sens:.0%} sensitivity, then applied unchanged to CheXpert.",
          "PPV depends on dataset prevalence; recompute for the assumed hospital prevalence.", "",
          md_table(df[["finding", "auroc_nih", "auroc_chexpert", "auroc_drop", "sens_nih", "sens_chexpert",
                       "sens_drop", "false_alarm_nih", "false_alarm_chexpert", "n_pos_chexpert"]])]
    (art("reports") / "cross_dataset.md").write_text("\n".join(md))
    print("\n".join(md))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(df)); w = 0.38
    err = lambda lo, hi, v: np.vstack([np.nan_to_num(v - lo), np.nan_to_num(hi - v)])
    ax[0].bar(x - w / 2, df.auroc_nih, w, yerr=err(df.nih_lo, df.nih_hi, df.auroc_nih), label="NIH test")
    ax[0].bar(x + w / 2, df.auroc_chexpert, w, yerr=err(df.cx_lo, df.cx_hi, df.auroc_chexpert), label="CheXpert")
    ax[0].set_title("AUROC: same dataset vs new dataset"); ax[0].set_ylim(0.4, 1)
    ax[1].bar(x - w / 2, df.sens_nih, w, label="NIH test"); ax[1].bar(x + w / 2, df.sens_chexpert, w, label="CheXpert")
    ax[1].axhline(a.target_sens, ls="--", c="k", lw=1); ax[1].set_title(f"Sensitivity at threshold fixed on NIH ({a.target_sens:.0%})")
    for z in ax:
        z.set_xticks(x); z.set_xticklabels(df.finding, rotation=40, ha="right"); z.legend()
    fig.tight_layout(); fig.savefig(art("reports", "cross_dataset.png"), dpi=150)
    print("saved reports/cross_dataset.{csv,md,png}")


if __name__ == "__main__":
    main()
