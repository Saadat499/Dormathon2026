"""Train the multi-label linear head on NIH features; report internal (same-dataset) metrics."""
import _common  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from chesttriage_ml.config import FINDINGS, art
from chesttriage_ml.headmodel import LinearHead
from chesttriage_ml.metrics import (auprc, auroc, bootstrap_ci, composite_label, composite_score, sigmoid)
from _common import backbone_name, get_policy, load_xy, md_table, stamp, stamp_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    ap.add_argument("--n_boot", type=int, default=200)
    a = ap.parse_args()

    X, Y, d = load_xy("nih", backbone_name())
    s = d["split"].values
    tr, va, te = s == "train", s == "val", s == "test"
    print(f"train {tr.sum()}  val {va.sum()}  test {te.sum()}")
    head = LinearHead(FINDINGS).fit(X[tr], Y[tr], X[va], Y[va], Cs=a.Cs)
    head.save(art("head.npz"))

    P = sigmoid(head.logits(X))
    ui = get_policy().urgent_idx()                      # follows policy.json after the clinician review
    rows = []
    groups = d["patient_id"].values[te]
    scopes = [(f, Y[te, k], P[te, k]) for k, f in enumerate(FINDINGS)]
    scopes.append(("ANY_URGENT", composite_label(Y[te], ui), composite_score(P[te], ui)))
    for name, y, p in scopes:
        lo, hi = bootstrap_ci(auroc, y, p, groups, a.n_boot)
        rows.append({"finding": name, "n_pos_test": int(np.nansum(y == 1)), "prevalence": float(np.nanmean(y)),
                     "auroc": auroc(y, p), "auroc_lo": lo, "auroc_hi": hi, "auprc": auprc(y, p),
                     "best_C": head.best_C.get(name), "val_auroc": head.val_auroc.get(name)})
    df = pd.DataFrame(rows)
    stamp(df).to_csv(art("reports", "internal_metrics.csv"), index=False)
    print(stamp_line())
    print(md_table(df))
    macro = df[df.finding != "ANY_URGENT"].auroc.mean()
    print(f"\nmean per-finding AUROC (NIH test): {macro:.3f}   head saved to {art('head.npz')}")
    print("Note: sample sizes for rare findings are small, so read the CIs, not just the point estimate.")


if __name__ == "__main__":
    main()
