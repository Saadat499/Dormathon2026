"""Fit the provisional policy on NIH validation data and export predictions for B, C and D.

Outputs (artifacts/predictions/):
  <dataset>_full.csv      labels, logits, probabilities, composite, ood, priority (for Member B)
  <dataset>_contract.jsonl one data-contract record per study (for Member C / D)
  artifacts/policy.json    thresholds, urgent list, temperatures (Member B edits / refits)
  reports/budget_table.csv  capacity budget -> sensitivity / PPV on NIH test and CheXpert (slider, Queue Race)

Default policy: Raised = top 20% of scans by score (thresholds from NIH validation quantiles).
Change with --raise_budget 0.15 / --unsure_budget / --mode sensitivity.

If Member B provides artifacts/calibration.json ({"temperature": {"effusion": 1.3, ...}}),
re-run with --refit_policy so thresholds are refit on calibrated probabilities.
"""
import _common  # noqa: F401
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from chesttriage_ml.config import FINDINGS, art
from chesttriage_ml.contract import make_record, model_version
from chesttriage_ml.data import LABEL_COLS
from chesttriage_ml.headmodel import LinearHead
from chesttriage_ml.metrics import composite_label, composite_score, operating_point
from chesttriage_ml.policy import Policy
from _common import backbone_name, get_policy, load_xy, md_table, stamp, stamp_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit_policy", action="store_true")
    ap.add_argument("--mode", choices=["budget", "sensitivity"], help="threshold rule (changes policy.json)")
    ap.add_argument("--raise_budget", type=float, help="share of scans in the Raised lane, e.g. 0.20")
    ap.add_argument("--unsure_budget", type=float, help="share in the AI unsure band below it, e.g. 0.10")
    ap.add_argument("--datasets", nargs="+", default=["nih", "chexpert"])
    a = ap.parse_args()

    bb = backbone_name()
    head = LinearHead.load(art("head.npz"))
    head_sha = hashlib.sha256(art("head.npz").read_bytes()).hexdigest()[:12]
    pol = get_policy()
    cal = art("calibration.json")
    pf = art("policy.json")
    refit = a.refit_policy or not pf.exists()
    if pf.exists() and pol.fitted.get("head") != head_sha:
        print("policy.json was not fitted on this head.npz -> refitting thresholds")
        refit = True
    for k in ("mode", "raise_budget", "unsure_budget"):
        if getattr(a, k) is not None and getattr(a, k) != getattr(pol, k):
            setattr(pol, k, getattr(a, k)); refit = True
    if cal.exists():
        temps = json.loads(cal.read_text()).get("temperature", {})
        if temps != pol.temperature:
            print("calibration.json differs from policy.json -> refitting thresholds on calibrated probabilities")
            pol.temperature, refit = temps, True
    ui = pol.urgent_idx()
    if refit:
        Xn, Yn, dn = load_xy("nih", bb)
        va = (dn.split == "val").values
        Pn = pol.probs(head.logits(Xn))
        pol.fit_thresholds(composite_label(Yn[va], ui), composite_score(Pn[va], ui))
        pol.fitted["head"] = head_sha
        pol.save(pf)
        f = pol.fitted
        print(f"policy ({pol.mode}) fitted on NIH validation: t_raise={pol.t_raise:.4f} -> Raised {f['raised_share']:.0%} "
              f"of scans, sensitivity {f['raised_sensitivity']:.0%}, PPV {f['raised_ppv']:.0%}; t_unsure={pol.t_unsure:.4f} -> "
              f"Raised+unsure {f['raised_or_unsure_share']:.0%} of scans, sensitivity {f['raised_or_unsure_sensitivity']:.0%}")
    for w in pol.warnings():
        print("WARNING:", w)
    ood = joblib.load(art("ood.joblib")) if art("ood.joblib").exists() else None
    if ood is None:
        print("WARNING: no ood.joblib found (run 04_build_ood.py); ood_flag will be False everywhere")
    version = model_version(art())
    print("model_version:", version)

    for name in a.datasets:
        X, Y, d = load_xy(name, bb)
        L = head.logits(X); P = pol.probs(L)
        o = ood.predict(X) if ood is not None else None
        flag = o["flag"] if o is not None else np.zeros(len(X), bool)
        dec = pol.decide(P, flag)
        dec_score = pol.decide(P, None)
        full = d[["study_id", "dataset", "split", "patient_id", "age", "sex", "view"] + LABEL_COLS].copy()
        for k, f in enumerate(FINDINGS):
            full[f"logit_{f}"] = L[:, k]; full[f"p_{f}"] = P[:, k]
        full["label_any_urgent"] = composite_label(Y, ui)
        full["composite_score"] = dec["composite_score"]
        full["ood_flag"] = flag
        full["ood_score"] = o["maha"] if o is not None else np.nan
        full["priority"], full["confidence_band"] = dec["priority"], dec["confidence_band"]
        full["priority_score_only"] = dec_score["priority"]      # ignores the OOD flag (for analysis)
        full["model_version"] = version
        full.to_csv(art("predictions", f"{name}_full.csv"), index=False)
        with open(art("predictions", f"{name}_contract.jsonl"), "w") as fh:
            for i in range(len(d)):
                rec = make_record(d.study_id[i], pol, P[i], dec["priority"][i], dec["confidence_band"][i],
                                  dec["composite_score"][i], flag[i], None if o is None else o["maha"][i],
                                  version=version)
                fh.write(json.dumps(rec) + "\n")
        print(f"\n[{name}] priority counts by split:\n{full.groupby('split').priority.value_counts().unstack(fill_value=0)}")
        for sp in ("test", "external"):
            m = (full.split == sp).values & ~np.isnan(full.label_any_urgent.values)
            if m.sum():
                op = operating_point(full.label_any_urgent[m], full.composite_score[m], pol.t_raise)
                print(f"[{name}/{sp}] at 'Raised' threshold: sensitivity {op['sensitivity']:.3f}, "
                      f"false-alarm rate {op['false_alarm_rate']:.3f}, flag rate {op['flag_rate']:.3f} "
                      f"(n={op['n']}, urgent n={op['n_pos']})")
    budget_table(head, pol, bb)
    print("\nexported to", art("predictions"))


def budget_table(head, pol, bb):
    """Capacity budget -> what the Raised lane captures, for NIH test and CheXpert external.

    mode fixed_cutoff: score cut-off = NIH VALIDATION quantile (never test), applied unchanged, so the
        realised flag rate drifts at a new site (CheXpert scores run higher).
    mode queue_rank:   top X% of the queue by rank (Policy.queue_rank); here the whole held-out set is
        treated as one queue, so the flag rate equals the budget by construction.
    ceiling_sensitivity = flag rate / prevalence, capped at 1: the best ANY ranker could do at that budget.
    random_sensitivity  = flag rate: what picking scans at random would catch.
    PPV is at each dataset's OWN prevalence (enriched public data); recompute for the assumed hospital
    prevalence: PPV = sens*p / (sens*p + FAR*(1-p)).
    """
    ui = pol.urgent_idx()
    Xn, Yn, dn = load_xy("nih", bb)
    Xc, Yc, dc = load_xy("chexpert", bb)
    sn, sc = composite_score(pol.probs(head.logits(Xn)), ui), composite_score(pol.probs(head.logits(Xc)), ui)
    yn, yc = composite_label(Yn, ui), composite_label(Yc, ui)
    va, te, ex = (dn.split == "val").values, (dn.split == "test").values, (dc.split == "external").values
    sets = (("nih_test", yn[te], sn[te]), ("chexpert_external", yc[ex], sc[ex]))
    rows = []
    for b in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        t_val = float(np.quantile(sn[va], 1 - b))
        for name, y, s in sets:
            m = ~np.isnan(y)
            y, s = y[m], s[m]
            for mode in ("fixed_cutoff", "queue_rank"):
                if mode == "fixed_cutoff":
                    flag, cut = s >= t_val, t_val
                else:
                    flag = Policy(findings=pol.findings, urgent=pol.urgent, raise_budget=b, unsure_budget=0.0
                                  ).queue_rank(s) == "Raised"
                    cut = float(s[flag].min())
                op = operating_point(y, flag.astype(float), 0.5)
                prev = float(y.mean())
                rows.append({"mode": mode, "budget": b, "dataset": name, "score_cutoff": cut,
                             "cutoff_source": "NIH validation" if mode == "fixed_cutoff" else "this queue",
                             "flag_rate": op["flag_rate"], "sensitivity": op["sensitivity"],
                             "ceiling_sensitivity": min(1.0, op["flag_rate"] / prev),
                             "random_sensitivity": op["flag_rate"], "false_alarm_rate": op["false_alarm_rate"],
                             "ppv_at_dataset_prevalence": op["ppv"], "dataset_prevalence": prev, "n": op["n"]})
    df = stamp(pd.DataFrame(rows))
    df.to_csv(art("reports", "budget_table.csv"), index=False)
    note = ("PPV is at each dataset's own (enriched) prevalence and must be recomputed at the assumed hospital "
            "prevalence p: PPV = sens*p / (sens*p + FAR*(1-p)). Cut-offs for fixed_cutoff come from NIH validation only.")
    show = df[["mode", "budget", "dataset", "flag_rate", "sensitivity", "ceiling_sensitivity", "false_alarm_rate",
               "ppv_at_dataset_prevalence"]]
    (art("reports") / "budget_table.md").write_text("\n".join(
        ["# Capacity budget table", "", stamp_line(), "", note, "", md_table(show, 3)]))
    print("\nCapacity budget table (reports/budget_table.{csv,md}):")
    print(show.round(3).to_string(index=False))
    print(note)


if __name__ == "__main__":
    main()
