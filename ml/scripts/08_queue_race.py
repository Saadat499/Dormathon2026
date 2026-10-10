"""Queue Race on REAL predictions - a STARTING POINT for Member B's simulator, not the final one.

Scans arrive at random (Poisson) over a shift and are read one at a time by `readers` radiologists.
Each simulated scan is a real held-out study (NIH test or CheXpert external) with its real label,
score and OOD flag, resampled so that the urgent share equals an ASSUMED hospital prevalence
(public data is enriched: NIH ~21%, CheXpert ~72% "any urgent").
Policies compared:
  fcfs         first come, first served (today's baseline)
  lanes_fixed  Raised / AI unsure / Normal from the fixed NIH-validation cut-offs; first-come within a lane
  lanes_rank   Raised = top `raise_budget` of the scans waiting NOW (Policy.queue_rank), then AI unsure
               (next `unsure_budget` + all OOD), then Normal; first-come within a lane
  score        order by the continuous score; OOD scans sit just above every Normal scan
Safety rules shared by all: a scan that has waited `wait_cap` minutes jumps the queue (oldest first),
nothing is removed, every scan is read. Clinician requests and the FCFS fallback switch are not modelled.

ASSUMPTIONS (all labelled in the outputs): arrival process, arrival rate grid, readers, reading-time
distribution, shift length, wait cap, prevalence. This shows improved ordering over arrival order under
these assumptions; it is a simulation, not clinical proof.
Outputs: reports/queue_race.csv, reports/queue_race.md, reports/queue_race.png
"""
import _common  # noqa: F401
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chesttriage_ml.config import art
from _common import get_policy, md_table, stamp, stamp_line

LANE = {"Raised": 0, "AI unsure": 1, "Normal": 2}
POLICIES = ("fcfs", "lanes_fixed", "lanes_rank", "score")


def simulate(pool, policy_name, pol, rng, rate_per_min, readers, read_min, hours, wait_cap, prevalence):
    n = rng.poisson(rate_per_min * hours * 60)
    arr = np.sort(rng.uniform(0, hours * 60, n))
    urg = rng.random(n) < prevalence
    idx = np.where(urg, rng.choice(pool["urg_idx"], n), rng.choice(pool["non_idx"], n))
    score, lane, ood = pool["score"][idx], pool["lane"][idx], pool["ood"][idx]
    key_score = np.where(ood, pol.t_unsure, score)                     # OOD never ranks below Normal
    sig = 0.5                                                          # lognormal reading time, CV ~0.53
    svc = rng.lognormal(np.log(read_min) - sig ** 2 / 2, sig, n)
    free = np.zeros(readers)
    start = np.full(n, np.nan)
    waiting, nxt, capped = [], 0, 0
    while nxt < n or waiting:
        r = int(np.argmin(free))
        t = free[r]
        if not waiting:                                                # idle reader: jump to next arrival
            t = max(t, arr[nxt])
        while nxt < n and arr[nxt] <= t:
            waiting.append(nxt); nxt += 1
        over = [i for i in waiting if t - arr[i] >= wait_cap]
        if over:
            j = min(over, key=lambda i: arr[i]); capped += 1
        elif policy_name == "fcfs":
            j = min(waiting, key=lambda i: arr[i])
        elif policy_name == "lanes_fixed":
            j = min(waiting, key=lambda i: (lane[i], arr[i]))
        elif policy_name == "lanes_rank":
            w = np.array(waiting)                                      # already in arrival order
            lr = pol.queue_rank(score[w], ood[w])
            j = int(min(range(len(w)), key=lambda q: (LANE[lr[q]], arr[w[q]])))
            j = int(w[j])
        else:
            j = min(waiting, key=lambda i: (-key_score[i], arr[i]))
        waiting.remove(j)
        start[j] = t
        free[r] = t + svc[j]
    return start - arr, urg, capped


def summarise(waits, urgs):
    w, u = np.concatenate(waits), np.concatenate(urgs)
    wu, wn = w[u], w[~u]
    return {"urgent_mean_wait": wu.mean(), "urgent_median_wait": np.median(wu), "urgent_p90_wait": np.percentile(wu, 90),
            "urgent_read_within_30min": (wu <= 30).mean(), "nonurgent_mean_wait": wn.mean(),
            "nonurgent_p90_wait": np.percentile(wn, 90), "max_wait_any": w.max(), "n_scans": len(w)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nih_test", "chexpert_external"])
    ap.add_argument("--prevalence", type=float, nargs="+", default=[0.10, 0.20],
                    help="ASSUMED urgent share of arriving hospital scans (scans are resampled to it)")
    ap.add_argument("--readers", type=int, default=2)
    ap.add_argument("--read_min", type=float, default=4.0, help="mean minutes to read one scan")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--wait_cap", type=float, default=180.0, help="minutes; capped scans jump the queue")
    ap.add_argument("--utilisation", type=float, nargs="+", default=[0.80, 0.90, 0.95, 1.00, 1.10],
                    help="arrival rate as a share of reading capacity (the arrival-rate grid)")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pol = get_policy()
    cap_per_hour = a.readers * 60 / a.read_min
    src = {"nih_test": ("nih", "test"), "chexpert_external": ("chexpert", "external")}
    rows = []
    for ds in a.datasets:
        name, sp = src[ds]
        d = pd.read_csv(art("predictions", f"{name}_full.csv"))
        d = d[(d.split == sp) & d.label_any_urgent.notna()].reset_index(drop=True)
        y = d.label_any_urgent.values == 1
        pool = {"score": d.composite_score.values, "lane": d.priority.map(LANE).values,
                "ood": d.ood_flag.values.astype(bool), "urg_idx": np.flatnonzero(y), "non_idx": np.flatnonzero(~y)}
        print(f"[{ds}] {len(d)} real studies (own prevalence {y.mean():.2f}, resampled to the assumed values)")
        for prev in a.prevalence:
            for u in a.utilisation:
                rate = u * cap_per_hour / 60
                for p in POLICIES:
                    rng = np.random.default_rng(a.seed)                # same arrivals for every policy
                    W, U, caps = [], [], 0
                    for _ in range(a.reps):
                        w, uu, c = simulate(pool, p, pol, rng, rate, a.readers, a.read_min, a.hours, a.wait_cap, prev)
                        W.append(w); U.append(uu); caps += c
                    rows.append({"dataset": ds, "assumed_prevalence": prev, "utilisation": u,
                                 "arrivals_per_hour": rate * 60, "policy": p, **summarise(W, U),
                                 "cap_triggers_per_shift": caps / a.reps})
            print(f"  prevalence {prev:.2f} done")
    df = stamp(pd.DataFrame(rows))
    df.to_csv(art("reports", "queue_race.csv"), index=False)

    assumptions = [
        "ASSUMPTIONS (change them; Member B owns the final simulator):",
        f"- arrivals: Poisson, {a.hours:g} h shift, then the queue drains; arrival-rate grid "
        f"{', '.join(f'{u * cap_per_hour:.0f}/h' for u in a.utilisation)} (utilisation {a.utilisation})",
        f"- readers: {a.readers}, reading time lognormal with mean {a.read_min:g} min (capacity {cap_per_hour:.0f} scans/h)",
        f"- prevalence: resampled to ASSUMED hospital values {a.prevalence} (public data is enriched)",
        f"- wait cap: {a.wait_cap:g} min; clinician requests and the FCFS fallback switch are not modelled",
        f"- policy: Raised budget {pol.raise_budget:.0%}, AI unsure budget {pol.unsure_budget:.0%}; "
        f"fixed cut-offs from NIH validation; {a.reps} shifts per cell, same arrivals for every policy",
        "- waits in minutes. A simulation, not clinical proof: it shows improved ordering over arrival order "
        "under these assumptions."]
    md = ["# Queue Race on real predictions (starting point for Member B)", "", stamp_line(), ""] + assumptions + [""]
    cols = ["utilisation", "arrivals_per_hour", "policy", "urgent_mean_wait", "urgent_p90_wait",
            "urgent_read_within_30min", "nonurgent_mean_wait", "nonurgent_p90_wait", "max_wait_any"]
    for ds in a.datasets:
        for prev in a.prevalence:
            t = df[(df.dataset == ds) & (df.assumed_prevalence == prev)][cols]
            md += [f"## {ds}, assumed prevalence {prev:.0%}", "", md_table(t, 2), ""]
    (art("reports") / "queue_race.md").write_text("\n".join(md))
    print("\n".join(md))

    combos = [(ds, pv) for ds in a.datasets for pv in a.prevalence]
    fig, axes = plt.subplots(len(combos), 2, figsize=(11, 3.6 * len(combos)), squeeze=False)
    colors = {"fcfs": "#888888", "lanes_fixed": "#2a7fd4", "lanes_rank": "#2aa876", "score": "#d4572a"}
    for row, (ds, pv) in zip(axes, combos):
        t = df[(df.dataset == ds) & (df.assumed_prevalence == pv)]
        for p in POLICIES:
            q = t[t.policy == p]
            row[0].plot(q.arrivals_per_hour, q.urgent_mean_wait, "o-", c=colors[p], label=p)
            row[1].plot(q.arrivals_per_hour, q.nonurgent_mean_wait, "o-", c=colors[p], label=p)
        row[0].set_title(f"{ds}, prevalence {pv:.0%}: mean wait, URGENT (min)", fontsize=9)
        row[1].set_title(f"{ds}, prevalence {pv:.0%}: mean wait, non-urgent (min)", fontsize=9)
        for z in row:
            z.set_xlabel(f"arrivals per hour (capacity {cap_per_hour:.0f}/h)"); z.legend(fontsize=7); z.grid(alpha=.3)
    fig.suptitle("Simulation under stated assumptions - not clinical proof", fontsize=10)
    fig.tight_layout(); fig.savefig(art("reports", "queue_race.png"), dpi=130)
    print("saved reports/queue_race.{csv,md,png}")


if __name__ == "__main__":
    main()
