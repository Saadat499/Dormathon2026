"""Queue Race: the same scans meet the same single reader twice, in arrival order (FCFS) and ranked
by VoxelSort's own queue rule. Cases are drawn from the SAVED predictions (real scores with true
labels), so misses and wrong flags come from the data, not from a formula.

This is a simulation under stated assumptions. It is not clinical proof."""
from __future__ import annotations

import math
import random

from . import config
from .cohorts import PredictionSet
from .ranking import PRIORITY_ORDER, derive_priority


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def run_queue_race(p: dict, ps: PredictionSet, threshold: float, max_wait_minutes: int) -> dict:
    try:
        arrivals, reads = float(p["arrivals_per_hour"]), float(p["reads_per_hour"])
        share, hours = float(p["urgent_share"]), float(p["hours"])
        seed = int(p.get("seed", 1))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Check the inputs: arrivals_per_hour, reads_per_hour, urgent_share, hours and seed are required numbers.") from None
    if not (0 < arrivals <= 500 and 0 < reads <= 500 and 0 < hours <= 72 and 0 < share < 1):
        raise ValueError("Check the inputs: arrivals and reads must be 1 to 500 per hour, hours 1 to 72, urgent share between 0 and 1.")
    if arrivals * hours > config.MAX_SIM_CASES:
        raise ValueError(f"Too many scans for the simulation (max {config.MAX_SIM_CASES}).")
    pos, neg = ps.positives, ps.negatives
    if not pos or not neg:
        raise ValueError("The saved predictions need both urgent and routine scans with labels to run the simulation.")

    rng = random.Random(seed)
    gap = 60.0 / arrivals
    cases, t = [], 0.0
    while True:
        t += -math.log(1.0 - rng.random()) * gap  # Poisson arrivals
        if t > hours * 60:
            break
        urgent = rng.random() < share
        src = rng.choice(pos if urgent else neg)
        flag = derive_priority(src.score, src.ood_flag, threshold)
        cases.append({"id": f"SIM-{len(cases) + 1:03d}", "arrive": t, "urgent": urgent, "flag": flag})
    if not cases:
        raise ValueError("No scans arrived in that time. Raise the arrival rate or the hours.")

    service = 60.0 / reads  # fixed read time: an assumption, stated below

    def run(pick):
        queue, done, clock, i = [], [], 0.0, 0
        while len(done) < len(cases):
            while i < len(cases) and cases[i]["arrive"] <= clock:
                queue.append(cases[i])
                i += 1
            if not queue:
                clock = cases[i]["arrive"]
                continue
            c = queue.pop(pick(queue, clock))
            done.append({**c, "start": clock, "wait": clock - c["arrive"]})
            clock += service
        return done

    fcfs = run(lambda q, clock: 0)  # queue is already in arrival order

    def voxel_pick(q, clock):
        best, key = 0, None
        for k, c in enumerate(q):
            kk = (0 if clock - c["arrive"] >= max_wait_minutes else 1, PRIORITY_ORDER[c["flag"]], c["arrive"])
            if key is None or kk < key:
                best, key = k, kk
        return best

    vox = run(voxel_pick)

    urg = lambda d: [c for c in d if c["urgent"]]  # noqa: E731
    oth = lambda d: [c for c in d if not c["urgent"]]  # noqa: E731
    r1 = lambda x: round(x * 10) / 10  # noqa: E731

    def order(d):
        return [
            {"id": c["id"], "wait_min": r1(c["wait"]), "truly_urgent": c["urgent"], "urgent": c["urgent"], "flag": c["flag"]}
            for c in sorted(d, key=lambda c: c["start"])[:12]
        ]

    return {
        "assumptions": {
            "arrivals_per_hour": arrivals, "reads_per_hour": reads, "urgent_share": share, "hours": hours, "seed": seed,
            "readers": 1, "read_time": "fixed, 60 / reads_per_hour minutes", "arrivals": "Poisson",
            "threshold": threshold, "max_wait_minutes": max_wait_minutes,
            "scores": f"drawn at random from {len(pos)} urgent and {len(neg)} routine saved predictions ({ps.source})",
            "note": (
                "A simulation under stated assumptions, not clinical proof. "
                + ("Scores come from saved model predictions. " if ps.is_real else
                   "SYNTHETIC DATA: the saved predictions are not available, so scores come from a made-up cohort and say nothing about the model. ")
                + "VoxelSort can only move scans up; misses and wrong flags are counted, not hidden."
            ),
        },
        "n_cases": len(cases),
        "n_urgent": len(urg(cases)),
        "fcfs": {"urgent_wait_min": r1(_mean([c["wait"] for c in urg(fcfs)])), "other_extra_wait_min": 0, "missed_urgent": None, "wrong_flags": None},
        "voxelsort": {
            "urgent_wait_min": r1(_mean([c["wait"] for c in urg(vox)])),
            "other_extra_wait_min": r1(_mean([c["wait"] for c in oth(vox)]) - _mean([c["wait"] for c in oth(fcfs)])),
            "missed_urgent": sum(1 for c in cases if c["urgent"] and c["flag"] == "normal"),
            "wrong_flags": sum(1 for c in cases if not c["urgent"] and c["flag"] == "raised"),
        },
        "read_order": {"fcfs": order(fcfs), "voxelsort": order(vox)},
    }
