"""Queue Race: the same scans meet the same single reader in different orders.

Cases are drawn (with replacement, seeded) from the REAL saved predictions of the held-out NIH test
split, urgent and non-urgent separately, so the simulated urgent share is the one requested and the
model's misses and wrong flags are the model's own, not a formula.

Three orderings of the one stream:
  fcfs        arrival order. Engine: Member B's `QueueSimulator.run_simulation(strategy="fifo")`.
  voxelsort   the LIVE queue rule (`services.queue.sort_key`): wait cap, then raised > unsure > normal,
              then oldest first. This is what the worklist actually does.
  b_engine    Member B's own `ai_triage` ordering (tier base, score bonus, ageing, 80%-of-SLA jump),
              run through B's `run_queue_race_comparison` on the same stream and reported alongside.

A simulation under stated assumptions. It is not clinical proof, and the response says so.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .. import config
from ..context import AppContext
from ..errors import ApiError
from ..evalkit.queue_simulator import QueueSimulator
from ..evalkit.urgency_mapping import PriorityLevel
from . import queue

# Contract vocabulary -> Member B's tiers (only used to feed B's simulator).
LANE_TO_TIER = {"raised": PriorityLevel.STAT.value, "unsure": PriorityLevel.URGENT.value, "normal": PriorityLevel.ROUTINE.value}
READ_SIGMA = 0.35          # same log-normal shape as B's simulator
SHOWN = 12


def _num(p: dict, key: str, lo: float, hi: float, integer: bool = False) -> float:
    v = p.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or math.isinf(v):
        raise ApiError(400, f"{key} must be a number between {lo:g} and {hi:g}")
    if not lo <= v <= hi or (integer and int(v) != v):
        raise ApiError(400, f"{key} must be {'a whole number ' if integer else ''}between {lo:g} and {hi:g}")
    return float(v)


def _mean(xs) -> Optional[float]:
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else None


def _r1(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x) * 10) / 10


def build_stream(ctx: AppContext, arrivals: float, reads: float, share: float, hours: float, seed: int) -> list[dict]:
    test = ctx.predictions.test
    pos, neg = np.flatnonzero(test.labels == 1), np.flatnonzero(test.labels == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ApiError(503, "The saved predictions need both urgent and non-urgent held-out scans to run the simulation.")
    lanes = ctx.lanes()
    rng = np.random.default_rng(seed)
    gap, median_read = 60.0 / arrivals, 60.0 / reads
    stream, t = [], 0.0
    while True:
        t += float(rng.exponential(gap))                       # Poisson arrivals
        if t >= hours * 60.0 or len(stream) >= config.MAX_SIM_CASES:
            break
        urgent = bool(rng.random() < share)
        i = int(rng.choice(pos if urgent else neg))
        score, ood = float(test.scores[i]), bool(test.ood[i])
        lane = queue.model_priority(score, ood, lanes)
        read = float(np.clip(rng.lognormal(mean=np.log(median_read), sigma=READ_SIGMA), 0.2 * median_read, 5.0 * median_read))
        stream.append({
            "study_id": f"SIM-{len(stream) + 1:03d}", "idx": len(stream), "arrival_time": t, "read_time_needed": read,
            "truly_urgent": urgent, "lane": lane, "ood_flag": ood, "source_study": str(test.study_ids[i]),
            # fields Member B's simulator reads:
            "priority": LANE_TO_TIER[lane], "urgency_score": 0.5 if ood else score,
            "true_tier": (PriorityLevel.STAT if urgent else PriorityLevel.ROUTINE).value,
        })
    return stream


def run_live_rule(stream: list[dict], max_wait_minutes: float, mode: str = "model") -> list[dict]:
    """One reader, the live queue rule. Returns the cases in the order they were read."""
    queue_, done, clock, i, n = [], [], 0.0, 0, len(stream)
    while len(done) < n:
        while i < n and stream[i]["arrival_time"] <= clock:
            queue_.append(stream[i])
            i += 1
        if not queue_:
            clock = stream[i]["arrival_time"]
            continue
        c = min(queue_, key=lambda c: queue.sort_key(
            clinician_urgent=False, over_cap=queue.is_over_cap(clock - c["arrival_time"], max_wait_minutes, mode),
            priority=c["lane"], arrival=c["arrival_time"], seq=c["idx"], mode=mode))
        queue_.remove(c)
        done.append({**c, "read_start_time": clock, "wait_time_minutes": clock - c["arrival_time"]})
        clock += c["read_time_needed"]
    return done


def _summary(rows: list[dict]) -> dict:
    urgent = [r["wait_time_minutes"] for r in rows if r["truly_urgent"]]
    other = [r["wait_time_minutes"] for r in rows if not r["truly_urgent"]]
    return {"urgent": _mean(urgent), "other": _mean(other), "max": max((r["wait_time_minutes"] for r in rows), default=None)}


def _order(rows: list[dict]) -> list[dict]:
    first = sorted(rows, key=lambda r: (r["read_start_time"], r["idx"]))[:SHOWN]
    return [{"id": r["study_id"], "wait_min": _r1(r["wait_time_minutes"]), "truly_urgent": bool(r["truly_urgent"]),
             "urgent": bool(r["truly_urgent"]), "flag": str(r["lane"])} for r in first]


def run_queue_race(ctx: AppContext, p: dict) -> dict:
    if not isinstance(p, dict):
        raise ApiError(400, "Send a JSON object with arrivals_per_hour, reads_per_hour, urgent_share, hours and seed.")
    arrivals = _num(p, "arrivals_per_hour", 1, 200)
    reads = _num(p, "reads_per_hour", 1, 200)
    share = _num(p, "urgent_share", 0.01, 0.99)
    hours = _num(p, "hours", 1, config.MAX_SIM_HOURS)
    seed = int(_num({"seed": p.get("seed", 1)}, "seed", 0, 2**31 - 1, integer=True))
    if arrivals * hours > config.MAX_SIM_CASES:
        raise ApiError(400, f"Too many scans for one simulation (arrivals_per_hour x hours must be at most {config.MAX_SIM_CASES}).")
    if not ctx.predictions.available or ctx.threshold is None:
        raise ApiError(503, "Saved predictions are not available on this server, so the Queue Race cannot run. "
                            + " ".join(ctx.predictions.problems))
    max_wait = ctx.max_wait_minutes
    with ctx.sim_lock:                                         # one simulation at a time on a small host
        stream = build_stream(ctx, arrivals, reads, share, hours, seed)
        if not stream:
            raise ApiError(400, "No scans arrived in that time. Raise the arrival rate or the hours.")
        # Member B's engine: FIFO and B's own triage ordering on the identical stream. B promotes a scan
        # once it has waited 80% of its tier's SLA, so SLA = max_wait / 0.8 puts that jump at the wait cap.
        sla = max_wait / 0.8
        sim = QueueSimulator(arrival_rate_per_hour=arrivals, reading_speed_per_hour=reads, num_radiologists=1,
                             stat_sla_minutes=sla, urgent_sla_minutes=sla, routine_sla_minutes=sla, seed=seed)
        race = sim.run_queue_race_comparison(duration_hours=hours, stream=stream)
        fifo = race["df_fifo"].to_dict("records")
        b_ai = race["df_ai"].to_dict("records")
        live = run_live_rule(stream, max_wait)

    s_fifo, s_live, s_b = _summary(fifo), _summary(live), _summary(b_ai)

    def extra(a: dict) -> Optional[float]:
        return None if a["other"] is None or s_fifo["other"] is None else _r1(a["other"] - s_fifo["other"])

    n_urgent = sum(1 for c in stream if c["truly_urgent"])
    lanes = ctx.lanes()
    test = ctx.predictions.test
    real = ctx.predictions.is_real
    note = ("A simulation under stated assumptions, not clinical proof. "
            + ("Scores are real saved model outputs for held-out NIH test images, resampled to the requested urgent share. "
               if real else "NOT REAL MODEL OUTPUT: the saved predictions on this server are a test fixture. ")
            + "VoxelSort can only move scans up; misses and wrong flags are counted, not hidden."
            + ("" if n_urgent else " No truly urgent scan arrived in this run, so urgent waits are not available."))
    return {
        "assumptions": {
            "arrivals_per_hour": arrivals, "reads_per_hour": reads, "urgent_share": share, "hours": hours, "seed": seed,
            "readers": 1, "arrivals": "Poisson",
            "read_time": f"log-normal, median {60.0 / reads:.1f} min (shape as in Member B's simulator)",
            "threshold": ctx.threshold, "score_cutoff": round(float(lanes.t_raise), 4), "max_wait_minutes": max_wait,
            "scores": f"sampled from {int((test.labels == 1).sum())} urgent and {int((test.labels == 0).sum())} non-urgent saved predictions ({ctx.predictions.source}, NIH test split)",
            "model_version": ctx.predictions.model_version,
            "ordering": "voxelsort = the live queue rule (wait cap, then raised, unsure, normal, oldest first); no clinician requests or overrides are simulated",
            "truth": "'truly urgent' = the dataset label 'any urgent finding' (text-mined NIH labels, an assumption about urgency)",
            "note": note,
        },
        "n_cases": len(stream),
        "n_urgent": n_urgent,
        "fcfs": {"urgent_wait_min": _r1(s_fifo["urgent"]), "other_extra_wait_min": 0, "missed_urgent": None, "wrong_flags": None,
                 "max_wait_min": _r1(s_fifo["max"])},
        "voxelsort": {
            "urgent_wait_min": _r1(s_live["urgent"]), "other_extra_wait_min": extra(s_live),
            "missed_urgent": sum(1 for c in stream if c["truly_urgent"] and c["lane"] == "normal"),
            "wrong_flags": sum(1 for c in stream if not c["truly_urgent"] and c["lane"] == "raised"),
            "max_wait_min": _r1(s_live["max"]),
            "flagged_unsure": sum(1 for c in stream if c["lane"] == "unsure"),
            "flagged_raised": sum(1 for c in stream if c["lane"] == "raised"),
        },
        "b_engine": {
            "urgent_wait_min": _r1(s_b["urgent"]), "other_extra_wait_min": extra(s_b), "max_wait_min": _r1(s_b["max"]),
            "rule": "Member B's QueueSimulator 'ai_triage' ordering (tier base + score bonus + ageing; a scan that has waited max_wait_minutes gets B's promotion). Shown for comparison; the worklist uses the 'voxelsort' rule.",
        },
        "read_order": {"fcfs": _order(fifo), "voxelsort": _order(live)},
        "data_source": ctx.predictions.source,
        "data_is_real": real,
    }
