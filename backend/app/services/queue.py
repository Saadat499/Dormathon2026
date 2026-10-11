"""The single queue-ranking rule. Pure functions: no database, no clock, no model.

The live worklist and the Queue Race simulation both call `sort_key`, so there is one rule.

Order (docs/API_CONTRACT.md, "Queue order"), mode "model":
  1. clinician_urgent                      a clinician's request beats everything the model says
  2. waited >= max_wait_minutes            anti-starvation promotion ("wait_cap")
  3. effective_priority                    raised, then unsure, then normal
  4. oldest uploaded_at first              ties broken by the arrival counter `seq` (stable)
Mode "fcfs": clinician requests first, then arrival order only. The model, the wait cap and
coordinator overrides play no part.

Safety properties (tested in tests/test_queue.py):
  * nothing is ever dropped or hidden: the output is a permutation of the input;
  * an out-of-distribution or unscored scan is never `normal`, so it never ranks below a normal scan
    of the same group;
  * the model can only move a scan up relative to arrival order; only a human override can lower one.

Precedence of the reasons shown to the user (`priority_reason`):
  clinician_request > wait_cap > coordinator_override > model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

PRIORITIES = ("raised", "unsure", "normal")
PRIORITY_ORDER = {"raised": 0, "unsure": 1, "normal": 2}
MODES = ("model", "fcfs")


@dataclass(frozen=True)
class Lanes:
    """Score cut-offs derived from the current threshold. None means "no cut-off is known":
    every scored scan is then `unsure` (never `normal`)."""
    t_raise: Optional[float]
    t_unsure: Optional[float]


def model_priority(score: Optional[float], ood_flag: bool, lanes: Lanes) -> str:
    """The model's own lane. OOD, a missing score, a NaN score or unknown cut-offs all give `unsure`."""
    if ood_flag or score is None or score != score:           # score != score is the NaN test
        return "unsure"
    if lanes.t_raise is None or lanes.t_unsure is None:
        return "unsure"
    if score >= lanes.t_raise:
        return "raised"
    if score >= lanes.t_unsure:
        return "unsure"
    return "normal"


def confidence_band(score: Optional[float], ood_flag: bool, lanes: Lanes) -> str:
    """Member A's definition: where the score sits relative to the cut-offs, not a probability of
    being right. 'low' = inside the unsure band, 'none' = not scored, 'high' = clear of the band."""
    if ood_flag or score is None or score != score or lanes.t_raise is None or lanes.t_unsure is None:
        return "none"
    return "low" if lanes.t_unsure <= score < lanes.t_raise else "high"


def effective_priority(model: Optional[str], override: Optional[str]) -> str:
    """A coordinator override replaces the model's lane. With neither, the scan is `unsure`."""
    if override in PRIORITY_ORDER:
        return override  # type: ignore[return-value]
    return model if model in PRIORITY_ORDER else "unsure"


def is_over_cap(waited_minutes: float, max_wait_minutes: float, mode: str) -> bool:
    return mode == "model" and waited_minutes >= max_wait_minutes


def priority_reason(clinician_urgent: bool, over_cap: bool, override: Optional[str]) -> str:
    if clinician_urgent:
        return "clinician_request"
    if over_cap:
        return "wait_cap"
    if override:
        return "coordinator_override"
    return "model"


def sort_key(*, clinician_urgent: bool, over_cap: bool, priority: str, arrival, seq: int, mode: str) -> tuple:
    """Smaller sorts first. `arrival` is any orderable arrival time (datetime or minutes)."""
    if mode == "fcfs":
        return (0 if clinician_urgent else 1, arrival, seq)
    return (0 if clinician_urgent else 1, 0 if over_cap else 1, PRIORITY_ORDER[priority], arrival, seq)


@dataclass
class QueueItem:
    study_id: str
    seq: int
    uploaded_at: datetime
    clinician_urgent: bool
    priority: str                 # effective priority
    waited_minutes: float
    reviewed: bool = False


def rank(items: Iterable[QueueItem], mode: str, max_wait_minutes: float) -> list[tuple[QueueItem, Optional[int]]]:
    """Returns every item exactly once as (item, rank). Open scans come first in queue order with
    rank 1..n; reviewed scans follow, newest first, with rank None."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    items = list(items)
    open_items = [i for i in items if not i.reviewed]
    done = [i for i in items if i.reviewed]
    open_items.sort(key=lambda i: sort_key(
        clinician_urgent=i.clinician_urgent, over_cap=is_over_cap(i.waited_minutes, max_wait_minutes, mode),
        priority=i.priority, arrival=i.uploaded_at, seq=i.seq, mode=mode))
    done.sort(key=lambda i: (i.uploaded_at, i.seq), reverse=True)
    return [(i, n) for n, i in enumerate(open_items, start=1)] + [(i, None) for i in done]
