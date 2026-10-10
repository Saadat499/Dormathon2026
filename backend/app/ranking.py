"""Priority derivation, queue ranking and fixed plain-language templates.

Pure functions only (no database, no clock), so the same rule ranks the live worklist and the
Queue Race simulation, and so it can be tested on its own.

Safety rules enforced here (docs/API_CONTRACT.md):
  1. The model can only move scans up. Nothing is hidden or auto-cleared.
  2. `unsure` and out-of-distribution scans always rank above `normal`.
  3. A clinician urgency request beats the model.
  4. A scan waiting past max_wait_minutes is promoted.
  5. mode "fcfs" ignores the model completely.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .config import UNSURE_BAND

PRIORITY_ORDER = {"raised": 0, "unsure": 1, "normal": 2}

# Findings Member B's urgency mapping files under ROUTINE
# (eval_safety/voxelsort_eval_safety/urgency_mapping.py). They do not push a scan up the queue.
# This split is an ASSUMPTION about urgency, not a clinical standard. Any finding name we do not
# recognise is treated as urgent: an unknown name must never push a scan down.
ROUTINE_FINDINGS = {"cardiomegaly", "atelectasis", "support devices", "no finding"}


def norm_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").lower().split())


def urgency_score(findings: Iterable[dict]) -> Optional[float]:
    """Highest probability among findings that count as urgent. None when there is no model output."""
    scores = [
        float(f["probability"])
        for f in findings
        if norm_name(f["name"]) not in ROUTINE_FINDINGS and f.get("probability") is not None
    ]
    if scores:
        return max(scores)
    # Only routine findings were reported: the urgent score is the best we know, which is "low".
    # With no findings at all we know nothing, so the caller must treat it as unsure.
    return 0.0 if any(f.get("probability") is not None for f in findings) else None


def derive_priority(score: Optional[float], ood: bool, threshold: float, band: float = UNSURE_BAND) -> str:
    """Model priority from a score. Missing score or an OOD image is `unsure`, never `normal`."""
    if ood or score is None or score != score:  # score != score is the NaN check
        return "unsure"
    if score >= threshold:
        return "raised"
    if score >= threshold - band:
        return "unsure"
    return "normal"


def _pct(p: float) -> int:
    return round(p * 100)


def plain_language(*, processing: bool, ood: bool, findings: list, priority: Optional[str], source: str) -> str:
    """Fixed templates filled with real outputs. Nothing here is free-written medical reasoning."""
    if processing:
        return "This scan is still being processed. It is already in the queue and will be read by a clinician."
    if ood:
        text = (
            "This image looks different from the scans VoxelSort was built on, so the system is not sure about it. "
            "It has been moved up the queue, not down. A clinician will read the scan."
        )
    elif not findings:
        text = (
            "No model output is available for this scan, so VoxelSort cannot assess it. "
            "It is placed ahead of routine scans, not behind them, so a person looks at it. A clinician will read the scan."
        )
    else:
        top = max(findings, key=lambda f: f["probability"])
        name, p = top["name"], _pct(top["probability"])
        if priority == "raised":
            text = (
                f"VoxelSort estimates a {p}% likelihood of {name} on this scan, so it was moved up the reading queue. "
                "This is a prioritisation aid, not a diagnosis. A clinician will read the scan."
            )
        elif priority == "unsure":
            text = (
                f"VoxelSort is not sure about this scan (highest estimate: {p}% {name}). "
                "It was placed ahead of routine scans so a person looks at it sooner. This is not a diagnosis."
            )
        else:
            text = (
                f"VoxelSort did not flag an urgent finding on this scan (highest estimate: {p}% {name}). "
                "It stays in the normal reading order. This is not an all-clear: a clinician will still read the scan."
            )
    if source == "synthetic_demo":
        return "SYNTHETIC DEMO RECORD, not a real patient and not a model output. " + text
    if source == "replay":
        return "Replayed from a saved model prediction (the model was not run on this image here). " + text
    return text


def rank_studies(items: list[dict], mode: str) -> list[dict]:
    """Order Study dicts and assign `rank`. Reviewed studies go last, newest first, with rank None.

    Queue rule (the single ranking rule in the contract):
      1. clinician_urgent  2. waited past max_wait_minutes  3. raised > unsure > normal  4. oldest first.
    In fcfs mode only rule 1 (a human request, not a model output) and arrival order apply.
    """
    open_items = [s for s in items if s["status"] != "reviewed"]
    done = sorted((s for s in items if s["status"] == "reviewed"), key=lambda s: (s["uploaded_at"], s["_seq"]), reverse=True)
    arrival = lambda s: (s["uploaded_at"], s["_seq"])  # noqa: E731
    if mode == "fcfs":
        open_items.sort(key=lambda s: (not s["clinician_urgent"], *arrival(s)))
    else:
        open_items.sort(
            key=lambda s: (
                not s["clinician_urgent"],
                s["priority_reason"] != "wait_cap",
                PRIORITY_ORDER[s["effective_priority"]],
                *arrival(s),
            )
        )
    for i, s in enumerate(open_items, start=1):
        s["rank"] = i
    for s in done:
        s["rank"] = None
    return open_items + done


def public(study: dict) -> dict:
    """Strip internal keys before sending a Study to the client."""
    return {k: v for k, v in study.items() if not k.startswith("_")}
