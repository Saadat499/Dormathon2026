"""`plain_language`: fixed templates filled with the real outputs. Nothing here is free-written.

Every sentence below is a constant with slots for numbers and names that come from the stored
model output (finding name, probability), the queue (priority, reason) and settings (wait cap).
No medical reasoning is generated, and no template claims the probabilities are calibrated.
"""
from __future__ import annotations

from typing import Optional

DECISION_SUPPORT = "Decision support only: a clinician reads every scan."

SOURCE_PREFIX = {
    "real": "",
    "replay": "Replayed from a saved model output for this exact image (the model was not run again). ",
    "saved_demo": "Demo record: a public NIH ChestX-ray14 image with its saved model output. ",
    "fake": "FAKE DEVELOPMENT OUTPUT, not a model result. ",
    "none": "",
}

PRIORITY_LABEL = {"raised": "urgent review", "unsure": "AI unsure", "normal": "routine"}


def _pct(p: float) -> int:
    return int(round(float(p) * 100))


def _top(findings: list, urgent: Optional[list]) -> Optional[dict]:
    """The finding that drove the score: the highest urgent finding (or the highest of all if the
    urgent list is unknown)."""
    pool = [f for f in findings if urgent is None or f["name"] in urgent] or findings
    return max(pool, key=lambda f: f["probability"]) if pool else None


def build(*, processing: bool, source: str, findings: list, model_priority: Optional[str], ood_flag: bool,
          ood_reason: Optional[str], model_note: Optional[str], reason: str, effective_priority: str,
          max_wait_minutes: int, urgent_findings: Optional[list] = None) -> str:
    if processing:
        return f"This scan is still being processed. It is already in the queue. {DECISION_SUPPORT}"

    prefix = SOURCE_PREFIX.get(source, "")
    top = _top(findings, urgent_findings)
    if ood_flag:
        why = f" ({ood_reason})" if ood_reason else ""
        body = (f"This image looks different from the scans VoxelSort was built on{why}, so the system is not sure "
                "about it and gives no finding scores. It was moved up the queue, not down.")
    elif top is None:
        why = f" ({model_note})" if model_note else ""
        body = (f"No model output is available for this scan{why}. It is placed ahead of routine scans, "
                "not behind them, so a person looks at it.")
    else:
        p, name = _pct(top["probability"]), top["name"]
        if model_priority == "raised":
            body = (f"The model's highest urgent-finding score on this scan is {p}% for {name}, which is at or above "
                    "the current cut-off, so the scan was moved up the reading queue. "
                    "This is a prioritisation aid, not a diagnosis.")
        elif model_priority == "unsure":
            body = (f"The model's highest urgent-finding score on this scan is {p}% for {name}, just below the "
                    "current cut-off, so VoxelSort is not sure. The scan is placed ahead of routine scans "
                    "so a person looks at it sooner. This is not a diagnosis.")
        else:
            body = (f"The model's highest urgent-finding score on this scan is {p}% for {name}, below the current "
                    "cut-off, so the scan keeps its normal place in the queue. This is not an all-clear.")

    if reason == "clinician_request":
        tail = " A clinician asked for an urgent read, which goes ahead of anything the model says."
    elif reason == "wait_cap":
        tail = f" It has waited longer than the {int(max_wait_minutes)}-minute limit, so it was promoted."
    elif reason == "coordinator_override":
        tail = f" A coordinator set its queue priority to {PRIORITY_LABEL.get(effective_priority, effective_priority)} by hand."
    else:
        tail = ""
    return f"{prefix}{body}{tail} {DECISION_SUPPORT}"
