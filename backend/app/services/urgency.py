"""Vocabulary bridge between Member A's model, Member B's safety kit and the API contract.

Nothing here changes the contract. It only states, in one place, how the three vocabularies line up.

Which findings are urgent (an ASSUMPTION that needs clinician review, stated by both A and B)
  The model's composite score is the highest probability among A's `urgent` list in policy.json:
  pneumothorax, effusion, edema, consolidation, pneumonia. B's catalogue (`CLINICAL_FINDING_RULES`)
  files the same findings under STAT or URGENT and files atelectasis and cardiomegaly under ROUTINE,
  so the two agree. B has no separate rule for consolidation (B's pneumonia rule covers "infectious
  consolidation"); it follows A's list. A finding nobody has classified is never treated as routine.
  tests/test_vendored_sync.py fails if the two lists stop agreeing.

Priority lanes
  contract `raised`  <- model score at or above the cut-off        ~ B's STAT and URGENT tiers
  contract `unsure`  <- score just below the cut-off, or OOD, or no model output
  contract `normal`  <- score below the unsure band                ~ B's ROUTINE tier
  B's `UrgencyClassifier` sends an out-of-distribution scan to STAT. The contract is stricter about
  wording: OOD is always `unsure` (still ranked above `normal`), never `raised`.

Site Readiness verdicts
  B's GO -> `ready`, WARNING -> `use_with_caution`, NO-GO -> `not_ready`; `not_enough_evidence` when the
  labelled sample is too small or the interval's lower end is below the target (see site_readiness.py).
"""
from __future__ import annotations

from ..evalkit.urgency_mapping import CLINICAL_FINDING_RULES, PriorityLevel

# A's canonical finding name -> the rule in B's catalogue that covers it (None: B has no rule).
A_TO_B_RULE = {
    "pneumothorax": "Pneumothorax",
    "effusion": "Pleural Effusion",
    "edema": "Acute Pulmonary Edema",
    "pneumonia": "Pneumonia",
    "consolidation": None,
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
}
TIER_TO_LANE = {PriorityLevel.STAT.value: "raised", PriorityLevel.URGENT.value: "raised", PriorityLevel.ROUTINE.value: "normal"}


def b_tier(finding: str) -> str | None:
    """B's tier for one of A's findings, or None when B's catalogue does not cover it."""
    rule = A_TO_B_RULE.get(finding)
    return CLINICAL_FINDING_RULES[rule].target_priority.value if rule else None


def urgent_by_b(finding: str) -> bool:
    """True unless B's catalogue explicitly files the finding under ROUTINE."""
    return b_tier(finding) != PriorityLevel.ROUTINE.value
