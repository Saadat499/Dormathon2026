"""Request and response shapes, mirroring docs/API_CONTRACT.md.

Contract fields come first in every model. Fields under "extra" comments are additions that the
contract does not name; they are listed in the contract's deviation table. Response models allow
extra keys so a service can add provenance without a schema change, but every contract field is
declared and type-checked here.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Priority = Literal["raised", "normal", "unsure"]
Actor = Optional[str]


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Out(BaseModel):
    model_config = ConfigDict(extra="allow")


# --- prediction record and study ---------------------------------------------------------------
class Finding(BaseModel):
    name: str
    probability: float


class StudyOut(_Out):
    study_id: str
    findings: list[Finding]
    priority: Optional[Priority]                       # null only while status is "processing"
    confidence_band: str
    ood_flag: bool
    heatmap_path: Optional[str]
    uploaded_at: str
    wait_seconds: int
    rank: Optional[int]                                # null once reviewed
    effective_priority: Priority
    priority_reason: Literal["model", "clinician_request", "coordinator_override", "wait_cap"]
    clinician_urgent: bool
    status: Literal["processing", "awaiting_review", "needs_review", "reviewed"]
    flagged_disagreement: bool
    plain_language: str
    # extra
    composite_score: Optional[float] = None
    ood_reason: Optional[str] = None
    model_source: str = "none"
    model_version: Optional[str] = None


class StudyListOut(_Out):
    mode: Literal["model", "fcfs"]
    max_wait_minutes: int
    items: list[StudyOut]


# --- review and overrides ----------------------------------------------------------------------
class ReviewIn(_In):
    action: str
    reason: Optional[str] = Field(None, max_length=500)
    notes: Optional[str] = Field(None, max_length=2000)
    human_priority: Optional[str] = None
    actor: Actor = Field(None, max_length=200)


class UrgentIn(_In):
    note: Optional[str] = Field(None, max_length=1000)
    actor: Actor = Field(None, max_length=200)


class OverrideIn(_In):
    priority: str
    reason: Optional[str] = Field(None, max_length=1000)
    actor: Actor = Field(None, max_length=200)


# --- queue settings and threshold --------------------------------------------------------------
class QueueSettingsOut(_Out):
    mode: Literal["model", "fcfs"]
    max_wait_minutes: int


class QueueSettingsIn(_In):
    mode: str
    max_wait_minutes: int
    actor: Actor = Field(None, max_length=200)


class ThresholdOut(_Out):
    value: Optional[float]
    min_sensitivity_target: float


class ThresholdPreviewIn(_In):
    value: float = Field(allow_inf_nan=False)


class ThresholdPreviewOut(_Out):
    abstention_rate: float
    sensitivity: Optional[float]
    false_alarm_rate: Optional[float]
    n: int
    n_positive: int


class ThresholdIn(_In):
    value: float = Field(allow_inf_nan=False)
    confirm: bool = False
    actor: Actor = Field(None, max_length=200)


# --- queue race --------------------------------------------------------------------------------
class RaceArm(_Out):
    urgent_wait_min: Optional[float]
    other_extra_wait_min: Optional[float]
    missed_urgent: Optional[int]
    wrong_flags: Optional[int]


class RaceRow(_Out):
    id: str
    wait_min: float
    truly_urgent: bool


class RaceOrder(BaseModel):
    fcfs: list[RaceRow]
    voxelsort: list[RaceRow]


class QueueRaceOut(_Out):
    assumptions: dict[str, Any]
    fcfs: RaceArm
    voxelsort: RaceArm
    read_order: RaceOrder


# --- site readiness ----------------------------------------------------------------------------
class JobOut(BaseModel):
    job_id: str


class DriftOut(_Out):
    score: Optional[float]
    warning: bool
    note: str


class SensitivityOut(_Out):
    point: float
    ci_low: float
    ci_high: float
    width: float
    method: str
    n_positive: int


class SiteReadinessOut(_Out):
    status: Literal["processing", "done"]
    drift: Optional[DriftOut] = None
    sensitivity: Optional[SensitivityOut] = None
    target: Optional[float] = None
    note: Optional[str] = None
    verdict: Optional[Literal["ready", "use_with_caution", "not_ready", "not_enough_evidence"]] = None


class SampleSizeOut(_Out):
    positives_needed: Optional[int]
    scans_needed: Optional[int]
    note: str


# --- ood, audit, health ------------------------------------------------------------------------
class OodOut(_Out):
    ood_flag: bool
    reason: str
    score: Optional[float]


class AuditOut(_Out):
    id: int
    timestamp: str
    actor: str
    action: str
    study_id: Optional[str]
    detail: str


class HealthOut(_Out):
    status: str
