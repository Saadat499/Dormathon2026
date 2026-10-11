"""Database tables."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Study(Base):
    __tablename__ = "studies"

    study_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)      # arrival counter: stable tie-break
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    filename: Mapped[Optional[str]] = mapped_column(String(200))
    sha256: Mapped[Optional[str]] = mapped_column(String(64))
    image_path: Mapped[Optional[str]] = mapped_column(String(400))              # relative to DATA_DIR or the artifacts folder
    heatmap_path: Mapped[Optional[str]] = mapped_column(String(400))
    # processing | done.  The contract's `status` (processing/awaiting_review/needs_review/reviewed) is derived.
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="processing")
    findings: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # [{"name","probability"}], highest first
    composite_score: Mapped[Optional[float]] = mapped_column(Float)             # max probability over urgent findings
    model_priority: Mapped[Optional[str]] = mapped_column(String(10))           # snapshot when scored; the API derives it live
    confidence_band: Mapped[Optional[str]] = mapped_column(String(20))
    ood_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ood_score: Mapped[Optional[float]] = mapped_column(Float)
    ood_reason: Mapped[Optional[str]] = mapped_column(String(300))
    # real | replay | saved_demo | fake | none   (where the numbers came from)
    model_source: Mapped[str] = mapped_column(String(20), nullable=False, default="none")
    model_version: Mapped[Optional[str]] = mapped_column(String(80))
    model_note: Mapped[Optional[str]] = mapped_column(String(400))              # e.g. why no model output exists
    clinician_urgent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    override_priority: Mapped[Optional[str]] = mapped_column(String(10))
    override_reason: Mapped[Optional[str]] = mapped_column(Text)
    flagged_disagreement: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class AuditEntry(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    actor: Mapped[str] = mapped_column(String(80), nullable=False)
    action: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    study_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    detail: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)    # {"text": "...", ...structured fields}


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(60), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)


class SiteJob(Base):
    __tablename__ = "site_jobs"

    job_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="processing")
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SavedPrediction(Base):
    """Member A's saved model outputs with true labels: the table behind the threshold preview,
    the sensitivity floor and the Queue Race."""
    __tablename__ = "saved_predictions"

    study_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    site: Mapped[str] = mapped_column(String(20), nullable=False)               # dataset, e.g. "nih"
    split: Mapped[str] = mapped_column(String(20), nullable=False)              # val | test
    true_label_any_urgent: Mapped[Optional[int]] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float, nullable=False)                 # composite score
    probs: Mapped[dict] = mapped_column(JSON, nullable=False)                   # {finding: probability}
    ood_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="real")  # real | synthetic_test_fixture

    __table_args__ = (Index("ix_saved_predictions_split", "split"),)
