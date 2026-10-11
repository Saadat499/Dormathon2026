"""Audit log: every state change writes one row. `actor` is whatever name was typed in (no login),
so the log records what happened, not who is proven to have done it."""
from __future__ import annotations

import csv
import io
import re
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..db import iso, now_utc
from ..models import AuditEntry

ACTIONS = (
    "upload", "mark_reviewed", "flag_disagreement", "urgent_request", "override", "queue_settings",
    "threshold_change", "site_readiness_run",
    "scoring",          # not in contract v0.3: the background model result for an upload (actor "system")
)
SYSTEM = "system"


def clean(text: Optional[str], limit: int = 500) -> str:
    """User text for the log: no control characters, bounded length."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).strip()[:limit]


def actor_name(actor: Optional[str]) -> str:
    return clean(actor, config.MAX_ACTOR_LEN) or "unnamed"


def record(s: Session, *, actor: Optional[str], action: str, study_id: Optional[str], text: str, **data) -> AuditEntry:
    if action not in ACTIONS:
        raise ValueError(f"unknown audit action {action!r}")
    row = AuditEntry(timestamp=now_utc(), actor=SYSTEM if actor == SYSTEM else actor_name(actor), action=action,
                     study_id=study_id, detail={"text": clean(text, 2000), **data})
    s.add(row)
    return row


def to_public(row: AuditEntry) -> dict:
    d = dict(row.detail or {})
    return {"id": row.id, "timestamp": iso(row.timestamp), "actor": row.actor, "action": row.action,
            "study_id": row.study_id, "detail": d.pop("text", ""), "data": d}


def query(s: Session, action: Optional[str] = None, q: Optional[str] = None, limit: int = 200) -> list[dict]:
    stmt = select(AuditEntry)
    if action:
        stmt = stmt.where(AuditEntry.action == action)
    rows = s.scalars(stmt.order_by(AuditEntry.id.desc())).all()
    out = [to_public(r) for r in rows]
    if q:
        needle = q.strip().lower()
        out = [r for r in out if needle in (r["study_id"] or "").lower() or needle in r["actor"].lower() or needle in r["detail"].lower()]
    return out[:limit]


def _safe_cell(v) -> str:
    """Stop spreadsheet programs from running user text as a formula."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def to_csv(rows: list[dict]) -> str:
    cols = ["id", "timestamp", "actor", "action", "study_id", "detail"]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(cols)
    for r in rows:
        w.writerow([r[c] if c in ("id", "timestamp") else _safe_cell(r[c]) for c in cols])
    return out.getvalue()
