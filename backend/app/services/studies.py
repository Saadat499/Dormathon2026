"""Studies: upload, background scoring, the ranked worklist, review, urgent requests and overrides.

Priority is derived when a study is READ, from its stored score and the current threshold, so a
threshold change re-ranks the whole queue at once. `Study.model_priority` keeps the lane the scan
had when it was scored, for the record.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import config
from ..context import AppContext
from ..db import as_utc, iso, now_utc
from ..errors import ApiError
from ..models import Study
from . import audit, heatmap, images, plain_language, queue
from .predictor import PredictionRecord, sha256_hex

log = logging.getLogger("voxelsort.studies")
_create_lock = threading.Lock()
STATUSES = ("processing", "awaiting_review", "needs_review", "reviewed")


# ------------------------------------------------------------------------------------- reading
def _view(ctx: AppContext, row: Study, now: datetime, lanes: queue.Lanes, mode: str, max_wait: int) -> dict:
    processing = row.status == "processing"
    reviewed = row.reviewed_at is not None
    model_pr = None if processing else queue.model_priority(row.composite_score, row.ood_flag, lanes)
    effective = queue.effective_priority(model_pr, row.override_priority)
    uploaded = as_utc(row.uploaded_at)
    end = as_utc(row.reviewed_at) if reviewed else now
    wait_s = max(0.0, (end - uploaded).total_seconds())
    over_cap = (not reviewed) and queue.is_over_cap(wait_s / 60.0, max_wait, mode)
    reason = queue.priority_reason(row.clinician_urgent, over_cap, row.override_priority)
    if processing:
        status = "processing"
    elif reviewed:
        status = "reviewed"
    elif effective == "unsure":
        status = "needs_review"
    else:
        status = "awaiting_review"
    return {
        "study_id": row.study_id,
        "findings": list(row.findings or []),
        "priority": model_pr,
        "confidence_band": "pending" if processing else queue.confidence_band(row.composite_score, row.ood_flag, lanes),
        "ood_flag": bool(row.ood_flag),
        "heatmap_path": f"/studies/{row.study_id}/heatmap" if row.heatmap_path else None,
        "uploaded_at": iso(uploaded),
        "wait_seconds": int(wait_s),
        "rank": None,
        "effective_priority": effective,
        "priority_reason": reason,
        "clinician_urgent": bool(row.clinician_urgent),
        "status": status,
        "flagged_disagreement": bool(row.flagged_disagreement),
        "plain_language": plain_language.build(
            processing=processing, source=row.model_source, findings=list(row.findings or []), model_priority=model_pr,
            ood_flag=bool(row.ood_flag), ood_reason=row.ood_reason, model_note=row.model_note, reason=reason,
            effective_priority=effective, max_wait_minutes=max_wait, urgent_findings=ctx.model.urgent),
        # Extra, not in contract v0.3 (documented in docs/API_CONTRACT.md):
        "composite_score": row.composite_score,
        "ood_reason": row.ood_reason if row.ood_flag else None,
        "model_source": row.model_source,
        "model_version": row.model_version,
        "_seq": row.seq, "_uploaded": uploaded, "_wait_min": wait_s / 60.0, "_reviewed": reviewed,
    }


def _public(v: dict) -> dict:
    return {k: val for k, val in v.items() if not k.startswith("_")}


def ranked(ctx: AppContext, s: Session) -> list[dict]:
    now, lanes, mode, max_wait = now_utc(), ctx.lanes(), ctx.mode, ctx.max_wait_minutes
    views = {r.study_id: _view(ctx, r, now, lanes, mode, max_wait) for r in s.scalars(select(Study)).all()}
    items = [queue.QueueItem(v["study_id"], v["_seq"], v["_uploaded"], v["clinician_urgent"], v["effective_priority"],
                             v["_wait_min"], v["_reviewed"]) for v in views.values()]
    out = []
    for item, n in queue.rank(items, mode, max_wait):
        v = views[item.study_id]
        v["rank"] = n
        out.append(v)
    return out


def list_studies(ctx: AppContext, priority: Optional[str] = None, status: Optional[str] = None, q: Optional[str] = None) -> dict:
    if priority and priority not in queue.PRIORITIES:
        raise ApiError(400, "priority must be raised, normal or unsure")
    if status and status not in STATUSES:
        raise ApiError(400, "status must be processing, awaiting_review, needs_review or reviewed")
    with ctx.db.session() as s:
        items = ranked(ctx, s)
    if priority:
        items = [v for v in items if v["effective_priority"] == priority]
    if status:
        items = [v for v in items if v["status"] == status]
    if q and q.strip():
        needle = q.strip().lower()
        items = [v for v in items if needle in v["study_id"].lower()]
    return {"mode": ctx.mode, "max_wait_minutes": ctx.max_wait_minutes, "items": [_public(v) for v in items]}


def get_study(ctx: AppContext, study_id: str) -> dict:
    with ctx.db.session() as s:
        for v in ranked(ctx, s):
            if v["study_id"] == study_id:
                return _public(v)
    raise ApiError(404, f"Study {study_id} not found")


def _row(s: Session, study_id: str) -> Study:
    row = s.get(Study, study_id)
    if row is None:
        raise ApiError(404, f"Study {study_id} not found")
    return row


def _resolve(ctx: AppContext, stored: Optional[str]) -> Optional[Path]:
    """Stored paths are 'data:<relative>' (uploads) or 'artifacts:<relative>' (shipped demo files)."""
    if not stored or ":" not in stored:
        return None
    kind, rel = stored.split(":", 1)
    base = {"data": ctx.settings.data_dir, "artifacts": ctx.settings.artifacts_dir}.get(kind)
    if base is None:
        return None
    p = (base / rel).resolve()
    if base.resolve() not in p.parents or not p.is_file():
        return None
    return p


def image_file(ctx: AppContext, study_id: str) -> Path:
    with ctx.db.session() as s:
        p = _resolve(ctx, _row(s, study_id).image_path)
    if p is None:
        raise ApiError(404, "No image is stored for this study.")
    return p


def heatmap_file(ctx: AppContext, study_id: str) -> Path:
    with ctx.db.session() as s:
        p = _resolve(ctx, _row(s, study_id).heatmap_path)
    if p is None:
        raise ApiError(404, "No heatmap exists for this study.")
    return p


# ------------------------------------------------------------------------------------- writing
def create_study(ctx: AppContext, data: bytes, filename: str, clinician_urgent: bool, actor: Optional[str]) -> dict:
    """Validates and stores the upload and puts it in the queue as `processing`. Scoring happens later
    in `process_study`. The scan is visible, and ranked as `unsure`, from this moment."""
    if len(data) > ctx.settings.max_upload_bytes:
        raise ApiError(413, f"The file is larger than {ctx.settings.max_upload_mb} MB.")
    try:
        img = images.load_image(data)
    except images.ImageError as e:
        raise ApiError(400, str(e)) from None
    name = audit.clean(filename, 120) or "upload"
    with _create_lock, ctx.db.session() as s:
        if (s.scalar(select(func.count()).select_from(Study)) or 0) >= config.MAX_STORED_STUDIES:
            raise ApiError(409, "The queue is full on this demo server. Restart the service to clear it.")
        seq = (s.scalar(select(func.max(Study.seq))) or 0) + 1
        study_id = f"CXR-{1000 + seq}"
        while s.get(Study, study_id) is not None:      # never reuse an id
            seq += 1
            study_id = f"CXR-{1000 + seq}"
        rel = f"images/{study_id}.jpg"
        images.to_display(img).save(ctx.settings.data_dir / rel, "JPEG", quality=90)
        s.add(Study(study_id=study_id, seq=seq, uploaded_at=now_utc(), filename=name, sha256=sha256_hex(data),
                    image_path=f"data:{rel}", status="processing", findings=[], model_source="none",
                    clinician_urgent=bool(clinician_urgent)))
        audit.record(s, actor=actor, action="upload", study_id=study_id,
                     text=f"File {name} added to the queue. Model mode: {ctx.model.active}."
                          + (" A clinician asked for an urgent read." if clinician_urgent else ""),
                     filename=name, clinician_urgent=bool(clinician_urgent), model_mode=ctx.model.active, bytes=len(data))
    return get_study(ctx, study_id)


def _apply(ctx: AppContext, row: Study, rec: PredictionRecord) -> None:
    lanes = ctx.lanes()
    row.status = "done"
    row.findings = rec.findings
    row.composite_score = rec.composite_score
    row.ood_flag = rec.ood_flag
    row.ood_score = rec.ood.score if rec.ood else None
    row.ood_reason = rec.ood.reason if rec.ood_flag else None
    row.model_source = rec.source
    row.model_version = rec.model_version
    row.model_note = rec.note
    row.model_priority = queue.model_priority(rec.composite_score, rec.ood_flag, lanes)
    row.confidence_band = queue.confidence_band(rec.composite_score, rec.ood_flag, lanes)


def process_study(ctx: AppContext, study_id: str, data: bytes) -> None:
    """Runs after the 202 response. Whatever happens, the study leaves `processing` and stays in the
    queue. If the model fails, the scan has no model output and ranks as `unsure`, never `normal`."""
    try:
        pil = images.load_image(data)
        rec = ctx.model.predict(data, pil)
        heat_rel = None
        if rec.cam is not None:
            with ctx.db.session() as s:
                disp = _resolve(ctx, _row(s, study_id).image_path)
            if disp is not None:
                from PIL import Image
                with Image.open(disp) as d:
                    size = d.size
                heat_rel = f"heatmaps/{study_id}.png"
                heatmap.overlay_rgba(rec.cam, size).save(ctx.settings.data_dir / heat_rel, "PNG")
        with ctx.db.session() as s:
            row = _row(s, study_id)
            _apply(ctx, row, rec)
            row.heatmap_path = f"data:{heat_rel}" if heat_rel else None
            top = rec.findings[0] if rec.findings else None
            audit.record(
                s, actor=audit.SYSTEM, action="scoring", study_id=study_id,
                text=(f"Model result ({rec.source}): {row.model_priority}"
                      + (f", top finding {top['name']} {top['probability']:.2f}" if top else ", no finding scores")
                      + (f", out of distribution: {rec.ood.reason}" if rec.ood_flag else "")
                      + (f". {rec.note}" if rec.note else "") + "."),
                source=rec.source, model_version=rec.model_version, model_priority=row.model_priority,
                composite_score=rec.composite_score, ood_flag=rec.ood_flag, threshold=ctx.threshold)
    except Exception as e:  # noqa: BLE001
        log.exception("processing %s failed; it stays in the queue as unsure", study_id)
        with ctx.db.session() as s:
            row = s.get(Study, study_id)
            if row is not None:
                _apply(ctx, row, PredictionRecord(source="none", note=f"processing failed ({type(e).__name__})"))
                audit.record(s, actor=audit.SYSTEM, action="scoring", study_id=study_id,
                             text="Processing failed. No model output: the scan stays in the queue as AI unsure.",
                             source="none", error=type(e).__name__)


def recover_interrupted(ctx: AppContext) -> int:
    """A restart loses background tasks. Scans left in `processing` are released as unscored (`unsure`)."""
    n = 0
    with ctx.db.session() as s:
        for row in s.scalars(select(Study).where(Study.status == "processing")).all():
            _apply(ctx, row, PredictionRecord(source="none", note="processing was interrupted by a restart"))
            audit.record(s, actor=audit.SYSTEM, action="scoring", study_id=row.study_id,
                         text="Processing was interrupted by a restart. No model output: the scan stays in the queue as AI unsure.",
                         source="none", error="interrupted")
            n += 1
    return n


def review(ctx: AppContext, study_id: str, action: str, reason: Optional[str], notes: Optional[str],
           human_priority: Optional[str], actor: Optional[str]) -> dict:
    if action not in ("mark_reviewed", "flag_disagreement"):
        raise ApiError(400, "action must be mark_reviewed or flag_disagreement")
    if human_priority is not None and human_priority not in queue.PRIORITIES:
        raise ApiError(400, "human_priority must be raised, normal or unsure")
    with ctx.db.session() as s:
        row = _row(s, study_id)
        if row.status == "processing":
            raise ApiError(409, "This study is still being processed.")
        if row.reviewed_at is not None:
            raise ApiError(409, "This study has already been reviewed.")
        row.reviewed_at = now_utc()
        if action == "mark_reviewed":
            text = "Reviewer marked the scan as read." + (f" Note: {audit.clean(notes, 1000)}" if audit.clean(notes) else "")
        else:
            row.flagged_disagreement = True
            text = (f"Reviewer disagreed with the model. Reason: {audit.clean(reason) or 'not given'}. "
                    f"Human priority: {human_priority or 'not given'}." + (f" Notes: {audit.clean(notes, 1000)}" if audit.clean(notes) else ""))
        audit.record(s, actor=actor, action=action, study_id=study_id, text=text, reason=audit.clean(reason) or None,
                     human_priority=human_priority, model_priority=queue.model_priority(row.composite_score, row.ood_flag, ctx.lanes()))
    return get_study(ctx, study_id)


def urgent_request(ctx: AppContext, study_id: str, note: Optional[str], actor: Optional[str]) -> dict:
    with ctx.db.session() as s:
        row = _row(s, study_id)
        if row.reviewed_at is not None:
            raise ApiError(409, "This study has already been reviewed.")
        if not row.clinician_urgent:
            row.clinician_urgent = True
            audit.record(s, actor=actor, action="urgent_request", study_id=study_id,
                         text=audit.clean(note, 1000) or "Clinician requested an urgent read.")
    return get_study(ctx, study_id)


def override(ctx: AppContext, study_id: str, priority: str, reason: Optional[str], actor: Optional[str]) -> dict:
    if priority not in queue.PRIORITIES:
        raise ApiError(400, "priority must be raised, normal or unsure")
    why = audit.clean(reason, 1000)
    if not why:
        raise ApiError(400, "A reason is required for an override.")
    with ctx.db.session() as s:
        row = _row(s, study_id)
        if row.reviewed_at is not None:
            raise ApiError(409, "This study has already been reviewed.")
        if priority == "normal" and row.ood_flag:
            raise ApiError(409, "This image looks unlike the training data, so it cannot be set to routine. "
                                "A clinician must read it and mark it reviewed.")
        before = queue.effective_priority(
            None if row.status == "processing" else queue.model_priority(row.composite_score, row.ood_flag, ctx.lanes()),
            row.override_priority)
        row.override_priority = priority
        row.override_reason = why
        audit.record(s, actor=actor, action="override", study_id=study_id,
                     text=f"Queue priority set to {priority} (was {before}). Reason: {why}",
                     priority=priority, previous=before, reason=why)
    return get_study(ctx, study_id)
