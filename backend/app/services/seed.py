"""Demo seed: so the worklist is never blank after a restart (the free host has no persistent disk).

Seeded studies are public NIH ChestX-ray14 images from Member A's demo bundle with their SAVED model
outputs (`model_source: "saved_demo"`). The numbers are real model outputs; the arrival times are
staged relative to start-up, and each `plain_language` starts with "Demo record". Seeding is
idempotent: it only runs when the studies table is empty.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import func, select

from ..context import AppContext
from ..db import now_utc
from ..models import Study
from . import audit, queue
from .predictor import sorted_findings

log = logging.getLogger("voxelsort.seed")
MINUTES_AGO = [4, 9, 15, 23, 31, 42, 56, 71, 88, 104, 133, 165]   # one is past the default 120-minute wait cap
REVIEWED_INDEX = 11                                               # the oldest one has already been read


def seed_demo(ctx: AppContext) -> int:
    picks = sorted((r for r in ctx.model.demo.values() if r.get("seed_order") is not None), key=lambda r: r["seed_order"])
    picks = picks[:len(MINUTES_AGO)]
    if not picks:
        return 0
    lanes = ctx.lanes()
    now = now_utc()
    with ctx.db.session() as s:
        if (s.scalar(select(func.count()).select_from(Study)) or 0) > 0:
            return 0
        for k, (r, mins) in enumerate(zip(picks, MINUTES_AGO), start=1):
            up = now - timedelta(minutes=mins)
            ood = bool(r.get("ood_flag"))
            score = r.get("composite_score")
            s.add(Study(
                study_id=r["study_id"], seq=k, uploaded_at=up, filename=None, sha256=r.get("sha256"),
                image_path=f"artifacts:{r['image']}" if r.get("image") else None,
                heatmap_path=f"artifacts:{r['heatmap']}" if r.get("heatmap") else None,
                status="done", findings=sorted_findings(r.get("findings") or {}), composite_score=score,
                model_priority=queue.model_priority(score, ood, lanes), confidence_band=queue.confidence_band(score, ood, lanes),
                ood_flag=ood, ood_score=r.get("ood_score"), ood_reason=r.get("ood_reason") if ood else None,
                model_source="saved_demo", model_version=r.get("model_version") or ctx.model.version,
                reviewed_at=(up + timedelta(minutes=25)) if k - 1 == REVIEWED_INDEX else None,
            ))
        audit.record(s, actor=audit.SYSTEM, action="upload", study_id=None,
                     text=(f"Demo seed at start-up: {len(picks)} public NIH demo images with their saved model outputs were placed "
                           "in the queue with staged arrival times."), seeded=len(picks))
    log.info("seeded %d demo studies", len(picks))
    return len(picks)
