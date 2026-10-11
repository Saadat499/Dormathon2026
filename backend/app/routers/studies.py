"""Worklist, upload, images, review, urgent requests and overrides."""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from .. import schemas
from ..context import AppContext, get_ctx
from ..errors import ApiError
from ..services import studies
from ..uploads import read_upload

router = APIRouter(tags=["worklist"])
MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
OK_TYPES = ("application/octet-stream", "application/dicom", "binary/octet-stream")


@router.get("/studies", response_model=schemas.StudyListOut)
def list_studies(priority: Optional[str] = None, status: Optional[str] = None, q: Optional[str] = None,
                 ctx: AppContext = Depends(get_ctx)) -> dict:
    return studies.list_studies(ctx, priority, status, q)


@router.post("/studies", status_code=202, response_model=schemas.StudyOut)
async def upload_study(background: BackgroundTasks, file: UploadFile = File(...), clinician_urgent: bool = Form(False),
                       actor: Optional[str] = Form(None), ctx: AppContext = Depends(get_ctx)) -> dict:
    ctype = (file.content_type or "").lower()
    if ctype and not (ctype.startswith("image/") or ctype in OK_TYPES):
        raise ApiError(400, "Only PNG and JPG images are supported.")
    data = await read_upload(file, ctx.settings.max_upload_bytes)
    study = await asyncio.to_thread(studies.create_study, ctx, data, file.filename or "upload", clinician_urgent, actor)
    background.add_task(studies.process_study, ctx, study["study_id"], data)
    return study


@router.get("/studies/{study_id}", response_model=schemas.StudyOut)
def get_study(study_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    return studies.get_study(ctx, study_id)


@router.get("/studies/{study_id}/image")
def get_image(study_id: str, ctx: AppContext = Depends(get_ctx)) -> FileResponse:
    p = studies.image_file(ctx, study_id)
    return FileResponse(p, media_type=MEDIA.get(p.suffix.lower(), "application/octet-stream"),
                        headers={"Cache-Control": "private, max-age=300", "Vary": "Origin"})


@router.get("/studies/{study_id}/heatmap")
def get_heatmap(study_id: str, ctx: AppContext = Depends(get_ctx)) -> FileResponse:
    p = studies.heatmap_file(ctx, study_id)
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "private, max-age=300", "Vary": "Origin"})


@router.post("/studies/{study_id}/review", response_model=schemas.StudyOut)
def review(study_id: str, body: schemas.ReviewIn, ctx: AppContext = Depends(get_ctx)) -> dict:
    return studies.review(ctx, study_id, body.action, body.reason, body.notes, body.human_priority, body.actor)


@router.post("/studies/{study_id}/urgent-request", response_model=schemas.StudyOut)
def urgent_request(study_id: str, body: schemas.UrgentIn, ctx: AppContext = Depends(get_ctx)) -> dict:
    return studies.urgent_request(ctx, study_id, body.note, body.actor)


@router.post("/studies/{study_id}/override", response_model=schemas.StudyOut)
def override(study_id: str, body: schemas.OverrideIn, ctx: AppContext = Depends(get_ctx)) -> dict:
    return studies.override(ctx, study_id, body.priority, body.reason, body.actor)
