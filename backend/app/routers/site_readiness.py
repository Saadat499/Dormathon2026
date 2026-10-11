"""Site Readiness: background job plus the live sample-size helper.

Route order matters: `/site-readiness/sample-size` is declared before `/site-readiness/{job_id}`.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile

from .. import config, schemas
from ..context import AppContext, get_ctx
from ..errors import ApiError
from ..services import site_readiness
from ..uploads import read_upload

router = APIRouter(tags=["site readiness"])


@router.get("/site-readiness/sample-size", response_model=schemas.SampleSizeOut)
def sample_size(target_sensitivity: float = Query(...), expected_sensitivity: float = Query(...),
                confidence: float = 0.95, prevalence: float = 0.2) -> dict:
    return site_readiness.sample_size(target_sensitivity, expected_sensitivity, confidence, prevalence)


@router.post("/site-readiness", status_code=202, response_model=schemas.JobOut)
async def start(background: BackgroundTasks, files: List[UploadFile] = File(default=[]),
                labels: Optional[UploadFile] = File(default=None), actor: Optional[str] = Form(None),
                ctx: AppContext = Depends(get_ctx)) -> dict:
    if not files and labels is None:
        raise ApiError(400, "Upload sample images and, if you have them, a labels CSV.")
    if len(files) > config.MAX_SITE_FILES:
        raise ApiError(400, f"At most {config.MAX_SITE_FILES} images per check.")
    payload, total = [], 0
    for f in files:
        data = await read_upload(f, ctx.settings.max_upload_bytes)
        total += len(data)
        if total > config.MAX_SITE_TOTAL_MB * 1024 * 1024:
            raise ApiError(413, f"The sample is larger than {config.MAX_SITE_TOTAL_MB} MB in total.")
        payload.append((f.filename or f"image-{len(payload) + 1}", data))
    labels_text = None
    if labels is not None:
        raw = await read_upload(labels, 2 * 1024 * 1024)
        try:
            labels_text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ApiError(400, "The labels CSV must be UTF-8 text.") from None
        try:
            site_readiness.parse_labels(labels_text)          # fail fast with a readable message
        except ValueError as e:
            raise ApiError(400, str(e)) from None
    job_id = site_readiness.start_job(ctx, len(payload), labels_text is not None, actor)
    background.add_task(site_readiness.run_job, ctx, job_id, payload, labels_text)
    return {"job_id": job_id}


@router.get("/site-readiness/{job_id}", response_model=schemas.SiteReadinessOut)
def result(job_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    return site_readiness.get_job(ctx, job_id)
