"""OOD check demo and the Evidence page."""
from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse

from .. import schemas
from ..context import AppContext, get_ctx
from ..errors import ApiError
from ..services import evidence, images
from ..uploads import read_upload

router = APIRouter(tags=["evidence"])
OOD_NOTE = ("A flag makes a scan 'AI unsure' (moved up the queue, never down). No flag does not prove the image is like the "
            "training data: the detector catches gross mistakes, not subtle site differences.")


@router.post("/ood-check", response_model=schemas.OodOut)
async def ood_check(file: UploadFile = File(...), ctx: AppContext = Depends(get_ctx)) -> dict:
    data = await read_upload(file, ctx.settings.max_upload_bytes)

    def work() -> dict:
        try:
            pil = images.load_image(data)
        except images.ImageError as e:
            raise ApiError(400, str(e)) from None
        r = ctx.model.check_ood(data, pil)
        return {"ood_flag": r.ood_flag, "reason": r.reason, "score": r.score, "method": r.method, "source": r.source,
                "score_meaning": "Mahalanobis distance from the training features (higher = further away)" if r.source != "basic"
                                 else "0 to 1 strength of the basic checks",
                "score_threshold": r.threshold, "probe_probability": r.probe_probability, "note": OOD_NOTE}

    return await asyncio.to_thread(work)


@router.get("/evidence")
def get_evidence(request: Request, ctx: AppContext = Depends(get_ctx)) -> dict:
    if ctx.evidence is None:
        raise ApiError(404, "No evidence file is installed on this server.")
    return evidence.public(ctx.evidence, str(request.base_url))


@router.get("/evidence/failures/{name}")
def failure_image(name: str, ctx: AppContext = Depends(get_ctx)) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_]+\.jpg", name):
        raise ApiError(404, "Not found")
    p = ctx.settings.artifacts_dir / "demo" / "failures" / name
    if not p.is_file():
        raise ApiError(404, "Not found")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})
