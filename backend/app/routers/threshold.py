"""Safety vs Speed threshold: preview (changes nothing) and guarded change."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import schemas
from ..context import AppContext, get_ctx
from ..services import threshold

router = APIRouter(tags=["threshold"])


@router.get("/threshold", response_model=schemas.ThresholdOut)
def get_threshold(ctx: AppContext = Depends(get_ctx)) -> dict:
    return threshold.info(ctx)


@router.post("/threshold/preview", response_model=schemas.ThresholdPreviewOut)
def preview(body: schemas.ThresholdPreviewIn, ctx: AppContext = Depends(get_ctx)) -> dict:
    return threshold.preview(ctx, body.value)


@router.put("/threshold", response_model=schemas.ThresholdOut)
def put_threshold(body: schemas.ThresholdIn, ctx: AppContext = Depends(get_ctx)) -> dict:
    return threshold.update(ctx, body.value, body.confirm, body.actor)
