"""Queue mode (model | fcfs) and the wait cap. `fcfs` is the fallback switch: it applies on the next read."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import schemas
from ..context import AppContext, get_ctx
from ..errors import ApiError
from ..services import audit, queue

router = APIRouter(tags=["queue settings"])


@router.get("/queue/settings", response_model=schemas.QueueSettingsOut)
def get_settings(ctx: AppContext = Depends(get_ctx)) -> dict:
    return {"mode": ctx.mode, "max_wait_minutes": ctx.max_wait_minutes}


@router.put("/queue/settings", response_model=schemas.QueueSettingsOut)
def put_settings(body: schemas.QueueSettingsIn, ctx: AppContext = Depends(get_ctx)) -> dict:
    if body.mode not in queue.MODES:
        raise ApiError(400, "mode must be model or fcfs")
    if not 10 <= body.max_wait_minutes <= 1440:
        raise ApiError(400, "max_wait_minutes must be between 10 and 1440")
    old = {"mode": ctx.mode, "max_wait_minutes": ctx.max_wait_minutes}
    new = {"mode": body.mode, "max_wait_minutes": int(body.max_wait_minutes)}
    with ctx.db.session() as s:
        ctx.set(s, "mode", new["mode"])
        ctx.set(s, "max_wait_minutes", new["max_wait_minutes"])
        audit.record(s, actor=body.actor, action="queue_settings", study_id=None,
                     text=(f"Queue mode {new['mode']}, wait cap {new['max_wait_minutes']} min "
                           f"(was {old['mode']}, {old['max_wait_minutes']} min)."
                           + (" First-come-first-served fallback: the model is ignored." if new["mode"] == "fcfs" else "")),
                     old=old, new=new)
    return {"mode": ctx.mode, "max_wait_minutes": ctx.max_wait_minutes}
