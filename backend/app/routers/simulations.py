"""Queue Race simulation."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Body, Depends

from .. import schemas
from ..context import AppContext, get_ctx
from ..services import simulation

router = APIRouter(tags=["queue race"])


@router.post("/simulations/queue-race", response_model=schemas.QueueRaceOut)
async def queue_race(params: Any = Body(...), ctx: AppContext = Depends(get_ctx)) -> dict:
    return await asyncio.to_thread(simulation.run_queue_race, ctx, params)
