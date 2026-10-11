"""Health and status. `/health` keeps the contract's {"status": "ok"} and adds what is running."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import config, schemas
from ..context import AppContext, get_ctx
from ..services import threshold

router = APIRouter(tags=["health"])
STORAGE_NOTE = ("SQLite and uploaded images live on the instance's local disk. On free hosting that disk is wiped on every "
                "redeploy, restart and idle spin-down: uploads, overrides, the audit log and threshold changes are lost and "
                "the demo queue is re-seeded.")


def status_body(ctx: AppContext) -> dict:
    db_ok = ctx.db.ping()
    model = ctx.model.describe()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": {"reachable": db_ok, "journal_mode": ctx.db.journal_mode, "workers": 1},
        "model_mode": model["active"],
        "model": model,
        "saved_predictions": ctx.predictions.summary(),
        "evidence": {"loaded": ctx.evidence is not None,
                     "data": (ctx.evidence or {}).get("provenance", {}).get("data") if ctx.evidence else None},
        "degraded": (not db_ok) or model["degraded"] or not ctx.predictions.available,
        "version": config.VERSION,
        "contract_version": config.CONTRACT_VERSION,
        "storage": {"persistent": False, "note": STORAGE_NOTE},
    }


@router.get("/")
def root() -> dict:
    return {"service": "VoxelSort API", "health": "/health", "status": "/status", "docs": "/docs",
            "note": "Decision support only. A clinician reads every scan."}


@router.get("/health", response_model=schemas.HealthOut)
def health(ctx: AppContext = Depends(get_ctx)) -> dict:
    return status_body(ctx)


@router.get("/status", response_model=schemas.HealthOut)
def status(ctx: AppContext = Depends(get_ctx)) -> dict:
    body = status_body(ctx)
    body["queue"] = {"mode": ctx.mode, "max_wait_minutes": ctx.max_wait_minutes}
    body["threshold"] = threshold.info(ctx)
    body["limits"] = {"max_upload_mb": ctx.settings.max_upload_mb, "max_site_files": config.MAX_SITE_FILES,
                      "max_site_total_mb": config.MAX_SITE_TOTAL_MB, "max_sim_cases": config.MAX_SIM_CASES}
    body["cors"] = {"open_to_any_origin": ctx.settings.cors_is_open}
    return body
