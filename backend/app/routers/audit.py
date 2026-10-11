"""Audit log, as JSON and as a CSV download. Newest first."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from .. import schemas
from ..context import AppContext, get_ctx
from ..services import audit

router = APIRouter(tags=["audit"])


@router.get("/audit", response_model=List[schemas.AuditOut])
def get_audit(action: Optional[str] = None, q: Optional[str] = None, limit: int = Query(200, ge=1, le=1000),
              ctx: AppContext = Depends(get_ctx)) -> list:
    with ctx.db.session() as s:
        return audit.query(s, action, q, limit)


@router.get("/audit.csv")
def get_audit_csv(action: Optional[str] = None, q: Optional[str] = None, ctx: AppContext = Depends(get_ctx)) -> Response:
    with ctx.db.session() as s:
        rows = audit.query(s, action, q, 100000)
    return Response(audit.to_csv(rows), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="voxelsort-audit.csv"'})
