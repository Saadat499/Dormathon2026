"""VoxelSort API. Endpoints follow docs/API_CONTRACT.md; deviations are listed there and in docs/BACKEND_NOTES.md.

Run with ONE worker (SQLite + in-process model):  uvicorn app.main:app --workers 1
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config
from .config import Settings, get_settings
from .context import AppContext
from .errors import ApiError
from .routers import audit, evidence, health, queue_settings, simulations, site_readiness, studies, threshold
from .services import evidence as evidence_service
from .services import seed as seed_service
from .services import studies as studies_service

log = logging.getLogger("voxelsort")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"t": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname, "logger": record.name,
               "msg": record.getMessage()}
        for k in ("method", "path", "status", "ms"):
            if hasattr(record, k):
                out[k] = getattr(record, k)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def setup_logging(level: str) -> None:
    root = logging.getLogger("voxelsort")
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        root.addHandler(h)
        root.propagate = False
    root.setLevel(level)


class TooLarge(Exception):
    pass


class GuardMiddleware:
    """Pure ASGI middleware, innermost: caps the request body, logs each request, and turns an
    unexpected error into {"detail": ...} while still inside the CORS layer (so a browser can read it)."""

    def __init__(self, app, max_body_bytes: int):
        self.app, self.max_body = app, max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started, status, seen = False, 500, 0
        t0 = time.perf_counter()

        async def limited_receive():
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max_body:
                    raise TooLarge()
            return msg

        async def tracked_send(msg):
            nonlocal started, status
            if msg["type"] == "http.response.start":
                started, status = True, msg["status"]
            await send(msg)

        async def reply(code: int, detail: str):
            body = json.dumps({"detail": detail}).encode()
            await tracked_send({"type": "http.response.start", "status": code,
                                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
            await tracked_send({"type": "http.response.body", "body": body})

        try:
            declared = dict(scope.get("headers") or {}).get(b"content-length")
            if declared and declared.isdigit() and int(declared) > self.max_body:
                await reply(413, f"The request is larger than {self.max_body // (1024 * 1024)} MB.")
            else:
                await self.app(scope, limited_receive, tracked_send)
        except TooLarge:
            if not started:
                await reply(413, f"The request is larger than {self.max_body // (1024 * 1024)} MB.")
        except Exception:  # noqa: BLE001
            log.exception("unhandled error on %s %s", scope.get("method"), scope.get("path"))
            if not started:
                await reply(500, "The server hit an unexpected error. Nothing was changed by this request.")
        finally:
            log.info("request", extra={"method": scope.get("method"), "path": scope.get("path"), "status": status,
                                       "ms": round((time.perf_counter() - t0) * 1000, 1)})


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = AppContext(settings)
        ctx.evidence = evidence_service.load(settings.artifacts_dir)
        released = studies_service.recover_interrupted(ctx)
        seeded = seed_service.seed_demo(ctx) if settings.seed_demo else 0
        app.state.ctx = ctx
        log.info("started: model=%s (requested %s), saved predictions=%s, threshold=%s, seeded=%d, released=%d, journal=%s",
                 ctx.model.active, ctx.model.requested, ctx.predictions.source, ctx.threshold, seeded, released, ctx.db.journal_mode)
        for p in ctx.model.problems + ctx.predictions.problems:
            log.warning(p)
        try:
            yield
        finally:
            ctx.close()

    app = FastAPI(
        title="VoxelSort API", version=config.VERSION, lifespan=lifespan,
        description="Chest X-ray worklist prioritiser. Decision support only: a clinician reads every scan. "
                    f"Implements docs/API_CONTRACT.md v{config.CONTRACT_VERSION}.")

    app.add_middleware(GuardMiddleware, max_body_bytes=(config.MAX_SITE_TOTAL_MB + 8) * 1024 * 1024)
    app.add_middleware(                                   # added last = outermost: every response gets CORS headers
        CORSMiddleware,
        allow_origins=["*"] if settings.cors_is_open else settings.allowed_origins,
        allow_origin_regex=None if settings.cors_is_open else settings.allowed_origin_regex,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"], allow_headers=["Content-Type"], max_age=600,
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return JSONResponse({"detail": e.detail}, status_code=e.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, e: StarletteHTTPException):
        return JSONResponse({"detail": e.detail if isinstance(e.detail, str) else "Request failed"}, status_code=e.status_code)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, e: RequestValidationError):
        first = e.errors()[0]
        where = ".".join(str(p) for p in first["loc"] if p not in ("body", "query", "form"))
        return JSONResponse({"detail": f"{where}: {first['msg']}" if where else str(first["msg"])}, status_code=400)

    for r in (health.router, studies.router, queue_settings.router, threshold.router, simulations.router,
              site_readiness.router, evidence.router, audit.router):
        app.include_router(r)
    return app


app = create_app()
