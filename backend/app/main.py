"""VoxelSort API. Endpoints follow docs/API_CONTRACT.md; deviations are listed in docs/BACKEND_NOTES.md."""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from . import config, readiness
from .core import Backend, Rejected, actor_name
from .evidence import build_evidence
from .ood import ImageError, check_image, load_image
from .simulation import run_queue_race
from .stats import sample_size

log = logging.getLogger("voxelsort")

LIMIT_NOTE = f"Files larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB are rejected."


class Body(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ReviewBody(Body):
    action: str
    reason: Optional[str] = Field(None, max_length=500)
    notes: Optional[str] = Field(None, max_length=2000)
    human_priority: Optional[str] = None
    actor: Optional[str] = Field(None, max_length=80)


class UrgentBody(Body):
    note: Optional[str] = Field(None, max_length=1000)
    actor: Optional[str] = Field(None, max_length=80)


class OverrideBody(Body):
    priority: str
    reason: Optional[str] = Field(None, max_length=1000)
    actor: Optional[str] = Field(None, max_length=80)


class QueueBody(Body):
    mode: str
    max_wait_minutes: int
    actor: Optional[str] = Field(None, max_length=80)


class PreviewBody(Body):
    value: float = Field(allow_inf_nan=False)


class ThresholdBody(Body):
    value: float = Field(allow_inf_nan=False)
    confirm: bool = False
    actor: Optional[str] = Field(None, max_length=80)


def create_app(data_dir: Optional[Path] = None, artifact_dir: Optional[Path] = None, seed_demo: Optional[bool] = None) -> FastAPI:
    kwargs = {}
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    if artifact_dir is not None:
        kwargs["artifact_dir"] = artifact_dir
    if seed_demo is not None:
        kwargs["seed_demo"] = seed_demo

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = Backend(**kwargs)
        app.state.evidence = build_evidence(config.EVIDENCE_PATH)
        yield

    app = FastAPI(title="VoxelSort API", version=config.VERSION, lifespan=lifespan,
                  description="Chest X-ray worklist prioritiser. Decision support only: a clinician reads every scan.")
    app.add_middleware(
        CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_origin_regex=config.CORS_ORIGIN_REGEX,
        allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type"],
    )

    def be(request: Request) -> Backend:
        return request.app.state.backend

    @app.exception_handler(Rejected)
    async def _rejected(_: Request, e: Rejected):
        return JSONResponse({"detail": e.detail}, status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, e: RequestValidationError):
        first = e.errors()[0]
        where = ".".join(str(p) for p in first["loc"] if p not in ("body", "query", "form"))
        return JSONResponse({"detail": f"{where}: {first['msg']}" if where else first["msg"]}, status_code=400)

    async def read_upload(f: UploadFile, limit: int = config.MAX_UPLOAD_BYTES) -> bytes:
        data = await f.read(limit + 1)
        if len(data) > limit:
            raise Rejected(400, f"File {f.filename or ''} is larger than {limit // (1024 * 1024)} MB.")
        return data

    # Health and status ----------------------------------------------------------------------
    @app.get("/")
    def root():
        return {"service": "VoxelSort API", "health": "/health", "docs": "/docs", "status": "/status"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/status")
    def status(request: Request):
        b = be(request)
        ev = request.app.state.evidence
        return {
            "version": config.VERSION,
            "model": {"mode": b.provider.mode, "runs_a_trained_model": False, "description": b.provider.description},
            "saved_predictions": {
                "source": b.predictions.source, "real": b.predictions.is_real, "n": len(b.predictions.items),
                "n_labelled": len(b.predictions.labelled), "description": b.predictions.description, "problems": b.predictions.problems,
                "expected_in": str(config.ARTIFACT_DIR),
            },
            "evidence": {"loaded": ev is not None, "data": ev["provenance"]["data"] if ev else None},
            "ood_check": "placeholder heuristic, not Member A's detector",
            "storage": {"persistent": False, "note": "SQLite in an ephemeral folder: studies and the audit log reset on redeploy or restart on free hosting."},
            "queue": b.queue_settings(),
            "threshold": b.threshold_info(),
        }

    # Worklist -------------------------------------------------------------------------------
    @app.get("/studies")
    def list_studies(request: Request, priority: Optional[str] = None, status: Optional[str] = None, q: Optional[str] = None):
        return be(request).list_studies(priority, status, q)

    @app.post("/studies", status_code=202)
    async def upload_study(
        request: Request, background: BackgroundTasks, file: UploadFile = File(...),
        clinician_urgent: bool = Form(False), actor: Optional[str] = Form(None),
    ):
        b = be(request)
        data = await read_upload(file)
        study = await asyncio.to_thread(b.create_study, data, file.filename or "upload", clinician_urgent, actor)
        background.add_task(b.process_study, study["study_id"], data, file.filename or "upload")
        return study

    @app.get("/studies/{study_id}")
    def get_study(request: Request, study_id: str):
        return be(request).get_study(study_id)

    @app.get("/studies/{study_id}/image")
    def get_image(request: Request, study_id: str):
        return FileResponse(be(request).image_path(study_id), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})

    @app.get("/studies/{study_id}/heatmap")
    def get_heatmap(request: Request, study_id: str):
        return FileResponse(be(request).heatmap_path(study_id), media_type="image/png")

    # Review and overrides -------------------------------------------------------------------
    @app.post("/studies/{study_id}/review")
    def review(request: Request, study_id: str, body: ReviewBody):
        return be(request).review(study_id, body.action, body.reason, body.notes, body.human_priority, body.actor)

    @app.post("/studies/{study_id}/urgent-request")
    def urgent_request(request: Request, study_id: str, body: UrgentBody):
        return be(request).urgent_request(study_id, body.note, body.actor)

    @app.post("/studies/{study_id}/override")
    def override(request: Request, study_id: str, body: OverrideBody):
        return be(request).override(study_id, body.priority, body.reason, body.actor)

    # Queue settings and threshold -----------------------------------------------------------
    @app.get("/queue/settings")
    def get_queue_settings(request: Request):
        return be(request).queue_settings()

    @app.put("/queue/settings")
    def put_queue_settings(request: Request, body: QueueBody):
        return be(request).put_queue_settings(body.mode, body.max_wait_minutes, body.actor)

    @app.get("/threshold")
    def get_threshold(request: Request):
        return be(request).threshold_info()

    @app.post("/threshold/preview")
    def threshold_preview(request: Request, body: PreviewBody):
        return be(request).preview(body.value)

    @app.put("/threshold")
    def put_threshold(request: Request, body: ThresholdBody):
        return be(request).put_threshold(body.value, body.confirm, body.actor)

    # Queue Race -----------------------------------------------------------------------------
    @app.post("/simulations/queue-race")
    async def queue_race(request: Request, params: dict):
        b = be(request)
        try:
            return await asyncio.to_thread(run_queue_race, params, b.predictions, b.threshold, b.max_wait)
        except ValueError as e:
            raise Rejected(400, str(e)) from None

    # Site Readiness (the fixed path must be declared before the {job_id} path) ---------------
    @app.get("/site-readiness/sample-size")
    def site_sample_size(target_sensitivity: float = Query(...), expected_sensitivity: float = Query(...),
                         confidence: float = 0.95, prevalence: float = 0.2):
        try:
            return sample_size(target_sensitivity, expected_sensitivity, confidence, prevalence)
        except ValueError as e:
            raise Rejected(400, str(e)) from None

    @app.post("/site-readiness", status_code=202)
    async def site_readiness_start(
        request: Request, background: BackgroundTasks, files: List[UploadFile] = File(default=[]),
        labels: Optional[UploadFile] = File(default=None), actor: Optional[str] = Form(None),
    ):
        b = be(request)
        if not files and labels is None:
            raise Rejected(400, "Upload sample images and/or a labels CSV.")
        if len(files) > config.MAX_SITE_FILES:
            raise Rejected(400, f"At most {config.MAX_SITE_FILES} images per check.")
        checks, total, skipped = [], 0, 0
        for f in files:
            data = await read_upload(f)
            total += len(data)
            if total > config.MAX_SITE_BYTES:
                raise Rejected(400, f"The sample is larger than {config.MAX_SITE_BYTES // (1024 * 1024)} MB in total.")
            try:
                r = check_image(load_image(data))
                checks.append({"ood_flag": r.ood_flag, "score": r.score})
            except ImageError:
                skipped += 1
        labels_text = None
        if labels is not None:
            raw = await read_upload(labels, 2 * 1024 * 1024)
            try:
                labels_text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise Rejected(400, "The labels CSV must be UTF-8 text.") from None
            try:
                readiness.parse_labels(labels_text)  # fail fast with a readable message
            except ValueError as e:
                raise Rejected(400, str(e)) from None
        if files and not checks and labels_text is None:
            raise Rejected(400, "None of the uploaded files could be read as PNG or JPG images.")
        job_id = f"sr-{uuid.uuid4().hex[:12]}"
        with b.jobs_lock:
            b.jobs[job_id] = {"status": "processing"}
            while len(b.jobs) > 50:
                b.jobs.pop(next(iter(b.jobs)))
        b.store.add_audit(actor_name(actor), "site_readiness_run", None,
                          f"{len(checks)} readable images ({skipped} unreadable), labels {'provided' if labels_text else 'not provided'}. Job {job_id}.")

        def work():
            try:
                res = readiness.evaluate(image_checks=checks, labels_text=labels_text, ps=b.predictions, threshold=b.threshold)
                if skipped:
                    res["note"] = ((res["note"] or "") + f" {skipped} file(s) could not be read as PNG or JPG and were skipped.").strip()
            except Exception as e:  # noqa: BLE001
                log.exception("site readiness job failed")
                res = {"status": "done", "drift": {"score": None, "warning": False, "note": "The check failed."}, "sensitivity": None,
                       "target": config.MIN_SENSITIVITY_TARGET, "verdict": "not_enough_evidence", "note": f"The check could not be completed: {e}"}
            with b.jobs_lock:
                b.jobs[job_id] = res

        background.add_task(work)
        return {"job_id": job_id}

    @app.get("/site-readiness/{job_id}")
    def site_readiness_result(request: Request, job_id: str):
        b = be(request)
        with b.jobs_lock:
            res = b.jobs.get(job_id)
        if res is None:
            raise Rejected(404, "Job not found. The server may have restarted; run the check again.")
        return res

    # OOD demo and evidence ------------------------------------------------------------------
    @app.post("/ood-check")
    async def ood_check(file: UploadFile = File(...)):
        data = await read_upload(file)
        try:
            r = await asyncio.to_thread(lambda: check_image(load_image(data)))
        except ImageError as e:
            raise Rejected(400, str(e)) from None
        return {"ood_flag": r.ood_flag, "reason": r.reason, "score": r.score, "method": r.method,
                "note": "A flag makes a scan 'unsure' (moved up the queue). No flag does not prove the image is in-distribution."}

    @app.get("/evidence")
    def evidence(request: Request):
        ev = request.app.state.evidence
        if ev is None:
            raise Rejected(404, "No evidence file has been loaded.")
        return ev

    # Audit ----------------------------------------------------------------------------------
    @app.get("/audit")
    def audit(request: Request, action: Optional[str] = None, q: Optional[str] = None, limit: int = Query(200, ge=1, le=1000)):
        return be(request).store.list_audit(action, q, limit)

    @app.get("/audit.csv")
    def audit_csv(request: Request, action: Optional[str] = None, q: Optional[str] = None):
        rows = be(request).store.list_audit(action, q, 100000)
        out = io.StringIO()
        w = csv.writer(out)
        cols = ["id", "timestamp", "actor", "action", "study_id", "detail"]
        w.writerow(cols)

        def safe(v):  # stop spreadsheet programs from running user text as a formula
            s = "" if v is None else str(v)
            return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s

        for r in rows:
            w.writerow([r[c] if c in ("id", "timestamp") else safe(r[c]) for c in cols])
        return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="voxelsort-audit.csv"'})

    return app


app = create_app()
