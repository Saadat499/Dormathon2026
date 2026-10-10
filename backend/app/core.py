"""Application logic: studies, queue, overrides, threshold, audit. main.py only does HTTP."""
from __future__ import annotations

import io
import json
import logging
import re
import threading
from datetime import timedelta
from pathlib import Path
from typing import Optional

from . import config
from .cohorts import PredictionSet, highest_threshold_meeting, load_prediction_set, operating_point
from .model import make_provider
from .ood import ImageError, check_image, load_image, to_display
from .ranking import PRIORITY_ORDER, derive_priority, plain_language, public, rank_studies, urgency_score
from .store import Store, iso, now_utc, parse_iso

log = logging.getLogger("voxelsort")
PRIORITIES = tuple(PRIORITY_ORDER)


class Rejected(Exception):
    """A request the server understood but will not do. `status` is the HTTP code (400, 404, 409)."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def clean(text: Optional[str], limit: int = 500) -> str:
    """User-supplied text for the audit log: no control characters, bounded length."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).strip()[:limit]


def actor_name(a: Optional[str]) -> str:
    return clean(a, 80) or "unnamed"


class Backend:
    def __init__(self, data_dir: Path = config.DATA_DIR, artifact_dir: Path = config.ARTIFACT_DIR, seed_demo: bool = config.SEED_DEMO):
        self.data_dir = Path(data_dir)
        (self.data_dir / "images").mkdir(parents=True, exist_ok=True)
        self.store = Store(self.data_dir / "voxelsort.db")
        self.predictions: PredictionSet = load_prediction_set(Path(artifact_dir))
        self.provider = make_provider(self.predictions)
        self.jobs: dict[str, dict] = {}
        self.jobs_lock = threading.Lock()
        if self.store.get_setting("mode") is None:
            self.store.set_setting("mode", "model")
            self.store.set_setting("max_wait_minutes", config.DEFAULT_MAX_WAIT_MINUTES)
        if self.store.get_setting("threshold") is None:
            self.store.set_setting("threshold", highest_threshold_meeting(self.predictions, config.MIN_SENSITIVITY_TARGET) or 0.5)
        if seed_demo and self.store.count_studies() == 0:
            self._seed()

    # settings -------------------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self.store.get_setting("mode", "model")

    @property
    def max_wait(self) -> int:
        return int(self.store.get_setting("max_wait_minutes", config.DEFAULT_MAX_WAIT_MINUTES))

    @property
    def threshold(self) -> float:
        return float(self.store.get_setting("threshold", 0.5))

    def queue_settings(self) -> dict:
        return {"mode": self.mode, "max_wait_minutes": self.max_wait}

    def put_queue_settings(self, mode: str, max_wait: int, actor: Optional[str]) -> dict:
        if mode not in ("model", "fcfs"):
            raise Rejected(400, "mode must be model or fcfs")
        if not 10 <= max_wait <= 1440:
            raise Rejected(400, "max_wait_minutes must be between 10 and 1440")
        self.store.set_setting("mode", mode)
        self.store.set_setting("max_wait_minutes", int(max_wait))
        self.store.add_audit(actor_name(actor), "queue_settings", None, f"Mode {mode}, max wait {int(max_wait)} min.")
        return self.queue_settings()

    # threshold ------------------------------------------------------------------------------
    def threshold_info(self) -> dict:
        return {"value": self.threshold, "min_sensitivity_target": config.MIN_SENSITIVITY_TARGET}

    def preview(self, value: float) -> dict:
        if not 0.0 < value < 1.0:
            raise Rejected(400, "value must be between 0 and 1")
        op = operating_point(self.predictions, value, config.UNSURE_BAND)
        op.update(min_sensitivity_target=config.MIN_SENSITIVITY_TARGET, unsure_band=config.UNSURE_BAND,
                  data_source=self.predictions.source, data_is_real=self.predictions.is_real)
        return op

    def put_threshold(self, value: float, confirm: bool, actor: Optional[str]) -> dict:
        if confirm is not True:
            raise Rejected(400, "confirm must be true to change the threshold.")
        if not 0.0 < value < 1.0:
            raise Rejected(400, "value must be between 0 and 1")
        op = operating_point(self.predictions, value, config.UNSURE_BAND)
        if op["sensitivity"] is None:
            raise Rejected(400, "The saved predictions have no urgent scans, so the sensitivity at this threshold cannot be checked.")
        if op["sensitivity"] < config.MIN_SENSITIVITY_TARGET:
            raise Rejected(400, f"Sensitivity {op['sensitivity'] * 100:.0f}% at {value:.2f} is below the fixed minimum target of {config.MIN_SENSITIVITY_TARGET * 100:.0f}%.")
        old = self.threshold
        self.store.set_setting("threshold", value)
        self.store.add_audit(actor_name(actor), "threshold_change", None,
                             f"Threshold changed from {old:.2f} to {value:.2f}. Sensitivity on saved predictions {op['sensitivity'] * 100:.1f}%, "
                             f"false alarms {op['false_alarm_rate'] * 100:.1f}% ({self.predictions.source}).")
        return self.threshold_info()

    # studies --------------------------------------------------------------------------------
    def _view(self, row: dict, now) -> dict:
        findings = json.loads(row["findings"])
        processing = row["proc_state"] == "processing"
        ood = bool(row["ood_flag"])
        score = urgency_score(findings) if findings else None
        model_priority = None if processing else derive_priority(score, ood, self.threshold)
        effective = row["override_priority"] or model_priority or "unsure"  # not yet predicted: never low risk
        reviewed = row["reviewed_at"] is not None
        up = parse_iso(row["uploaded_at"])
        wait = ((parse_iso(row["reviewed_at"]) if reviewed else now) - up).total_seconds()
        over_cap = (not reviewed) and wait / 60 >= self.max_wait and self.mode == "model"
        if row["clinician_urgent"]:
            reason = "clinician_request"
        elif over_cap:
            reason = "wait_cap"
        elif row["override_priority"]:
            reason = "coordinator_override"
        else:
            reason = "model"
        if processing:
            status = "processing"
        elif reviewed:
            status = "reviewed"
        elif effective == "unsure":
            status = "needs_review"
        else:
            status = "awaiting_review"
        return {
            "study_id": row["study_id"],
            "findings": findings,
            "priority": model_priority,
            "confidence_band": "pending" if processing else (row["confidence_band"] or "n/a"),
            "ood_flag": ood,
            "heatmap_path": f"/studies/{row['study_id']}/heatmap" if row["has_heatmap"] else None,
            "uploaded_at": row["uploaded_at"],
            "wait_seconds": max(0, int(wait)),
            "rank": None,
            "effective_priority": effective,
            "priority_reason": reason,
            "clinician_urgent": bool(row["clinician_urgent"]),
            "status": status,
            "flagged_disagreement": bool(row["flagged_disagreement"]),
            "plain_language": plain_language(processing=processing, ood=ood, findings=findings, priority=model_priority, source=row["model_source"]),
            "_seq": row["seq"],
        }

    def ranked(self) -> list[dict]:
        now = now_utc()
        return rank_studies([self._view(r, now) for r in self.store.all_studies()], self.mode)

    def list_studies(self, priority: Optional[str] = None, status: Optional[str] = None, q: Optional[str] = None) -> dict:
        items = self.ranked()
        if priority:
            items = [s for s in items if s["effective_priority"] == priority]
        if status:
            items = [s for s in items if s["status"] == status]
        if q:
            items = [s for s in items if q.lower() in s["study_id"].lower()]
        return {"mode": self.mode, "max_wait_minutes": self.max_wait, "items": [public(s) for s in items]}

    def get_study(self, study_id: str) -> dict:
        for s in self.ranked():
            if s["study_id"] == study_id:
                return public(s)
        raise Rejected(404, f"Study {study_id} not found")

    def _row(self, study_id: str) -> dict:
        row = self.store.get_study(study_id)
        if not row:
            raise Rejected(404, f"Study {study_id} not found")
        return row

    def image_path(self, study_id: str) -> Path:
        row = self._row(study_id)
        p = self.data_dir / "images" / f"{row['study_id']}.jpg"
        if not row["has_image"] or not p.is_file():
            raise Rejected(404, "No image is stored for this study.")
        return p

    def heatmap_path(self, study_id: str) -> Path:
        row = self._row(study_id)
        p = self.data_dir / "images" / f"{row['study_id']}.heatmap.png"
        if not row["has_heatmap"] or not p.is_file():
            raise Rejected(404, "No heatmap has been generated for this study.")
        return p

    def create_study(self, data: bytes, filename: str, clinician_urgent: bool, actor: Optional[str]) -> dict:
        if self.store.count_studies() >= config.MAX_STORED_STUDIES:
            raise Rejected(409, "The demo queue is full. Restart the service to clear it.")
        if not data:
            raise Rejected(400, "The uploaded file is empty.")
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise Rejected(400, f"File is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        try:
            img = load_image(data)
        except ImageError as e:
            raise Rejected(400, str(e)) from None
        seq = self.store.next_seq()
        study_id = f"CXR-{1000 + seq}"
        to_display(img).save(self.data_dir / "images" / f"{study_id}.jpg", "JPEG", quality=88)
        self.store.insert_study({
            "study_id": study_id, "seq": seq, "uploaded_at": iso(now_utc()), "filename": clean(filename, 120), "has_image": 1,
            "findings": "[]", "ood_flag": 0, "model_source": "none", "proc_state": "processing",
            "clinician_urgent": 1 if clinician_urgent else 0,
        })
        self.store.add_audit(actor_name(actor), "upload", study_id,
                             f"File {clean(filename, 120)}. Model: {self.provider.mode}.{' Clinician asked for urgency.' if clinician_urgent else ''}")
        return self.get_study(study_id)

    def process_study(self, study_id: str, data: bytes, filename: str) -> None:
        """Runs after the 202 response. Whatever happens, the study leaves `processing` as `unsure`, never `normal`."""
        try:
            ood = check_image(load_image(data))
            pred = self.provider.predict(data, filename)
            self.store.update_study(
                study_id, proc_state="done", findings=json.dumps(pred.findings if pred else []),
                confidence_band=pred.confidence_band if pred else None,
                ood_flag=1 if (ood.ood_flag or (pred and pred.ood_flag)) else 0, ood_reason=ood.reason,
                model_source=pred.source if pred else "none",
            )
        except Exception:  # noqa: BLE001
            log.exception("processing %s failed; leaving it as unsure", study_id)
            self.store.update_study(study_id, proc_state="done", findings="[]", ood_flag=1, ood_reason="processing failed", model_source="none")

    def review(self, study_id: str, action: str, reason: Optional[str], notes: Optional[str], human_priority: Optional[str], actor: Optional[str]) -> dict:
        row = self._row(study_id)
        if action not in ("mark_reviewed", "flag_disagreement"):
            raise Rejected(400, "action must be mark_reviewed or flag_disagreement")
        if human_priority is not None and human_priority not in PRIORITIES:
            raise Rejected(400, "human_priority must be raised, normal or unsure")
        if row["proc_state"] == "processing":
            raise Rejected(409, "This study is still being processed.")
        if row["reviewed_at"]:
            raise Rejected(409, "This study has already been reviewed.")
        fields: dict = {"reviewed_at": iso(now_utc())}
        if action == "mark_reviewed":
            detail = "Reviewer acknowledged the AI output."
        else:
            fields["flagged_disagreement"] = 1
            detail = f"{clean(reason) or 'No reason'}. Human priority: {human_priority or 'n/a'}. {clean(notes, 1000)}".strip()
        self.store.update_study(study_id, **fields)
        self.store.add_audit(actor_name(actor), action, study_id, detail)
        return self.get_study(study_id)

    def urgent_request(self, study_id: str, note: Optional[str], actor: Optional[str]) -> dict:
        row = self._row(study_id)
        if row["reviewed_at"]:
            raise Rejected(409, "This study has already been reviewed.")
        if not row["clinician_urgent"]:
            self.store.update_study(study_id, clinician_urgent=1)
            self.store.add_audit(actor_name(actor), "urgent_request", study_id, clean(note, 1000) or "Clinician requested an urgent read.")
        return self.get_study(study_id)

    def override(self, study_id: str, priority: str, reason: Optional[str], actor: Optional[str]) -> dict:
        row = self._row(study_id)
        if priority not in PRIORITIES:
            raise Rejected(400, "priority must be raised, normal or unsure")
        if not clean(reason):
            raise Rejected(400, "A reason is required for an override.")
        if row["reviewed_at"]:
            raise Rejected(409, "This study has already been reviewed.")
        if priority == "normal" and row["ood_flag"]:
            raise Rejected(409, "This image looks unlike the training data, so it cannot be set to routine. A clinician must read it and mark it reviewed.")
        self.store.update_study(study_id, override_priority=priority)
        self.store.add_audit(actor_name(actor), "override", study_id, f"Priority set to {priority}. Reason: {clean(reason, 1000)}")
        return self.get_study(study_id)

    # demo seed ------------------------------------------------------------------------------
    def _seed(self) -> None:
        """A few queue entries so the worklist is not empty. Always labelled in `plain_language`.
        Replayed from A's saved predictions when present, otherwise synthetic. Arrival times are staged."""
        items = sorted(self.predictions.items, key=lambda i: -(i.score or 0))
        if len(items) < 8:
            return
        step = len(items) / 8
        pick = [items[int(i * step)] for i in range(8)]
        minutes_ago = [6, 14, 22, 35, 51, 78, 131, 160]
        source = "replay" if self.predictions.is_real else "synthetic_demo"
        now = now_utc()
        for k, (it, mins) in enumerate(zip(pick, minutes_ago)):
            up = now - timedelta(minutes=mins)
            self.store.insert_study({
                "study_id": it.study_id, "seq": self.store.next_seq(), "uploaded_at": iso(up), "filename": None, "has_image": 0,
                "findings": json.dumps(it.findings), "confidence_band": it.confidence_band, "ood_flag": 1 if it.ood_flag else 0,
                "model_source": source, "proc_state": "done",
                "reviewed_at": iso(up + timedelta(minutes=20)) if k == 7 else None,
            })
