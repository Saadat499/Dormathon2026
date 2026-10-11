"""Everything one running app needs, built once at start-up and shared by the routers."""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

from fastapi import Request
from sqlalchemy.orm import Session

from .config import Settings
from .db import Database
from .models import Setting
from .services import audit
from .services.predictor import ModelService
from .services.queue import Lanes
from .services.saved_predictions import PredictionStore, load_store

log = logging.getLogger("voxelsort")
SCALE_NOTE = ("percentile of the composite score on the NIH validation (reference) split: "
              "value 0.41 raises every scan scoring above the 41st percentile of that set")


class AppContext:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.images_dir.mkdir(parents=True, exist_ok=True)
        settings.heatmaps_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(settings.database_url)
        self.db.create_all()
        self.predictions: PredictionStore = load_store(settings.artifacts_dir, self.db)
        self.model = ModelService(settings.artifacts_dir, settings.model_mode, settings.onnx_threads)
        self.sim_lock = threading.Lock()
        self.evidence: Optional[dict] = None
        self.unsure_budget = float(settings.unsure_budget if settings.unsure_budget is not None
                                   else (self.model.policy or {}).get("unsure_budget", 0.10))
        self._init_settings()

    # --- key/value settings ----------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self.db.session() as s:
            row = s.get(Setting, key)
            return default if row is None else row.value

    def set(self, s: Session, key: str, value: Any) -> None:
        row = s.get(Setting, key)
        if row is None:
            s.add(Setting(key=key, value=value))
        else:
            row.value = value

    @property
    def mode(self) -> str:
        return self.get("mode", "model")

    @property
    def max_wait_minutes(self) -> int:
        return int(self.get("max_wait_minutes", self.settings.default_max_wait_minutes))

    @property
    def threshold(self) -> Optional[float]:
        v = self.get("threshold")
        return None if v is None else float(v)

    @property
    def target(self) -> float:
        return self.settings.min_sensitivity_target

    def lanes(self) -> Lanes:
        """Score cut-offs for the current threshold. Without saved predictions the percentile scale
        does not exist; Member A's own fitted cut-offs (policy.json) are used if present, otherwise
        there are no cut-offs and every scored scan is `unsure`."""
        t = self.threshold
        if t is not None and self.predictions.available:
            return self.predictions.lanes(t, self.unsure_budget)
        pol = self.model.policy or {}
        if "t_raise" in pol and "t_unsure" in pol:
            return Lanes(float(pol["t_raise"]), float(pol["t_unsure"]))
        return Lanes(None, None)

    def lanes_source(self) -> str:
        if self.threshold is not None and self.predictions.available:
            return "threshold on the saved-prediction percentile scale"
        if "t_raise" in (self.model.policy or {}):
            return "Member A's policy.json cut-offs (saved predictions unavailable, threshold endpoints disabled)"
        return "none: every scored scan is 'unsure'"

    # --- start-up --------------------------------------------------------------------------
    def _init_settings(self) -> None:
        with self.db.session() as s:
            if s.get(Setting, "mode") is None:
                self.set(s, "mode", "model")
            if s.get(Setting, "max_wait_minutes") is None:
                self.set(s, "max_wait_minutes", int(self.settings.default_max_wait_minutes))
            self.set(s, "min_sensitivity_target", float(self.target))
            row = s.get(Setting, "threshold")
            current = None if row is None else row.value
            if not self.predictions.available:
                return
            if current is None:
                v = self.predictions.highest_value_meeting(self.target)
                self.set(s, "threshold", v)
                if v is None:
                    log.warning("no threshold meets the %.2f sensitivity target on the reference split", self.target)
                else:
                    log.info("default threshold %.2f (highest value meeting the %.2f sensitivity target)", v, self.target)
                return
            sens = self.predictions.sensitivity_at(float(current))
            if sens is None or sens < self.target:
                v = self.predictions.highest_value_meeting(self.target)
                self.set(s, "threshold", v)
                audit.record(s, actor=audit.SYSTEM, action="threshold_change", study_id=None,
                             text=(f"Stored threshold {float(current):.2f} no longer meets the minimum sensitivity target "
                                   f"{self.target:.2f} on the reference split; reset to {v if v is None else format(v, '.2f')} at start-up."),
                             old=current, new=v, reason="floor_violation_at_startup")

    def close(self) -> None:
        self.db.dispose()


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)
