"""Runtime settings, all read from environment variables (see backend/.env.example).

`get_settings()` builds a fresh Settings object from the environment; `create_app()` accepts one
explicitly so tests can point everything at a temporary folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
APP_DIR = Path(__file__).resolve().parent                     # backend/app/
VERSION = "1.0.0"
CONTRACT_VERSION = "0.4"

MODEL_MODES = ("auto", "real", "replay", "fake", "none")

# Fixed limits for a small single-process host (documented in docs/BACKEND_NOTES.md).
MAX_IMAGE_PIXELS = 30_000_000
MAX_IMAGE_SIDE = 1024            # stored display copy is downsampled to at most this many pixels per side
MAX_STORED_STUDIES = 1000
MAX_SITE_FILES = 100
MAX_SITE_TOTAL_MB = 120
MAX_SIM_HOURS = 24
MAX_SIM_CASES = 1500           # bounds the run time of one simulation on a 0.15-CPU host
MIN_POSITIVES_FOR_VERDICT = 30   # fewer labelled urgent scans than this is always "not enough evidence"
MIN_DRIFT_SAMPLE_FOR_NOT_READY = 50   # a drift NO-GO on a smaller sample is only a warning
MIN_DRIFT_SAMPLE = 20            # Member A's drift score refuses smaller samples
MAX_ACTOR_LEN = 80


def _env(name: str, *aliases: str, default: str | None = None) -> str | None:
    for key in (name, *aliases):
        v = os.getenv(key)
        if v is not None and v.strip() != "":
            return v.strip()
    return default


def _flag(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class Settings:
    env: str = "dev"                                   # dev | prod
    data_dir: Path = BACKEND_DIR / "data"              # SQLite file, stored images, heatmaps
    database_url: str = ""                             # default: sqlite file inside data_dir
    artifacts_dir: Path = BACKEND_DIR / "model_artifacts"
    allowed_origins: list[str] = field(default_factory=list)
    allowed_origin_regex: str | None = None
    max_upload_mb: int = 15
    default_max_wait_minutes: int = 120
    min_sensitivity_target: float = 0.9
    unsure_budget: float | None = None                 # None: take Member A's policy.json value (0.10)
    model_mode: str = "auto"                           # auto | real | replay | fake | none
    seed_demo: bool = True
    onnx_threads: int = 1
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        if not self.database_url:
            self.database_url = f"sqlite:///{(self.data_dir / 'voxelsort.db').as_posix()}"
        if self.model_mode not in MODEL_MODES:
            raise ValueError(f"MODEL_MODE must be one of {MODEL_MODES}, got {self.model_mode!r}")
        if not 0.0 < self.min_sensitivity_target < 1.0:
            raise ValueError("MIN_SENSITIVITY_TARGET must be between 0 and 1")
        if not 10 <= self.default_max_wait_minutes <= 1440:
            raise ValueError("DEFAULT_MAX_WAIT_MINUTES must be between 10 and 1440")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def heatmaps_dir(self) -> Path:
        return self.data_dir / "heatmaps"

    @property
    def cors_is_open(self) -> bool:
        return "*" in self.allowed_origins


def get_settings() -> Settings:
    env = (_env("APP_ENV", "ENV", default="dev") or "dev").lower()
    raw_origins = _env("ALLOWED_ORIGINS", "CORS_ORIGINS")
    if raw_origins is None:
        # Open CORS only on a developer machine. In production an explicit list is required.
        origins = ["*"] if env != "prod" else ["http://localhost:5173"]
    else:
        origins = [o.strip().rstrip("/") for o in raw_origins.split(",") if o.strip()]
    unsure = _env("UNSURE_BUDGET")
    return Settings(
        env=env,
        data_dir=Path(_env("DATA_DIR", "VOXELSORT_DATA_DIR", default=str(BACKEND_DIR / "data"))),
        database_url=_env("DATABASE_URL", default="") or "",
        artifacts_dir=Path(_env("MODEL_ARTIFACTS_DIR", "VOXELSORT_ARTIFACT_DIR", default=str(BACKEND_DIR / "model_artifacts"))),
        allowed_origins=origins,
        allowed_origin_regex=_env("ALLOWED_ORIGIN_REGEX", "CORS_ORIGIN_REGEX"),
        max_upload_mb=int(_env("MAX_UPLOAD_MB", default="15")),
        default_max_wait_minutes=int(_env("DEFAULT_MAX_WAIT_MINUTES", default="120")),
        min_sensitivity_target=float(_env("MIN_SENSITIVITY_TARGET", default="0.9")),
        unsure_budget=float(unsure) if unsure is not None else None,
        model_mode=(_env("MODEL_MODE", default="auto") or "auto").lower(),
        seed_demo=_flag(_env("SEED_DEMO", "VOXELSORT_SEED_DEMO"), True),
        onnx_threads=int(_env("ONNX_THREADS", default="1")),
        log_level=(_env("LOG_LEVEL", default="INFO") or "INFO").upper(),
    )
