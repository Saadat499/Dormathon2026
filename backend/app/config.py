"""Runtime settings. Everything is read from environment variables so Render can override it."""
import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
APP_DIR = Path(__file__).resolve().parent

# Free-tier hosts have an ephemeral disk: this folder is wiped on redeploy and often on wake-up.
DATA_DIR = Path(os.getenv("VOXELSORT_DATA_DIR") or Path(tempfile.gettempdir()) / "voxelsort-data")
# Where Member A's model output is expected (predictions.json / predictions.csv, optional model files).
ARTIFACT_DIR = Path(os.getenv("VOXELSORT_ARTIFACT_DIR") or BASE_DIR / "model_artifacts")
EVIDENCE_PATH = APP_DIR / "static_evidence" / "locked_evidence.json"

VERSION = "0.1.0"

# Queue and safety constants (documented in docs/BACKEND_NOTES.md as assumptions).
MIN_SENSITIVITY_TARGET = 0.9      # contract example value; fixed first, false alarms minimised after
UNSURE_BAND = 0.15                # scores within this distance below the threshold are "unsure"
DEFAULT_MAX_WAIT_MINUTES = 120
MIN_POSITIVES_FOR_VERDICT = 30    # below this the site check says "not enough evidence"
DRIFT_WARNING_PSI = 0.10          # PSI >= 0.10 is a moderate shift (same cut-off B uses)

# Abuse and memory limits for a small free instance.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_STORED_STUDIES = 1000
MAX_SITE_FILES = 100
MAX_SITE_BYTES = 120 * 1024 * 1024
MAX_SIM_CASES = 3000

SEED_DEMO = os.getenv("VOXELSORT_SEED_DEMO", "1") not in ("0", "false", "False", "")

# Browsers allowed to call the API. Vercel preview/production domains match the regex.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()] + [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", r"https://([a-z0-9-]+\.)*vercel\.app")
