"""Shared test set-up.

Every test app gets its own temporary database and upload folder (`tmp_path`). Most tests run
against a small SYNTHETIC artifacts folder built here (`make_artifacts`): it is labelled
`synthetic_test_fixture` in SOURCE.json and the API reports it as not real. Tests that exercise
Member A's real model use the `real_client` fixture and are skipped when the model files are absent.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.dont_write_bytecode = True

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

REAL_ARTIFACTS = BACKEND / "model_artifacts"
FINDINGS = ["atelectasis", "cardiomegaly", "consolidation", "edema", "effusion", "pneumonia", "pneumothorax"]
URGENT = ["pneumothorax", "effusion", "edema", "consolidation", "pneumonia"]


def cxr_png(w: int = 320, h: int = 400, seed: int = 0, colour: bool = False) -> bytes:
    """A made-up, roughly upright, left-right symmetric grayscale picture. Not a medical image."""
    rng = np.random.RandomState(seed)
    x = np.linspace(-1, 1, w)[None, :]
    y = np.linspace(-1, 1, h)[:, None]
    a = 120 + 60 * np.exp(-((x * 3) ** 2)) * (1 - 0.3 * y) + rng.normal(0, 6, (h, w))
    a = np.clip(a, 0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    if colour:
        img = Image.merge("RGB", (img, Image.fromarray(255 - a), Image.fromarray(a // 2)))
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def make_artifacts(folder: Path, n: int = 800, seed: int = 3) -> Path:
    """Synthetic saved predictions + three synthetic 'demo' images with made-up saved outputs."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lines = ["study_id,split,label_any_urgent,composite_score," + ",".join(f"p_{f}" for f in FINDINGS) + ",ood_flag"]
    for i in range(n):
        split = "val" if i % 2 == 0 else "test"
        pos = rng.random() < 0.25
        score = float(np.clip(rng.beta(5, 3) if pos else rng.beta(2, 6), 0.001, 0.999))
        probs = {f: round(score * float(rng.random()) * 0.8, 5) for f in FINDINGS}
        probs["pneumothorax"] = round(score, 5)
        ood = i % 97 == 0
        lines.append(f"FX-{i:04d},{split},{int(pos)},{score:.5f}," + ",".join(str(probs[f]) for f in FINDINGS) + f",{ood}")
    with gzip.open(folder / "saved_predictions_nih.csv.gz", "wt", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    (folder / "policy.json").write_text(json.dumps({"findings": FINDINGS, "urgent": URGENT, "unsure_budget": 0.1, "raise_budget": 0.2,
                                                    "t_raise": 0.5, "t_unsure": 0.4}))
    (folder / "SOURCE.json").write_text(json.dumps({"data": "synthetic_test_fixture", "model_version": "test-fixture-0",
                                                    "artifacts_folder": "none", "dataset": "synthetic", "findings": FINDINGS, "urgent": URGENT}))
    (folder / "demo" / "images").mkdir(parents=True, exist_ok=True)
    studies = []
    plan = [("DEMO-HIGH", 0.91, False, 1), ("DEMO-LOW", 0.02, False, 2), ("DEMO-OOD", None, True, 3)]
    for k, (sid, score, ood, order) in enumerate(plan):
        data = cxr_png(seed=100 + k)
        (folder / "demo" / "images" / f"{sid}.png").write_bytes(data)
        findings = {} if ood else {f: (score if f == "pneumothorax" else round(score * 0.3, 4)) for f in FINDINGS}
        studies.append({"study_id": sid, "category": "fixture", "dataset": "synthetic", "label_any_urgent": None, "true_findings": None,
                        "findings": findings, "composite_score": score, "top_urgent_finding": None if ood else "pneumothorax",
                        "ood_flag": ood, "ood_score": 400.0 if ood else 100.0, "ood_reason": "test fixture: flagged" if ood else None,
                        "sha256": hashlib.sha256(data).hexdigest(), "image": f"demo/images/{sid}.png", "heatmap": None,
                        "model_version": "test-fixture-0", "seed_order": order})
    (folder / "demo" / "demo_studies.json").write_text(json.dumps({"model_version": "test-fixture-0", "studies": studies}))
    return folder


def settings_for(tmp_path: Path, artifacts: Path, **over) -> Settings:
    base = dict(env="dev", data_dir=tmp_path / "data", artifacts_dir=artifacts, allowed_origins=["*"], model_mode="auto",
                seed_demo=False, log_level="WARNING")
    base.update(over)
    return Settings(**base)


@pytest.fixture
def artifacts(tmp_path) -> Path:
    return make_artifacts(tmp_path / "artifacts")


@pytest.fixture
def client(tmp_path, artifacts):
    """Empty queue, synthetic saved predictions, replay mode (no model files in the fixture folder)."""
    with TestClient(create_app(settings_for(tmp_path, artifacts))) as c:
        yield c


@pytest.fixture
def seeded_client(tmp_path, artifacts):
    with TestClient(create_app(settings_for(tmp_path, artifacts, seed_demo=True))) as c:
        yield c


@pytest.fixture
def bare_client(tmp_path):
    """No artifacts at all: no model, no saved predictions, no evidence."""
    empty = tmp_path / "no-artifacts"
    empty.mkdir()
    with TestClient(create_app(settings_for(tmp_path, empty))) as c:
        yield c


def have_real_model() -> bool:
    return all((REAL_ARTIFACTS / n).is_file() for n in ("backbone_densenet121-onnx.onnx", "head.npz", "ood.joblib", "policy.json", "backbone.json"))


@pytest.fixture(scope="session")
def real_client(tmp_path_factory):
    """The shipped artifacts with Member A's real model. Skipped when the model files are not installed."""
    if not have_real_model():
        pytest.skip("model files are not installed (run scripts/fetch_model.py)")
    tmp = tmp_path_factory.mktemp("real")
    with TestClient(create_app(settings_for(tmp, REAL_ARTIFACTS, seed_demo=True, model_mode="real"))) as c:
        yield c


def ctx_of(c: TestClient):
    return c.app.state.ctx


def upload(c: TestClient, data: bytes | None = None, name: str = "scan.png", urgent: bool = False, actor: str = "tester"):
    return c.post("/studies", files={"file": (name, data if data is not None else cxr_png(), "image/png")},
                  data={"clinician_urgent": "true" if urgent else "false", "actor": actor})


def wait_done(c: TestClient, study_id: str, timeout: float = 20.0) -> dict:
    """TestClient runs background tasks before returning, but poll anyway like the frontend does."""
    end = time.time() + timeout
    while True:
        s = c.get(f"/studies/{study_id}").json()
        if s["status"] != "processing" or time.time() > end:
            return s
        time.sleep(0.05)


def demo_bytes(artifacts: Path, study_id: str) -> bytes:
    return (artifacts / "demo" / "images" / f"{study_id}.png").read_bytes()
