import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

from app.main import create_app  # noqa: E402


def cxr_png(w=320, h=400, seed=0, colour=False) -> bytes:
    """A made-up, roughly upright, left-right symmetric grayscale image. Not a medical image."""
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


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(data_dir=tmp_path / "data", artifact_dir=tmp_path / "no-artifacts")) as c:
        yield c


@pytest.fixture
def empty_client(tmp_path):
    with TestClient(create_app(data_dir=tmp_path / "data", artifact_dir=tmp_path / "no-artifacts", seed_demo=False)) as c:
        yield c


@pytest.fixture
def replay_client(tmp_path):
    """A tiny stand-in for Member A's predictions.json (test data, not real model output)."""
    art = tmp_path / "art"
    art.mkdir()
    recs = []
    for i in range(60):
        pos = i % 5 == 0
        p = 0.9 if pos else 0.1 + (i % 7) * 0.02
        recs.append({"study_id": f"A-{i:03d}", "label": int(pos), "findings": [{"name": "pneumothorax", "probability": p}, {"name": "atelectasis", "probability": 0.3}]})
    (art / "predictions.json").write_text(json.dumps(recs))
    with TestClient(create_app(data_dir=tmp_path / "data", artifact_dir=art)) as c:
        yield c


def upload(c, name="scan.png", data=None, urgent=False, actor="tester"):
    return c.post("/studies", files={"file": (name, data or cxr_png(), "image/png")},
                  data={"clinician_urgent": "true" if urgent else "false", "actor": actor})
