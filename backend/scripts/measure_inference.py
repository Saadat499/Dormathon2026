"""Measure what real inference costs on this machine: memory after each import and after loading the
model, and the time per image. Used to decide whether the model fits the host (docs/BACKEND_NOTES.md).

    python scripts/measure_inference.py [threads]      # needs: pip install psutil, and the model files
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import psutil

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
ART = BACKEND / "model_artifacts"
proc = psutil.Process()


def rss(tag: str) -> None:
    m = proc.memory_info()
    print(f"{tag:46s} rss {m.rss / 1e6:7.1f} MB   peak {getattr(m, 'peak_wset', m.rss) / 1e6:7.1f} MB", flush=True)


def main() -> int:
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rss("python started")
    import fastapi, pandas, scipy.stats, sqlalchemy  # noqa: F401,E401
    rss("+ fastapi, sqlalchemy, pandas, scipy")
    import onnxruntime, sklearn  # noqa: F401,E401
    rss("+ onnxruntime, scikit-learn")
    from PIL import Image

    from app.services.predictor import RealModel
    t = time.perf_counter()
    model = RealModel(ART, threads=threads)
    print(f"model loaded in {time.perf_counter() - t:.2f} s  ({model.version}, {model.provider}, {threads} thread(s))")
    rss("model loaded")
    studies = json.loads((ART / "demo" / "demo_studies.json").read_text(encoding="utf-8"))["studies"][:20]
    times = []
    for s in studies:
        pil = Image.open(ART / s["image"])
        pil.load()
        t = time.perf_counter()
        model.predict(pil)
        times.append(time.perf_counter() - t)
    rss(f"after {len(times)} predictions")
    times.sort()
    print(f"per image: median {times[len(times) // 2] * 1000:.0f} ms, slowest {times[-1] * 1000:.0f} ms (first call includes warm-up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
