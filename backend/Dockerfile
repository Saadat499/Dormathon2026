# Build context: backend/   (Render: dockerContext ./backend, dockerfilePath ./backend/Dockerfile)
# Python 3.13 on purpose: it is the version Member A froze the model with (MANIFEST.json), and
# ood.joblib is a pickle. The backend test-suite runs on the same version.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=prod \
    DATA_DIR=/app/data \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Member A's model files live in the repo under ml/ (outside this build context). They are downloaded
# from the repository at a pinned commit and every SHA-256 is checked. If the download fails the image
# still builds and the service runs in replay mode, which GET /health reports. Build with
# --build-arg INCLUDE_MODEL=0 for a replay-only image.
ARG INCLUDE_MODEL=1
RUN if [ "$INCLUDE_MODEL" = "1" ]; then \
      python scripts/fetch_model.py || echo "WARNING: model files not installed; the service will run in replay mode"; \
    fi

RUN useradd --create-home --uid 10001 voxel \
 && mkdir -p /app/data \
 && chown -R voxel:voxel /app/data
USER voxel

# One worker: studies live in one SQLite file (WAL) and the model is loaded once in this process.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
