# Dormathon2026

VoxelSort: a chest X-ray worklist prioritiser. Decision support only; a clinician reads every scan.

| Folder | What | Start here |
|---|---|---|
| `backend/` | FastAPI service (Member C) | [docs/BACKEND_NOTES.md](docs/BACKEND_NOTES.md): how to run, what is real, limits |
| `frontend/` | Vite + React app (Member D) | [frontend/README.md](frontend/README.md) |
| `ml/` | Model, evaluation, OOD detector (Member A) | [ml/README.md](ml/README.md) |
| `eval_safety/` | Evaluation and safety kit (Member B) | `eval_safety/voxelsort_eval_safety/assumptions_and_claims.md` |
| `docs/` | [API contract v0.4](docs/API_CONTRACT.md) | |

Live API: https://chesttriage-api.onrender.com (`/health`, `/docs`). Check it end to end with `scripts/verify_deploy.sh`.
