# ChestTriage API contract (draft)

Agree this with the whole team in hour one. Everyone builds against it.

## Prediction record

```json
{
  "study_id": "string",
  "findings": { "pneumothorax": 0.0, "effusion": 0.0 },
  "priority": "Raised | Normal | AI unsure",
  "confidence_band": "string",
  "ood_flag": false,
  "heatmap_path": "string | null"
}
```

## Endpoints

| Method | Path | Purpose | Status |
|--------|------|---------|--------|
| GET | /health | Liveness check, returns `{"status":"ok"}` | live |

More endpoints (auth, upload, queue, override, audit, Queue Race, Site Readiness) to be added here.
