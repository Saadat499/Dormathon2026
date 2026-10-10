# VoxelSort API contract (DRAFT v0.3 for team review)

Everyone builds against this file. Change it by agreement, not silently.
Base URL (live): `https://chesttriage-api.onrender.com`  ·  Interactive docs: `/docs`

> Decision support only. A clinician reads every scan. All numbers shown in the demo come from real saved model outputs; simulations are not clinical proof.

## Conventions

- JSON in and out. Timestamps are ISO 8601 UTC (`2026-10-11T02:00:00Z`).
- Errors: `{"detail": "message"}` with a normal HTTP status (400 bad input, 404 not found, 409 not allowed).
- **No login and no accounts.** Every action that changes something takes an optional `actor` string (e.g. `"coordinator"`). It is only recorded in the audit log, it is not a security control.
- Long jobs (inference, Site Readiness) return `202` straight away. The frontend polls the matching `GET` every 2 to 3 seconds. No WebSockets.
- The frontend only talks to the backend through `src/services/api.js`.

## Prediction record (shared with A and B)

```json
{
  "study_id": "CXR-1042",
  "findings": [
    {"name": "pneumothorax", "probability": 0.82},
    {"name": "effusion", "probability": 0.26}
  ],
  "priority": "raised",
  "confidence_band": "0.71-0.89",
  "ood_flag": false,
  "heatmap_path": "/studies/CXR-1042/heatmap"
}
```

- `findings` is an ordered list (highest probability first), probabilities are calibrated.
  *Change from the original plan, which used `{name: probability}`. A and B to confirm.*
- `priority` is the model's own output: `raised | normal | unsure`.
- `ood_flag: true` means the image looks unlike the training data. It always forces `priority: "unsure"`, never `normal`.
- `heatmap_path` is `null` until the heatmap exists.

## Study object (what the app receives)

The prediction record plus queue fields:

| Field | Type | Meaning |
|---|---|---|
| `uploaded_at` | timestamp | When the scan entered the queue |
| `wait_seconds` | int | Computed by the server, used for wait timers |
| `rank` | int | 1 = read first. The queue order is decided only by the server |
| `effective_priority` | `raised \| normal \| unsure` | What the queue actually used after overrides and the wait cap |
| `priority_reason` | string | `model`, `clinician_request`, `coordinator_override` or `wait_cap` |
| `clinician_urgent` | bool | A clinician asked for urgency |
| `status` | string | `processing \| awaiting_review \| needs_review \| reviewed` |
| `flagged_disagreement` | bool | A reader disagreed with the model |
| `plain_language` | string | Fixed-template text filled with the real outputs. Never free-written |

Frontend display labels (suggested): `raised` = "Urgent review", `unsure` = "AI unsure, review", `normal` = "Routine".

## Endpoints

### Health
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{"status":"ok"}` (live) |

### Worklist
| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/studies` | query: `priority`, `status`, `q` (search id) | `{"mode":"model\|fcfs","max_wait_minutes":120,"items":[Study]}` already in queue order |
| GET | `/studies/{id}` | | `Study` |
| POST | `/studies` | multipart `file`, optional `clinician_urgent` (bool) | `202` + `Study` with `status:"processing"`. Poll `GET /studies/{id}` |
| GET | `/studies/{id}/image` | | The image (downsampled) |
| GET | `/studies/{id}/heatmap` | | Heatmap image, or `404` if none yet |

### Review and overrides (every call writes an audit entry)
| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/studies/{id}/review` | `{"action":"mark_reviewed\|flag_disagreement","reason":"...","notes":"...","human_priority":"raised\|normal\|unsure","actor":"..."}` | `Study` |
| POST | `/studies/{id}/urgent-request` | `{"note":"...","actor":"..."}` | `Study` with `clinician_urgent: true` |
| POST | `/studies/{id}/override` | `{"priority":"raised\|normal\|unsure","reason":"required","actor":"..."}` | `Study` |

### Queue settings
| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/queue/settings` | | `{"mode":"model","max_wait_minutes":120}` |
| PUT | `/queue/settings` | `{"mode":"model\|fcfs","max_wait_minutes":120,"actor":"..."}` | updated settings |

`mode: "fcfs"` is the fallback switch: pure first-come-first-served, the model is ignored.

### Safety vs Speed threshold
| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/threshold/preview` | `{"value":0.5}` | `{"abstention_rate":..,"sensitivity":..,"false_alarm_rate":..,"n":..,"n_positive":..}` recalculated from saved predictions. Changes nothing |
| GET | `/threshold` | | `{"value":0.5,"min_sensitivity_target":0.9}` |
| PUT | `/threshold` | `{"value":0.45,"confirm":true,"actor":"..."}` | updated threshold. `400` unless `confirm` is `true`, and `400` if the value gives sensitivity below `min_sensitivity_target`. Logged |

Sensitivity target is fixed first, then false alarms are minimised at that target. Error never reaches zero, so the UI must not imply it can.

### Queue Race
| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/simulations/queue-race` | `{"arrivals_per_hour":20,"reads_per_hour":18,"urgent_share":0.15,"hours":8,"seed":1}` | see below |

```json
{
  "assumptions": {"...": "echoed back so the slide can state them"},
  "fcfs":        {"urgent_wait_min": 0, "other_extra_wait_min": 0, "missed_urgent": 0, "wrong_flags": 0},
  "voxelsort":   {"urgent_wait_min": 0, "other_extra_wait_min": 0, "missed_urgent": 0, "wrong_flags": 0},
  "read_order": {"fcfs": [{"id":"SIM-001","wait_min":0,"truly_urgent":true}], "voxelsort": [{"id":"SIM-001","wait_min":0,"truly_urgent":true}]}
}
```

FCFS has no model, so its `missed_urgent` and `wrong_flags` are `null` (shown as n/a). Runs on real saved predictions, including misses and wrong flags. Owned by B (engine) and C (endpoint).

### Site Readiness
| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/site-readiness` | multipart: sample images and an optional labels CSV | `202` `{"job_id":"..."}` |
| GET | `/site-readiness/{job_id}` | | result below |
| GET | `/site-readiness/sample-size` | query: `target_sensitivity`, `expected_sensitivity`, `confidence` (default 0.95), `prevalence` | `{"positives_needed":..,"scans_needed":..,"note":".."}` computed live |

```json
{
  "status": "processing | done",
  "drift": {"score": 0.0, "warning": false, "note": "warning signal only, not a proven predictor"},
  "sensitivity": {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "width": 0.0, "method": "clopper-pearson", "n_positive": 0},
  "target": 0.9,
  "note": "optional, e.g. no labels supplied",
  "verdict": "ready | use_with_caution | not_ready | not_enough_evidence"
}
```

`not_enough_evidence` when the sample is too small or `ci_low` is below the target. Verdict rules and the sample-size formula are owned by B.

### OOD check demo
| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/ood-check` | multipart `file` | `{"ood_flag":true,"reason":"not a chest X-ray \| rotated \| different distribution","score":0.0}` |

### Evidence page
| Method | Path | Response |
|---|---|---|
| GET | `/evidence` | `{"calibration":{...},"cross_dataset":{...},"per_finding":[...],"shortcut_audit":{...},"failures":[...]}` |

Static results produced by A and B and served as-is by C. Exact inner shapes: **TBD by A and B**.

### Audit log
| Method | Path | Notes |
|---|---|---|
| GET | `/audit` | query: `action`, `q`, `limit`. Newest first. Entry: `{"id","timestamp","actor","action","study_id","detail"}` |
| GET | `/audit.csv` | Same data as CSV download |

Logged actions: `upload`, `mark_reviewed`, `flag_disagreement`, `urgent_request`, `override`, `queue_settings`, `threshold_change`, `site_readiness_run`.

## Safety rules the backend enforces

1. The model can only move scans **up**. It never removes, hides or auto-clears a scan.
2. `unsure` and `ood_flag` are never treated as low risk. They always rank above `normal`.
3. A clinician urgency request always beats the model.
4. A scan waiting longer than `max_wait_minutes` is promoted, so false positives cannot starve routine patients.
5. `mode: "fcfs"` switches back to first-come-first-served instantly.
6. Threshold changes need `confirm: true` and are logged.
7. Every state change writes an audit entry.

**Known limitation:** with no login, "clinical lead only" for threshold changes cannot be enforced. The confirmation step and the audit trail are the only controls. State this honestly in the demo.

## Queue order (the single ranking rule)

1. `clinician_urgent`
2. waited past `max_wait_minutes`
3. `effective_priority`: `raised`, then `unsure`, then `normal`
4. Oldest `uploaded_at` first within the same group

## Open questions for the team

- A and B: confirm `findings` as an ordered list.
- A: list of finding names and the urgency mapping (an assumption, to be stated on a slide).
- B: verdict thresholds, sample-size formula, `/evidence` inner shapes.
- D: the live `App.jsx` currently uses `id`, "Urgent / Review / Routine" and `[[name, prob]]`. Map to this contract inside `api.js` only.
