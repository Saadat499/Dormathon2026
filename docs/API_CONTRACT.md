# VoxelSort API contract (v0.4, as implemented by the backend)

Everyone builds against this file. Change it by agreement, not silently.
Base URL (live): `https://chesttriage-api.onrender.com`  ·  Interactive docs: `/docs`

> Decision support only. A clinician reads every scan. Numbers shown in the demo come from real saved model outputs; simulations are not clinical proof.

**v0.4 = v0.3 plus what the backend actually does.** No v0.3 field was renamed or removed. Everything that was
added, specified more precisely, or behaves differently from a literal reading of v0.3 is listed in
[Changes from v0.3](#changes-from-v03). The backend test `backend/tests/test_contract.py` checks this file's
endpoint tables against the running app.

## Conventions

- JSON in and out. Timestamps are ISO 8601 UTC (`2026-10-11T02:00:00Z`).
- Errors: `{"detail": "message"}` with a normal HTTP status: 400 bad input, 404 not found, 409 not allowed,
  413 file or request too large, 503 a needed data file is not installed on the server, 500 unexpected error.
- **No login and no accounts.** Every action that changes something takes an optional `actor` string (e.g. `"coordinator"`). It is only recorded in the audit log, it is not a security control.
- Long jobs (inference, Site Readiness) return `202` straight away. The frontend polls the matching `GET` every 2 to 3 seconds. No WebSockets.
- The frontend only talks to the backend through `src/services/api.js`.

## Prediction record (shared with A and B)

```json
{
  "study_id": "CXR-1042",
  "findings": [
    {"name": "effusion", "probability": 0.57},
    {"name": "edema", "probability": 0.24}
  ],
  "priority": "raised",
  "confidence_band": "high",
  "ood_flag": false,
  "heatmap_path": "/studies/CXR-1042/heatmap"
}
```

- `findings` is an ordered list (highest probability first) of the model's seven findings: `atelectasis`,
  `cardiomegaly`, `consolidation`, `edema`, `effusion`, `pneumonia`, `pneumothorax`. It is `[]` when the scan was not
  scored: still processing, out of distribution (Member A's rule: an OOD image is not scored), or no model output.
  Probabilities are the model's own outputs (logistic regression on frozen features, no temperature applied).
  Measured calibration is on the Evidence page; **nothing in the API claims the probabilities are well calibrated.**
- `priority` is the model's own output: `raised | normal | unsure`. It is `null` only while `status` is `processing`.
  It is derived from the **composite score** (the highest probability among the urgent findings: pneumothorax,
  effusion, edema, consolidation, pneumonia; a provisional assumption that needs clinician review) and the current
  threshold. A scan with no model output is `unsure`.
- `confidence_band` is Member A's categorical band: `high` (score clear of the unsure band), `low` (inside the unsure
  band), `none` (not scored), or `pending` (processing). It says where the score sits relative to the cut-offs. It is
  not a probability of the model being right.
- `ood_flag: true` means the image looks unlike the training data. It always forces `priority: "unsure"`, never `normal`.
- `heatmap_path` is `null` until the heatmap exists. The heatmap is a transparent PNG the same size as the image
  (a saliency map: where the model's evidence came from, not proof of reasoning).

## Study object (what the app receives)

The prediction record plus queue fields:

| Field | Type | Meaning |
|---|---|---|
| `uploaded_at` | timestamp | When the scan entered the queue |
| `wait_seconds` | int | Computed by the server, used for wait timers. Stops when the scan is reviewed |
| `rank` | int or null | 1 = read first. The queue order is decided only by the server. `null` once reviewed |
| `effective_priority` | `raised \| normal \| unsure` | The lane the queue used: the coordinator override if there is one, otherwise the model's priority |
| `priority_reason` | string | `model`, `clinician_request`, `coordinator_override` or `wait_cap` (highest applicable rule wins, in the order clinician request, wait cap, override, model) |
| `clinician_urgent` | bool | A clinician asked for urgency |
| `status` | string | `processing \| awaiting_review \| needs_review \| reviewed` (`needs_review` = effective priority `unsure`) |
| `flagged_disagreement` | bool | A reader disagreed with the model |
| `plain_language` | string | Fixed-template text filled with the real outputs. Never free-written |

Extra fields (not in v0.3): `composite_score`, `ood_reason`, `model_source` (`real | replay | saved_demo | fake | none`), `model_version`.

Frontend display labels (suggested): `raised` = "Urgent review", `unsure` = "AI unsure, review", `normal` = "Routine".

## Endpoints

### Health
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{"status":"ok", ...}` plus `database`, `model_mode`, `model`, `saved_predictions`, `degraded`, `storage` |
| GET | `/status` | Same as `/health` plus current queue settings, threshold and limits (not in v0.3) |

### Worklist
| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/studies` | query: `priority`, `status`, `q` (search id) | `{"mode":"model\|fcfs","max_wait_minutes":120,"items":[Study]}` already in queue order |
| GET | `/studies/{id}` | | `Study` |
| POST | `/studies` | multipart `file`, optional `clinician_urgent` (bool), optional `actor` | `202` + `Study` with `status:"processing"`. Poll `GET /studies/{id}` |
| GET | `/studies/{id}/image` | | The image (downsampled to at most 1024 px) |
| GET | `/studies/{id}/heatmap` | | Heatmap image, or `404` if none yet |

Uploads: PNG or JPG, at most 15 MB. A DICOM file is refused with a message asking for a PNG/JPG export.

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

`mode: "fcfs"` is the fallback switch: first-come-first-served, the model is ignored. `max_wait_minutes` must be 10 to 1440.

### Safety vs Speed threshold
| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/threshold/preview` | `{"value":0.5}` | `{"abstention_rate":..,"sensitivity":..,"false_alarm_rate":..,"n":..,"n_positive":..}` recalculated from saved predictions. Changes nothing |
| GET | `/threshold` | | `{"value":0.4,"min_sensitivity_target":0.9}` |
| PUT | `/threshold` | `{"value":0.35,"confirm":true,"actor":"..."}` | updated threshold. `400` unless `confirm` is `true`, and `400` if the value gives sensitivity below `min_sensitivity_target`. Logged |

Sensitivity target is fixed first, then false alarms are minimised at that target. Error never reaches zero, so the UI must not imply it can.

**Threshold scale (specified in v0.4).** `value` (0.01 to 0.99) is a **percentile of the composite score on the
reference set** (the NIH validation split), not a raw probability: `0.40` raises every scan scoring above the 40th
percentile of that set. The raw score cut-off is returned as `score_cutoff`. Reason: at the 90% sensitivity target the
raw cut-off is about 0.05, so a raw 0.1 to 0.9 slider could never satisfy the floor. `sensitivity`, `false_alarm_rate`,
`abstention_rate`, `n` and `n_positive` are measured on the **validation** split (the same data the floor is checked
on); the held-out test split is reported under `held_out_test` and is never used to choose a threshold.
`sensitivity` = share of urgent scans that are `raised`. `abstention_rate` = share of scans that are `unsure`.

### Queue Race
| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/simulations/queue-race` | `{"arrivals_per_hour":20,"reads_per_hour":18,"urgent_share":0.15,"hours":8,"seed":1}` | see below |

```json
{
  "assumptions": {"...": "echoed back so the slide can state them"},
  "fcfs":        {"urgent_wait_min": 0, "other_extra_wait_min": 0, "missed_urgent": null, "wrong_flags": null},
  "voxelsort":   {"urgent_wait_min": 0, "other_extra_wait_min": 0, "missed_urgent": 0, "wrong_flags": 0},
  "read_order": {"fcfs": [{"id":"SIM-001","wait_min":0,"truly_urgent":true}], "voxelsort": [{"id":"SIM-001","wait_min":0,"truly_urgent":true}]}
}
```

FCFS has no model, so its `missed_urgent` and `wrong_flags` are `null` (shown as n/a). Runs on real saved predictions
(held-out NIH test split), including misses and wrong flags. Owned by B (engine) and C (endpoint). Limits: rates 1 to
200 per hour, `urgent_share` 0.01 to 0.99, `hours` 1 to 24, and `arrivals_per_hour x hours` at most 1500.

### Site Readiness
| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/site-readiness` | multipart: `files` (sample images), optional `labels` CSV, optional `actor` | `202` `{"job_id":"..."}` |
| GET | `/site-readiness/sample-size` | query: `target_sensitivity`, `expected_sensitivity`, `confidence` (default 0.95), `prevalence` | `{"positives_needed":..,"scans_needed":..,"note":".."}` computed live |
| GET | `/site-readiness/{job_id}` | | result below |

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

Labels CSV: a header row with `label` (1 = urgent finding, 0 = not) and `filename` (to link a row to an uploaded
image). `sensitivity` is `null` when it could not be measured. Verdict rules (B's GO / WARNING / NO-GO mapped to the
contract's words; details and limits in `docs/BACKEND_NOTES.md`):

| Verdict | When |
|---|---|
| `not_ready` | the interval's upper end is below the target, or B's drift rule says NO-GO beyond what sample size alone explains |
| `not_enough_evidence` | no labels, fewer than 30 labelled urgent scans, or `ci_low` below the target |
| `ready` | `ci_low` at or above the target, at least 30 urgent scans, and B's drift rule says GO |
| `use_with_caution` | `ci_low` at or above the target, at least 30 urgent scans, but drift WARNING or drift check not run |

### OOD check demo
| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/ood-check` | multipart `file` | `{"ood_flag":true,"reason":"...","score":0.0}` |

`score` is the Mahalanobis distance of Member A's detector (higher = further from the training data); `score_threshold`
and `method` are returned with it. When the trained detector is not loaded, `method` says so.

### Evidence page
| Method | Path | Response |
|---|---|---|
| GET | `/evidence` | `{"calibration":{...},"cross_dataset":[...],"per_finding":[...],"shortcut_audit":{...},"failures":[...]}` plus `provenance` and `extras` |

Inner shapes (as the Evidence page renders them): `calibration` = `{finding, before:[{confidence,accuracy,count}], after:[...], ece_before, ece_after, per_finding, note}`;
`cross_dataset` and `per_finding` = lists of flat rows; `shortcut_audit` = flat object; `failures` = `[{study_id, caption, image_url}]`.
Main keys hold real results only. Member B's simulated-cohort file is passed through under
`extras.member_b_simulated_cohorts`, labelled synthetic.

### Audit log
| Method | Path | Notes |
|---|---|---|
| GET | `/audit` | query: `action`, `q`, `limit`. Newest first. Entry: `{"id","timestamp","actor","action","study_id","detail"}` |
| GET | `/audit.csv` | Same data as CSV download |

Logged actions: `upload`, `mark_reviewed`, `flag_disagreement`, `urgent_request`, `override`, `queue_settings`,
`threshold_change`, `site_readiness_run`, and `scoring` (v0.4: the background model result for an upload, actor `system`).

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

In `fcfs` mode: `clinician_urgent` first, then arrival order only. Reviewed scans are listed last with `rank: null`.

## Changes from v0.3

Nothing in v0.3 was removed or renamed. This table is everything else.

| Area | v0.3 said | v0.4 (implemented) | Why |
|---|---|---|---|
| Threshold `value` | example `0.5`, scale not defined | a percentile of the reference scores (see above); `score_cutoff` returned too | at the 90% target the raw cut-off is about 0.05, below any usable slider range |
| `/threshold/preview` | 5 fields | plus `value`, `score_cutoff`, `unsure_score_cutoff`, `raised_share`, `meets_target`, counts of misses and wrong flags, `held_out_test`, `data_source`, `data_is_real`, `note` | provenance; validation and test kept apart |
| `/threshold` | 2 fields | plus `scale`, `score_cutoff`, `unsure_budget`, `max_allowed_value`, `data_source` | same |
| `confidence_band` | example `"0.71-0.89"` | `high \| low \| none \| pending` | this is what Member A's model defines; no interval per scan exists |
| `findings` when OOD | not stated | `[]` | Member A does not score OOD images |
| `priority` while processing | not stated | `null` | no model output yet (`effective_priority` is `unsure`) |
| Study | 15 fields | plus `composite_score`, `ood_reason`, `model_source`, `model_version` | say where each number came from |
| `fcfs` mode | "pure first-come-first-served" | clinician requests are still first | safety rule 3 says a clinician request always beats the model |
| OOD scan override | not stated | setting an OOD scan to `normal` is refused with `409` | safety rule 2 |
| Reviewed scans | not stated | review, override or urgent-request on a reviewed scan gives `409`; reviewing a scan that is still processing gives `409` | avoid silent double actions |
| `POST /studies` | `file`, `clinician_urgent` | also accepts `actor` (the frontend sends it) | audit |
| Errors | 400, 404, 409 | also 413 (too large), 503 (data file missing), 500; validation errors are `400` with a string `detail`, not FastAPI's 422 | one error shape for the frontend |
| Queue Race | 4 keys | plus `n_cases`, `n_urgent` (the UI reads them), per row `urgent` (the UI's name) and `flag`, per arm `max_wait_min`, and `b_engine` (Member B's own ordering on the same stream) | the UI needs them; B's rule shown for comparison |
| Site Readiness result | 6 keys | plus `n_images`, `n_images_flagged_ood`, `n_label_rows`, `model_mode`, `verdict_rules`; `drift` adds `b_verdict`, `b_verdict_counted_as`, `max_psi`, `max_ks`, `ood_rate`, `domain_auroc`, `chance_level_95`; `sensitivity` adds `n_caught`, `confidence` | show how the verdict was reached |
| Sample size | 3 keys | plus `method`, `normal_approximation_positives`, `exact_positives` | both of the two calculations are shown |
| `/ood-check` | 3 keys | plus `method`, `source`, `score_meaning`, `score_threshold`, `probe_probability`, `note` | which detector answered |
| `/evidence` | inner shapes TBD | defined above; plus `provenance`, `extras` | agreed with what the Evidence page renders |
| Audit | 8 actions; entry has 6 fields | plus action `scoring`; entry also has `data` (structured detail) | background model results and failures are traceable |
| New endpoints | | `GET /status`, `GET /evidence/failures/{name}` (images for the failure gallery) | |

## Open questions for the team

- **A, B, clinical lead:** the contract's 90% sensitivity floor forces the threshold to 0.40, which raises about 60% of
  scans. Member A's shipped policy raises the top 20% (about 47% sensitivity) and cannot be selected under this floor.
  Which do we present? (`MIN_SENSITIVITY_TARGET` is an environment variable.)
- **D:** static wording on three pages conflicts with the evidence (list in `docs/BACKEND_NOTES.md`).
- **B:** the Site Readiness drift cut-offs fire by chance on small samples; the backend guards against that. Agree or adjust.
