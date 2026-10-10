# Backend notes (Member C)

Written autonomously while Member C was unavailable. This file records every assumption, deviation and gap.
**Read "Inputs that were missing" first.**

## Inputs that were missing

The brief said to read these first. They are **not in the repository** (checked `main`, the only branch, and the PR list):

| Expected | Found | Consequence |
|---|---|---|
| `docs/TASK_C_PROMPT.md` (instructions, Steps 1 to 12, definition of done, "section 1" safety rules) | **missing** | I built from `docs/API_CONTRACT.md` (its "Safety rules the backend enforces"), the frontend, B's code and the written brief. Steps 1 to 12 and the exact definition of done were not available, so I cannot say that checklist is satisfied. |
| `backend/model_artifacts/` (Member A's model output) | **missing** | There is no model output to load. Step 9 could not be done "for real". The backend has a loader ready (`backend/model_artifacts/README.md`) and switches to replay mode when `predictions.json` or `predictions.csv` appears. |

## What is real and what is fake

| Area | Status |
|---|---|
| Queue ranking rule, wait cap, FCFS fallback, clinician urgency, overrides, review, audit log, CSV export | **Real.** Implemented and tested (`backend/tests`). |
| Uploads: validation, downsampled image storage, background processing | **Real.** PNG and JPG only. DICOM is rejected with a clear message. |
| **Model inference** | **Fake / absent.** No model runs. Without A's files an upload gets no findings and is placed in the queue as `unsure` (never `normal`). With `predictions.json`, replay mode matches an upload to a saved result by file name (= study id) or `sha256`; the model is still not run. |
| Saved predictions behind Safety vs Speed and Queue Race | **Synthetic** until A's file exists: a seeded made-up cohort (`cohorts.synthetic_cohort`). It is computed, not hard-coded, but says nothing about any model. `GET /status` and `data_source` in `/threshold/preview` say which source is active. |
| Seed queue (8 demo rows) | Synthetic (ids `SYN-xxxx`), each `plain_language` starts "SYNTHETIC DEMO RECORD". Arrival times are staged relative to start-up. |
| Statistics: Clopper-Pearson interval, sample-size search, PSI drift | **Real calculations** (scipy). |
| Site Readiness | Real machinery. Sensitivity needs model scores, which this host cannot produce, so they must come from a `score` column in the labels CSV or from A's saved predictions matched by file name. Otherwise: no sensitivity, verdict `not_enough_evidence`. Verdict rules are mine (below). |
| OOD check | **Placeholder heuristic** (colour, size, shape, contrast, upright symmetry). Not Member A's detector, never validated. A flag makes a scan `unsure`; no flag proves nothing. |
| `/evidence` | B's `locked_evidence.json`, adapted to the page's shapes. **Those numbers come from simulated cohorts** (`evidence_exporter.py` builds them with `np.random`), not from a trained model on real scans. The response carries a `provenance` block and a `data` column on the per-finding table. Cross-dataset, failure gallery and the uncalibrated ("before") curve are A's or were not saved by B, so they are left out and the page says "Not available yet". |
| Heatmaps | None exist. `heatmap_path` is `null`, `/heatmap` returns 404. |
| Persistence | **None across restarts.** SQLite in an ephemeral folder (`VOXELSORT_DATA_DIR`, default the system temp dir). Studies, uploads and the audit log are lost on redeploy and usually on free-tier sleep and wake. |

## Safety rules (API_CONTRACT.md) and how they are enforced

1. *Model only moves scans up*: nothing is hidden or deleted. The only way down is a human coordinator override, with a required reason, logged.
2. *`unsure` and OOD never low risk*: missing or NaN score, no findings, OOD flag and "still processing" all derive `unsure`. A coordinator cannot set an OOD scan to `normal` (409).
3. *Clinician urgency beats the model*: first sort key. It also applies in FCFS mode (see deviations).
4. *Wait cap*: a scan waiting at least `max_wait_minutes` (model mode) is the second sort key.
5. *FCFS fallback*: `PUT /queue/settings {"mode":"fcfs"}` takes effect on the next read.
6. *Threshold*: needs `confirm: true`, is refused if sensitivity on the saved predictions falls below 0.9, and is logged.
7. *Every state change is audited* (tested for all eight logged actions).

Never described as "well calibrated": B's own numbers show temperature scaling barely changed ECE (e.g. Pneumothorax 0.443 to 0.432; Massive effusion got slightly worse). The Queue Race response states it is a simulation and not clinical proof.

## Assumptions I made

- **Single urgency score**: the highest probability among findings not in B's ROUTINE group (`cardiomegaly`, `atelectasis`, `support devices`, `no finding`; a test keeps this equal to B's `urgency_mapping.py`). An unrecognised finding name counts as urgent. This urgency mapping is an assumption, not a clinical standard.
- **`unsure` band**: scores within 0.15 below the threshold are `unsure` (same constant as the frontend demo).
- **Minimum sensitivity target** is fixed at 0.9 (the contract example). Default threshold = highest value on a 0.01 grid that meets it on the saved predictions (`0.51` on the synthetic cohort).
- **Priority is derived at read time** from stored probabilities and the current threshold, so a threshold change re-ranks the whole queue immediately (the UI says it does).
- **Site Readiness verdict**: `not_ready` if the interval's upper end is below 0.9; `not_enough_evidence` if fewer than 30 urgent scans or the lower end is below 0.9; otherwise `use_with_caution` when drift warns, else `ready`. Drift warns at PSI >= 0.10. Sensitivity counts a scan as caught when its score is at or above the current threshold.
- **Queue Race engine** is mine, not B's `queue_simulator.py`, because B's uses STAT/URGENT/ROUTINE tiers and its own aging rule while the contract and the live worklist use `raised/unsure/normal` plus the wait cap. Using the same rule everywhere keeps the demo consistent. Cases are drawn from the saved predictions with the requested urgent share; one reader; fixed read time (60 / reads_per_hour min); Poisson arrivals. B's own queue-race results are passed through untouched in `/evidence` under `extras`.
- Limits: 15 MB per upload, 1000 stored studies, 100 site images / 120 MB, 3000 simulated scans.

## Deviations from the contract

- `/threshold/preview` also returns `value`, `min_sensitivity_target`, `unsure_band`, `data_source`, `data_is_real` (extra fields only).
- `/simulations/queue-race` adds `n_cases`, `n_urgent` (the UI reads them) and, per `read_order` row, `urgent` (the UI reads this name) next to the contract's `truly_urgent`, plus `flag`.
- Site Readiness: sensitivity `method` is `clopper-pearson` (as the contract says; the frontend demo used Wilson). Extra fields: `n_images`, `n_images_flagged_ood`, `n_label_rows`, `drift.kind`. When no model scores exist the drift `score` is the share of images flagged by the placeholder OOD check, and `drift.note` says so.
- `/site-readiness/sample-size` also returns `method`.
- `/evidence` adds `provenance` and `extras`; `calibration` has `before: []` (see above).
- `/ood-check` adds `method` and `note`.
- New `GET /status` (not in the contract): active model mode, data sources, storage warning.
- Validation errors return **400** with `{"detail": "<string>"}` instead of FastAPI's default 422.
- In `fcfs` mode a clinician urgency request is still honoured (rule 3 says it always beats the model). The contract says "pure first-come-first-served".
- `POST /studies` accepts the `actor` form field the frontend sends.
- DICOM is rejected (400) because no DICOM reader is installed.
- CORS: not `*` any more. Allowed: `https://*.vercel.app`, `http://localhost:5173`, plus `CORS_ORIGINS` (comma separated) for a custom domain.
- Reviewing, overriding or urgent-requesting an already reviewed study returns 409; reviewing a study that is still processing returns 409.

## Environment variables

| Name | Default | Purpose |
|---|---|---|
| `VOXELSORT_DATA_DIR` | system temp dir `/voxelsort-data` | SQLite file and images |
| `VOXELSORT_ARTIFACT_DIR` | `backend/model_artifacts` | Member A's files |
| `VOXELSORT_SEED_DEMO` | `1` | `0` starts with an empty queue |
| `CORS_ORIGINS` | empty | extra allowed origins |
| `CORS_ORIGIN_REGEX` | `https://([a-z0-9-]+\.)*vercel\.app` | allowed origin pattern |

## Running and testing

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest -q tests          # 37 tests
uvicorn app.main:app --reload
```

## Deployment

- Render service `chesttriage-api` (free plan, Docker, context `./backend`, auto-deploy from `main`). A deploy of commit `49e9c23` reached **live**; its log shows `Application startup complete` and `GET / 200`.
- **Not verified from the build sandbox:** the sandbox's network policy blocks `onrender.com`, so I could not curl the public URL. Run `scripts/verify_deploy.sh` (15 checks, including an upload) from any machine. The same script passes against a local instance.
- Pushes to `main` did not trigger a Render build by themselves in this session; I triggered one through the Render API. If a later push does not deploy, press "Manual Deploy" in the dashboard.
- **Free-tier limits not verified:** the service sleeps after idle time (the first request may take about a minute), the disk is wiped on redeploy and restart, and memory is limited. Locally the process peaks at about 105 MB RSS with a few uploads, which should fit, but this was not measured on Render.
- **Vercel:** the connected Vercel account contains only a project named `chesttriage-cors-test`, not the VoxelSort frontend, so I could not set `VITE_API_URL` or check the live frontend. Locally I ran the unmodified frontend (Vite) against the backend with headless Chromium: Worklist, Upload, Study review (urgent request, override, mark reviewed), Safety vs Speed, Queue Race, Site readiness, Evidence tabs, OOD check and Audit log all rendered real responses with no API errors.

## Frontend wording that conflicts with the evidence (not changed: outside the allowed files)

These are static strings in `frontend/src/pages/`. They need Member D's decision:

- `StudyReview.jsx`: "Calibrated model output" and "Calibrated probability". B's own numbers show little calibration improvement, and in the current backend there is no model output at all.
- `SafetySpeed.jsx`: "Computed from N saved predictions" is shown even while the cohort is synthetic.
- `Evidence.jsx`: no on-screen label that B's results are simulated (only the `data` column and the audit-row text carry it). The Calibration tab in particular shows ECE 0.443 then 0.432 with no synthetic label. `/evidence` returns a `provenance` object the page could display.
- The live UI has no banner for "no model attached"; `GET /status` has the facts.

## Hand-off: what A must provide to make this real

1. `backend/model_artifacts/predictions.json` (or `.csv`) as described in `backend/model_artifacts/README.md`, with true labels. Then `/status` shows `saved_predictions.real: true`, and Safety vs Speed, Queue Race, the default threshold and the seeded queue switch to it automatically.
2. A decision on running the network on the host (needs weights plus memory). If it fits, implement `predict` in `app/model.py` and return findings, heatmap path and a real confidence band.
3. A's OOD detector to replace the placeholder in `app/ood.py`.
