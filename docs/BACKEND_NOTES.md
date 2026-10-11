# Backend notes (Member C, Task C)

How the VoxelSort backend works, what is real and what is not, and how to run it.
The API itself is described in [API_CONTRACT.md](API_CONTRACT.md) (v0.4, with every change from v0.3 listed).

Decision support only. A clinician reads every scan.

## 1. Run it on Windows

Python 3.13 and Node 20 or newer. Commands are run from the repository root (`C:\dev\Dormathon2026`).

**PowerShell**

```powershell
cd backend
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts\fetch_model.py          # copies Member A's model files from ..\ml and checks every SHA-256
.\.venv\Scripts\python.exe -m pytest tests -q              # 423 tests, about 80 seconds
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --workers 1
```

In a second PowerShell window, the frontend against that backend:

```powershell
cd frontend
npm install
$env:VITE_API_URL = "http://localhost:8000"
npm run dev                                                # http://localhost:5173
```

**Command Prompt**

```bat
cd backend
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe scripts\fetch_model.py
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --workers 1
```

```bat
cd frontend
npm install
set VITE_API_URL=http://localhost:8000
npm run dev
```

Always one worker: studies live in one SQLite file (WAL mode) and the model is loaded once in that process.
API docs: `http://localhost:8000/docs`. What is running: `http://localhost:8000/health`.
Docker (not needed for local work): `docker compose up --build` from `backend\`.
Member B's own tests: `cd eval_safety\voxelsort_eval_safety` then `..\..\backend\.venv\Scripts\python.exe -m pytest tests -q` (19 tests).

## 2. Environment variables

Full list with comments: [`backend/.env.example`](../backend/.env.example). The app reads real environment variables; it does not load a `.env` file.

| Name | Default | Purpose |
|---|---|---|
| `APP_ENV` | `dev` (the Docker image sets `prod`) | In `prod`, CORS is never open by default |
| `DATA_DIR` | `backend/data` | SQLite file, stored images, heatmaps |
| `DATABASE_URL` | SQLite file in `DATA_DIR` | SQLAlchemy URL |
| `MODEL_ARTIFACTS_DIR` | `backend/model_artifacts` | Model files, saved predictions, evidence, demo bundle |
| `ALLOWED_ORIGINS` | any origin in `dev`; `http://localhost:5173` in `prod` | Comma-separated browser origins |
| `ALLOWED_ORIGIN_REGEX` | none | Extra origins by pattern (for example one Vercel project's preview URLs) |
| `MAX_UPLOAD_MB` | `15` | Per-file upload limit |
| `DEFAULT_MAX_WAIT_MINUTES` | `120` | Wait cap at first start (then changed through `PUT /queue/settings`) |
| `MIN_SENSITIVITY_TARGET` | `0.9` | Floor for threshold changes |
| `UNSURE_BUDGET` | Member A's `policy.json` (`0.10`) | Width of the "AI unsure" band, as a share of scans |
| `MODEL_MODE` | `auto` | `auto`, `real`, `replay`, `fake` (development only), `none` |
| `SEED_DEMO` | `1` | `0` starts with an empty queue |
| `ONNX_THREADS` | `1` | Threads for the backbone |
| `LOG_LEVEL` | `INFO` | |

The older names `VOXELSORT_DATA_DIR`, `VOXELSORT_ARTIFACT_DIR`, `VOXELSORT_SEED_DEMO`, `CORS_ORIGINS` and `CORS_ORIGIN_REGEX` are still read.

## 3. The model: which one, and how it was chosen

**Selected: `artifacts_v2_imagenet_fullnih`, `model_version` `partA-0.1.0+4ca280b3cd`**, from Member A's delivery
(`C:\Users\User\Downloads\chesttriage_ml\chesttriage_ml`). Chosen on evidence, not on the folder name
(`backend/model_artifacts/SOURCE.json` records each check):

| Check | Result |
|---|---|
| A's README names it the frozen model | yes |
| SHA-256 of every deploy file equals `MANIFEST.json` | yes, all five |
| `model_version` recomputed from `backbone.json` + `head.npz` + `policy.json` | equals the manifest |
| `policy.json` was fitted for this `head.npz` (`fitted.head`) | yes |
| Every row of `predictions/nih_full.csv` carries this version | yes |
| Composite score equals the highest urgent probability in every row | yes |
| Better than v1 on the same 21,845 held-out NIH test images | AUROC "any urgent" 0.811 against 0.801, paired difference 0.010 (0.007 to 0.013) |
| Live inference reproduces A's saved outputs | 59 demo images re-scored: largest probability difference 0.0, all OOD flags equal |

Rejected: `artifacts_v1_imagenet_frozen` (same backbone, head trained on a 30,000-image subset, lower on every finding)
and `artifacts` (an intermediate copy identical to v1, used by no report).

**Runtime.** Preprocessing: grayscale, stretched to 224 x 224, ImageNet normalisation. Backbone: torchvision's
ImageNet DenseNet121 as an ONNX graph, run by ONNX Runtime on CPU (no PyTorch). Head: one logistic regression per
finding (`head.npz`), no temperature. Policy: seven findings, five of them urgent; composite score = highest urgent
probability. OOD: A's detector (`ood.joblib`, scikit-learn 1.7.2). All of this is A's code, copied unchanged to
`backend/chesttriage_ml/`. The backend only swaps the ONNX session options (one thread, no memory arena) and colours
the saliency map itself, so matplotlib is not needed. Python 3.13 and the library versions are the ones in A's manifest.

**Artifact placement.** See [`backend/model_artifacts/README.md`](../backend/model_artifacts/README.md). The model
files are already in the repo under `ml/artifacts_v2_imagenet_fullnih/` and are not committed a second time;
`scripts/fetch_model.py` copies or downloads them and verifies each checksum. One exception: `policy.json` is
committed under `backend/model_artifacts/` with `-text`, because git stores the `ml/` copy with LF line endings while
A froze it with CRLF. **On Linux and macOS the `ml/` copy therefore has a different SHA-256 and A's engine computes a
different `model_version` from it.** That is worth fixing in `ml/` too (a `.gitattributes` line), but it is A's folder.

## 4. What is real, replayed, synthetic, unavailable or unverified

| Area | Status |
|---|---|
| Model inference on uploads | **Real** when the model files are installed (`/health` `model_mode: "real"`): A's frozen model runs on every upload |
| Out-of-distribution check | **Real**: A's trained detector. A flagged image is not scored and becomes `unsure` |
| Heatmaps | **Real saliency maps** (A's exact class-activation map for the top urgent finding), coloured by the backend. Coarse (7 x 7 grid). They show where the evidence came from, not that the model looked at the right thing: A's own audit found maps on image borders and a text marker |
| Saved predictions behind the threshold preview, the floor and the Queue Race | **Real**: 11,341 NIH validation and 21,845 NIH test predictions with labels |
| Demo queue at start-up (12 studies) | **Real saved outputs** for public NIH images; **staged** arrival times; each text starts "Demo record". Chosen so every lane is visible, not a performance sample |
| Replay mode | Only when the model is not installed. An upload byte-identical (SHA-256) to a saved demo image returns its saved output, labelled "Replayed". Any other upload gets **no model output** and stays in the queue as `unsure`. Never matched by file name |
| Model fails or is missing | **No fabricated numbers.** The scan has no findings, is `unsure`, stays visible, and `/health` shows `degraded` |
| Fake predictor | Exists only behind `MODEL_MODE=fake` for development. Never the default; labelled FAKE in the text, the audit log and `/health` |
| `/evidence` main keys | **Real**: A's reports, plus B's calibration and AP/PA code run on A's real predictions |
| B's `locked_evidence.json` | **Synthetic** (B's exporter draws cohorts with `np.random`). Passed through under `extras.member_b_simulated_cohorts`, labelled; never shown as results |
| Queue Race | A **simulation** on real scores. Not clinical proof |
| Chest-tube shortcut audit | **Unavailable**: no chest-tube annotation exists in either dataset copy |
| Unit-test fixture | **Synthetic**, built in `backend/tests/conftest.py`, labelled `synthetic_test_fixture`; the API reports `data_is_real: false` for it. Never shipped |
| CheXpert | Aggregate numbers only (A's report tables, and calibration error computed at build time). No CheXpert image, label or per-image row is in the repo or on the server |
| Unverified | Hospital prevalence, label quality, and any behaviour on real non-chest X-rays (A tested synthetic non-chest images only) |

## 5. Queue order and precedence

One function, `backend/app/services/queue.py`, ranks the live worklist and the simulated one.

Mode `model`:
1. `clinician_urgent`
2. waited at least `max_wait_minutes` (reason `wait_cap`)
3. effective priority: `raised`, then `unsure`, then `normal`
4. oldest `uploaded_at` first; ties broken by arrival counter (stable)

Mode `fcfs`: clinician requests first, then arrival order. The model, the wait cap and overrides play no part.
Reviewed scans are listed last, `rank: null`. Nothing is ever dropped or hidden.

How the rules interact (all tested in `tests/test_queue.py` and `tests/test_studies.py`):

- **Model lane.** `raised` if the composite score is at or above the cut-off, `unsure` if it is in the band just below,
  `normal` otherwise. Out of distribution, no score, or no cut-off available: always `unsure`, never `normal`.
- **"Unsure ranks above normal"** holds inside the model-priority step (step 3). The explicit higher rules can still
  place a `normal` scan first: a clinician request (step 1) or the wait cap (step 2).
- **Inside the clinician group and the wait-cap group** steps 3 and 4 still apply.
- **Coordinator override** replaces the model's lane in step 3 (`effective_priority`); the model's own `priority` is
  never rewritten. It needs a reason and is logged. It can raise any scan and can lower a scored, in-distribution
  scan. It cannot set an out-of-distribution scan to `normal` (409). A human-lowered scan is still promoted by the
  wait cap and by a clinician request.
- **Reason shown** (`priority_reason`): clinician request, else wait cap, else override, else model.
- **Within a lane the order is oldest first**, as the contract says. The task brief allowed "higher confidence first
  inside raised"; the contract does not, so it is not done. See open decision 2 in section 11.

### Vocabulary mappings (`backend/app/services/urgency.py`)

| Contract | Member A | Member B |
|---|---|---|
| `raised` | `Raised` | `STAT`, `URGENT` |
| `unsure` | `AI unsure` (score band, or OOD) | no tier; B's classifier sends OOD to `STAT`, the contract says `unsure` |
| `normal` | `Normal` | `ROUTINE` |
| `ready` / `use_with_caution` / `not_ready` | | `GO` / `WARNING` / `NO-GO` |
| `not_enough_evidence` | | sample too small or interval lower end below target |

Urgent findings: A's list (pneumothorax, effusion, edema, consolidation, pneumonia). B's catalogue files the same
findings under STAT or URGENT and atelectasis and cardiomegaly under ROUTINE, so they agree; B has no separate rule for
consolidation, which follows A's list. A test fails if the two lists diverge. This urgency split is a provisional
assumption that needs clinician review, as both A and B say.

## 6. Threshold, and where each number comes from

- **Scale.** The threshold `value` is a percentile of the composite score on the reference set, not a raw probability
  (reason in the contract). `unsure` = the next `unsure_budget` (10%) of scans below the cut-off, plus every OOD scan.
- **Splits.** A made patient-level NIH splits: train 78,934, validation 11,341, test 21,845. The backend ships validation
  and test. **The default threshold, the floor check and the top-level preview numbers use validation only. The test
  split is reported under `held_out_test` and is never used to pick a threshold.**
- **Default.** The highest value on a 0.01 grid whose validation sensitivity is at least the target. The raw cut-off
  comes from B's `select_threshold_at_sensitivity`; B rounds it to four decimals (0.05), which by itself drops
  sensitivity to 89.99%, so the backend snaps down to the grid and re-checks with B's `compute_binary_metrics`.
  Result: **0.40** (score cut-off 0.0477).
- **What 0.40 means on validation:** sensitivity 90.1%, false alarms 51.7%, AI unsure 10.5%. On held-out test:
  sensitivity 91.8%, false alarms 52.2%, 384 of 4,659 urgent scans not raised (203 of them left `normal`).
- **Zero is reported when it is measured, never implied.** The preview returns counts with `n`; its note says error
  is never zero in practice and that public-dataset prevalence is not hospital prevalence.

## 7. Queue Race

`POST /simulations/queue-race`. Cases are drawn with replacement from the held-out NIH test predictions (urgent and
non-urgent separately, to hit the requested share). One reader, Poisson arrivals, log-normal read times.

- `fcfs`: Member B's `QueueSimulator.run_simulation(strategy="fifo")`.
- `voxelsort`: the live queue rule on the same stream. A test checks that the two event loops agree on FCFS.
- `b_engine`: B's own `ai_triage` ordering through B's `run_queue_race_comparison`, for comparison.
- `missed_urgent` = truly urgent and `normal`; `wrong_flags` = not urgent and `raised`. FCFS has `null` for both.

Example (defaults, seed 1, 169 scans, 25 urgent): urgent wait 66.8 min FCFS, 35.2 min VoxelSort, others wait 5.8 min
longer, 1 missed urgent scan, 70 wrong flags; B's ordering gives 9.5 min because it orders by score inside a tier.
"Truly urgent" is the dataset label (text-mined, noisy). Clinician requests and overrides are not simulated.
**B's headline in `locked_evidence.json` (STAT breaches 52% to 0%) is from B's synthetic cohort and is not shown as a result.**

## 8. Site Readiness

Rules and their limits are written at the top of `backend/app/services/site_readiness.py`. In short:

- Sensitivity is on the composite label "any urgent finding", counting a scan as caught when it is `raised` at the
  current cut-off, with an exact (Clopper-Pearson) two-sided 95% interval. `target` = `min_sensitivity_target`.
- `not_ready` needs demonstrated evidence: the interval's upper end below the target, or B's NO-GO beyond chance.
  A small or straddling sample is `not_enough_evidence`. `ready` needs lower end at or above target, at least 30
  urgent scans, and B's GO.
- **Limit of B's drift rule.** PSI on a small sample is large by chance (about 9 / n expected), so B's cut-offs
  (0.10, 0.25) fire on small samples drawn from the reference data itself. Measured here: 8 in-distribution images
  give max PSI 5.0 and B's rule says NO-GO. The backend therefore counts a PSI NO-GO only if it exceeds the 95th
  percentile of the same statistic for random reference samples of the same size, and an OOD-rate NO-GO only with at
  least 50 images. With B's cut-offs a clean GO needs a few hundred images. B should confirm or change this.
- **Limit of the mapping.** A drift NO-GO gives `not_ready`, but the frontend's fixed sentence for `not_ready` talks
  about the sensitivity interval. The API `note` states the real reason.
- Member A's drift score (domain AUROC, at least 20 images) is reported as `drift.domain_auroc`. It reaches about 1.0
  for training images merely re-saved as JPEG, so it means "these images differ, check them", nothing more.
- Sample size: `positives_needed` is the larger of B's normal-approximation formula (margin = expected minus target)
  and the exact search the verdict actually applies. For 0.90 target and 0.95 expected: 73 and 127; 127 is returned.
- Labels CSV: `filename,label`. A `score` column is accepted and reported as uploader-supplied.

## 9. Evaluation provenance and limitations

Everything on `/evidence` is for `partA-0.1.0+4ca280b3cd`.

- **Ranking (NIH test, 21,845 images):** "any urgent finding" AUROC 0.811 (0.801 to 0.822). Pneumonia is weak (0.687).
- **Different hospital's dataset (CheXpert):** 0.811 to 0.755. Labels, prevalence and image format all contribute.
- **Calibration, measured, not claimed.** B's calibration code on real predictions, fitted on validation, measured on
  test: expected calibration error 0.0005 to 0.015 per finding on NIH test, and temperature scaling changes almost
  nothing (temperatures 0.95 to 1.03). On CheXpert the same probabilities have calibration error up to 0.30
  (effusion). **Do not describe the model as well calibrated.** B's earlier figure (0.443 to 0.432) is from B's
  synthetic cohort and is not shown as a result.
- **AP and PA views.** AUROC 0.749 (AP) against 0.834 (PA). B's rule checks only the AUROC gap (0.084, limit 0.10) and
  says "robust", but at the current cut-off AP scans are raised far more often (false-alarm rate 0.705 against 0.425).
  Both are shown.
- **Enriched prevalence.** "Any urgent" is 21% of NIH and 72% of CheXpert. Positive predictive value must be recomputed
  for a hospital's own prevalence.
- **Labels are noisy** (NIH text-mined; CheXpert labeller-extracted). The urgent list is provisional.
- **OOD detector** flagged 99 to 100% of gross mistakes in A's tests, but the same mistake types trained it, and it
  was not tested on real non-chest X-rays. It flags 6.7% of CheXpert: it is not a site-shift detector.

## 10. Hosting, persistence and deployment

Details of the live deployment are in section 12.

- **Render free web service** (verified): spins down after 15 minutes without traffic and takes about a minute to wake
  ([Render docs](https://render.com/docs/free)); the filesystem is ephemeral, no persistent disk on the free plan;
  Render may restart the instance at any time; 750 instance hours per month. This service's limits, read from
  Render's metrics API: **512 MiB memory, 0.15 CPU**.
- **What survives a restart, redeploy or wake-up:** the code, the model, the saved predictions, the evidence and the
  12-study demo queue (re-seeded). **What is lost:** uploads and their images, overrides, reviews, urgent requests, the
  audit log, queue settings and any threshold change (it returns to 0.40). A real deployment needs a persistent
  database; `DATABASE_URL` accepts any SQLAlchemy URL, but only SQLite has been tested.
- **Memory.** Measured, not assumed. On Windows (`scripts/measure_inference.py` and the running server): about 245 MB
  with the model loaded, 285 MB after uploads, a Site Readiness run and a Queue Race (peak 293 MB). **On Render
  (its metrics API): 170 MB at start-up, 284 MB after real inference on uploads, a Site Readiness run and the largest
  allowed Queue Race, against the 512 MiB limit.** So the full runtime (ONNX backbone, head, scikit-learn OOD detector,
  pandas and scipy for B's code) fits with about 230 MB to spare. The largest Site Readiness run (100 images) was not
  load-tested on Render.
- **No login.** `actor` is a typed name. "Clinical lead only" for threshold changes cannot be enforced: the
  confirmation and the audit log are the only controls. The audit log is not tamper-proof.
- **CORS.** `ALLOWED_ORIGINS` plus an optional pattern. Never `*` in production.
- **Vercel.** `VITE_API_URL` is read at build time, so it must be set in the Vercel project and the site redeployed.

## 11. Open decisions for the team (not hidden)

1. **Sensitivity floor against A's budget.** The contract's 90% floor forces threshold 0.40, which raises about 60% of
   scans (false alarms 52%). A's own policy raises the top 20% (sensitivity 47%) because, in A's words, at 90% "the
   queue would barely change". Under the floor A's setting cannot be applied. To adopt it: set
   `MIN_SENSITIVITY_TARGET=0.45`, then `PUT /threshold {"value": 0.80, "confirm": true}`. This is a clinical-lead decision.
2. **Order inside the raised lane.** Oldest first (contract) gives 35 min urgent wait in the default race; ordering by
   score inside the lane (B's simulator, and what the task brief permitted) gives about 10 min. Changing it changes the contract.
3. **B's drift cut-offs on small samples** (section 8).
4. **`policy.json` line endings under `ml/`** (section 3).
5. **Frontend wording** (Member D's files, not changed):
   - `StudyReview.jsx`: "Calibrated model output", "Calibrated probability", "Grad-CAM heatmap". The probabilities are
     the model's raw outputs, and the map is a class-activation map of a linear head.
   - `Worklist.jsx`: an out-of-distribution or unscored scan shows "Processing..." as its top finding, because `findings` is empty.
   - `Evidence.jsx`: the calibration panel shows two curves without naming the finding or the dataset, and does not
     show `calibration.note` or `provenance`. The Cross-dataset table does carry the calibration error for both datasets.
   - `SiteReadiness.jsx`: the fixed sentence for `not_ready` mentions only the interval; the API `note` has the real reason.
   - `SafetySpeed.jsx`: "Decision threshold 0.40" does not say it is a percentile; `GET /threshold` returns `scale` and `score_cutoff`.

## 12. Deployment record

**Backend: deployed and verified on 2026-10-11.**

| | |
|---|---|
| Service | Render `chesttriage-api` (`srv-db57lk5ckfvc7398gkn0`), free plan, Singapore, Docker, context `./backend`, branch `main` |
| URL | `https://chesttriage-api.onrender.com` (docs at `/docs`) |
| Commit | `3f6f7e1`, deploy `dep-db5e1h0473hc73aqo07g`, status live |
| Environment set on Render | `APP_ENV=prod`, `MODEL_MODE=auto`, `MIN_SENSITIVITY_TARGET=0.9`, `DATA_DIR=/app/data`, `ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`, `ALLOWED_ORIGIN_REGEX=https://([a-z0-9-]+\.)*vercel\.app` |
| `/health` | `status ok`, `model_mode real`, `degraded false`, model `partA-0.1.0+4ca280b3cd`, 11,341 + 21,845 real saved predictions, SQLite WAL |
| End-to-end check | `scripts/verify_deploy.sh`: 27 of 27 checks passed against the live URL (health, worklist, upload with real inference, image, OOD check, override, audit, review, threshold confirm and floor, Queue Race, Site Readiness, sample size, evidence, CORS allowed and refused) |
| Speed on Render | about 0.5 s per image for a forward pass; default Queue Race 0.2 s; the largest allowed Queue Race (1,500 scans) 11.6 s; threshold preview 0.2 s |
| CI | GitHub Actions "backend tests" passed on Linux, Python 3.13, including the real-model tests |

Things to know:

- **A push to `main` does not start a Render build on this service** (observed again: the build came from the API
  call, not from the push). After a backend change, press "Manual Deploy" in the Render dashboard or trigger a deploy
  through the API.
- **Docker was not built locally** (Docker is not installed on the development machine). The image was built and run
  by Render only.
- The model files are downloaded during the Docker build from this repository at a pinned commit and checked against
  `SOURCE.json`. If GitHub is unreachable at build time the image still builds and the service runs in replay mode;
  `/health` then shows `model_mode: "replay"`.
- **CORS is wider than it should be:** any `https://*.vercel.app` origin is allowed (kept from the previous backend,
  because the frontend's final address is not known). Once it is, set `ALLOWED_ORIGINS` to that address and narrow or
  remove `ALLOWED_ORIGIN_REGEX`.

**Frontend: not deployed by me.** The Vercel account connected to this session contains one project,
`chesttriage-cors-test` (a CORS test page). There is no VoxelSort frontend project in it and none linked to the GitHub
repository, and I was asked not to create replacement hosting. What was verified instead:

- `npm run build` succeeds with `VITE_API_URL=https://chesttriage-api.onrender.com` (the address is in the bundle).
- The unmodified frontend, run locally (`npm run dev`) against the **live** backend in a real browser: the worklist
  showed the live queue, Study Review loaded the live image and heatmap, and "Clinician requests urgent read" moved the
  scan to position 1 and appeared in the Audit log. Against a local backend every page was exercised: Worklist, Upload
  (multipart from the browser origin), Study Review with heatmap, Safety vs speed, Queue Race, Site readiness, Evidence
  (calibration, failure gallery), Audit log.
- **No change to `frontend/src/services/api.js` was needed**: the backend matches what it sends and expects.
- The frontend has no tests of its own (`package.json` has no test script).

To put the frontend on Vercel: create or open the project with root directory `frontend`, framework Vite, add the
environment variable `VITE_API_URL=https://chesttriage-api.onrender.com` for Production and Preview, and deploy.
`VITE_` variables are read at build time, so a change needs a new deployment. Then run
`scripts/verify_deploy.sh https://chesttriage-api.onrender.com https://<your-frontend>.vercel.app` to check CORS for it.

## 13. Ten-line demo script

Replace the base URL with `http://localhost:8000` for a local run. `scripts/verify_deploy.sh` runs all of this with checks.

```bash
B=https://chesttriage-api.onrender.com
curl -s $B/health                                                        # what is running: model_mode, degraded, saved predictions
curl -s $B/studies                                                       # the ranked queue (12 demo studies after a restart)
curl -s -o scan.png $B/studies/nih_00015355_007/image                    # a public NIH demo image
curl -s -F "file=@scan.png;type=image/png" -F actor=demo $B/studies      # 202, status "processing"; note the study_id
curl -s $B/studies/CXR-1013                                              # poll with YOUR study_id: findings, priority, plain_language, heatmap_path
curl -s -X POST -H "Content-Type: application/json" -d '{"priority":"raised","reason":"clinical concern","actor":"demo"}' $B/studies/CXR-1013/override
curl -s "$B/audit?q=CXR-1013"                                            # upload, scoring and override entries
curl -s -X POST -H "Content-Type: application/json" -d '{"value":0.4}' $B/threshold/preview        # sensitivity, false alarms, held_out_test
curl -s -X PUT  -H "Content-Type: application/json" -d '{"value":0.8,"confirm":true}' $B/threshold # refused: below the 90% floor
curl -s -X POST -H "Content-Type: application/json" -d '{"arrivals_per_hour":20,"reads_per_hour":18,"urgent_share":0.15,"hours":8,"seed":1}' $B/simulations/queue-race
```
