# ChestTriage - Part A (ML lead)

Everything Member A owns: data, cached features, multi-label head, cross-dataset test, OOD detector,
drift score, saliency maps, capacity-budget policy, exported predictions, a live inference engine for
the app, and a real-prediction Queue Race as a starting point for Member B. Decision support only:
a clinician reads every scan.

## Model folders (read this first)
| Folder | What | Status |
|---|---|---|
| `artifacts_v2_imagenet_fullnih/` | ImageNet DenseNet121 (frozen) + linear head trained on all NIH **train** patients (78,934 images) | **Frozen model**, `partA-0.1.0+4ca280b3cd` |
| `artifacts_v1_imagenet_frozen/` | Same backbone, head trained on a 30,000-image NIH subset | Fallback, untouched |
| `artifacts/` | Intermediate working copy from before the version split | Not used by any report; safe to delete |

Every report states the folder and `model_version` it came from (CSV columns `artifacts_folder`,
`model_version`; a line at the top of each `.md`). Point scripts at a folder with
`CHESTTRIAGE_ARTIFACTS=artifacts_v2_imagenet_fullnih`. The version is a hash of `backbone.json` +
`head.npz` + `policy.json`, so any change to the model, calibration or thresholds changes it.

**v1 vs v2** (`scripts/09_compare_models.py`, report in v2 `reports/model_comparison.md`): both heads scored on the
same 21,845 NIH test images (6,162 patients) that neither model used for training or for choosing settings.
v2 is higher on every finding (paired bootstrap difference CI above zero), by +0.008 to +0.039 AUROC
(+0.010 on "any urgent finding"); the marginal CIs of the two models overlap. Same model type, more data.

**Not in the git repo (regenerate with `run_all.sh`):** the datasets, feature caches (`features/`), patient subsets,
per-study predictions (`predictions/`), demo images, heatmap panels and `where_it_fails.csv`. They are large, contain
local paths, or are derived from NIH/CheXpert images, and CheXpert's research-use agreement does not allow
redistribution in a public repo. The model files, `MANIFEST.json` and the summary reports are included.

## Run it
```bash
pip install -r requirements.txt
export CHESTTRIAGE_ARTIFACTS=artifacts_v2_imagenet_fullnih
./run_all.sh                    # or the scripts one by one (order below)
python tests/test_pipeline.py   # dry run on synthetic data, no torch, no downloads (~2 min)
```
**No torch? Use `--backbone densenet121-onnx`** (what v1 and v2 use). Same torchvision ImageNet DenseNet121, same
weights, run by ONNX Runtime (`pip install onnxruntime-directml onnx` on Windows GPU). Use it where Windows Smart
App Control blocks torch's unsigned DLLs, or to deploy without torch. Checked: an ImageNet test image (Grace Hopper)
gives top-1 'military uniform'; CPU and GPU outputs agree to 4e-6.

## What each step does
| Script | Output |
|---|---|
| `00_check_setup.py` | Hour-one gate: finds both datasets, patient-level splits, label-overlap table, backbone smoke test. Prints `GATE: PASS/FAIL` (FAIL names the AquaTriage fallback) |
| `00b_finetune_backbone.py` | OPTIONAL, NOT RUN. Fine-tunes the last DenseNet block on NIH train (val for early stopping), exports to ONNX. `--workers` 0-2 only |
| `01_extract_features.py` | Cached backbone features (resumable; cache keyed by backbone tag) |
| `02_train_head.py` | `head.npz`, `reports/internal_metrics.csv` (per-finding AUROC, patient-level bootstrap CI) |
| `03_cross_dataset_eval.py` | Train NIH, test CheXpert with thresholds fixed on NIH validation. `reports/cross_dataset.{csv,md,png}` |
| `04_build_ood.py` | `ood.joblib`, `reports/ood_report.csv`, `drift_reference.npy`, `reports/drift_report.csv` |
| `04b_drift_controls.py` | JPEG / resize+JPEG / NIH-vs-NIH controls for the drift score, and the effect of re-encoding on predictions. `reports/drift_controls.md` |
| `05_export_predictions.py` | `policy.json` (budget cut-offs from NIH validation), `predictions/*_full.csv`, `*_contract.jsonl`, `reports/budget_table.{csv,md}` |
| `06_heatmaps_and_failures.py` | `reports/where_it_fails.csv`, demo bundle (`predictions/demo_contract.jsonl`, `demo_images/`, `heatmaps/`), `reports/heatmap_sheet.png`, `reports/heatmap_attention.md` |
| `07_freeze.py` | Phase 4 freeze: `MANIFEST.json` (model_version, policy, SHA-256 of deploy files, library versions). Fails if any export is stale |
| `08_queue_race.py` | Queue Race on real predictions, assumed hospital prevalence, arrival-rate grid. Starting point for Member B. `reports/queue_race.{csv,md,png}` |
| `09_compare_models.py` | Two model folders on the same clean NIH test patients, paired bootstrap |

## Results (v2, `partA-0.1.0+4ca280b3cd`)
**Ranking on NIH test** (21,845 images, never used for training or settings), AUROC with 95% patient-bootstrap CI:
"any urgent finding" **0.811 (0.801-0.822)**; effusion 0.836, edema 0.863, pneumothorax 0.839, cardiomegaly 0.821,
atelectasis 0.756, consolidation 0.750. Pneumonia is weak (0.687) and is not a headline finding.

**Capacity budget** (`reports/budget_table.md`). With the Raised lane set to the top 20% of scans, it holds about half
of the urgent scans on NIH test (sensitivity 0.50; random picking 0.20; the best possible ranker 0.94), at PPV 0.53
at NIH's own prevalence. PPV must be recomputed at the assumed hospital prevalence. A fixed NIH cut-off raised 40% of
CheXpert scans instead of 20%; the rank-based mode (top X% of the current queue) holds the budget at any site.

**Queue Race** (`reports/queue_race.md`, a simulation under stated assumptions, not clinical proof): improved
ordering over arrival order. Example, NIH test, assumed 10% prevalence, readers 95% busy: mean wait of urgent scans
11.2 min first-come-first-served, 4.7 min with rank-based lanes, 2.9 min ordered by score; non-urgent scans wait a
little longer (11.4 -> 12.3 to 12.4 min). The lowest-scoring scans wait up to the 180 min cap under score/rank ordering,
which is why the wait cap is part of the design.

**Different hospital's dataset** (`reports/cross_dataset.md`): performance dropped on a different hospital's dataset
(CheXpert): "any urgent finding" AUROC 0.811 -> 0.755, per-finding drops 0.07-0.14. Part of this may come from label
definitions, prevalence and image format (CheXpert is JPEG; re-encoding NIH as JPEG alone costs about 0.007 AUROC), so
do not attribute it to scanners or patients alone.

**OOD check** (`reports/ood_report.csv`): catches most gross mistakes in our tests (rotated, flipped, inverted, very
noisy, blurred, low-contrast, synthetic non-X-ray images: 99-100% flagged) while flagging 0.6% of NIH test images. These
mistake types were also used to train the check, so real-world catch rates will be lower; it has not been tested on real
non-chest X-rays (use `04_build_ood.py --nonchest_dir`). It flags only 6.7% of CheXpert, so it is not a site-shift
detector.

**Drift score** (`reports/drift_controls.md`): **it detects file format.** NIH vs NIH scores 0.48; NIH vs other NIH
images re-saved as JPEG scores 0.998, and resized+JPEG 0.997, about the same as NIH vs CheXpert (1.00). Same-hospital
AP-only or PA-only samples also score 0.71-0.81. So a high site score means "these images differ from training,
check them", not "different patients or scanner", and never "accuracy dropped". Only labelled site data shows that.
Site samples under 20 images are refused.

**Saliency maps** (`reports/heatmap_sheet.png`, `reports/heatmap_attention.md`): coarse 7x7 maps; they show where the
model's evidence came from, not that it looked at the right thing. By eye: NIH effusion hits sit on the lower chest
(lower zones, diaphragm, heart base). CheXpert pneumothorax hits and several false alarms sit on lateral edges,
shoulders, tubes and lines, and at least one sits on a text marker ("PORT SEMI-UPRIGHT"), which suggests shortcuts.
Missed cases sit on the top band (neck, clavicles). Use CheXpert maps in "Where it fails", not as explanations.

## Design decisions (state these on a slide)
- **Findings:** the 7 shared by NIH and CheXpert. Multi-label, one probability each.
- **Labels:** NIH text-mined. CheXpert blank = negative, **uncertain (-1) = masked**, frontal views only.
- **Splits:** by patient. NIH 70/10/20 train/val/test; CheXpert is the external set (`valid.csv` kept as `gold_valid`).
  NIH test is never used for training, thresholds or model choice.
- **Backbone:** frozen ImageNet DenseNet121, deliberately not a chest X-ray model trained on NIH/CheXpert (that would
  contaminate the cross-dataset test).
- **Head:** one logistic regression per finding, no class re-weighting.
- **Urgency (PROVISIONAL, needs clinician review):** pneumothorax, effusion, edema, consolidation, pneumonia.
  Composite score = max urgent probability.
- **Policy (`policy.json`, mode `budget`):** Raised = top 20% of scans by score, AI unsure = next 10% plus every OOD
  image, Normal = the rest; cut-offs are NIH validation quantiles. `Policy.queue_rank()` applies the same budget to the
  current queue instead. On NIH validation, Raised holds 47% of urgent scans and Raised + AI unsure 65%
  (`min_sensitivity` check: 60%, warns if missed). A 90%-sensitivity cut-off would raise about 60% of all scans, so
  the queue would barely change. The budget and the minimum sensitivity are clinical-lead decisions.
  Nothing is ever removed from the queue.
- **De-duplication:** one row per image file.

## Handoffs
**Member B:** see [HANDOFF_B.md](HANDOFF_B.md).

**Member C** (engine)
```python
from chesttriage_ml.engine import ChestTriageEngine
engine = ChestTriageEngine("artifacts_v2_imagenet_fullnih")   # load once at startup
rec = engine.predict("scan.png", study_id="S1", heatmap_dir="static/heatmaps")   # path, PIL image or bytes
recs = engine.predict_batch([...], [...])
lanes = engine.policy.queue_rank([r["composite_score"] or 0 for r in queue], [r["ood_flag"] for r in queue])
rows = engine.analyse_batch([...], [...])   # Site Readiness: scores even for OOD images, plus ood_flag
drift = engine.site_drift([...])            # Site Readiness: domain AUROC (20+ images), OOD rate; a warning only
engine.reload()                             # after the clinical lead approves a policy.json threshold edit
```
Queue semantics the safety design needs: `Raised` moves a scan up. `Normal` keeps its first-come-first-served place
and is never moved down or removed. `AI unsure` (including every OOD image) never ranks below `Normal`. Clinician
requests, the wait cap and the FCFS fallback override all of these.
Demo seeding: `predictions/demo_contract.jsonl` (paths relative to the artifacts folder; OOD rule applied, as live).
Deploy: `backbone.json`, `backbone_densenet121-onnx.onnx`, `head.npz`, `policy.json`, `ood.joblib`,
`drift_reference.npy` (checksums in `MANIFEST.json`; `ood.joblib` is a pickle, use the scikit-learn version listed).

**Member D** (contract record)
`study_id, findings{}, priority (Raised|Normal|AI unsure), confidence_band (high|low|none), ood_flag, heatmap_path`,
plus `composite_score, top_urgent_finding, flagged_findings, ood_score, model_version` (and `image_path` in the demo
bundle). For OOD images `findings` is `{}`. Build the plain-language card from these fields only. Do not describe the
heatmap as proof of anything.

## Known limits (be upfront)
- Frozen ImageNet features give a modest model (0.81 on "any urgent finding"). Fine-tuning the last block is
  scripted (`00b`) but not run.
- NIH labels are noisy and CheXpert labels are model-extracted. The urgency tiers are an assumption.
- Public data is enriched (NIH 21%, CheXpert 72% "any urgent"); PPV and queue results depend on prevalence.
- The Queue Race is a simulation; clinician requests and the fallback switch are not modelled.
- **Licences:** `--download` uses Kaggle mirrors. CheXpert is under the Stanford AIMI research-use agreement and NIH
  ChestX-ray14 has its own terms. Check both before you show or share results built on them.
