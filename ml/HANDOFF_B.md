# Handoff to Member B (evaluation and safety)

Frozen model: `artifacts_v2_imagenet_fullnih/`, `model_version` `partA-0.1.0+4ca280b3cd`.
Fallback: `artifacts_v1_imagenet_frozen/`. Every file below carries `artifacts_folder` and `model_version`
(CSV columns, or a line at the top of the `.md`). If you see a different version, the numbers are not comparable.

## Files (all inside the artifacts folder)
| File | What it is |
|---|---|
| `predictions/nih_full.csv`, `predictions/chexpert_full.csv` | One row per image: labels, logits, probabilities, scores, decisions (columns below) |
| `reports/budget_table.csv` / `.md` | Capacity budget -> sensitivity, false-alarm rate, flag rate, PPV, ceiling (for the slider and simulator) |
| `reports/internal_metrics.csv` | NIH test AUROC per finding with patient-bootstrap CIs |
| `reports/cross_dataset.csv` / `.md` / `.png` | NIH -> CheXpert drop at thresholds fixed on NIH validation |
| `reports/model_comparison.md` | v1 vs v2 on the same clean NIH test patients (paired bootstrap) |
| `reports/ood_report.csv` | What the OOD check flags (test cases and real data) |
| `reports/drift_report.csv`, `reports/drift_controls.md` | Drift score on real samples and the JPEG / resize controls |
| `reports/format_effect_on_predictions.csv` | AUROC on NIH test as PNG vs re-saved JPEG |
| `reports/queue_race.csv` / `.md` / `.png` | Queue Race on real predictions (starting point for your simulator) |
| `reports/where_it_fails.csv`, `reports/heatmap_sheet.png`, `reports/heatmap_attention.md` | Failure examples and saliency audit |
| `policy.json` | Urgent list, budgets, cut-offs (NIH validation), temperatures, what was measured at fit time |
| `features/<dataset>_densenet121-onnx/feats.npy` | Backbone features for the shortcut audit, aligned to `subsets/<dataset>.csv` rows where `ok.npy` is 1 |

## Columns of `*_full.csv`
- `study_id, dataset, split, patient_id, age, sex, view` (NIH split: `train` / `val` / `test`; CheXpert: `external` / `gold_valid`)
- `y_<finding>` label: 1, 0, or empty (= CheXpert "uncertain", masked)
- `logit_<finding>`, `p_<finding>`: raw logit and probability (after any temperature in `policy.json`)
- `label_any_urgent`: 1 if any urgent finding is positive, 0 if none, empty if none positive but one is masked
- `composite_score`: max probability over the urgent findings (the queue score)
- `ood_flag`, `ood_score`: OOD check result and distance
- `priority`, `confidence_band`: the decision the app shows (fixed cut-offs, OOD -> AI unsure)
- `priority_score_only`: the decision ignoring OOD (use this for performance analysis)
- `model_version`

Use NIH `val` for anything you fit (calibration, thresholds). Report on NIH `test` and CheXpert `external` only.

## Caveats you must carry into every number
1. **Enriched prevalence.** "Any urgent" is 21% in NIH and 72% in CheXpert, far above a typical hospital worklist.
   PPV in the budget table is at each dataset's own prevalence. Recompute it at the assumed hospital prevalence p:
   `PPV = sens*p / (sens*p + FAR*(1-p))`. Make p an explicit simulator parameter.
2. **Noisy labels.** NIH labels are text-mined from reports; CheXpert labels are extracted by a labeller.
3. **Masked uncertain labels.** CheXpert "uncertain" (-1) is ignored (empty), not counted as negative or positive.
4. **Provisional urgent list.** pneumothorax, effusion, edema, consolidation, pneumonia. It needs clinician review;
   after any change, edit `policy.json` and run `05_export_predictions.py --refit_policy` (the version changes).
5. **Drift confound.** The drift score (domain AUROC) is 0.998 for NIH images merely re-saved as JPEG and 0.71-0.81 for
   same-hospital AP-only / PA-only samples, vs 0.48 for NIH vs NIH. It detects file format and case mix as much as
   site shift. Present it as "these images differ from training, check them", never as "accuracy dropped".
6. **Cross-dataset drop.** 0.811 -> 0.755 on "any urgent finding". Label definitions, prevalence and format (CheXpert is
   JPEG; JPEG alone costs about 0.007 AUROC on NIH) all contribute. Say "performance dropped on a different hospital's
   dataset".
7. **OOD check.** Catches most gross mistakes in our tests, but the test mistake types were also used to train it, and it
   has not been tested on real non-chest X-rays. It is not a site-shift detector (flags 6.7% of CheXpert).
8. **Pneumonia** AUROC is 0.69 on NIH test; do not headline it.
9. **Saliency maps** are coarse and some CheXpert maps sit on edges, tubes and a text marker. They are not evidence that
   the model is right.

## Using the budget table
Columns: `mode, budget, dataset, score_cutoff, cutoff_source, flag_rate, sensitivity, ceiling_sensitivity,
random_sensitivity, false_alarm_rate, ppv_at_dataset_prevalence, dataset_prevalence, n`.
- `mode = fixed_cutoff`: the score cut-off is the NIH-validation quantile for that budget, applied unchanged. At
  CheXpert the realised `flag_rate` is about double the budget (scores run higher). Use this to show why a fixed
  cut-off drifts at a new site.
- `mode = queue_rank`: top X% of the queue by rank (`Policy.queue_rank`). `flag_rate` equals the budget at any site.
  This is the recommended mode for the app and the slider.
- `ceiling_sensitivity` = best any ranker could do at that budget (flag rate / prevalence, capped at 1). Show it next to
  `sensitivity`, especially for CheXpert, where a 20% budget can catch at most 28% of urgent scans.
- `random_sensitivity` = what random picking catches; the gap to `sensitivity` is what the model adds.

**Safety vs Speed slider:** x-axis = budget (5-40%). Plot `sensitivity` (with `ceiling_sensitivity` and
`random_sensitivity` as reference lines) and `false_alarm_rate`; compute PPV live from the slider's assumed prevalence
with the formula above. Use the `queue_rank` rows. Changing the deployed budget needs the clinical lead's confirmation
(the app writes `policy.json`, then `engine.reload()`; the version changes).

**Simulator:** prefer real per-scan scores over the summary table. Sample studies from `*_full.csv` (NIH `test` or
CheXpert `external`), resample to the assumed prevalence, and rank with `composite_score` (OOD scans at least above every
Normal scan) or with `Policy.queue_rank(scores, ood_flags)` on the current queue. `scripts/08_queue_race.py` does exactly
this and is yours to replace; its assumptions (Poisson arrivals over 8 h, 2 readers, 4 min lognormal reads, 180 min
wait cap, prevalence 10%/20%, arrival grid 24-33/h) are labelled in `reports/queue_race.md`. It does not model clinician
requests or the FCFS fallback switch. Under score or rank ordering the lowest-scoring scans wait up to the cap, so report
the maximum wait alongside the urgent wait.

## Site Readiness helpers
```python
from chesttriage_ml.engine import ChestTriageEngine
eng = ChestTriageEngine("artifacts_v2_imagenet_fullnih")
eng.site_drift(images)        # {'n_site', 'domain_auroc', 'ood_flag_rate', 'median_ood_score', 'model_version'}; refuses < 20 images
eng.analyse_batch(images)     # per image: p_<finding>, composite_score, priority_score_only, ood_flag, ood_score
```
- Reference values for the drift verdict: NIH vs NIH about 0.5 (`drift_report.csv`, `drift_controls.md`); JPEG-only 0.998.
- For the exact interval on the labelled site sample, use `label_any_urgent` (built the same way as
  `metrics.composite_label`) against the Raised (or Raised + AI unsure) decision from `analyse_batch` or `queue_rank`.
- Calibration: write `calibration.json` as `{"temperature": {"effusion": 1.3, ...}}` fitted on NIH `val` logits, then
  re-run `05`, `03`, `06`, `07` (the version changes). Only per-finding temperature is supported.
