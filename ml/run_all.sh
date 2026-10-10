#!/usr/bin/env bash
# Full Part A run into a versioned folder. Order matters: 04 (OOD) before 05 (export).
set -e
export CHESTTRIAGE_ARTIFACTS=${CHESTTRIAGE_ARTIFACTS:-artifacts_v2_imagenet_fullnih}
python scripts/00_check_setup.py --download --nih_n 200000 --backbone densenet121-onnx   # or densenet121 with torch
# python scripts/00b_finetune_backbone.py --workers 2      # optional, timeboxed; compare with 09 before adopting
python scripts/01_extract_features.py                 # slowest step; resumable
python scripts/02_train_head.py
python scripts/03_cross_dataset_eval.py
python scripts/04_build_ood.py                        # add --nonchest_dir some_folder for real non-chest images
python scripts/04b_drift_controls.py
python scripts/05_export_predictions.py               # budget policy; --raise_budget / --mode sensitivity to change
python scripts/06_heatmaps_and_failures.py
python scripts/07_freeze.py                           # Phase 4: MANIFEST.json, consistency checks
python scripts/08_queue_race.py                       # starting point for Member B
# python scripts/09_compare_models.py --a artifacts_v1_imagenet_frozen --b $CHESTTRIAGE_ARTIFACTS
