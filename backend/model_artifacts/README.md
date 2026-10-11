# model_artifacts/

Everything the backend needs from Member A's model, for the selected version
**`artifacts_v2_imagenet_fullnih`, `model_version` `partA-0.1.0+4ca280b3cd`**.
`SOURCE.json` records how each file was produced and verified.

| File | In git | What it is |
|---|---|---|
| `MANIFEST.json` | yes | Member A's manifest (checksums, policy, library versions), unchanged |
| `SOURCE.json` | yes | Provenance and verification results written by `scripts/build_model_artifacts.py` |
| `saved_predictions_nih.csv.gz` | yes | Real model outputs with true labels: NIH validation (11,341) and held-out test (21,845) images. Scores only, no images |
| `evidence.json` | yes | Real evaluation results served by `GET /evidence` |
| `demo/demo_studies.json`, `demo/images`, `demo/heatmaps`, `demo/failures` | yes | 59 public NIH ChestX-ray14 demo images (plus A's synthetic non-chest test images) with their saved model outputs, keyed by image SHA-256 |
| `backbone.json`, `backbone_densenet121-onnx.onnx`, `head.npz`, `policy.json`, `ood.joblib`, `drift_reference.npy` | **no** (git-ignored) | The model itself. Already in the repo under `ml/artifacts_v2_imagenet_fullnih/`; copied or downloaded here by `scripts/fetch_model.py`, which verifies every SHA-256 |

```powershell
# from backend\
.\.venv\Scripts\python.exe scripts\fetch_model.py          # get the model files (copies from ..\ml\ when present)
.\.venv\Scripts\python.exe scripts\fetch_model.py --check  # verify only
```

With the model files present the backend runs **real** inference. Without them it runs in **replay**
mode: an upload that is byte-identical to one of the demo images returns that image's saved output;
any other upload gets no model output and stays in the queue as "AI unsure". `GET /health` says which.

Rebuilding from Member A's full delivery (needs its `predictions/`, `demo_images/` and `reports/`
folders, which are not in git):

```powershell
.\.venv\Scripts\python.exe scripts\build_model_artifacts.py --delivery "C:\Users\User\Downloads\chesttriage_ml\chesttriage_ml"
```

CheXpert images, labels and per-image predictions are deliberately not here: the Stanford research-use
agreement does not allow redistribution. CheXpert appears only as aggregate numbers in `evidence.json`.
NIH ChestX-ray14 is public (NIH Clinical Center; cite Wang et al., CVPR 2017).
