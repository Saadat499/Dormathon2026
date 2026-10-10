# model_artifacts/

Drop Member A's output here. The backend reads it at start-up.

Expected files (the loader in `app/cohorts.py` accepts either):

* `predictions.json`: a list (or `{"predictions": [...]}`) of records
  `{"study_id": "...", "findings": [{"name": "pneumothorax", "probability": 0.82}, ...],
    "label": 1, "ood_flag": false, "confidence_band": "0.71-0.89", "sha256": "<optional image hash>"}`
  `label` = 1 if the scan is truly urgent, 0 if not (needed for Safety vs Speed and Queue Race).
* `predictions.csv`: columns `study_id,label` plus one probability column per finding
  (optional `ood_flag,confidence_band,sha256`).

Check what the backend found with `GET /status`. If the files are missing or unreadable the API
falls back to a clearly labelled synthetic cohort and says so in `/status`.
