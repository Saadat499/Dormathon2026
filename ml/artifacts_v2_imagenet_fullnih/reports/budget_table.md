# Capacity budget table

Artifacts folder: `artifacts_v2_imagenet_fullnih`, model_version: `partA-0.1.0+4ca280b3cd`

PPV is at each dataset's own (enriched) prevalence and must be recomputed at the assumed hospital prevalence p: PPV = sens*p / (sens*p + FAR*(1-p)). Cut-offs for fixed_cutoff come from NIH validation only.

| mode | budget | dataset | flag_rate | sensitivity | ceiling_sensitivity | false_alarm_rate | ppv_at_dataset_prevalence |
|---|---|---|---|---|---|---|---|
| fixed_cutoff | 0.050 | nih_test | 0.049 | 0.147 | 0.228 | 0.022 | 0.646 |
| queue_rank | 0.050 | nih_test | 0.050 | 0.151 | 0.235 | 0.023 | 0.643 |
| fixed_cutoff | 0.050 | chexpert_external | 0.099 | 0.125 | 0.137 | 0.032 | 0.912 |
| queue_rank | 0.050 | chexpert_external | 0.050 | 0.062 | 0.069 | 0.018 | 0.900 |
| fixed_cutoff | 0.100 | nih_test | 0.095 | 0.265 | 0.443 | 0.048 | 0.597 |
| queue_rank | 0.100 | nih_test | 0.100 | 0.279 | 0.469 | 0.051 | 0.595 |
| fixed_cutoff | 0.100 | chexpert_external | 0.201 | 0.249 | 0.277 | 0.074 | 0.898 |
| queue_rank | 0.100 | chexpert_external | 0.100 | 0.126 | 0.138 | 0.032 | 0.912 |
| fixed_cutoff | 0.150 | nih_test | 0.140 | 0.373 | 0.659 | 0.078 | 0.566 |
| queue_rank | 0.150 | nih_test | 0.150 | 0.394 | 0.703 | 0.084 | 0.560 |
| fixed_cutoff | 0.150 | chexpert_external | 0.295 | 0.363 | 0.408 | 0.117 | 0.890 |
| queue_rank | 0.150 | chexpert_external | 0.150 | 0.188 | 0.207 | 0.052 | 0.905 |
| fixed_cutoff | 0.200 | nih_test | 0.192 | 0.480 | 0.902 | 0.114 | 0.532 |
| queue_rank | 0.200 | nih_test | 0.200 | 0.496 | 0.938 | 0.120 | 0.529 |
| fixed_cutoff | 0.200 | chexpert_external | 0.399 | 0.482 | 0.552 | 0.182 | 0.874 |
| queue_rank | 0.200 | chexpert_external | 0.200 | 0.248 | 0.276 | 0.073 | 0.899 |
| fixed_cutoff | 0.300 | nih_test | 0.297 | 0.656 | 1.000 | 0.200 | 0.471 |
| queue_rank | 0.300 | nih_test | 0.300 | 0.660 | 1.000 | 0.203 | 0.469 |
| fixed_cutoff | 0.300 | chexpert_external | 0.596 | 0.703 | 0.824 | 0.315 | 0.854 |
| queue_rank | 0.300 | chexpert_external | 0.300 | 0.369 | 0.415 | 0.119 | 0.890 |
| fixed_cutoff | 0.400 | nih_test | 0.405 | 0.785 | 1.000 | 0.302 | 0.414 |
| queue_rank | 0.400 | nih_test | 0.400 | 0.780 | 1.000 | 0.297 | 0.416 |
| fixed_cutoff | 0.400 | chexpert_external | 0.749 | 0.850 | 1.000 | 0.485 | 0.821 |
| queue_rank | 0.400 | chexpert_external | 0.400 | 0.483 | 0.553 | 0.182 | 0.874 |