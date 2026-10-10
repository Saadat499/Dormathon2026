# Cross-dataset results (train NIH -> test CheXpert)

Artifacts folder: `artifacts_v2_imagenet_fullnih`, model_version: `partA-0.1.0+4ca280b3cd`

Thresholds fixed on NIH validation at 90% sensitivity, then applied unchanged to CheXpert.
PPV depends on dataset prevalence; recompute for the assumed hospital prevalence.

| finding | auroc_nih | auroc_chexpert | auroc_drop | sens_nih | sens_chexpert | sens_drop | false_alarm_nih | false_alarm_chexpert | n_pos_chexpert |
|---|---|---|---|---|---|---|---|---|---|
| atelectasis | 0.756 | 0.612 | -0.144 | 0.910 | 0.789 | -0.120 | 0.603 | 0.634 | 2315 |
| cardiomegaly | 0.821 | 0.681 | -0.140 | 0.893 | 0.663 | -0.229 | 0.459 | 0.415 | 1839 |
| consolidation | 0.750 | 0.647 | -0.103 | 0.884 | 0.898 | 0.013 | 0.555 | 0.723 | 1046 |
| edema | 0.863 | 0.721 | -0.143 | 0.890 | 0.827 | -0.063 | 0.317 | 0.511 | 3810 |
| effusion | 0.836 | 0.768 | -0.068 | 0.919 | 0.894 | -0.025 | 0.456 | 0.543 | 6217 |
| pneumonia | 0.687 | 0.589 | -0.098 | 0.943 | 0.980 | 0.037 | 0.813 | 0.975 | 397 |
| pneumothorax | 0.839 | 0.697 | -0.142 | 0.943 | 0.991 | 0.048 | 0.586 | 0.958 | 1406 |
| ANY_URGENT | 0.811 | 0.755 | -0.057 | 0.920 | 0.966 | 0.046 | 0.512 | 0.734 | 9429 |