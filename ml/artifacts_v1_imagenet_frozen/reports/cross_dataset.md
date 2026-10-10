# Cross-dataset results (train NIH -> test CheXpert)

Thresholds fixed on NIH validation at 90% sensitivity, then applied unchanged to CheXpert.
PPV depends on dataset prevalence; recompute for the assumed hospital prevalence.

| finding | auroc_nih | auroc_chexpert | auroc_drop | sens_nih | sens_chexpert | sens_drop | false_alarm_nih | false_alarm_chexpert | n_pos_chexpert |
|---|---|---|---|---|---|---|---|---|---|
| atelectasis | 0.752 | 0.602 | -0.151 | 0.924 | 0.826 | -0.098 | 0.638 | 0.686 | 2315 |
| cardiomegaly | 0.810 | 0.657 | -0.154 | 0.933 | 0.628 | -0.306 | 0.586 | 0.409 | 1839 |
| consolidation | 0.732 | 0.624 | -0.108 | 0.887 | 0.944 | 0.056 | 0.569 | 0.839 | 1046 |
| edema | 0.825 | 0.695 | -0.130 | 0.779 | 0.562 | -0.217 | 0.261 | 0.299 | 3810 |
| effusion | 0.822 | 0.758 | -0.064 | 0.902 | 0.830 | -0.072 | 0.447 | 0.462 | 6217 |
| pneumonia | 0.641 | 0.547 | -0.094 | 0.885 | 0.927 | 0.042 | 0.718 | 0.909 | 397 |
| pneumothorax | 0.806 | 0.659 | -0.147 | 0.940 | 0.981 | 0.041 | 0.681 | 0.965 | 1406 |
| ANY_URGENT | 0.796 | 0.753 | -0.043 | 0.917 | 0.952 | 0.035 | 0.543 | 0.717 | 9429 |