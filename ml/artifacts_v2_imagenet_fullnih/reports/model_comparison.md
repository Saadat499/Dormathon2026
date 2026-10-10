# Model comparison on common clean NIH test patients

A = `artifacts_v1_imagenet_frozen` (partA-0.1.0+8d12efa246), B = `artifacts_v2_imagenet_fullnih` (partA-0.1.0+4ca280b3cd)
Test: 21845 images, 6162 patients, unseen by both models in training and validation. Patient-level bootstrap, 1000 resamples; diff CI is paired.

| finding | n_pos | auroc_A | A_lo | A_hi | auroc_B | B_lo | B_hi | diff_B_minus_A | diff_lo | diff_hi | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| atelectasis | 2225 | 0.740 | 0.726 | 0.755 | 0.756 | 0.743 | 0.769 | 0.016 | 0.010 | 0.021 | B better |
| cardiomegaly | 587 | 0.789 | 0.761 | 0.812 | 0.821 | 0.797 | 0.842 | 0.033 | 0.021 | 0.044 | B better |
| consolidation | 926 | 0.742 | 0.721 | 0.762 | 0.750 | 0.730 | 0.770 | 0.008 | 0.001 | 0.015 | B better |
| edema | 444 | 0.851 | 0.831 | 0.871 | 0.863 | 0.846 | 0.882 | 0.013 | 0.007 | 0.019 | B better |
| effusion | 2819 | 0.823 | 0.810 | 0.834 | 0.836 | 0.824 | 0.847 | 0.013 | 0.010 | 0.017 | B better |
| pneumonia | 262 | 0.647 | 0.614 | 0.686 | 0.687 | 0.649 | 0.722 | 0.039 | 0.018 | 0.059 | B better |
| pneumothorax | 1004 | 0.819 | 0.795 | 0.842 | 0.839 | 0.816 | 0.859 | 0.020 | 0.009 | 0.031 | B better |
| ANY_URGENT | 4659 | 0.801 | 0.790 | 0.812 | 0.811 | 0.800 | 0.821 | 0.010 | 0.007 | 0.013 | B better |