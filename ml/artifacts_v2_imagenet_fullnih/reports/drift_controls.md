# Drift score controls

Artifacts folder: `artifacts_v2_imagenet_fullnih`, model_version: `partA-0.1.0+4ca280b3cd`

Domain AUROC (cross-validated logistic regression on backbone features), 200 vs 200 images, 5 repeats, different images in each group.

| comparison | mean | std | min | max |
|---|---|---|---|---|
| baseline | 0.481 | 0.070 | 0.384 | 0.554 |
| jpeg | 0.998 | 0.001 | 0.996 | 1.000 |
| resize_jpeg | 0.997 | 0.002 | 0.994 | 0.999 |
| chexpert | 1.000 | 0.000 | 1.000 | 1.000 |

**Verdict:** The drift score DOES react to file processing alone: re-encoding NIH images (no change of patients or scanner) scores up to 1.00 vs 0.48 for NIH vs NIH. A high site score therefore cannot be read as a different population or scanner; it may only mean different image processing. Use it as 'these images differ from training, check them', compared against these control values.

## Effect of the same re-encoding on the model's predictions (NIH test, any urgent finding)

| images | n | any_urgent_auroc | mean_score |
|---|---|---|---|
| original | 3000 | 0.810 | 0.129 |
| jpeg | 3000 | 0.804 | 0.142 |
| resize_jpeg | 3000 | 0.803 | 0.131 |

A drift score near 1.0 for re-encoded images, with only a small AUROC change, means the classifier detects compression artefacts easily even when they barely affect predictions. Part of the NIH -> CheXpert drop may still come from format (CheXpert is JPEG), so attribute that drop to 'a different hospital's dataset', not to scanners or patients alone.