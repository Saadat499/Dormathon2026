# Heatmap attention proxies

Artifacts folder: `artifacts_v2_imagenet_fullnih`, model_version: `partA-0.1.0+4ca280b3cd`

Mean share of saliency mass in each region. Above the area share = the map is drawn to that region. Proxies only: check reports/heatmap_sheet.png by eye.

| category | border_10pct | corners_15pct | top_band_12pct |
|---|---|---|---|
| chexpert_confident_hit | 0.375 | 0.067 | 0.122 |
| chexpert_false_alarm | 0.303 | 0.049 | 0.120 |
| chexpert_missed_urgent | 0.458 | 0.119 | 0.256 |
| nih_confident_hit | 0.295 | 0.056 | 0.085 |
| nih_false_alarm | 0.278 | 0.063 | 0.088 |
| nih_missed_urgent | 0.442 | 0.113 | 0.344 |
| uniform map (= area share) | 0.361 | 0.089 | 0.121 |