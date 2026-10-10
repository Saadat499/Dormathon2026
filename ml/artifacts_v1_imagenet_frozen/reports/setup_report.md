# Setup report

## NIH: 30007 images, 8280 patients
splits: {'train': 21248, 'test': 5558, 'val': 3201}
sample image sizes: ['224x224 L']

## CheXpert: 15202 images, 5177 patients
splits: {'external': 15000, 'gold_valid': 202}
sample image sizes: ['320x329 L', '351x320 L', '369x320 L', '389x320 L', '390x320 L']

## Label overlap (shared findings)
| finding | in_nih_as | in_chexpert_as | nih_pos_rate | nih_n_pos | chexpert_pos_rate | chexpert_n_pos | chexpert_uncertain_rate | provisional_urgent |
|---|---|---|---|---|---|---|---|---|
| atelectasis | Atelectasis | Atelectasis | 0.102 | 3048 | 0.183 | 2315 | 0.157 | False |
| cardiomegaly | Cardiomegaly | Cardiomegaly | 0.023 | 684 | 0.127 | 1839 | 0.035 | False |
| consolidation | Consolidation | Consolidation | 0.038 | 1147 | 0.080 | 1046 | 0.124 | True |
| edema | Edema | Edema | 0.021 | 633 | 0.271 | 3810 | 0.062 | True |
| effusion | Effusion | Pleural Effusion | 0.108 | 3254 | 0.435 | 6217 | 0.048 | True |
| pneumonia | Pneumonia | Pneumonia | 0.012 | 347 | 0.029 | 397 | 0.082 | True |
| pneumothorax | Pneumothorax | Pneumothorax | 0.047 | 1406 | 0.095 | 1406 | 0.014 | True |

ANY_URGENT prevalence (provisional urgent set ['pneumothorax', 'effusion', 'edema', 'consolidation', 'pneumonia']): NIH 0.195, CheXpert(external) 0.724

Notes: public-data prevalence is NOT hospital prevalence. The queue simulation must use an assumed prevalence as a parameter. CheXpert 'uncertain' labels are masked (ignored); blanks are treated as 0.0. NIH labels are text-mined and noisy.

## Backbone smoke test
densenet121-onnx: embedding (16, 1024), finite=True, 7.2 img/s (first call includes warm-up)
