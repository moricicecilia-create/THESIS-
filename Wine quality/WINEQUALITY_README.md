# Wine Quality — Reference Classification Results


| Split | Observations | Percentage |
|---|---:|---:|
| Training/context | 959 | 59.97% |
| Validation | 160 | 10.01% |
| Calibration | 240 | 15.01% |
| Test | 240 | 15.01% |
| **Total** | **1,599** | **100.00%** |

## Reference model comparison



| Model | AUROC ↑ | Balanced accuracy ↑ | Log loss ↓ | Brier score ↓ | ECE ↓ | Fit time (s) | Prediction time (s) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TabPFN-v2 | 0.9373 | 0.7503 | 0.2108 | 0.0642 | 0.0301 | 0.28 | 143.11 | Success |
| TabICLv2 | **0.9568** | 0.7934 | **0.1666** | **0.0517** | 0.0258 | 0.86 | 49.15 | Success |
| TabDPT | 0.9521 | 0.7934 | 0.1729 | 0.0526 | **0.0087** | 50.86 | 90.83 | Success |
| CatBoost | 0.9338 | 0.6212 | 0.3366 | 0.0936 | 0.1340 | 0.57 | 0.01 | Success |
| MLP | 0.8649 | **0.8019** | 2.6198 | 0.0959 | 0.0964 | 0.23 | 0.00 | Success |


