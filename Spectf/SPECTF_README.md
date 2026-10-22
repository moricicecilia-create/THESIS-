# SPECTF — Binary Classification Baseline Results

| Partition | Observations |
|---|---:|
| Training / context | 159 |
| Validation | 27 |
| Calibration | 40 |
| Test | 41 |
| **Total** | **267** |



| Model | AUROC ↑ | Balanced Accuracy ↑ | Log Loss ↓ | Brier Score ↓ | ECE ↓ | Fit time (s) | Prediction time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabPFN-v2 | 0.8712 | 0.6875 | 0.3594 | 0.1186 | 0.1284 | 0.67 | 85.73 |
| TabICLv2 | 0.8258 | 0.5170 | 0.3739 | 0.1264 | 0.1633 | 4.64 | 23.94 |
| TabDPT | 0.8144 | 0.5947 | 0.3814 | 0.1285 | 0.0606 | 19.14 | 19.18 |
| CatBoost | 0.6970 | 0.4848 | 0.6225 | 0.2150 | 0.2270 | 0.67 | 0.01 |
| MLP | 0.7936 | 0.6269 | 2.3284 | 0.1898 | 0.1849 | 0.07 | 0.01 |


