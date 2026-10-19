# Breast Cancer Wisconsin (Diagnostic): Reference Results

## Model performance and computational time

| Model | AUROC ↑ | Balanced Accuracy ↑ | Log Loss ↓ | Brier Score ↓ | ECE ↓ | Fit Time (s) | Prediction Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabPFN-v2 | 1.0000 | 1.0000 | 0.0289 | 0.0047 | 0.0260 | 0.516 | 180.574 |
| TabICLv2 | 1.0000 | 1.0000 | 0.0307 | 0.0053 | 0.0275 | 1.520 | 31.935 |
| TabDPT | 1.0000 | 1.0000 | 0.0346 | 0.0056 | 0.0313 | 22.059 | 28.132 |
| CatBoost | 0.9994 | 0.9688 | 0.2197 | 0.0456 | 0.1681 | 0.996 | 0.009 |
| MLP | 0.9734 | 0.9688 | 0.4509 | 0.0219 | 0.0244 | 0.045 | 0.003 |

