# Diabetic Retinopathy: Reference Experiment Results

## 3. Reference performance

| Model | AUROC ↑ | Balanced Accuracy ↑ | Log Loss ↓ | Brier Score ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| TabPFN-v2 | 0.8527 | 0.8002 | 0.4576 | 0.1516 | 0.0554 |
| TabICLv2 | 0.8533 | 0.7770 | 0.4615 | 0.1517 | 0.0307 |
| TabDPT | 0.8580 | 0.8049 | 0.4562 | 0.1507 | 0.0533 |
| CatBoost | 0.7575 | 0.7041 | 0.5738 | 0.1967 | 0.0408 |
| MLP | 0.7869 | 0.7029 | 6.0148 | 0.2859 | 0.2889 |

## 4. Computational cost

| Model | Fit / Preparation Time (s) | Prediction Time (s) | 
|---|---:|---:|---|
| TabPFN-v2 | 0.306 | 159.730 | 
| TabICLv2 | 0.799 | 40.369 | 
| TabDPT | 31.982 | 57.259 | 
| CatBoost | 1.113 | 0.005 | 
| MLP | 0.535 | 0.003 |

