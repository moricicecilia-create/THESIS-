# Thoracic Surgery — Reference Classification Results


| Subset | Observations | Share |
|---|---:|---:|
| Training / context | 281 | 59.8% |
| Validation | 47 | 10.0% |
| Calibration | 71 | 15.1% |
| Test | 71 | 15.1% |
| **Total** | **470** | **100.0%** |


## Reference results

The following metrics are computed on the **test subset** in the reference condition. Higher values are better for AUROC and Balanced Accuracy (↑); lower values are better for Log Loss, Brier Score and Expected Calibration Error (ECE) (↓). Values are rounded to four decimals.

| Model | AUROC ↑ | Balanced Accuracy ↑ | Log Loss ↓ | Brier Score ↓ | ECE ↓ | 
|---|---:|---:|---:|---:|---:|---|
| **TabPFN-v2** | 0.5970 | 0.5000 | **0.4399** | **0.1361** | 0.0784 | 
| **TabICLv2** | 0.5894 | 0.4833 | 0.4657 | 0.1433 | **0.0479** | 
| **TabDPT** | 0.5848 | 0.4917 | 0.4487 | 0.1361 | 0.0551 | 
| **CatBoost** | **0.6235** | 0.5000 | 0.6585 | 0.2327 | 0.3197 | 
| **MLP** | 0.5894 | **0.5492** | 5.7535 | 0.1952 | 0.1955 | 

| Model | Fit time (s) | Prediction time (s) |
|---|---:|---:|
| TabPFN-v2 | 0.501 | 61.966 |
| TabICLv2 | 1.120 | 24.111 |
| TabDPT | 15.497 | 22.155 |
| CatBoost | 0.339 | 0.007 |
| MLP | 0.260 | 0.003 |


