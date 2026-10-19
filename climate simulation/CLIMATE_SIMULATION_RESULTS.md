# Climate Model Simulation Crashes (Pop Failures) — Reference Results

## Experimental setup

- **Dataset:** Climate Model Simulation Crashes (Pop Failures).
- **Observations:** 540; **predictors:** 18 numeric features (excluding `Study`, `Run`, and the target).
- **Task:** binary classification (`outcome` = 0 or 1).
- **Evaluation:** one stratified split with random seed **42**; no artificial missingness and no post-hoc calibration.
- **Models:** TabPFN-v2, TabICLv2, TabDPT, CatBoost, and MLP.
- **Split:** train/context: 323, test: 81, calibration: 81, validation: 55.

## Reference-condition results

| Model | AUROC ↑ | Balanced Accuracy ↑ | Log Loss ↓ | Brier Score ↓ | ECE ↓ | Fit Time (s) | Prediction Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabPFN-v2 | 0.8668 | 0.7790 | 0.1921 | 0.0373 | 0.0278 | 0.472 | 65.052 |
| TabICLv2 | 0.8996 | 0.7857 | 0.2072 | 0.0338 | 0.0445 | 0.795 | 19.250 |
| TabDPT | 0.8977 | 0.8571 | 0.1806 | 0.0321 | 0.0372 | 19.289 | 29.540 |
| CatBoost | 0.9382 | 0.5000 | 0.5464 | 0.1776 | 0.3226 | 0.616 | 0.007 |
| MLP | 0.8986 | 0.7075 | 0.8611 | 0.0606 | 0.0611 | 0.045 | 0.003 |

**Execution status:** all five models completed successfully according to the model status file.

## Timing notes

- Fit time measures the timed model preparation/training section in the respective script; it does not consistently isolate initial model loading across implementations.
- Prediction time includes probability prediction on both calibration and test subsets.
- These timings are from **a single run**, not means across seeds.
- Peak memory consumption and separate loading/caching costs were not measured.

## Important limitations

- Results refer only to the reference condition; missingness experiments and temperature scaling have not yet been run for this dataset.
- The test subset has 81 observations. Metric estimates may vary with the random split.
- An AUROC above 0.5 together with balanced accuracy of 0.5 (as for CatBoost in this run) indicates that probability ranking and classification at the fixed 0.5 threshold give different pictures; further investigation is needed before drawing conclusions.

## Reproducibility

Run `pop_failures.py` with `pop_failures.dat` in the same directory. The script writes output to `results_pop_failures/`. The original source dataset and its license should be linked in the repository README.
