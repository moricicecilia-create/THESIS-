## Experimental setup

- Five classifiers: TabPFN-v2, TabICLv2, TabDPT, CatBoost, and MLP.


## 1. Reference performance (no missingness)

| Model | AUROC | Balanced accuracy | Log loss | Brier score | ECE |
| --- | --- | --- | --- | --- | --- |
| TabPFN-v2 | 0.910 | 0.778 | 0.369 | 0.124 | 0.076 |
| TabICLv2 | 0.937 | 0.871 | 0.287 | 0.092 | 0.122 |
| TabDPT | 0.929 | 0.841 | 0.336 | 0.107 | 0.063 |
| CatBoost | 0.918 | 0.778 | 0.516 | 0.165 | 0.200 |
| MLP | 0.898 | 0.820 | 1.669 | 0.182 | 0.209 |

## 2. Missingness robustness


### AUROC

| Model | Reference | Cell 10% | Cell 30% | Cell 50% | Feature 10% | Feature 30% | Feature 50% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabPFN-v2 | 0.910 | 0.878 | 0.827 | 0.765 | 0.898 | 0.859 | 0.796 |
| TabICLv2 | 0.937 | 0.922 | 0.835 | 0.847 | 0.894 | 0.914 | 0.855 |
| TabDPT | 0.929 | 0.906 | 0.827 | 0.824 | 0.894 | 0.890 | 0.816 |
| CatBoost | 0.918 | 0.922 | 0.675 | 0.757 | 0.878 | 0.753 | 0.824 |
| MLP | 0.898 | 0.882 | 0.757 | 0.784 | 0.914 | 0.808 | 0.725 |

### Brier score 

| Model | Reference | Cell 10% | Cell 30% | Cell 50% | Feature 10% | Feature 30% | Feature 50% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabPFN-v2 | 0.124 | 0.141 | 0.175 | 0.205 | 0.129 | 0.159 | 0.193 |
| TabICLv2 | 0.092 | 0.112 | 0.166 | 0.164 | 0.103 | 0.122 | 0.162 |
| TabDPT | 0.107 | 0.128 | 0.177 | 0.174 | 0.129 | 0.149 | 0.183 |
| CatBoost | 0.165 | 0.178 | 0.236 | 0.232 | 0.178 | 0.229 | 0.213 |
| MLP | 0.182 | 0.251 | 0.326 | 0.290 | 0.153 | 0.216 | 0.328 |

### ECE 

| Model | Reference | Cell 10% | Cell 30% | Cell 50% | Feature 10% | Feature 30% | Feature 50% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabPFN-v2 | 0.076 | 0.099 | 0.152 | 0.124 | 0.174 | 0.163 | 0.190 |
| TabICLv2 | 0.122 | 0.108 | 0.104 | 0.146 | 0.185 | 0.128 | 0.109 |
| TabDPT | 0.063 | 0.071 | 0.115 | 0.123 | 0.100 | 0.106 | 0.017 |
| CatBoost | 0.200 | 0.181 | 0.164 | 0.196 | 0.203 | 0.185 | 0.073 |
| MLP | 0.209 | 0.262 | 0.348 | 0.287 | 0.140 | 0.219 | 0.348 |

## 3. Estimated temperatures


| Model | Reference calibration T | Augmented calibration T |
| --- | --- | --- |
| TabPFN-v2 | 1.179 | 1.244 |
| TabICLv2 | 0.978 | 0.898 |
| TabDPT | 0.664 | 0.627 |
| CatBoost | 0.274 | 0.287 |
| MLP | 9.654 | 10.321 |

## 4. Calibration comparison


| Model | Scenario | Uncalibrated ECE | Reference-calibrated ECE | Augmented-calibrated ECE |
| --- | --- | --- | --- | --- |
| TabPFN-v2 | reference | 0.076 | 0.064 | 0.066 |
| TabPFN-v2 | cell_30 | 0.152 | 0.096 | 0.084 |
| TabPFN-v2 | feature_30 | 0.163 | 0.078 | 0.087 |
| TabICLv2 | reference | 0.122 | 0.121 | 0.046 |
| TabICLv2 | cell_30 | 0.104 | 0.104 | 0.109 |
| TabICLv2 | feature_30 | 0.128 | 0.125 | 0.116 |
| TabDPT | reference | 0.063 | 0.099 | 0.063 |
| TabDPT | cell_30 | 0.115 | 0.089 | 0.087 |
| TabDPT | feature_30 | 0.106 | 0.092 | 0.088 |
| CatBoost | reference | 0.200 | 0.125 | 0.131 |
| CatBoost | cell_30 | 0.164 | 0.259 | 0.252 |
| CatBoost | feature_30 | 0.185 | 0.309 | 0.294 |
| MLP | reference | 0.209 | 0.084 | 0.083 |
| MLP | cell_30 | 0.348 | 0.112 | 0.112 |
| MLP | feature_30 | 0.219 | 0.087 | 0.060 |

