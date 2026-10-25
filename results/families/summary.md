# Classification summary

Data: `features_ds004504.npz` | unit: subject | spatial: channels | outer: 10-fold x 1 repeat(s) | inner: 5-fold, 15 trials | nested: True

Subject-level, out-of-fold metrics (mean ± SD over repeats). Positive class = disease (FTD for AD vs FTD).

| task | model | feature_set | band | n_features | balanced_accuracy | balanced_accuracy_sd | sensitivity | specificity | roc_auc | choices |
|---|---|---|---|---|---|---|---|---|---|---|
| AD_CN | LDA | riemann | all | 2166 | 0.830 | nan | 0.833 | 0.828 | 0.872 |  |
| AD_CN | LDA | all | all | 3762 | 0.789 | nan | 0.750 | 0.828 | 0.801 |  |
| AD_CN | LDA | complexity | all | 114 | 0.737 | nan | 0.750 | 0.724 | 0.772 |  |
| AD_CN | LDA | spectral | all | 342 | 0.692 | nan | 0.694 | 0.690 | 0.800 |  |
| AD_CN | LDA | connectivity | all | 1140 | 0.675 | nan | 0.694 | 0.655 | 0.727 |  |
| AD_FTD | LDA | complexity | all | 114 | 0.606 | nan | 0.435 | 0.778 | 0.589 |  |
| AD_FTD | LDA | riemann | all | 2166 | 0.580 | nan | 0.522 | 0.639 | 0.606 |  |
| AD_FTD | LDA | spectral | all | 342 | 0.557 | nan | 0.391 | 0.722 | 0.645 |  |
| AD_FTD | LDA | all | all | 3762 | 0.517 | nan | 0.478 | 0.556 | 0.580 |  |
| AD_FTD | LDA | connectivity | all | 1140 | 0.450 | nan | 0.261 | 0.639 | 0.432 |  |
| FTD_CN | LDA | spectral | all | 342 | 0.757 | nan | 0.652 | 0.862 | 0.850 |  |
| FTD_CN | LDA | complexity | all | 114 | 0.744 | nan | 0.696 | 0.793 | 0.741 |  |
| FTD_CN | LDA | all | all | 3762 | 0.705 | nan | 0.652 | 0.759 | 0.780 |  |
| FTD_CN | LDA | riemann | all | 2166 | 0.701 | nan | 0.609 | 0.793 | 0.729 |  |
| FTD_CN | LDA | connectivity | all | 1140 | 0.632 | nan | 0.609 | 0.655 | 0.720 |  |

## Published subject-level results on ds004504

| Study | AD vs CN | FTD vs CN | AD vs FTD |
|---|---|---|---|
| Miltiadous et al. 2023 (dataset paper): RBP + RF/MLP, LOSO | 77.0 acc | 73.1 acc | - |
| AHEPA review 2026: mean of Validity-1 studies | 82.1 acc | 75.2 acc | 71.4 acc |
| Riemannian FC stacking (PMC12782479), LOSO | 85 acc | 74 acc | 74 acc |
| FC stacking, Sci Rep 2026, LOSO | 81.8 AUC | 71.4 AUC | 65.1 AUC |
| Leakage-safe benchmark (arXiv 2609.22092), LOSO | 87.7 acc | 82.7 acc | 78.0 acc |

15 configurations were compared. If more than one, treat the top row per task as exploratory: only `auto` rows (or a pre-registered single configuration) give an unbiased estimate.
