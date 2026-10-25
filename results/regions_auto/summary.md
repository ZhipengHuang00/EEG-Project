# Classification summary

Data: `features_ds004504.npz` | unit: subject | spatial: regions | outer: 10-fold x 3 repeat(s) | inner: 5-fold, 30 trials | nested: True

Subject-level, out-of-fold metrics (mean ± SD over repeats). Positive class = disease (FTD for AD vs FTD).

| task | model | feature_set | band | n_features | balanced_accuracy | balanced_accuracy_sd | sensitivity | specificity | roc_auc | choices |
|---|---|---|---|---|---|---|---|---|---|---|
| AD_CN | auto | all | all | 2406 | 0.796 | 0.023 | 0.741 | 0.851 | 0.848 | models: LDAx12, RandomForestx7, SVMx6, LogRegx4, FamilyVotex1; families: riemannx27, spectralx19, complexityx11, connectivityx9 |
| AD_FTD | auto | all | all | 2406 | 0.598 | 0.058 | 0.594 | 0.602 | 0.599 | models: FamilyVotex12, LDAx11, LogRegx3, SVMx3, RandomForestx1; families: riemannx21, spectralx19, complexityx19, connectivityx15 |
| FTD_CN | auto | all | all | 2406 | 0.712 | 0.019 | 0.609 | 0.816 | 0.738 | models: FamilyVotex11, RandomForestx10, LDAx5, LogRegx3, SVMx1; families: riemannx22, spectralx19, complexityx18, connectivityx13 |

## Published subject-level results on ds004504

| Study | AD vs CN | FTD vs CN | AD vs FTD |
|---|---|---|---|
| Miltiadous et al. 2023 (dataset paper): RBP + RF/MLP, LOSO | 77.0 acc | 73.1 acc | - |
| AHEPA review 2026: mean of Validity-1 studies | 82.1 acc | 75.2 acc | 71.4 acc |
| Riemannian FC stacking (PMC12782479), LOSO | 85 acc | 74 acc | 74 acc |
| FC stacking, Sci Rep 2026, LOSO | 81.8 AUC | 71.4 AUC | 65.1 AUC |
| Leakage-safe benchmark (arXiv 2609.22092), LOSO | 87.7 acc | 82.7 acc | 78.0 acc |

3 configurations were compared. If more than one, treat the top row per task as exploratory: only `auto` rows (or a pre-registered single configuration) give an unbiased estimate.
