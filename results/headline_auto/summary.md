# Classification summary

Data: `features_ds004504.npz` | unit: subject | spatial: channels | outer: 10-fold x 3 repeat(s) | inner: 5-fold, 30 trials | nested: True

Subject-level, out-of-fold metrics (mean ± SD over repeats). Positive class = disease (FTD for AD vs FTD).

| task | model | feature_set | band | n_features | balanced_accuracy | balanced_accuracy_sd | sensitivity | specificity | roc_auc | choices |
|---|---|---|---|---|---|---|---|---|---|---|
| AD_CN | auto | all | all | 3762 | 0.785 | 0.040 | 0.731 | 0.839 | 0.805 | models: LDAx11, SVMx8, FamilyVotex4, LogRegx4, RandomForestx3; families: riemannx24, spectralx19, complexityx18, connectivityx8 |
| AD_FTD | auto | all | all | 3762 | 0.607 | 0.008 | 0.565 | 0.648 | 0.618 | models: FamilyVotex18, LDAx7, RandomForestx3, LogRegx2; families: complexityx23, spectralx18, riemannx13, connectivityx5 |
| FTD_CN | auto | all | all | 3762 | 0.723 | 0.041 | 0.652 | 0.793 | 0.732 | models: LDAx9, RandomForestx8, LogRegx7, FamilyVotex5, SVMx1; families: spectralx22, riemannx19, complexityx11, connectivityx11 |

## Published subject-level results on ds004504

| Study | AD vs CN | FTD vs CN | AD vs FTD |
|---|---|---|---|
| Miltiadous et al. 2023 (dataset paper): RBP + RF/MLP, LOSO | 77.0 acc | 73.1 acc | - |
| AHEPA review 2026: mean of Validity-1 studies | 82.1 acc | 75.2 acc | 71.4 acc |
| Riemannian FC stacking (PMC12782479), LOSO | 85 acc | 74 acc | 74 acc |
| FC stacking, Sci Rep 2026, LOSO | 81.8 AUC | 71.4 AUC | 65.1 AUC |
| Leakage-safe benchmark (arXiv 2609.22092), LOSO | 87.7 acc | 82.7 acc | 78.0 acc |

3 configurations were compared. If more than one, treat the top row per task as exploratory: only `auto` rows (or a pre-registered single configuration) give an unbiased estimate.
