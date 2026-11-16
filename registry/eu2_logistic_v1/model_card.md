# eu2_logistic_v1

Evaluation 2 family, trained on every European field-year in READY.txt (AT 2022-2024, FR 2022, NL 2025) with the 13:58 design

Trained on European parcel registry labels only: 8000 fields (4000 silage), countries ['AT', 'FR', 'NL'], years ['2022', '2023', '2024', '2025'], imagery collections ['planetary:sentinel-2-l2a', 'planetary:sentinel-2-l2a+mixed', 'sentinel-2-c1-l2a']. Training run `0484d7e43267`, created 2026-11-14T17:48:36. No US data was used for training, tuning or feature design.

Model: logit = -0.25*z(harv_days_since_peak) -0.31*z(harv_decline_days) -0.24*z(pre_ndvi_frac) +0.58*z(pre_ndmi) +0.31*z(sen_ndvi_slope_rel) -0.44*z(post_ndti) -0.00; Platt calibration p = sigmoid(0.98*logit(score) -0.00)

## European spatial cross-validation (5 folds grouped by 10 km block)

- All fields: silage precision 0.721 [0.69, 0.75], silage recall 0.730 [0.69, 0.76], grain precision 0.726 [0.71, 0.74], grain recall 0.718 [0.67, 0.76], FPR 0.282 [0.24, 0.33], ROC-AUC 0.768 [0.75, 0.79], n 8000 (4000 silage)
- Confident subset: silage precision 0.764 [0.74, 0.79], silage recall 0.738 [0.69, 0.78], grain precision 0.720 [0.70, 0.74], grain recall 0.748 [0.69, 0.80], FPR 0.252 [0.20, 0.31], ROC-AUC 0.783 [0.76, 0.81], n 5598 (2941 silage), coverage 0.70

Confusion (all fields):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 2919 | 1081 |
| grain | 1130 | 2870 |

Always-grain baseline on the same fields: accuracy 0.500.

Calibration on held-out folds: Brier 0.195, ECE 0.044.

At a 7 % silage share (own threshold): precision 0.163, recall 0.730.
At a 7 % silage share (prior-shifted): precision 0.311, recall 0.001.

## Known limits

- Trained on one or a few European regions; the US harvest calendar and management differ. Leave-one-country-out results, when available, are the best estimate of transfer.
- Needs a usable Sentinel-2 observation before and after the harvest drop; cloud gaps widen the harvest bracket and push fields into abstention.
- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage precision in practice is far lower than the balanced numbers (see the 7 % line).
- European labels are farmer declarations; some are wrong.
