# eu2_gbm_v1

Evaluation 2 family, trained on every European field-year in READY.txt (AT 2022-2024, FR 2022, NL 2025) with the 13:58 design

Trained on European parcel registry labels only: 8000 fields (4000 silage), countries ['AT', 'FR', 'NL'], years ['2022', '2023', '2024', '2025'], imagery collections ['planetary:sentinel-2-l2a', 'planetary:sentinel-2-l2a+mixed', 'sentinel-2-c1-l2a']. Training run `0484d7e43267`, created 2026-11-14T17:48:36. No US data was used for training, tuning or feature design.

Model: LightGBM, 135 features, 300 trees; Platt calibration p = sigmoid(0.86*logit(score) -0.01)

## European spatial cross-validation (5 folds grouped by 10 km block)

- All fields: silage precision 0.768 [0.74, 0.79], silage recall 0.762 [0.74, 0.78], grain precision 0.764 [0.75, 0.78], grain recall 0.769 [0.73, 0.80], FPR 0.231 [0.20, 0.27], ROC-AUC 0.830 [0.81, 0.84], n 8000 (4000 silage)
- Confident subset: silage precision 0.781 [0.75, 0.81], silage recall 0.800 [0.77, 0.83], grain precision 0.772 [0.75, 0.79], grain recall 0.752 [0.70, 0.79], FPR 0.248 [0.21, 0.30], ROC-AUC 0.837 [0.82, 0.85], n 5598 (2941 silage), coverage 0.70

Confusion (all fields):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 3047 | 953 |
| grain | 922 | 3078 |

Always-grain baseline on the same fields: accuracy 0.500.

Calibration on held-out folds: Brier 0.167, ECE 0.031.

At a 7 % silage share (own threshold): precision 0.199, recall 0.762.
At a 7 % silage share (prior-shifted): precision 0.324, recall 0.030.

## Known limits

- Trained on one or a few European regions; the US harvest calendar and management differ. Leave-one-country-out results, when available, are the best estimate of transfer.
- Needs a usable Sentinel-2 observation before and after the harvest drop; cloud gaps widen the harvest bracket and push fields into abstention.
- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage precision in practice is far lower than the balanced numbers (see the 7 % line).
- European labels are farmer declarations; some are wrong.
