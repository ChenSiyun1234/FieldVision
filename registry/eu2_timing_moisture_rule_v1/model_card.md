# eu2_timing_moisture_rule_v1

Evaluation 2 family, trained on every European field-year in READY.txt (AT 2022-2024, FR 2022, NL 2025) with the 13:58 design

Trained on European parcel registry labels only: 8000 fields (4000 silage), countries ['AT', 'FR', 'NL'], years ['2022', '2023', '2024', '2025'], imagery collections ['planetary:sentinel-2-l2a', 'planetary:sentinel-2-l2a+mixed', 'sentinel-2-c1-l2a']. Training run `0484d7e43267`, created 2026-11-14T17:48:36. No US data was used for training, tuning or feature design.

Model: silage if harv_days_since_peak <= 76.0 AND pre_ndmi >= 0.173; no harvest detected -> score 0.62; Platt calibration p = sigmoid(0.59*logit(score) -0.01)

## European spatial cross-validation (5 folds grouped by 10 km block)

- All fields: silage precision 0.687 [0.66, 0.71], silage recall 0.683 [0.65, 0.72], grain precision 0.685 [0.67, 0.70], grain recall 0.689 [0.64, 0.73], FPR 0.311 [0.27, 0.36], ROC-AUC 0.734 [0.72, 0.75], n 8000 (4000 silage)
- Confident subset: silage precision 0.739 [0.71, 0.77], silage recall 0.677 [0.64, 0.72], grain precision 0.673 [0.65, 0.69], grain recall 0.735 [0.69, 0.78], FPR 0.265 [0.22, 0.31], ROC-AUC 0.760 [0.74, 0.78], n 5598 (2941 silage), coverage 0.70

Confusion (all fields):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 2734 | 1266 |
| grain | 1244 | 2756 |

Always-grain baseline on the same fields: accuracy 0.500.

Calibration on held-out folds: Brier 0.209, ECE 0.026.

At a 7 % silage share (own threshold): precision 0.142, recall 0.683.
At a 7 % silage share (prior-shifted): precision n/a, recall 0.000.

## Known limits

- Trained on one or a few European regions; the US harvest calendar and management differ. Leave-one-country-out results, when available, are the best estimate of transfer.
- Needs a usable Sentinel-2 observation before and after the harvest drop; cloud gaps widen the harvest bracket and push fields into abstention.
- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage precision in practice is far lower than the balanced numbers (see the 7 % line).
- European labels are farmer declarations; some are wrong.
