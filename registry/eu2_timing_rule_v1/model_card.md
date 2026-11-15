# eu2_timing_rule_v1

Evaluation 2 family, trained on every European field-year in READY.txt (AT 2022-2024, FR 2022, NL 2025) with the 13:58 design

Trained on European parcel registry labels only: 8000 fields (4000 silage), countries ['AT', 'FR', 'NL'], years ['2022', '2023', '2024', '2025'], imagery collections ['planetary:sentinel-2-l2a', 'planetary:sentinel-2-l2a+mixed', 'sentinel-2-c1-l2a']. Training run `0484d7e43267`, created 2026-11-14T17:48:36. No US data was used for training, tuning or feature design.

Model: silage if harv_mid_doy <= 257.5; no harvest detected -> score 0.62; Platt calibration p = sigmoid(0.26*logit(score) +0.09)

## European spatial cross-validation (5 folds grouped by 10 km block)

- All fields: silage precision 0.607 [0.59, 0.63], silage recall 0.653 [0.61, 0.71], grain precision 0.625 [0.61, 0.65], grain recall 0.578 [0.52, 0.63], FPR 0.422 [0.37, 0.48], ROC-AUC 0.660 [0.64, 0.68], n 8000 (4000 silage)
- Confident subset: silage precision 0.620 [0.60, 0.64], silage recall 0.674 [0.62, 0.73], grain precision 0.600 [0.58, 0.63], grain recall 0.542 [0.46, 0.61], FPR 0.458 [0.39, 0.54], ROC-AUC 0.653 [0.63, 0.68], n 5598 (2941 silage), coverage 0.70

Confusion (all fields):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 2614 | 1386 |
| grain | 1689 | 2311 |

Always-grain baseline on the same fields: accuracy 0.500.

Calibration on held-out folds: Brier 0.232, ECE 0.026.

At a 7 % silage share (own threshold): precision 0.104, recall 0.653.
At a 7 % silage share (prior-shifted): precision 0.070, recall 0.000.

## Known limits

- Trained on one or a few European regions; the US harvest calendar and management differ. Leave-one-country-out results, when available, are the best estimate of transfer.
- Needs a usable Sentinel-2 observation before and after the harvest drop; cloud gaps widen the harvest bracket and push fields into abstention.
- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage precision in practice is far lower than the balanced numbers (see the 7 % line).
- European labels are farmer declarations; some are wrong.
