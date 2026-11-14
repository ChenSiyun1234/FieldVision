# eu_timing_rule_v1

Timing rule (date of the sustained NDVI drop) with its threshold fitted on Austria 2024 only

Trained on European parcel registry labels only: 2400 fields (1200 silage), countries ['AT'], years ['2024'], imagery collections ['sentinel-2-c1-l2a']. Training run `24500e02a9e0`, created 2026-11-14T12:07:52. No US data was used for training, tuning or feature design.

Model: silage if harv_mid_doy <= 251.5; no harvest detected -> score 0.46; Platt calibration p = sigmoid(0.19*logit(score) +0.15)

## European spatial cross-validation (5 folds grouped by 10 km block)

- All fields: silage precision 0.601 [0.57, 0.65], silage recall 0.745 [0.69, 0.80], grain precision 0.665 [0.64, 0.69], grain recall 0.506 [0.39, 0.61], FPR 0.494 [0.39, 0.61], ROC-AUC 0.665 [0.63, 0.70], n 2400 (1200 silage)
- Confident subset: silage precision 0.666 [0.62, 0.73], silage recall 0.715 [0.65, 0.79], grain precision 0.674 [0.65, 0.70], grain recall 0.621 [0.49, 0.74], FPR 0.379 [0.26, 0.51], ROC-AUC 0.698 [0.66, 0.73], n 1606 (825 silage), coverage 0.67

Confusion (all fields):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 894 | 306 |
| grain | 593 | 607 |

Always-grain baseline on the same fields: accuracy 0.500.

Calibration on held-out folds: Brier 0.234, ECE 0.051.

At a 7 % silage share (own threshold): precision 0.102, recall 0.745.
At a 7 % silage share (prior-shifted): precision 0.000, recall 0.000.

## Zero-shot on the organisers' Beltsville set (Evaluation 1, run once)

Frozen model scored on 44 labelled field-years without any fitting on them; exact (Clopper-Pearson) 95 % intervals.

- All: silage precision 0.714 [0.42, 0.92], silage recall 0.833 [0.52, 0.98], grain precision 0.933 [0.78, 0.99], grain recall 0.875 [0.71, 0.96], FPR 0.125 [0.04, 0.29], ROC-AUC 0.904, n 44 (12 silage), accuracy 0.864
- Confident subset: silage precision 0.750 [0.43, 0.95], silage recall 0.900 [0.55, 1.00], grain precision 0.958 [0.79, 1.00], grain recall 0.885 [0.70, 0.98], FPR 0.115 [0.02, 0.30], ROC-AUC 0.929, n 36 (10 silage), coverage 0.82
- 2021: silage precision 0.600 [0.15, 0.95], silage recall 0.750 [0.19, 0.99], grain precision 0.923 [0.64, 1.00], grain recall 0.857 [0.57, 0.98], FPR 0.143 [0.02, 0.43], ROC-AUC 0.857, n 18 (4 silage)
- 2022: silage precision 0.800 [0.28, 0.99], silage recall 1.000 [0.40, 1.00], grain precision 1.000 [0.16, 1.00], grain recall 0.667 [0.09, 0.99], FPR 0.333 [0.01, 0.91], ROC-AUC 0.708, n 7 (4 silage)
- 2023: silage precision 0.750 [0.19, 0.99], silage recall 0.750 [0.19, 0.99], grain precision 0.933 [0.68, 1.00], grain recall 0.933 [0.68, 1.00], FPR 0.067 [0.00, 0.32], ROC-AUC 0.933, n 19 (4 silage)

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 10 | 2 |
| grain | 4 | 28 |

Always grain on the same field-years: accuracy 0.727.

## Known limits

- Trained on one or a few European regions; the US harvest calendar and management differ. Leave-one-country-out results, when available, are the best estimate of transfer.
- Needs a usable Sentinel-2 observation before and after the harvest drop; cloud gaps widen the harvest bracket and push fields into abstention.
- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage precision in practice is far lower than the balanced numbers (see the 7 % line).
- European labels are farmer declarations; some are wrong.
