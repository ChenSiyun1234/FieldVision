# eu_logistic_v1

Six-feature logistic regression (timing, dry-down, pre-harvest greenness and moisture, slope, residue), Platt-calibrated, trained on Austria 2024 registry labels

**Exactly the model scored as a secondary result in Evaluation 1 (42 of 44 on Beltsville). It was not the pre-declared primary; do not read its Beltsville score as a model choice.**

Trained on European parcel registry labels only: 2400 fields (1200 silage), countries ['AT'], years ['2024'], imagery collections ['sentinel-2-c1-l2a']. Training run `24500e02a9e0`, created 2026-11-14T12:07:52. No US data was used for training, tuning or feature design.

Model: logit = -0.03*z(harv_days_since_peak) -0.51*z(harv_decline_days) -0.40*z(pre_ndvi_frac) +0.67*z(pre_ndmi) +0.42*z(sen_ndvi_slope_rel) -0.53*z(post_ndti) +0.01; Platt calibration p = sigmoid(0.94*logit(score) -0.04)

## European spatial cross-validation (5 folds grouped by 10 km block)

- All fields: silage precision 0.754 [0.71, 0.80], silage recall 0.715 [0.68, 0.75], grain precision 0.729 [0.70, 0.76], grain recall 0.767 [0.71, 0.82], FPR 0.233 [0.18, 0.29], ROC-AUC 0.783 [0.75, 0.81], n 2400 (1200 silage)
- Confident subset: silage precision 0.838 [0.81, 0.87], silage recall 0.710 [0.66, 0.77], grain precision 0.736 [0.70, 0.77], grain recall 0.855 [0.81, 0.89], FPR 0.145 [0.11, 0.19], ROC-AUC 0.814 [0.78, 0.84], n 1606 (825 silage), coverage 0.67

Confusion (all fields):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 858 | 342 |
| grain | 280 | 920 |

Always-grain baseline on the same fields: accuracy 0.500.

Calibration on held-out folds: Brier 0.188, ECE 0.057.

At a 7 % silage share (own threshold): precision 0.187, recall 0.715.
At a 7 % silage share (prior-shifted): precision 0.158, recall 0.004.

## Zero-shot on the organisers' Beltsville set (Evaluation 1, run once)

Frozen model scored on 44 labelled field-years without any fitting on them; exact (Clopper-Pearson) 95 % intervals.

- All: silage precision 0.917 [0.62, 1.00], silage recall 0.917 [0.62, 1.00], grain precision 0.969 [0.84, 1.00], grain recall 0.969 [0.84, 1.00], FPR 0.031 [0.00, 0.16], ROC-AUC 0.958, n 44 (12 silage), accuracy 0.955
- Confident subset: silage precision 1.000 [0.66, 1.00], silage recall 0.900 [0.55, 1.00], grain precision 0.963 [0.81, 1.00], grain recall 1.000 [0.87, 1.00], FPR 0.000 [0.00, 0.13], ROC-AUC 0.962, n 36 (10 silage), coverage 0.82
- 2021: silage precision 1.000 [0.29, 1.00], silage recall 0.750 [0.19, 0.99], grain precision 0.933 [0.68, 1.00], grain recall 1.000 [0.77, 1.00], FPR 0.000 [0.00, 0.23], ROC-AUC 0.929, n 18 (4 silage)
- 2022: silage precision 0.800 [0.28, 0.99], silage recall 1.000 [0.40, 1.00], grain precision 1.000 [0.16, 1.00], grain recall 0.667 [0.09, 0.99], FPR 0.333 [0.01, 0.91], ROC-AUC 1.000, n 7 (4 silage)
- 2023: silage precision 1.000 [0.40, 1.00], silage recall 1.000 [0.40, 1.00], grain precision 1.000 [0.78, 1.00], grain recall 1.000 [0.78, 1.00], FPR 0.000 [0.00, 0.22], ROC-AUC 1.000, n 19 (4 silage)

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 11 | 1 |
| grain | 1 | 31 |

Always grain on the same field-years: accuracy 0.727.

## Known limits

- Trained on one or a few European regions; the US harvest calendar and management differ. Leave-one-country-out results, when available, are the best estimate of transfer.
- Needs a usable Sentinel-2 observation before and after the harvest drop; cloud gaps widen the harvest bracket and push fields into abstention.
- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage precision in practice is far lower than the balanced numbers (see the 7 % line).
- European labels are farmer declarations; some are wrong.
