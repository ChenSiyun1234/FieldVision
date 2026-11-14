# eu_timing_moisture_rule_v1

Timing + moisture rule: silage if the drop comes soon enough after the field's own peak OR the canopy is still wet just before it; thresholds fitted on Austria 2024 only

Trained on European parcel registry labels only: 2400 fields (1200 silage), countries ['AT'], years ['2024'], imagery collections ['sentinel-2-c1-l2a']. Training run `24500e02a9e0`, created 2026-11-14T12:07:52. No US data was used for training, tuning or feature design.

Model: silage if harv_days_since_peak <= 26.7 OR pre_ndmi >= 0.213; no harvest detected -> score 0.46; Platt calibration p = sigmoid(0.60*logit(score) +0.12)

## European spatial cross-validation (5 folds grouped by 10 km block)

- All fields: silage precision 0.688 [0.64, 0.74], silage recall 0.708 [0.66, 0.76], grain precision 0.700 [0.67, 0.73], grain recall 0.679 [0.58, 0.77], FPR 0.321 [0.23, 0.42], ROC-AUC 0.750 [0.71, 0.78], n 2400 (1200 silage)
- Confident subset: silage precision 0.809 [0.76, 0.85], silage recall 0.679 [0.62, 0.74], grain precision 0.710 [0.68, 0.74], grain recall 0.831 [0.76, 0.88], FPR 0.169 [0.12, 0.24], ROC-AUC 0.805 [0.77, 0.84], n 1606 (825 silage), coverage 0.67

Confusion (all fields):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 850 | 350 |
| grain | 385 | 815 |

Always-grain baseline on the same fields: accuracy 0.500.

Calibration on held-out folds: Brier 0.203, ECE 0.035.

At a 7 % silage share (own threshold): precision 0.142, recall 0.708.
At a 7 % silage share (prior-shifted): precision n/a, recall 0.000.

## Zero-shot on the organisers' Beltsville set (Evaluation 1, run once)

Frozen model scored on 44 labelled field-years without any fitting on them; exact (Clopper-Pearson) 95 % intervals.

- All: silage precision 0.579 [0.33, 0.80], silage recall 0.917 [0.62, 1.00], grain precision 0.960 [0.80, 1.00], grain recall 0.750 [0.57, 0.89], FPR 0.250 [0.11, 0.43], ROC-AUC 0.906, n 44 (12 silage), accuracy 0.795
- Confident subset: silage precision 0.692 [0.39, 0.91], silage recall 0.900 [0.55, 1.00], grain precision 0.957 [0.78, 1.00], grain recall 0.846 [0.65, 0.96], FPR 0.154 [0.04, 0.35], ROC-AUC 0.935, n 36 (10 silage), coverage 0.82
- 2021: silage precision 1.000 [0.29, 1.00], silage recall 0.750 [0.19, 0.99], grain precision 0.933 [0.68, 1.00], grain recall 1.000 [0.77, 1.00], FPR 0.000 [0.00, 0.23], ROC-AUC 0.893, n 18 (4 silage)
- 2022: silage precision 0.571 [0.18, 0.90], silage recall 1.000 [0.40, 1.00], grain precision n/a, grain recall 0.000 [0.00, 0.71], FPR 1.000 [0.29, 1.00], ROC-AUC 0.667, n 7 (4 silage)
- 2023: silage precision 0.444 [0.14, 0.79], silage recall 1.000 [0.40, 1.00], grain precision 1.000 [0.69, 1.00], grain recall 0.667 [0.38, 0.88], FPR 0.333 [0.12, 0.62], ROC-AUC 1.000, n 19 (4 silage)

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 11 | 1 |
| grain | 8 | 24 |

Always grain on the same field-years: accuracy 0.727.

## Known limits

- Trained on one or a few European regions; the US harvest calendar and management differ. Leave-one-country-out results, when available, are the best estimate of transfer.
- Needs a usable Sentinel-2 observation before and after the harvest drop; cloud gaps widen the harvest bracket and push fields into abstention.
- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage precision in practice is far lower than the balanced numbers (see the 7 % line).
- European labels are farmer declarations; some are wrong.
