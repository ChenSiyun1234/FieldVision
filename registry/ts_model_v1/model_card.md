# ts_model_v1

LightGBM on self-referenced harvest features, trained on Austria 2024 registry labels

Trained on European parcel registry labels only: 2400 fields (1200 silage), countries ['AT'], years ['2024'], imagery collections ['sentinel-2-c1-l2a']. Training run `24500e02a9e0`, created 2026-11-14T12:07:52. No US data was used for training, tuning or feature design.

Model: LightGBM, 135 features, 300 trees; Platt calibration p = sigmoid(0.61*logit(score) +0.11)

## European spatial cross-validation (5 folds grouped by 10 km block)

- All fields: silage precision 0.815 [0.78, 0.85], silage recall 0.771 [0.73, 0.81], grain precision 0.783 [0.76, 0.81], grain recall 0.825 [0.78, 0.86], FPR 0.175 [0.14, 0.22], ROC-AUC 0.857 [0.83, 0.88], n 2400 (1200 silage)
- Confident subset: silage precision 0.844 [0.81, 0.87], silage recall 0.782 [0.73, 0.83], grain precision 0.786 [0.75, 0.82], grain recall 0.848 [0.80, 0.89], FPR 0.152 [0.11, 0.20], ROC-AUC 0.864 [0.84, 0.89], n 1606 (825 silage), coverage 0.67

Confusion (all fields):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 925 | 275 |
| grain | 210 | 990 |

Always-grain baseline on the same fields: accuracy 0.500.

Calibration on held-out folds: Brier 0.149, ECE 0.042.

At a 7 % silage share (own threshold): precision 0.249, recall 0.771.
At a 7 % silage share (prior-shifted): precision 0.429, recall 0.058.

## Zero-shot on the organisers' Beltsville set (Evaluation 1, run once)

Frozen model scored on 44 labelled field-years without any fitting on them; exact (Clopper-Pearson) 95 % intervals.

- All: silage precision 0.429 [0.18, 0.71], silage recall 0.500 [0.21, 0.79], grain precision 0.800 [0.61, 0.92], grain recall 0.750 [0.57, 0.89], FPR 0.250 [0.11, 0.43], ROC-AUC 0.714, n 44 (12 silage), accuracy 0.682
- Confident subset: silage precision 0.455 [0.17, 0.77], silage recall 0.500 [0.19, 0.81], grain precision 0.800 [0.59, 0.93], grain recall 0.769 [0.56, 0.91], FPR 0.231 [0.09, 0.44], ROC-AUC 0.704, n 36 (10 silage), coverage 0.82
- 2021: silage precision 1.000 [0.29, 1.00], silage recall 0.750 [0.19, 0.99], grain precision 0.933 [0.68, 1.00], grain recall 1.000 [0.77, 1.00], FPR 0.000 [0.00, 0.23], ROC-AUC 0.911, n 18 (4 silage)
- 2022: silage precision 0.333 [0.01, 0.91], silage recall 0.250 [0.01, 0.81], grain precision 0.250 [0.01, 0.81], grain recall 0.333 [0.01, 0.91], FPR 0.667 [0.09, 0.99], ROC-AUC 0.417, n 7 (4 silage)
- 2023: silage precision 0.250 [0.03, 0.65], silage recall 0.500 [0.07, 0.93], grain precision 0.818 [0.48, 0.98], grain recall 0.600 [0.32, 0.84], FPR 0.400 [0.16, 0.68], ROC-AUC 0.717, n 19 (4 silage)

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 6 | 6 |
| grain | 8 | 24 |

Always grain on the same field-years: accuracy 0.727.

## Known limits

- Trained on one or a few European regions; the US harvest calendar and management differ. Leave-one-country-out results, when available, are the best estimate of transfer.
- Needs a usable Sentinel-2 observation before and after the harvest drop; cloud gaps widen the harvest bracket and push fields into abstention.
- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage precision in practice is far lower than the balanced numbers (see the 7 % line).
- European labels are farmer declarations; some are wrong.
