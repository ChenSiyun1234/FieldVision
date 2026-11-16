# eu2_gbm_thin_v1

Evaluation 2 family member candidate: the 135-feature LightGBM of eu2_gbm_v1, trained on the same European field-years (AT 2022-2024, FR 2022, NL 2025) plus 3 date-thinned copies of every field (keep probability 0.7, one 10-20 day gap, seed 101), from the three-country augmentation experiment

**Candidate only: date thinning was picked after six augmentation arms were compared on a three-country hold-out; to be confirmed once on new data.**

Trained on European parcel registry labels only: 8000 fields (4000 silage), countries ['AT', 'FR', 'NL'], years ['2022', '2023', '2024', '2025'], imagery collections ['planetary:sentinel-2-l2a', 'planetary:sentinel-2-l2a+mixed', 'sentinel-2-c1-l2a']. Training run `341a9a0d1cef`, created 2026-11-14T18:39:09. No US data was used for training, tuning or feature design.

Model: LightGBM, 135 features, 300 trees; Platt calibration p = sigmoid(0.95*logit(score) +0.00); trained on 8000 fields plus 3 date-thinned copies of each (keep probability 0.7, one 10-20 day gap, seed 101)

## European spatial cross-validation (5 folds grouped by 10 km block)

- All fields: silage precision 0.775 [0.75, 0.80], silage recall 0.771 [0.75, 0.79], grain precision 0.772 [0.76, 0.79], grain recall 0.776 [0.74, 0.81], FPR 0.224 [0.19, 0.26], ROC-AUC 0.834 [0.82, 0.85], n 8000 (4000 silage)
- Confident subset: silage precision 0.786 [0.76, 0.81], silage recall 0.812 [0.78, 0.84], grain precision 0.784 [0.76, 0.81], grain recall 0.755 [0.70, 0.80], FPR 0.245 [0.20, 0.30], ROC-AUC 0.838 [0.82, 0.86], n 5598 (2941 silage), coverage 0.70

Confusion (all fields):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 3084 | 916 |
| grain | 895 | 3105 |

## Known limits

- Trained on one or a few European regions; the US harvest calendar and management differ. Leave-one-country-out results, when available, are the best estimate of transfer.
- Needs a usable Sentinel-2 observation before and after the harvest drop; cloud gaps widen the harvest bracket and push fields into abstention.
- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage precision in practice is far lower than the balanced numbers (see the 7 % line).
- European labels are farmer declarations; some are wrong.
