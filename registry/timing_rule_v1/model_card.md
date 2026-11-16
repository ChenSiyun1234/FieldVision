# timing_rule_v1

Thresholds on the harvest drop date and on the greenness just before it, fitted on the organisers' Beltsville field-years and validated by leave-one-year-out (train on two years, test on the third). This is not the European model: it uses Beltsville labels, so its numbers are an in-site estimate, and the folds share fields (the same polygon appears in several years).

Final rule (fitted on all 44 field-years): silage if harv_mid_doy <= 257.5 AND pre_ndvi >= 0.708; no harvest detected -> score 0.27

## Field-years by year and class

```
 year  grain  silage
 2021     14       4
 2022      3       4
 2023     15       4
```

## Leave-one-year-out results (exact Clopper-Pearson 95 % intervals)

| set | n (silage) | silage called silage | silage called grain | grain called silage | grain called grain | accuracy | silage precision | silage recall | grain precision | grain recall |
|---|---|---|---|---|---|---|---|---|---|---|
| test 2021 | 18 (4) | 3 | 1 | 0 | 14 | 0.944 [0.73, 1.00] | 1.000 [0.29, 1.00] | 0.750 [0.19, 0.99] | 0.933 [0.68, 1.00] | 1.000 [0.77, 1.00] |
| test 2022 | 7 (4) | 4 | 0 | 1 | 2 | 0.857 [0.42, 1.00] | 0.800 [0.28, 0.99] | 1.000 [0.40, 1.00] | 1.000 [0.16, 1.00] | 0.667 [0.09, 0.99] |
| test 2023 | 19 (4) | 3 | 1 | 1 | 14 | 0.895 [0.67, 0.99] | 0.750 [0.19, 0.99] | 0.750 [0.19, 0.99] | 0.933 [0.68, 1.00] | 0.933 [0.68, 1.00] |
| pooled (all 3 folds) | 44 (12) | 10 | 2 | 2 | 30 | 0.909 [0.78, 0.97] | 0.833 [0.52, 0.98] | 0.833 [0.52, 0.98] | 0.938 [0.79, 0.99] | 0.938 [0.79, 0.99] |
| always grain, pooled | 44 (12) | 0 | 12 | 0 | 32 | 0.727 [0.57, 0.85] | n/a | 0.000 [0.00, 0.26] | 0.727 [0.57, 0.85] | 1.000 [0.89, 1.00] |
| pooled, not abstained | 36 (10) | 9 | 1 | 1 | 25 | 0.944 [0.81, 0.99] | 0.900 [0.55, 1.00] | 0.900 [0.55, 1.00] | 0.962 [0.80, 1.00] | 0.962 [0.80, 1.00] |

Abstain rate (low evidence: harvest bracket too wide, too few dates, no or unconfirmed drop): 0.182 (8 of 44). The rule still makes a call on those; the row above leaves them out.

Each field-year is 1/44 of the pooled accuracy, about 2.3 points; each silage field-year is 1/12 of silage recall, about 8.3 points. With 12 silage field-years the intervals above are wide, and differences of one or two fields are not meaningful.

Baseline: always grain scores 0.727 (32 of 44) with silage recall 0.

## Rule fitted in each fold

- test 2021 (trained on 26, 8 silage): silage if harv_mid_doy <= 257.5 AND pre_ndvi >= 0.688; no harvest detected -> score 0.31
- test 2022 (trained on 37, 8 silage): silage if harv_mid_doy <= 257.5 AND pre_ndvi >= 0.714; no harvest detected -> score 0.22
- test 2023 (trained on 25, 8 silage): silage if harv_mid_doy <= 253.1; no harvest detected -> score 0.32

## Known limits

- Fitted and validated on one farm in Maryland; the thresholds encode its calendar and will not hold where corn is planted or cut at other times.
- Leave-one-year-out reuses the same fields across folds, so it measures a new season on known fields, not a new place.
- The score is a rule margin, not a calibrated probability.
- Needs a clear observation before and after the drop.

## Note added for the worked example (after export)

`results/MD_<year>.jsonl` shows this model's out-of-fold predictions: each year predicted by the
rule fitted on the other two years (`evidence.fold = "test-<year>"`). Running this frozen folder
on the same 44 field-years gives in-sample predictions, because it was fitted on all three years.
