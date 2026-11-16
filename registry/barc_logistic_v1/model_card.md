# barc_logistic_v1

Six-feature logistic regression fitted on the 44 Beltsville field-years only (the team's plan: train on Beltsville, test on held-out Beltsville).

**In-site model: it is fitted on the Beltsville labels, so it applies to Beltsville only and its numbers come from cross-validation on the same 44 field-years.**

Fitted model: logit = +0.05*z(harv_days_since_peak) -0.81*z(harv_decline_days) +0.45*z(pre_ndvi_frac) +0.41*z(pre_ndmi) +0.02*z(sen_ndvi_slope_rel) -1.79*z(post_ndti) -1.11

In the comparison below it is `logistic_barc`.

## Cross-validation on Beltsville (exact Clopper-Pearson intervals)

| validation | model | correct of 44 [95% CI] | silage found of 12 | grain called silage of 32 | abstain rate | correct where not abstained |
|---|---|---|---|---|---|---|
| leave-one-year-out | timing_rule_v1 procedure | 40 [0.78, 0.97] | 10 | 2 | 0.18 | 34 of 36 |
| leave-one-year-out | logistic_barc | 43 [0.88, 1.00] | 11 | 0 | 0.18 | 35 of 36 |
| leave-one-year-out | rule_barc | 37 [0.70, 0.93] | 9 | 4 | 0.18 | 30 of 36 |
| leave-one-year-out | eu_logistic_adj | 43 [0.88, 1.00] | 11 | 0 | 0.18 | 35 of 36 |
| leave-one-year-out | logistic_at_barc | 43 [0.88, 1.00] | 11 | 0 | 0.18 | 35 of 36 |
| leave-one-field-out | timing_rule_v1 procedure | 40 [0.78, 0.97] | 10 | 2 | 0.18 | 34 of 36 |
| leave-one-field-out | logistic_barc | 43 [0.88, 1.00] | 11 | 0 | 0.18 | 35 of 36 |
| leave-one-field-out | rule_barc | 39 [0.75, 0.96] | 10 | 3 | 0.18 | 32 of 36 |
| leave-one-field-out | eu_logistic_adj | 43 [0.88, 1.00] | 11 | 0 | 0.18 | 35 of 36 |
| leave-one-field-out | logistic_at_barc | 43 [0.88, 1.00] | 11 | 0 | 0.18 | 35 of 36 |
| reference | always grain | 32 |  |  |  |  |
| reference | timing_rule_v1 (leave-one-year-out) | 40 |  |  |  |  |
| reference | European logistic, zero-shot (Evaluation 1, secondary result) | 42 |  |  |  |  |

Leave-one-field-out (all seasons of a polygon held out together) is the honest number for a new field; leave-one-year-out is a new season on fields already seen. Three logistic variants tie at 43 of 44 under both designs; the single miss in every case is barc-327-2021 (2021 fold: 17 of 18; 2022: 7 of 7; 2023: 19 of 19), a declared silage field that was dry before the cut and looks like grain on every measure. The 8 abstained field-years still get a call and are counted in the 43. Everything fitted (imputation, scaling, the logistic) is fitted inside the training fold; the threshold is a fixed 0.5 and the abstain flag uses only each field's own data quality.

44 field-years cannot separate models that differ by one or two fields: each field-year is about 2.3 points of accuracy and each silage field-year about 8.3 points of silage recall.

## Label-permutation check (leave-one-year-out, 200 seeded shuffles of the training labels within each fold)

| model | median correct of 44 | 95th percentile | max | shuffles reaching 43 | shuffles reaching 40 |
|---|---|---|---|---|---|
| logistic_barc | 23 | 35 | 40 | 0 of 200 | 1 of 200 |
| logistic_at_barc | 34 | 41 | 42 | 0 of 200 | 25 of 200 |

With the training labels shuffled it reaches at most 40 of 44 (median 23); the real 43 is far in the tail of that distribution.

The out-of-fold rows in `results/MD_<year>.jsonl` come from leave-one-field-out.

## Known limits

- Fitted on one farm; do not apply it to other counties (the toolbox's in_site policy is limited to FIPS 24033).
- 12 silage field-years in total; the intervals above are wide.
- Probabilities are class-balanced, not calibrated to Beltsville's silage share.
