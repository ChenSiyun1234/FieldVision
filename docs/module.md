# module_v1: silage or grain, field by field and county by county

What we deliver is a module of several small models, each doing one job, not a single model.
It runs as the toolbox policy `module_v1` (`fieldvision.toolbox.Toolbox`, see `docs/toolbox.md`).
Every model in it is a frozen folder in `registry/`.

```
Sentinel-2 series per field
   │
   ▼
1 evidence check ──► abstain (call kept, flagged)       fieldvision.features (no model)
   │
   ▼
2 field classifier ──► p_silage, label                  registry/composite_v1
   │                                                     (logistic + timing rule + timing+moisture rule)
   ▼
3 second opinion ──► tier: agree / disagree / abstained  registry/anomaly_v1 (being built)
   │
   ▼
4 county total ──► headline share and its range         reference county + European error rates
```

## The layers

**1. Evidence check and abstain.** From each field's own series: at least 6 usable dates from
July to November, a harvest drop found after the corn peak (the peak window ignores spring cover
crops and a second green-up), the drop confirmed by a later observation, and the bracket between
the last standing and the first harvested image at most 20 days. A field that fails still gets a
call, but `abstain` is true and the reason is given. No model, nothing fitted.

**2. Field classifier.** `composite_v1`: the mean of the calibrated probabilities of the
six-feature logistic (`eu_logistic_v1`) and the two transparent rules (`eu_timing_rule_v1`,
`eu_timing_moisture_rule_v1`), all trained on Austrian registry labels only. Output per field:
`p_silage`, `label`, the members' probabilities, and the evidence a person can check (drop date,
greenness and moisture before the cut, residue after). It also abstains when the members split
their votes and are at least 0.5 apart. For the Beltsville fields only, the in-site model
`eu_barc_logistic_v1` (fitted on those fields) can be used instead (`in_site` policy).

**3. Second opinion.** An anomaly scorer that has learned what normal grain looks like, scored on
European data by the pretrained-comparison work. Each field gets a tier in `evidence.tier`:
`agree` (classifier says silage and the field is not grain-like, or grain and grain-like),
`disagree`, or `abstained` (layer 1 failed). Until `registry/anomaly_v1` exists the tier is
`classifier_only` and `evidence.second_opinion` says so. A field with `disagree` should be read as
uncertain even when its probability is high.

**4. County total.** `aggregate` keeps the stub's keys and reports, side by side: the raw sum of
probabilities, the counted share (area called silage), the counted share corrected with European
error rates, its spread across European validation designs, a prior-shift re-estimate, and the
**reference-calibrated share**: a nearly all-grain county of the same region and season (its
Census share known) gives the classifier's local false-positive rate,
FPR_ref = (counted share there - TPR x its Census share) / (1 - its Census share), and the target
county's share is (counted - FPR_ref) / (TPR - FPR_ref), clipped to 0-1, with TPR from Europe and a
range that resamples both counties' fields and TPR.

**Module headline** (`module_headline_share`): the reference-calibrated share when a reference
county is configured (`data/counties/references.json` or `set_reference`), otherwise the
Europe-rate adjusted share. The counted share stays in `silage_share_headline`, the rule logged
before any full-county total; both rules are recorded in the evaluation access log with times.

## What was validated where

| piece | where | result |
|---|---|---|
| classifier family | Austria 2024, 5-fold spatial CV (2,400 fields) | logistic ROC-AUC 0.78, timing rule 0.67, LightGBM 0.86 |
| classifier family | Austria 2022 vs 2024, each year held out | logistic 0.69 / 0.74, composite 0.72 / 0.74, LightGBM 0.78 / 0.83 |
| European models on US fields | Beltsville 44 field-years, Evaluation 1 (zero-shot, once) | logistic 42 of 44 (secondary), timing rule 38, LightGBM 30, always grain 32 |
| in-site models | Beltsville, leave-one-year-out and leave-one-field-out | 43 of 44; labels-shuffled check: Beltsville-only model at most 40 |
| county estimators | Austrian pseudo-counties, 1-50 % silage, same year and a held-out year | mean abs. error of the share, held-out year: counted 0.15, Europe-adjusted 0.10, reference-calibrated 0.10, raw 0.30 |
| headline counted share | Queen Anne's MD 2022, pre-registered check | counted 22.4 % vs Census 1.3 %: failed, in the direction stated in advance |
| anomaly scorer | European data (in progress) | not yet in the module |

Known limits: the European error rates did not hold on the Maryland counties seen so far
(Queen Anne's implies a local false-positive rate of about 0.22 against Europe's 0.16); the
reference-calibrated total uses the reference county's Census share, so it is not independent of
Census; Queen Anne's and the partial Frederick snapshot have been seen, so neither is a clean test
of the county step; a third Maryland county is kept untouched for that.
