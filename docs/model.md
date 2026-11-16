# Features, models and evaluation

Code: `src/fieldvision/features/` (indices, per-field season features) and
`src/fieldvision/model/` (models, evaluation, CLIs). Run outputs go to a data folder outside
the repo (`<data>` below), never into git; frozen models go to `registry/` (see
`registry/README.md`), and the agent toolbox is described in `docs/toolbox.md`.

Until the package is installable from the merged `pyproject.toml`, run from the
repo root with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`).

## Results on held-out data (times are rows of the evaluation access log, 14 Nov)

| what | when | result |
|---|---|---|
| Evaluation 1, Beltsville, Austria-trained models (zero-shot, once) | 13:53 | LightGBM (primary) 30 of 44, timing rule 38, logistic 42 (secondary), always grain 32 |
| Evaluation 2 primary chosen by leave-one-country-out (AT / FR / NL) | 17:49, corrected 18:01 | logistic .765 mean ROC-AUC, LightGBM .763, composite .760, timing+moisture rule .728, timing rule .662; the 17:41 reading of "tie" would have chosen the timing rule; all three readings logged |
| Evaluation 2, Beltsville, all-Europe models (once) | 18:02 | logistic `eu2_logistic_v1` (primary) 42 of 44, LightGBM 37, composite 36, timing+moisture rule 36, timing rule 35, always grain 32 |
| Third Beltsville scoring (coordinator's decision, candidate asked for by the user) | 19:15-19:16 | date-thinned LightGBM `eu2_gbm_thin_v1` 39 of 44, above the 37 recorded beforehand; the set is now closed |
| Maryland county check (Kent, Carroll; module_v2 local threshold) | 18:11, 18:16 | Kent 21.0 % against Census 5.5 %: fail; Carroll 0 % against 8.9 %: weak |
| Michigan 2022 tracking test (counted clear share) | 19:07-19:11 | Branch 2.1 % (Census 1.8 %), Gratiot 30.0 % (24.2 %): pass |
| Maryland 2025 state figure (counted clear share, 150-block sample) | 19:12-19:14 | 4.75 % [3.4, 6.3], 22,700 silage acres; NASS 2025 8.9 % (40,000 acres): reads 4.2 points low |
| Maryland 2025 with double-crop corn (after the fact) | 19:21-19:23 | 6.7 % [5.3, 8.2] |

How the county and state figures are built, and why: `docs/module.md`.

## Commands

Train and evaluate (spatial CV, leave-one-country-out, leave-one-year-out,
ablations, importance, collection-shift check, optional region holdout):

```
python -m fieldvision.model.train --timeseries <data>/timeseries/<country>_<year>/timeseries.parquet ... \
    --fields <data>/labels/<country>_<year>/fields.geojson ... --out <data>/model/runs/<name> \
    [--holdout-blocks "AT-E46*"] [--primary gbm] [--variant relative]
```

Per-field predictions (what the demo reads):

```
python -m fieldvision.model.predict --model <run dir> --timeseries <...> --fields <...> \
    --out <predictions.parquet> [--keep-columns county_fips]
```

External evaluation, zero-shot (nothing fit on the set; single use, recorded in a ledger):

```
python -m fieldvision.model.evaluate --model <run dir> --timeseries <...> --fields <file.gpkg> \
    --out <dir> [--rename field_id=<col> label=<col> year=<col>] [--label-map <value>=silage <value>=grain]
```

For the organisers' Beltsville file, first expand it to one row per field-year
(`barc-<OBJECTID>-<year>`, EPSG:4326; blank years kept as unlabelled and never scored). The
same GeoJSON is what the extractor should use, so the time-series `field_id`s match:

```
python -m fieldvision.model.barc --gpkg BARC_Corn.2021-2023.gpkg --out barc_field_years.geojson
python -m fieldvision.model.evaluate --model <run dir> --timeseries <barc series> \
    --fields barc_field_years.geojson --keep-columns OBJECTID --out <dir> \
    --access-log <data>/eval/ACCESS_LOG.md
```

Separate, clearly labelled variant that refits only the decision threshold on
`--n-calib` evaluation fields (fixed seed, stratified by label) and reports on
the rest, next to the zero-shot numbers on the same fields:

```
python -m fieldvision.model.evaluate_adapted <same arguments> --n-calib 10 --seed 0
```

How early can we know (time series cut at each date, features rebuilt, models retrained with
the same spatial folds):

```
python -m fieldvision.model.asof --timeseries <...> --fields <...> --out <dir> \
    [--cutoffs 08-01 08-15 09-01 ...] [--exclude-blocks <holdout patterns>]
```

Tests (synthetic data only, a software check):

```
python -m pytest tests -q --basetemp=<scratch dir> -o tmp_path_retention_policy=failed
```

## Rules the code enforces

- Splits are made from whole `block_id` groups, countries or years; never field by field.
- Everything learned (rule thresholds, imputation, scaling, trees) is fit in `model.fit()` on
  the training part only. There is no feature selection step.
- Every table has the majority-class baseline next to the models.
- Intervals are block-bootstrap percentiles. With fewer than 8 blocks (e.g. one research farm)
  the external evaluation resamples fields instead and says that the interval is too narrow.
- The holdout and the external sets are recorded in `single_use_ledger.jsonl` (next to the
  output dir, or `FIELDVISION_LEDGER`). A second zero-shot evaluation of the same set with the
  same run is refused unless `--allow-repeat`.
- Low evidence is decided from data quality only: too few usable dates, no harvest found,
  harvest interval wider than 20 days, or a harvest seen only once at the end of a gap.
- Anything run on synthetic data carries a banner saying it is a software check only.

## Probabilities, class balance, explanations

- `p_silage` is Platt-calibrated inside each training fold (inner CV grouped by block), so it
  is calibrated for the training class balance, which is 50/50 by design. The train report
  shows a reliability table and `figures/calibration.png` on held-out folds.
- About 6-8 % of US corn is silage. The train report re-weights held-out metrics to 7 % and 85 %
  silage (`--prevalences`), at the model's threshold and at the Bayes-shifted threshold, next to
  the always-one-class baseline. `predict --prevalence 0.07` adds `p_silage_at_prevalence`.
- Each prediction carries the numbers behind it (NDVI before/after the drop, days from peak,
  greenness and NDMI before, NDTI residue after, regrowth), how many of five physical signals
  look like silage compared with the training medians of each class, a one-sentence `reason`,
  and `top_model_factors` (LightGBM contributions or logistic coef*z, + = towards silage).
- The peak is the longest green cycle peaking inside `peak_window` (default day 160-280), so a
  short second crop does not pass for corn. For regions with an early calendar (Texas) set
  `--peak-window` from the crop calendar.

## Predictions table

`field_id, p_silage, predicted_label, low_evidence, low_evidence_reason, harvest_start,
harvest_end, harvest_interval_days, pre_harvest_ndvi, pre_harvest_ndmi, n_usable_dates,
max_gap_days`, plus `year, country, block_id, area_ha, centroid_lon, centroid_lat, label,
model, threshold, run_id` and any `--keep-columns`. `harvest_start`/`harvest_end` are the last
usable observation still standing and the first clearly harvested; `low_evidence_reason` is
empty for confident fields and `no_timeseries` when the extractor returned nothing.

## USDA Quick Stats table

`python -m fieldvision.model.quickstats --csv corn_grain_silage_acres_2012_2025.csv --out <dir>`
keeps exactly one record per geography, year, crop use and measurement (the rule is in the module
docstring: the three plain series, domain TOTAL, reference period YEAR, county rows from CENSUS
only and without "OTHER (COMBINED) COUNTIES", state rows from SURVEY). `(D)` stays missing,
never zero. Checks on the 2012-2025 file: Iowa grain harvested 12.45-13.71 million acres per
year; US silage share of harvested corn 6.2-7.8 % (2022: 7.8 % over the 1,410 counties
reporting both numbers). Maryland 2022 grain is 422 thousand acres summed over counties;
Prince George's County (24033) has 3,713 grain acres and a withheld silage figure. Summing
CENSUS and SURVEY county rows together roughly doubles the totals.

## County aggregation

`fieldvision.model.county.county_silage_share(predictions, county_col)` gives, per county, the
area-weighted silage share among confident fields, the same with low-evidence fields included,
the score-weighted share, and the low-evidence share of corn area. `reference_share` and
`compare_county_shares` turn a USDA-style table (silage and grain acres per county) into shares
and compare. Column names are arguments, since we have not seen the files yet.
`add_adjusted_shares(shares, tpr, fpr)` corrects the counted share with recall and FPR measured on
held-out data, (raw - FPR) / (TPR - FPR); without it a few percent of false positives on 93 %
grain area inflate the silage total badly.

## In-site models (trained on Beltsville)

Fitted on the organisers' 44 Beltsville field-years themselves, by design, in the separate
package `fieldvision.beltsville` (`timing_rule`, `in_site`). Nothing here feeds the European
models or the second evaluation, and every run is recorded in the evaluation access log.

```
python -m fieldvision.beltsville.in_site --timeseries <barc series x3> --fields <barc fields x3> \
    --austria-features <data>/model/runs/at2024_full/features.parquet --out <dir> [--export-to registry]
```

Two validation designs. Leave-one-field-out (all seasons of a polygon held out together) is
the honest number for **a new field**; leave-one-year-out (train on two seasons, test on the
third) is **a new season on fields already seen**, because the same polygons appear in
several seasons.

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

44 field-years cannot separate models that differ by one or two fields: each field-year is about 2.3 points of accuracy and each silage field-year about 8.3 points of silage recall.

`logistic_barc`: six-feature logistic fitted on Beltsville only. `rule_barc`: timing+moisture
rule fitted on Beltsville only. `eu_logistic_adj`: the European logistic with only its intercept
and threshold refitted on the training folds. `logistic_at_barc`: six-feature logistic on Austria
2024 plus the Beltsville training folds, both sources weighted equally. The three logistic
variants make the same single mistake (barc-327-2021, a declared silage field that was dry
before the cut and looks like grain on every measure). Per fold (leave-one-year-out): 2021 17 of
18, 2022 7 of 7, 2023 19 of 19. The 8 abstained field-years still get a call and are counted.

Label-permutation check (leave-one-year-out, training labels shuffled within each fold, 200
seeded shuffles): `logistic_barc` reaches a median of 23 of 44 and at most 40, so its 43 is far
in the tail. `logistic_at_barc` still reaches a median of 34 and at most 42 with the Beltsville
labels shuffled, because the Austrian rows carry the signal: the Beltsville labels add about one
field over the European logistic's zero-shot 42.

Frozen as `registry/barc_logistic_v1` (`logistic_barc`, the team's plan: Beltsville only) and
`registry/eu_barc_logistic_v1` (`logistic_at_barc`); the toolbox's `in_site` policy uses
`eu_barc_logistic_v1`, for FIPS 24033 only.
