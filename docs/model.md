# Features, models and evaluation

Code: `src/fieldvision/features/` (indices, per-field season features) and
`src/fieldvision/model/` (models, evaluation, CLIs). Outputs go to
`C:\t\fv\data\model\`, never into git.

Until the package is installable from the merged `pyproject.toml`, run from the
repo root with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`).

## Commands

Train and evaluate (spatial CV, leave-one-country-out, leave-one-year-out,
ablations, importance, collection-shift check, optional region holdout):

```
python -m fieldvision.model.train --timeseries C:\t\fv\data\timeseries\<country>_<year> ... \
    --fields C:\t\fv\data\labels\<...>.geojson ... --out C:\t\fv\data\model\runs\<name> \
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
    --fields barc_field_years.geojson --keep-columns OBJECTID --out <dir> \n    --access-log C:	vdataevalACCESS_LOG.md
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
python -m pytest tests -q --basetemp=C:/t/fv/data/model/pytest_tmp -o tmp_path_retention_policy=failed
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
