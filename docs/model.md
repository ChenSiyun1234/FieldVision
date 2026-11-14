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

Separate, clearly labelled variant that refits only the decision threshold on
`--n-calib` evaluation fields (fixed seed, stratified by label) and reports on
the rest, next to the zero-shot numbers on the same fields:

```
python -m fieldvision.model.evaluate_adapted <same arguments> --n-calib 10 --seed 0
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

## Predictions table

`field_id, p_silage, predicted_label, low_evidence, low_evidence_reason, harvest_start,
harvest_end, harvest_interval_days, pre_harvest_ndvi, pre_harvest_ndmi, n_usable_dates,
max_gap_days`, plus `year, country, block_id, area_ha, centroid_lon, centroid_lat, label,
model, threshold, run_id` and any `--keep-columns`. `harvest_start`/`harvest_end` are the last
usable observation still standing and the first clearly harvested; `low_evidence_reason` is
empty for confident fields and `no_timeseries` when the extractor returned nothing.

## County aggregation

`fieldvision.model.county.county_silage_share(predictions, county_col)` gives, per county, the
area-weighted silage share among confident fields, the same with low-evidence fields included,
the score-weighted share, and the low-evidence share of corn area. `reference_share` and
`compare_county_shares` turn a USDA-style table (silage and grain acres per county) into shares
and compare. Column names are arguments, since we have not seen the files yet.
