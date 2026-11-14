# FieldVision demo app

A static map of corn fields coloured by predicted harvest use (silage or grain), with the satellite evidence behind each call. Plain HTML, CSS and ES modules, no build step. Leaflet is vendored in `vendor/`, so the app works offline. Only the satellite basemap needs the network, and without it the fields are drawn on a plain background.

## Run it

From the repository root:

```bash
python -m http.server 8000 -d app
```

Then open <http://localhost:8000>. Opening `index.html` straight from disk does not work, because browsers block `fetch` on `file://`.

The committed data in `app/data/` is **synthetic**. Every synthetic region shows a yellow "Sample data · synthetic" badge, so nobody takes it for a result. When a real export overwrites a region folder, the badge for that region disappears.

## One-minute demo path

1. **Map**: fields are coloured by prediction, sky blue for silage and orange for grain. Fields with a dashed outline are low evidence. Zoomed out, each field is a dot; zoom in to see its outline.
2. **Click a field** on the map or in the list. The panel shows the call and its confidence, one or two plain-English sentences (for example "Cut between Sep 14 and Sep 17 while still green (NDVI 0.68): silage-like"), the harvest window, the NDVI and NDMI just before the cut, and both season curves with every Sentinel-2 acquisition. Esc goes back.
3. **Check vs declared** (button at the top right of the map) recolours the map: wrong calls turn magenta and correct ones grey. The **Model wrong** tab lists the mistakes, most confident first.
4. **Region and year**: the buttons in the top bar.
5. **Headline numbers** for the region sit at the top of the panel: precision, recall and false positive rate (silage is the positive class), the number of fields, the share flagged low evidence, and the share called silage. With no labels or no predictions, these show "—" and say why.
6. **Check these first** lists the least certain predictions. These are where a survey call or field visit tells us the most.
7. **Counties** tab (US regions exported with `--counties`): predicted silage share per county against the USDA share, as a scatter and a table. If county outlines were exported, the map also shades each county by predicted share.

The URL keeps the state (`#r=at-2024&f=<field_id>&mode=check`), so a bookmarked link reopens the same field.

## How to read the curves

| Mark | Meaning |
|---|---|
| filled dot | usable observation: `clear_fraction` ≥ the export's `--min-clear` (default 0.7) |
| hollow dot | excluded: some clear pixels, but below the threshold (usually cloud or haze) |
| tick under the axis | acquisition with no clear pixel at all |
| line | joins usable observations only, and never across a gap longer than `--gap-days` (default 20). Longer gaps stay open and are labelled "N d no clear view" |
| shaded band | harvest window from the predictions: last observation still standing to first observation clearly harvested |
| ringed value | the model's NDVI / NDMI just before the cut |

NDVI = (B08 − B04) / (B08 + B04) and NDMI = (B8A − B11) / (B8A + B11), the same definitions as `fieldvision.s2.indices`.

The cue words in the evidence sentences ("still green", "silage-like", "already dry") come from fixed display thresholds stored in `meta.json` (`cues`). They describe the curve, not the model's rule, so a sentence can disagree with the prediction. That is deliberate: a disagreement is worth noticing.

## Exporting real data

```bash
python -m fieldvision.app.export --fields <fields.geojson> --timeseries <parquet file or folder> --predictions <predictions.parquet|csv> --out app/data/at-2024 --title "Austria 2024" --note "spatial 5-fold CV, out-of-fold"
```

Run it with `src` on the import path (in PowerShell `$env:PYTHONPATH="src"`; in bash `PYTHONPATH=src`). The output folder name is the region id. Use the ids the sample data uses (`at-2024`, `nl-2025`, `fr-2022`, `barc-2021`, `barc-2022`, `barc-2023`, `us-2022`) and the real export replaces the sample in place, with no code changes. The export also adds or updates the region in `app/data/regions.json`.

Useful options:

| Option | Default | |
|---|---|---|
| `--year` | from the fields | required when the fields file spans several years (BARC 2021–2023 in one GeoPackage export → run once per year) |
| `--group` | the title | groups regions in the top bar, e.g. `--group "BARC Maryland"` for the three BARC years |
| `--min-clear` | 0.7 | clear fraction at or above which an observation counts as usable. Match the model's threshold |
| `--threshold` | 0.5 | `p_silage` cut-off when the predictions carry no `pred_label` |
| `--gap-days` | 20 | curves are not joined across longer gaps |
| `--max-fields` | all | draw a random subset in the browser for very large regions. Metrics and county shares still use every field |
| `--note` | none | short line under the metrics, e.g. how the predictions were produced |
| `--counties`, `--county-shapes` | none | US county comparison, see below |
| `--synthetic` | off | marks the region as sample data |

Predictions and time series are both optional. Without predictions the region shows fields and curves, says "No predictions for this region yet", and invents no number.

**The metrics are computed here**, from the declared labels in the fields file and the predictions you pass. For labelled regions, pass **out-of-fold** predictions (for example `cv_predictions.parquet` from spatial cross-validation). Predictions from a model trained on the same fields would give training-set numbers.

### Predictions format (what the model step needs to write)

One row per field, Parquet or CSV. All seven columns must be present; values may be null.

| Column | Type | Meaning |
|---|---|---|
| `field_id` | string | matches `field_id` in the fields GeoJSON |
| `p_silage` | float 0–1 | probability of silage (positive class). Null = no prediction for this field |
| `harvest_start` | date | last usable observation where the crop is still standing. Null if no drop was found |
| `harvest_end` | date | first usable observation that is clearly harvested |
| `pre_ndvi` | float | NDVI at `harvest_start` |
| `pre_ndmi` | float | NDMI at `harvest_start` |
| `low_evidence` | bool | the model's abstention flag |

Optional columns:

| Column | Meaning |
|---|---|
| `low_evidence_reason` | short text shown to the presenter, e.g. `harvest window 38 days wide` or `only 4 clear observations` |
| `pred_label` | `silage` / `grain`, if the decision is not simply `p_silage >= --threshold` |

Extra columns are ignored. Predictions for fields that are not in the fields file are ignored, with a note on stderr.

### US county comparison

`--counties usda.csv` with columns `county_fips, county_name, state, silage_acres, grain_acres` (USDA corn acres harvested for silage and for grain). Each field is assigned to a county by its `county_fips` property if it has one. Otherwise it is assigned by its location inside `--county-shapes` (any GeoJSON or GeoPackage with `county_fips` or `GEOID`, e.g. Census TIGER cartographic boundaries). The predicted share is area-weighted: the area of fields called silage over the area of all predicted fields in the county.

## Files the app loads

```
app/data/regions.json            list of regions for the top bar
app/data/<region>/meta.json      title, year, synthetic flag, thresholds, headline metrics
app/data/<region>/fields.geojson field outlines + per-field prediction, label, coverage
app/data/<region>/series.json    {field_id: {t: [day of year, 0 = Jan 1], ndvi, ndmi, cf, q}}, q: 2 usable, 1 excluded, 0 no clear pixel
app/data/<region>/counties.json  (optional) county table + summary (mean absolute difference, Pearson r)
app/data/<region>/counties.geojson (optional) simplified county outlines
```

These files are produced by the scripts and never edited by hand.

## Sample data

```bash
python -m fieldvision.app.sample_data
```

This writes contract-format inputs (fields GeoJSON, long-format Sentinel-2 Parquet, predictions, a USDA-style county table) to `C:\t\fv\data\app\sample\`, then runs the normal export into `app/data/`. The toy model reads the simulated curve around the true harvest day, so the curves, harvest band and prediction agree. The generator plants the edge cases: fields with no usable observation, overcast spells around harvest (wide window, low evidence), atypical fields that the toy model gets wrong, fields with no declared label, a region with no predictions (BARC 2023), and a region with no labels but county numbers (US 2022). Use `--only at-2024 --n 40` to rebuild one region.

## Tests

```bash
python -m pytest tests/test_app_export.py --basetemp C:/t/fv/pytest-app
```

## Layout

```
index.html        page shell
css/app.css       styles (dark UI, large type, Okabe–Ito colours)
js/main.js        state, data loading, panel rendering
js/map.js         Leaflet layers and field styles
js/chart.js       season curves (SVG)
js/counties.js    county scatter and table (SVG / HTML)
js/evidence.js    evidence sentences, certainty
js/format.js      number and date formatting (missing → "—")
vendor/leaflet/   Leaflet 1.9.4 (BSD-2), integrity matches the official release
```
