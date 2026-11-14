# FieldVision demo app

A static map of corn fields coloured by predicted harvest use (silage or grain), with the satellite evidence behind each call. Plain HTML, CSS and ES modules, no build step. Leaflet is vendored in `vendor/`, so the app works offline. Only the satellite basemap needs the network, and without it the fields are drawn on a plain background.

## Run it

From the repository root:

```bash
python -m http.server 8000 -d app
```

Then open <http://localhost:8000>. Opening `index.html` straight from disk does not work, because browsers block `fetch` on `file://`.

The committed data in `app/data/` is **synthetic**. Every synthetic region shows a yellow "Sample data · synthetic" badge, so nobody takes it for a result. When a real export overwrites a region folder, the badge for that region disappears.

## Demo path for a five-minute talk

1. **Opening**: one sentence says what the tool answers, over the dimmed map. Any click or key (Enter, Space, Esc) shows the map. Clicking the FieldVision logo brings the sentence back. Add `#intro=0` to the URL to skip it.
2. **Map**: fields are coloured by prediction, sky blue for silage and orange for grain. Low-evidence fields have a dashed outline. Zoomed out, each field is a dot.
3. **Click a field: defend this call.** The panel shows:
   - the call, and the probability as a bar from grain to silage;
   - in words, why the call is low evidence when it is (too few clear dates, clouds over the harvest window);
   - three pieces of evidence a skeptic can check against the curves below them:
     - **when it was cut**: the harvest window, how far NDVI fell, and how that compares with the region's typical silage and grain dates;
     - **the crop just before the cut**: NDVI and NDMI, and whether it stayed green or dried down slowly;
     - **the ground after the cut**: a residue index, NDTI. Stalks left behind mean a combine; short stubble and bare soil mean a chopper.
   - Each row gets a "silage-like" or "grain-like" tag. Then comes **what would have made it the other call**: which signals already point the other way, and which would also have to change, closest first. If the model supplies its own reason or counterfactual text, it is shown too.
4. **As-of slider** (bottom of the map): replays the season. Each step shows the calls made with satellite data up to that date. Observations after the date disappear from the curves, and the metrics change to that date. The small chart above the slider plots accuracy (and the share of confident calls) against the cutoff date. ▶ plays through the season.
5. **Check vs declared** (top right of the map): wrong calls turn magenta. The **Model wrong** tab lists them, most confident first.
6. **Headline numbers** for the region: precision, recall and false positive rate (silage is the positive class), the number of fields, the share flagged low evidence, and the share called silage.
7. **Check these first**: the least certain predictions, which are where a survey call or field visit tells us the most.
8. **Totals check** (US regions exported with `--counties`):
   - our silage and grain acres per county and summed to state, next to USDA's reported acres;
   - the gap stated in words ("13% more than USDA"), and how much of USDA's corn our fields cover;
   - the county silage-share scatter.

The URL keeps the state (`#r=at-2024&f=<field_id>&asof=2024-09-12&mode=check`), so a bookmarked link reopens exactly that view.

## How to read the curves

| Mark | Meaning |
|---|---|
| filled dot | usable observation: `clear_fraction` ≥ the export's `--min-clear` (default 0.7) |
| hollow dot | excluded: some clear pixels, but below the threshold (usually cloud or haze) |
| tick under the axis | acquisition with no clear pixel at all |
| line | joins usable observations only, and never across a gap longer than `--gap-days` (default 20). Longer gaps stay open and are labelled "N d no clear view" |
| shaded band | harvest window: last observation still standing to first observation clearly harvested |
| solid ring | the value just before the cut (NDVI, NDMI) |
| dashed ring | the value just after the cut (NDVI, NDTI) |
| dashed line on NDTI | residue threshold: above means stalks left on the ground, below means bare soil |
| yellow line | the as-of date. Everything right of it is hidden |

Index definitions:
- NDVI = (B08 − B04) / (B08 + B04)
- NDMI = (B8A − B11) / (B8A + B11)
- NDTI = (B11 − B12) / (B11 + B12)

NDVI and NDMI match `fieldvision.s2.indices`.

The "silage-like" and "grain-like" tags come from fixed display thresholds stored in `meta.json` (`cues`) and from the region's own median cutting dates (`harvest_ref`, from confident calls). They describe the curve, not the model's rule, so a row can disagree with the prediction. That is deliberate: a disagreement is exactly what a skeptic should see. The panel says this in small print.

## Files the modelling step must write

All files are one row per field (or per field and cutoff), Parquet or CSV, with `field_id` matching the fields GeoJSON. Columns listed as required must be present, but their values may be null. Extra columns are ignored.

### 1. `predictions.parquet`: full-season call per field (`--predictions`)

| Column | Required | Type | Meaning |
|---|---|---|---|
| `field_id` | yes | string | |
| `p_silage` | yes | float 0–1 | probability of silage (positive class). Null = no prediction |
| `harvest_start` | yes | date | last usable observation with the crop still standing. Null if no drop found |
| `harvest_end` | yes | date | first usable observation clearly harvested |
| `pre_ndvi` | yes | float | NDVI at `harvest_start` |
| `pre_ndmi` | yes | float | NDMI at `harvest_start` |
| `low_evidence` | yes | bool | the model's abstention flag |
| `low_evidence_reason` | no | string | e.g. `harvest window 38 days wide`; shown as "Model's flag" |
| `explanation` | no | string | short reason for the call, e.g. `early cut while still green and wet`; shown as "Model's reason" |
| `flip_reason` | no | string | what would have changed the call, in the model's terms |
| `post_ndvi` | no | float | NDVI just after the cut. If missing, read from the time series at `harvest_end` |
| `post_ndti` | no | float | residue index after the cut. If missing, the median NDTI of the first (up to three) usable views within 30 days of `harvest_end` |
| `pred_label` | no | `silage`/`grain` | only if the decision is not `p_silage >= --threshold` |

For labelled regions these must be **out-of-fold** predictions (spatial cross-validation), because the export computes precision, recall and FPR from them.

### 2. `predictions_by_cutoff.parquet`: early-season calls (`--asof`)

Long format, one row per field per cutoff date:

| Column | Required | Type | Meaning |
|---|---|---|---|
| `field_id` | yes | string | |
| `cutoff_date` | yes | date | only satellite data on or before this date was used |
| `p_silage` | yes | float 0–1 | probability of silage at that date. Null = no call possible yet |
| `low_evidence` | yes | bool | abstention flag at that date |
| `pred_label` | no | `silage`/`grain` | as above |

Use the same cutoffs for every field, e.g. every two weeks from Aug 1 to early November. The full season does not need its own cutoff; the app uses `predictions.parquet` for it. As with file 1, labelled regions need out-of-fold predictions.

### 3. Time series (from the extractor, `--timeseries`)

The existing long-format contract. The app needs `field_id, date, n_pixels, n_clear, clear_fraction, B04, B08, B8A, B11`, plus **`B12`** for the residue index. Without B12 the "ground after the cut" row says it was not seen.

### 4. USDA county table (`--counties`, US only)

CSV with `county_fips, county_name, state, silage_acres, grain_acres`: USDA corn acres harvested for silage and for grain. Each field is assigned to a county by its `county_fips` property if it has one. Otherwise it is assigned by its location inside `--county-shapes` (GeoJSON or GeoPackage with `county_fips` or `GEOID`, e.g. Census TIGER). Our acres are the areas of the fields we call silage or grain (hectares × 2.47105), and include low-evidence calls; the view states how many acres those are.

## Exporting a region

```bash
python -m fieldvision.app.export --fields <fields.geojson> --timeseries <parquet file or folder> --predictions predictions.parquet --asof predictions_by_cutoff.parquet --out app/data/at-2024 --title "Austria 2024" --note "spatial 5-fold CV, out-of-fold"
```

Run it with `src` on the import path (in PowerShell `$env:PYTHONPATH="src"`; in bash `PYTHONPATH=src`), or after `pip install -e .`. The output folder name is the region id. Use the ids the sample data uses (`at-2024`, `nl-2025`, `fr-2022`, `barc-2021`, `barc-2022`, `barc-2023`, `us-2022`) and the real export replaces the sample in place, with no code changes. The export also adds or updates the region in `app/data/regions.json`.

| Option | Default | |
|---|---|---|
| `--year` | from the fields | required when the fields file spans several years (BARC: run once per year) |
| `--group` | the title | groups regions in the top bar, e.g. `--group "BARC Maryland"` |
| `--min-clear` | 0.7 | clear fraction at or above which an observation counts as usable. Match the model's threshold |
| `--threshold` | 0.5 | `p_silage` cut-off when there is no `pred_label` |
| `--gap-days` | 20 | curves are not joined across longer gaps |
| `--max-fields` | all | draw a random subset in the browser for very large regions. Metrics, as-of metrics and county totals still use every field |
| `--note` | none | short line under the metrics, e.g. how the predictions were produced |
| `--counties`, `--county-shapes` | none | US totals check |
| `--synthetic` | off | marks the region as sample data |

Predictions, as-of predictions and time series are all optional. Whatever is missing shows as missing ("—", "not seen", "no predictions yet"), never as a made-up number.

## Files the app loads

```
app/data/regions.json              list of regions for the top bar
app/data/<region>/meta.json        title, year, synthetic flag, thresholds, cues, harvest_ref, headline metrics
app/data/<region>/fields.geojson   outlines + per-field call, label, coverage, evidence numbers
app/data/<region>/series.json      {field_id: {t: [day, 0 = Jan 1], ndvi, ndmi, ndti, cf, q}}, q: 2 usable, 1 excluded, 0 no clear pixel
app/data/<region>/asof.json        (optional) {cutoffs, metrics per cutoff, fields: {id: {p: [...], low: [...]}}}
app/data/<region>/counties.json    (optional) county rows, state totals, share summary
app/data/<region>/counties.geojson (optional) simplified county outlines
```

These files are produced by the scripts and never edited by hand.

## Sample data

```bash
python -m fieldvision.app.sample_data
```

This writes all four input files in the formats above to `C:\t\fv\data\app\sample\`, then runs the normal export into `app/data/`.
- The toy model reads the simulated curve around the true harvest day, so the curves, harvest band and call agree.
- Greenness and moisture at the cut follow how a field looks. The ground after the cut follows what really happened. So the few atypical fields the toy model gets wrong (late, dry silage; early, green grain) show a residue row that contradicts the call. That is the kind of mismatch a skeptic should catch.
- The generator also plants:
  - fields with no usable observation;
  - overcast spells over the harvest;
  - fields with no declared label;
  - fields with and without an `explanation` string;
  - a region with no predictions (BARC 2023);
  - a region with no labels but USDA county acres (US 2022).

Use `--only at-2024 --n 40` to rebuild one region.

## Tests

```bash
python -m pytest tests/test_app_export.py --basetemp C:/t/fv/pytest-app
```

## Layout

```
index.html        page shell
css/app.css       styles (dark UI, large type, Okabe–Ito colours)
js/main.js        state, data loading, panel rendering, as-of view
js/map.js         Leaflet layers and field styles
js/chart.js       season curves (SVG)
js/timeline.js    as-of slider with the accuracy-by-date chart
js/counties.js    totals check: state cards, county table, share scatter
js/evidence.js    defend-this-call rows, counterfactual lines, low-evidence wording
js/format.js      number and date formatting (missing → "—")
vendor/leaflet/   Leaflet 1.9.4 (BSD-2), integrity matches the official release
```
