# FieldVision demo app

A static map of corn fields coloured by predicted harvest use (silage or grain), with the satellite evidence behind each call. Plain HTML, CSS and ES modules, no build step. Leaflet is vendored in `vendor/`, so the app works offline. Only the satellite basemap needs the network, and without it the fields are drawn on a plain background.

## Run it

From the repository root of a fresh clone:

```bash
python app/run_demo.py
```

Then open <http://localhost:8000>. This one command:
1. reads the Beltsville files from the team data folder (`C:\t\fv\data`; use `--data-root` for another location);
2. writes the small JSON files the app loads into `app/data/`;
3. serves the app on port 8000.

Needs Python with numpy, pandas, pyarrow, geopandas and shapely (`pip install -e .` installs them). No npm, no build step.

| Read (read-only) | What |
|---|---|
| `labels/barc_<year>/fields.geojson` | field outlines and the organisers' truth (2021: 18, 2022: 7, 2023: 19) |
| `timeseries/barc_<year>/timeseries.parquet` | Sentinel-2 per field |
| `app/predictions_eu_logistic_v1/barc-<year>/predictions.parquet` | the stored predictions, with `explanation` and `flip_reason` |
| `registry/eu_logistic_v1/provenance.json` → its training run's `run.json` | the class medians of the Austrian training fields, for the evidence rows (optional) |

**The Beltsville-derived files are not in git.** `app/data/` is ignored and rebuilt by the command above. Other options:
- `--no-serve` only rebuilds the data;
- `--serve-only` serves what is already built;
- `--sample` builds synthetic sample regions instead, for a machine without the team's data. These regions are marked "Sample data · synthetic".

**Network.** Leaflet and everything else are in the repo, so the app runs without network access. The one exception is the satellite base map: Esri World Imagery tiles from `server.arcgisonline.com` over HTTPS, with no key or account. Without them the fields are drawn on a plain dark background, and the "Plain" button switches to that background deliberately. Opening `index.html` straight from disk does not work, because browsers block `fetch` on `file://`; use the command above.

### What the Beltsville view shows

- **Opening card**: the organisers' question, then the result in one block:
  - 42 of 44 field-years right, against 32 of 44 for "always grain";
  - 11 of 12 silage fields found;
  - the model and what it was trained on;
  - one sentence that this is a small test (44 field-years on one farm, 12 of them silage).

  The three requirements follow underneath.
- **Year switch** (2021, 2022, 2023) in the top bar.
- **Map**: fields are coloured by the stored prediction. A **miss** (prediction differs from the organisers' truth) keeps its colour but gets a thick magenta edge; the legend counts them. Calls the model abstains on (`low_evidence`) are drawn with a dashed outline. "Check vs truth" recolours every field as right or wrong.
- **Click a field**: the call with a probability bar, and the organisers' truth next to it. Then the traps that apply, and three evidence rows (cliff or fade, green or browned before the drop, bare soil or residue after).
  - Each row gives the field's own number next to the typical silage and grain values of the training fields. A row points to whichever value is closer: the same rule and the same numbers the model's own summary uses, recomputed with the unchanged feature code. The model's summary and its biggest counter-pull are shown below the rows, in plain words.
  - Below that come the greenness, leaf-moisture and residue curves, with the detected drop marked as a band.
- There are no cut-off-date predictions for Beltsville, so the as-of slider is hidden.

## Demo path for a five-minute talk

The demo mirrors the organisers' wording. Their challenge, "From space: which cornfields were chopped, and which were combined?", comes with three requirements: **public satellite data, earlier than January, explainable to a skeptic**. Their signal slide reads "Silage falls off a cliff. Grain fades out."

1. **Opening**: their question over the dimmed map, with the three requirements each answered by the loaded region's numbers:
   - **Public satellite data**: Sentinel-2, and the median number of looks and clear looks per field.
   - **Earlier than January**: the first cutoff from which accuracy stays within 3 points of the full season (or the confident share, for regions without labels). NASS publishes the split in January.
   - **Explainable to a skeptic**: every call shows its curve.

   Any click or key (Enter, Space, Esc) shows the map. Clicking the FieldVision logo brings the card back. Add `#intro=0` to the URL to skip it.
2. **Map**: fields are coloured by prediction, sky blue for silage (chopped) and orange for grain (combined). Low-evidence fields have a dashed outline. Zoomed out, each field is a dot.
3. **Click a field.** The panel shows:
   - the call ("Silage · chopped" or "Grain · combined"), and the probability as a bar from grain to silage;
   - the three requirements for this one field:
     - which public data the call rests on (clear looks out of all looks);
     - from which date the call was settled (from the as-of predictions);
     - that the checks follow below;
   - **traps**, named in the organisers' words wherever one applies:
     - "Clouds hid the week it was cut";
     - "Clouds: only N clear looks all season";
     - "A second green-up after harvest" (double crop or cover crop);
     - "Early does not mean silage" (cut early but already browned).

     For a low-evidence call this box explains why the call is low evidence.
   - **Explainable to a skeptic**: three rows in plain words first, with the index names as small secondary labels. Each can be checked against the curves:
     - **The drop: cliff or fade?** A cliff is still green one look and bare the next. A fade is weeks of browning, then the harvest. The row also gives the cut date against the region's typical silage and grain dates.
     - **Right before the drop**: green and moist, or browned and dry.
     - **After harvest: bare soil or residue?** Stalk residue means a combine; almost nothing left means a chopper.
   - Each row gets a "silage-like" or "grain-like" tag. Then comes **what would have made it the other call**: which signals already point the other way, and which would also have to change, closest first. If the model supplies its own reason or counterfactual text, it is shown too.
4. **As-of slider** (bottom of the map): replays the season. Each step shows the calls made with satellite data up to that date. Observations after the date disappear from the curves, and the metrics change to that date. The small chart above the slider plots accuracy, the share of confident calls, and the "always grain" yardstick against the cutoff date. ▶ plays through the season.
5. **Check vs declared** (top right of the map): wrong calls turn magenta. The **Model wrong** tab lists them, most confident first.
6. **Headline numbers** for the region:
   - precision, recall and false positive rate (silage is the positive class);
   - the number of fields, the share flagged low evidence, and the share called silage;
   - accuracy next to what "always grain" would score. About 94% of US corn is grain, so that baseline looks deceptively good.
7. **Check these first**: the least certain predictions, which are where a survey call or field visit tells us the most.
8. **Totals vs NASS** (US regions exported with `--counties`), the organisers' sanity check:
   - our silage and grain acres per county and summed to state, next to the acres USDA NASS reports;
   - the gap stated in words ("13% more than NASS"), and how much of NASS's corn acreage our fields cover;
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
| dashed line on ground residue | above means stalk residue on the ground, below means bare soil |
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
| `--evidence-reference` | none | a training run's `run.json` (or `{feature: {silage, grain}}`). The export recomputes the model's evidence numbers with `fieldvision.features` and marks which class median each is closer to, so the evidence rows match the model's own summary |
| `--label-name` | `Declared` | what the label column is called in the app, e.g. `"Organisers' truth"` |
| `--model-name`, `--trained-on` | none | shown in the opening summary |

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
