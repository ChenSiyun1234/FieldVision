# Handing us your own corn-field data

You collected your own fields and want to swap them in for the European training set. This page says what we need and how to hand it over. It takes about ten minutes if your file is in one of the shapes below.

## What we need

1. **A polygon for every field.** The outline of the field, not a dot. The satellite signal is averaged over the whole field.
2. **A label for every field: silage or grain.** Any wording is fine ("Silage", "Corn silage", "SIL"), as long as you tell us which value means which class.
3. **The harvest year** of each label. One field can have a label in several years.
4. Optional but useful: the harvest date, a field or farm name, anything else you know. We copy extra columns along if you ask; we never need them to train.

Any common file works: GeoPackage, Shapefile, GeoJSON or GeoParquet, in any coordinate system as long as the file says which one.

## The three layouts we accept

**A. One year for the whole file.** One row per field.

| field | use | geometry |
|---|---|---|
| 12 | Silage | polygon |
| 13 | Grain | polygon |

`--label-column use --year 2025`

**B. A year column.** One row per field and year, the same field may appear once per year.

| field | use | harvest_year | geometry |
|---|---|---|---|
| 12 | Silage | 2024 | polygon |
| 12 | Grain | 2025 | polygon |

`--label-column use --year-column harvest_year`

**C. Wide table.** One row per field, one label column per year (this is how the organisers' file looks). Blank means "no label that year".

| field | Corn_2022 | Corn_2023 | geometry |
|---|---|---|---|
| 12 | Silage | Grain | polygon |
| 13 | | Grain | polygon |

`--year-columns Corn_2022=2022 Corn_2023=2023`

## The command

From the repository root in PowerShell (output goes outside the repo; the folder must not already contain a `fields.geojson`):

```powershell
$env:PYTHONPATH='src'
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.from_file your_file.gpkg `
    --layer your_layer `
    --id-column field --label-column use --year 2025 `
    --map Silage=silage --map Grain=grain `
    --country US --source team-data --prefix team --out C:\t\fv\data\labels\team_data
```

Use `--year-column` or `--year-columns` for layouts B and C instead of `--year`. Leave out `--layer` for single-layer files. `--map` is not case-sensitive; blank labels and values you did not map are dropped and counted. Other useful options: `--keep COL COL` to carry columns along, `--split NAME` to mark a test set, `--crs EPSG:xxxx` if your file has no coordinate system, `--by-year-dirs` for one file per year. `python -m fieldvision.labels.from_file --help` lists everything.

It writes `fields.geojson`, `fields.parquet` and `summary.json` (counts per class and year, and everything it dropped and why). Send us the `summary.json` first: if a class count looks wrong, the cause is usually the `--map` values.
Every field gets an id `prefix-<id>-<year>`, so your id column must be unique per field.

## What happens next

We extract Sentinel-2 time series for every field-year (free public imagery, no login), train the model on your fields exactly as on the European ones, and compare. Extraction speed depends on how close fields sit to each other; fields packed into a few small areas are far cheaper than scattered ones. Time per thousand fields: to be confirmed, because our first large European run was still going when this was written (our 18-field test set took about two minutes). Imagery for 2022 is thin in the main catalogue and needs a fallback source; see `docs/s2-extractor.md` on the s2 branch.

## What will not work

* **Points only.** A dot per field cannot be averaged over a field. We stop with a clear message.
* **County-level or farm-level labels.** We need the label per field.
* **Fields smaller than about a third of a hectare.** Fewer than about 30 pixels, mostly edge; the converter keeps them but counts them (`fields_below_third_hectare`), and we recommend dropping them.
* **Labels without a year,** or a year that is the planting year rather than the harvest year.
* **A label column that only says "Corn".** We need to tell silage and grain apart.
* **A file without a coordinate system.** Tell us which one (`--crs`) or we cannot place the fields.
* **Repeated ids** (the same field twice for one year): the converter refuses.
