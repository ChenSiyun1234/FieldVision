# Maize declaration labels

The files describe the crop farmers declared for a season, not an observed harvest.
A field declared as silage may eventually be harvested for grain, or vice versa.
Use the labels as supervised proxies and inspect harvest curves separately.

## What is available

All under `C:\t\fv\data\labels\` (outside git). One line per finished file is appended to `READY.txt` there.

| Folder | Source and release | Fields per class | Blocks |
|---|---|---:|---:|
| `at_2024` | Austria INVEKOS Schläge `2024_2` | 1200 | 20 |
| `at_2023` | Austria INVEKOS Schläge `2023_2` | 600 | 10 (subset of the `at_2024` blocks) |
| `at_2022` | Austria INVEKOS Schläge `2022_2` | 600 | 10 (same blocks as `at_2023`) |
| `nl_2025` | Netherlands BRP 2025 (PDOK WFS) | 800 | 25 |
| `fr_2022` | France RPG 2022 (IGN Géoplateforme WFS) | 800 | 21 |

Each folder holds `fields.geojson` (FeatureCollection, EPSG:4326), `fields.parquet` (GeoParquet, same CRS),
`summary.json`, `quicklook.png`, `country_boundary.geojson` and `candidates.geojson`
(every eligible parcel found in the fetched blocks, for auditing). Properties of every feature:
`field_id, label, year, country, source, crop_code, crop_name, area_ha, block_id, centroid_lon, centroid_lat`,
plus `region` (the sampling region the block belongs to; useful for region hold-out). No farm or applicant
identifiers are kept: raw properties are whitelisted before anything is cached or written.

## How the sample is built

Run from the repository root (PowerShell):

```powershell
$env:PYTHONPATH='src'
& C:\t\fv\venv\Scripts\python.exe -m pytest -q
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.fetch --country AT --year 2024 --out C:\t\fv\data\labels\at_2024 --target-per-class 1200
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.fetch --country AT --year 2023 --out C:\t\fv\data\labels\at_2023 --target-per-class 600 --blocks-from C:\t\fv\data\labels\at_2024\fields.geojson
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.fetch --country AT --year 2022 --out C:\t\fv\data\labels\at_2022 --target-per-class 600 --blocks-from C:\t\fv\data\labels\at_2024\fields.geojson
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.fetch --country NL --year 2025 --out C:\t\fv\data\labels\nl_2025 --target-per-class 800
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.fetch --country FR --year 2022 --out C:\t\fv\data\labels\fr_2022 --target-per-class 800
```

1. **Ranking.** Each country has a few sampling windows (lon/lat boxes, one per region). In each window the
   scarcer class of that region is downloaded (Austria: silage; Netherlands: grain; France: grain in the silage-heavy
   west, silage in the grain-heavy south-west) and counted per 10 km block of the EPSG:3035 grid.
2. **Block download.** Blocks are tried best-first, taking one block from each region in turn. For every tried block
   both classes are downloaded completely (block box, then filtered to parcels whose centroid is in the block).
   This continues until the balanced sample reaches the target and at least three regions are represented.
3. **Selection.** In each block `n = min(silage, grain, 60)` parcels of each class are taken (seed 42, random after sorting
   ids); blocks with `n < 5` are not used; the last block may be cut to hit the target exactly.
   Both classes are therefore present in every block with identical counts.
4. **Filters.** Only polygons of at least 1 ha (area in EPSG:3035), valid after `make_valid` repair
   (polygonal parts kept; counts in `summary.json`), fully inside the Eurostat GISCO 2024 1:1M country outline.
   The outline is coarse, so some valid border parcels are dropped (see `outside_country_boundary`).
5. **Reusing blocks across years** (`--blocks-from`). Austria 2023 and 2022 use blocks of `at_2024` taken in the same
   best-first order, so both years are the same 10 blocks and the geography is comparable. Parcels are re-downloaded from each
   year's own collection and sampled again; field ids differ between years, parcels change from year to year.
6. **Caching.** Every API page is stored in `C:\t\fv\data\labels\_cache\<folder>` (properties already whitelisted).
   Re-running the same command reads the cache and gives byte-identical `fields.geojson`
   (checked by md5 for all five files). Delete the cache folder to refetch.

Split train/test by `block_id`, never by random field; hold out whole regions for a stronger test.
Blocks are 10 km cells, so neighbouring blocks may still be spatially correlated.
The quick-look PNGs show real field polygons per region and centroid dots on the country overview. No imagery is used.

The exporter re-reads the GeoJSON and checks unique ids, two labels only, country and year, valid geometry,
recomputed areas, whole-polygon containment in the country outline, equal class counts per block and at least three regions;
the GeoParquet is re-read for ids and CRS.

## Austria: INVEKOS Schläge

Provider: Agrarmarkt Austria (AMA), OGC API Features at `https://gis.lfrz.gv.at/api/geodata/i009501/ogc/features/v1`.
Farmer-declared management units for the vegetation period. Collections exist for 2015 to 2026-1; from 2022 on each year
has a `-1` and a `-2` release. **Used: `2022_2`, `2023_2`, `2024_2`** (the `-2` revision, available for every year we use).
For 2024 the `-2` release is the winter revision with temporal reference **2024-12-31** (data.gv.at metadata, publication 2025-01-16).
That reference date was only read for 2024; for 2022 and 2023 the metadata was not opened, so we assume the same convention.
National counts of the two classes differ little between `-1` and `-2` (2024: silage 51,447 / 51,624, grain 107,418 / 107,687).

Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), linked from the collection and opened.
Attribute "Agrarmarkt Austria, INVEKOS Schläge Österreich". Our derivative selects maize, filters/repairs geometry,
balances samples and adds area/grid properties.

| Code | Registry name | Use | National parcels 2022_2 / 2023_2 / 2024_2 |
|---|---|---|---|
| 109 | SILOMAIS | silage | 53,309 / 51,488 / 51,624 |
| 105 | KÖRNERMAIS | grain | 109,413 / 113,212 / 107,687 |
| 106 | MAIS CORN-COB-MIX (CCM) | excluded: cob mix | 2024: 8,358 |
| 107 | ZUCKERMAIS | excluded: sweet corn | 2024: 223 |
| 135 | GRÜNMAIS | excluded: ambiguous green maize use | 2024: 209 |
| 173 | SAATMAISVERMEHRUNG | excluded: seed production | 2024: 2,757 |

Notes. Filtering needs `filter-lang=cql2-text&filter=snar_code IN ('109','105')`; a plain property parameter is silently
ignored. The server caps pages at 100 features; we page with `startIndex` and compare the total with `numberMatched`.
Field id is `at<year>-<geo_id>`. The farm key `fs_kennung` is never written. Sampling windows: Upper Austria
(13.45–14.45 E, 48.0–48.5 N), Lower Austria (14.9–15.9 E, 48.0–48.5 N), Styria (15.2–16.05 E, 46.75–47.2 N).
Of the parcels received in the 2024 blocks, 8,959 of 20,516 were under 1 ha and dropped.

## Netherlands: BRP gewaspercelen 2025

Provider: RVO, distributed through PDOK. Annual crop declarations with reference date **15 May**; live records say
`jaar=2025`, `status=Definitief`. The OGC API (`https://api.pdok.nl/rvo/gewaspercelen/ogc/v1`) serves only the latest year
and rejects crop-code filters (HTTP 400), so we use PDOK's WFS `https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0`
with FES XML filters on code, year and bounding box (EPSG URN axis order: latitude, longitude). A filter with a single
`<Or>` operand is rejected by this MapServer, so one-code requests send the bare comparison.

Licence: the API links the [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/deed.nl); the ATOM
feed states [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/deed.nl). Both pages were opened. Credit RVO/PDOK.

Codes checked against the official RVO [2025 crop list (22 April)](https://www.rvo.nl/sites/default/files/2025-04/Tabel-2e-gewascodelijst-stikstofgebruiksnorm-2025-april.pdf).
National live WFS counts on 2026-11-14 before geometry/area filters:

| Code | Dutch name | Parcels | Use |
|---|---|---:|---|
| 259 | Mais, snij- | 95,192 | silage |
| 316 | Mais, korrel- | 3,992 | grain |
| 317 | Mais, corncob mix | 1,525 | excluded: cob mix |
| 814 | Mais, suiker- | 395 | excluded: sweet corn |
| 1935 | Maiskolvensilage | 812 | excluded: cob silage, not whole plant |
| 2032 | Mais, energie | 24 | excluded: energy use, not an unambiguous grain label |

Only 2025 is usable: the WFS holds 2,331,084 records for 2025 and none for 2024 or 2026; the 2024 GeoPackage
is 3.11 GB (2025: 2.93 GB), above our 2 GB limit, so no bulk file was fetched. Sampling windows: Limburg, North Brabant, Eastern
Gelderland, Drenthe (boxes in `adapters.py`). Grain is scarce (about 1,480 qualifying parcels in the four windows), which is why this
file has 25 blocks and 15 of them hold fewer than 30 pairs (four have 5 to 7). Field id is the PDOK feature id, `nl2025-brp_gewas.<uuid>`.
The audit script `python -m fieldvision.labels.audit --out <file>` refreshes national counts and download sizes.

## France: RPG 2022

Provider: IGN / ASP, Géoplateforme WFS `https://data.geopf.fr/wfs/ows`, layer `RPG.2022:parcelles_graphiques`,
anonymised graphic parcels of the CAP declaration. Per the data.gouv.fr dataset page, the layer holds the boundaries
declared in year N as known and fixed on **1 January of year N+1**; the crop is the main crop of the parcel.

Licence: Licence Ouverte / Open Licence 2.0 (Etalab), shown on the data.gouv.fr dataset page and confirmed on
the [licence page](https://www.data.gouv.fr/pages/legal/licences/etalab-2.0/). Attribution must name the publisher and the date of the latest update.

| Code | Registry name (`codes_cultures`) | Parcels, France 2022 | Use |
|---|---|---:|---|
| MIE | Maïs ensilage | 357,080 | silage |
| MIS | Maïs | 406,261 | grain-type maize (the registry's plain maize class) |
| MID | Maïs doux | 2,126 | excluded: sweet corn |

**Never use 2023 or later**: from 2023 the registry merges silage into `MIS` (research notes: 2023 MIE = 0), so the label would be
meaningless. `MIS` is the registry's generic maize and includes grain maize; it also may include maize cut green for other uses.
The `the_geom` field name is specific to 2022 (2024 uses `geom`). Bounding boxes are latitude,longitude inside `CQL_FILTER`.
Windows: Ille-et-Vilaine and Mayenne (silage-heavy west, grain is the scarcer class) and Landes, Béarn and Gers (grain-heavy
south-west, silage is scarcer). Landes and Gers have few silage parcels, so they contribute only 52 and 34 pairs;
Béarn gives 247. Parcels with fewer than 1 ha are dropped by our own area computation (registry `surf_parc` is not used);
465 invalid polygons in the fetched blocks were repaired. Field id is `fr2022-<id_parcel>`.
The RPG `id_parcel` is the anonymised parcel id; farm identifiers are not in this layer.

## Known limits

* A declared use is not an observed harvest; farmers can change plans, declarations are checked only partly.
* The registries do not say what happened after the declaration; any maize can be cut early or left to dry, so some labels are wrong by an unknown share. We have not measured it.
* Dutch grain maize is rare (3,992 parcels nationally against 95,192 silage), so the Dutch blocks are small; Drenthe adds only 42 pairs.
* The windows are targeted samples, not national probability samples; counts per block differ because the scarce class sets the pace.
* Sentinel-2 imagery of 2022 and 2023 comes from an older processing collection than 2024-2025; Austria 2022 and 2023 exist so the model can be checked on that older imagery.

## Evaluation file (Beltsville, BARC) and other external files

`python -m fieldvision.labels.from_file` converts any polygon file to the same contract (see `docs/data-handoff.md`).
The organisers' evaluation GeoPackage is reproduced with this command (do not point `--out` at the existing `barc_*` folders;
the converter refuses to overwrite a `fields.geojson`):

```powershell
$env:PYTHONPATH='src'
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.from_file C:\t\fv\data\eval\barc\BARC_Corn.2021-2023.gpkg `
    --layer barc_silage_grain_corn20212023 --id-column OBJECTID `
    --year-columns Corn_2021=2021 Corn_2022=2022 Corn_2023=2023 `
    --map Silage=silage --map Grain=grain --country US --source barc-gpkg-2021-2023 --prefix barc `
    --split external_eval --block-id US-BARC --keep FARM FIELD --by-year-dirs --out C:\t\fv\data\labels\_check_barc
```

Result on 2026-11-14 against the files used for extraction: same 44 field-years (2021: 18, 2022: 7, 2023: 19; 32 grain,
12 silage), same ids and labels, same polygons (zero symmetric difference). Differences: `area_ha` is 0.002 ha larger at most
(0.018 %) because the converter uses an equal-area projection while the older file used a UTM zone; centroids differ by
under 1e-6 degrees; `crop_name` is the source value (`Grain`) instead of `Corn (Grain)`; single-part polygons are `Polygon`
instead of `MultiPolygon`; the older files carry extra `acres_reported`, `barc_*` and `source_label` properties.
Without `--block-id` the converter assigns 10 km grid blocks like the European files.
