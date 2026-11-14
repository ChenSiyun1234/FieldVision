# Sentinel-2 field time series

Run from the repository root in PowerShell. No login or API key is needed.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.s2.extract --fields fields.geojson --year 2025 --start 04-01 --end 11-30 --out C:\t\fv\data\timeseries\nl_2025 --workers 16
& C:\t\fv\venv\Scripts\python.exe -m pytest -q
```

Set `PYTHONPATH` once per shell. The package also supports an ordinary editable
installation in a team's own environment. Runtime dependencies are in
`pyproject.toml`; the supplied environment already has them.

## Input and output

Input is a WGS84 GeoJSON FeatureCollection. Each feature needs a unique string
`field_id`, an integer `year` matching the CLI, and polygon geometry. The other
contract properties (`label`, `country`, `source`, `crop_code`, `area_ha`, optional
`block_id`) are retained in input metadata but not repeated in the time series.
Join labels by `field_id`; extraction never uses them. When `block_id` is present,
fields sharing it are kept together and then split into read blocks of at most
6 km (cut at the middle of the extent), so a 10 km label block becomes about four
read blocks instead of being cut along an arbitrary 6 km grid.

The output directory contains:

- `timeseries.parquet`: final field/date table with the requested columns; dates
  are ISO `YYYY-MM-DD` strings, bands are dimensionless reflectance in [0, 1].
- `blocks/*.parquet`: independently resumable block tables.
- `blocks/*.json`: scene completion checkpoints, errors and reflectance diagnostics.
- `catalog/*.json`: cached STAC items, including original asset calibration metadata.
- `manifest.json`: exact science options, input/code fingerprint, timing, counts,
  per-block status, scene errors, and calibration checks.
- `fields_coverage.csv`: every input ID, status, maximum observed interior pixel
  count, usable date count, first/last date, and maximum gap in days.

One row per field/calendar date survives, choosing the greatest `n_clear`, then
clear fraction, then Collection 1 on ties. Rows with no clear pixels have NaN in
every band. `n_pixels` is the number of interior pixel centers in the polygon,
including cloudy or missing pixels; `n_clear` requires SCL 4/5/6/7 and valid
reflectance in all ten bands. `clear_fraction = n_clear / n_pixels`, or zero for a
zero-pixel polygon. Pixel counts are on a 10 m grid; resampling does not create
independent 10 m measurements for the native 20 m bands.

A usable date defaults to at least one clear pixel and clear fraction >= 0.2.
Change `--min-clear-pixels` and `--min-clear-fraction` for stricter modeling.
Coverage gaps include start-to-first and last-to-end gaps within the inclusive
requested window. A field with no usable dates has a gap spanning the window.

## Catalog and calibration

The catalog is <https://earth-search.aws.element84.com/v1>. Collection 1 is the
default. Direct probes on 2026-11-14 found the following item counts for April 1
through November 30, with no cloud filter and a 0.1 degree bbox per region:

| Region | 2022 C1 / older | 2024 C1 / older | 2025 C1 / older |
|---|---:|---:|---:|
| Austria (15.7,48.0,15.8,48.1) | 0 / 97 | 95 / 95 | 144 / 144 |
| Netherlands (5.5,52.0,5.6,52.1) | 1 / 94 | 104 / 104 | 142 / 140 |
| Western France (-1.0,47.0,-0.9,47.1) | 1 / 100 | 103 / 103 | 146 / 146 |
| US Midwest (-93.7,41.9,-93.6,42.0) | 1 / 98 | 96 / 96 | 121 / 120 |

Whole-year queries return 10–14 C1 items for 2022 in these locations, which can
misleadingly suggest coverage. The growing season is effectively missing.

Beltsville (BARC, bbox -76.96,39.00,-76.84,39.07, all MGRS 18SUJ), April 1 to
November 30, unique dates with scene cloud < 80% (`scripts/probe_beltsville.py`):

| Year | ES Collection 1 | ES older `sentinel-2-l2a` | Planetary Computer `sentinel-2-l2a` |
|---|---:|---:|---:|
| 2021 | 70 (baseline 05.00) | 73 (mixed 03.0x / 05.00) | 63 (02.12 / 03.00) |
| 2022 | **0** | 61 (04.00; 7 items flag=False) | 61 (04.00, 25 dates also as 05.10) |
| 2023 | 72 (05.09) | 72 | 72 |
| 2024 | 69 | 69 | 67 |

For 2022 and earlier, acquisitions missing from C1 (by MGRS tile/date/platform)
are added from the fallback source, by default Microsoft Planetary Computer
(`--fallback-source planetary:sentinel-2-l2a`, no account: the reader fetches an
anonymous SAS token from the public token endpoint and appends it to each COG URL;
the `planetary-computer` package is not needed). Where PC has both the original and
ESA's reprocessed product for one acquisition, the newest baseline is used. For
newer years, fallback occurs only when the block has no C1 scenes. `--no-fallback`
disables it; `--collection` selects any single source. Every row's `collection`
column names its source: `sentinel-2-c1-l2a`, `sentinel-2-l2a` (Earth Search older)
or `planetary:sentinel-2-l2a`.

Calibration rules:

- Earth Search C1: `raster:bands` scale and offset (0.0001, -0.1).
- Planetary Computer: unmodified ESA products, so reflectance = DN * 0.0001 - 0.1
  from processing baseline 04.00 on, and DN * 0.0001 before it. Items without a
  baseline are refused.
- Earth Search older collection: `earthsearch:boa_offset_applied=true` means the
  offset is already in the numbers (zero additional offset). **`false` is refused**,
  as is a missing flag with a nonzero declared offset. See the measurements below;
  also <https://github.com/Element84/earth-search/issues/71>.

Measured agreement (`scripts/compare_s2_sources.py` then
`scripts/analyze_s2_calibration.py`; 25 squares of 200 m at Beltsville, per-field
medians, fields >= 90% clear in both rows; numbers are median absolute difference
in reflectance over all field/date pairs, worst band in brackets):

| Comparison | Pairs (dates) | Rule used | MAD | Opposite offset rule |
|---|---:|---|---|---|
| 2024 PC b05.11 vs C1 | 145 (5) | baseline rule | 0.0000 all bands | 0.100 |
| 2024 PC b05.10 vs C1 | 56 (3) | baseline rule | 0.0004–0.0016 (B8A) | 0.100 |
| 2024 ES older flag=true vs C1 | 254 (8) | no extra offset | 0.0000 all bands | 0.100 |
| 2022 ES older flag=true vs PC original b04.00 | 265 (8) | no extra offset | 0.0004–0.0033 (B8A); NIR bias -0.0026 | 0.100 |
| 2022 ES older flag=false vs PC original b04.00 | 89 (5) | **no extra offset** fits | 0.0005–0.0028 | 0.100 (red -0.02 to -0.04) |
| 2022 PC reprocessed b05.10 vs PC original b04.00 | 116 (6) | baseline rule | 0.0006–0.0029 (B8A) | 0.100 |
| 2021 PC b02.12 vs C1 b05.00 | 44 (2) | no offset (pre-04.00) | 0.0012–0.0035 (B8A) | 0.100 |
| 2021 PC b03.00 vs C1 b05.00 | 65 (4) | no offset (pre-04.00) | 0.0019–0.0050 (B03); green bias +0.005 | 0.100 |

So every flag=false scene tested was already offset, and believing the flag would
make red negative. The PC baseline rule reproduces C1 exactly where the products
are identical; remaining differences (up to about 0.003 in 2022 and 0.005 for the
pre-04.00 products of 2021) come from different processor versions, not offsets.
Agreement is poorest in the NIR/red-edge bands and for the 2021 pre-04.00 products.

For blocks with at least 20 SCL vegetation pixels, require median red in (0, 0.35),
NIR > red + 0.02, and NIR < 1 before writing. A failed check records the scene error.
After that check, out-of-range reflectance is marked missing rather than clipped.
This conservative test may reject unusual vegetation/illumination; inspect errors.

### Known gap: fog and cloud that SCL calls clear

SCL sometimes labels low stratus/fog as class 5 (not vegetated), so it survives the
mask. Example: S2A_MSIL2A_20221128T095341_R079_T33TWM (Graz basin fog): 5613 field
pixels in SCL 5 with median B02 0.46 and B11 0.48. It happens in both sources and
all years, mostly October-November in Austria. A row with median B02 > 0.25 is
almost certainly such a case (B11 ~0.47, so not snow): 0.67% of usable rows in
at_2022, 0.83% in at_2024, ~0 elsewhere. Thresholds of 0.15-0.18 would also remove
spring bare-soil/haze rows, so 0.25 is the suggested floor. The extractor does not
drop these rows; `scripts/report_s2_run.py` counts them in each READY line, and the
filter belongs in feature building where it stays reversible.

### Difference from the organisers' starter notebook

The starter notebook reads Planetary Computer `sentinel-2-l2a` with scene cloud
cover < 40% and keeps SCL classes 4, 5, 6, 7. The mask here is the same, but the
scene-level filter is looser (`--max-cloud 80`): masking is per pixel, so a scene
that is 60% cloudy over the tile can still be clear over a field, and harvest dates
are the signal. To reproduce the starter's scene set, pass `--max-cloud 40`. Our
default source for 2021 and 2023 is Earth Search Collection 1 (ESA reprocessed
baselines 05.xx); 2022 comes from Planetary Computer like the starter. On identical
products the two sources agree to 0.0000 reflectance (table above).

### Beltsville evaluation fields

The evaluation GeoPackage holds 83 multipolygons in Maryland State Plane (US feet);
convert to EPSG:4326 per the contract. Small or oddly shaped research plots are
never dropped: every `field_id` appears in `fields_coverage.csv` with `n_pixels`
and a status (`no_pixels` for parcels with no 10 m pixel centre inside). The 5 m
inset is skipped for a field when fewer than four pixel centres would remain.
Expected sources: 2021 Collection 1 (70 dates) plus a few PC scenes, 2022 PC
(61 dates), 2023 Collection 1 (72 dates).

## Spatial and operational behavior

Fields are grouped in their centroid's local UTM zone (and label `block_id` if
present) into compact blocks up to 6 km wide. Each block is reprojected to each scene's own CRS and snapped to its
10 m grid. Whole parcels stay together even across a zone or MGRS boundary.
A single parcel wider than 6 km gets its own block; a window exceeding four million
pixels fails explicitly rather than exhausting memory. This is intended for farm
parcels, not regional masks or globally distributed multipart geometries.

The polygon is inset 5 m only if at least four actual pixel centers remain;
otherwise the original polygon is used. Pixel centers must fall inside polygons;
subpixel parcels can have zero centers and are reported as `no_pixels`.
Shared pixels in overlapping polygons are aggregated for each field without
additional imagery reads.

The worker pool reads `(scene, band)` windows through GDAL COG range requests.
Three scenes are in flight by default (`--scene-batch`); blocks run sequentially.
Both spectral bands and SCL use nearest-neighbor resampling, preserving original
values and nodata. SCL classes 4, 5, 6, 7 remain; bare post-harvest soil is retained.
Actual STAC footprints filter candidate scenes; asset nodata handles internal
missing footprints. Each band nodata is excluded before scale/offset conversion.

HTTP and raster reads retry with backoff. A failed scene does not abort a block.
Rerunning the same command skips complete blocks and successful scenes in partial
blocks. Inputs, scientific options and source code must match the saved
fingerprint; use a new output directory after changes. Do not run two processes
against the same output directory. The final table is rebuilt from block files.
Exit code 0 means processing complete, 2 means partial (scene/block/input errors),
and 1 means an invalid invocation or fatal setup error. Complete does not imply
that every field has a usable date; always inspect coverage.

## Indices

`fieldvision.s2.indices` supplies pure `ndvi`, `ndmi` (B8A/B11), `evi`, `nbr` and
`psri` functions accepting a DataFrame or mapping of array-like band columns.
`add_indices(table)` returns a copy with all five. PSRI is the Sentinel-2
approximation `(B04 - B02) / B06`; zero denominators return NaN. These are indices
of per-field band medians, not per-pixel index medians.

## Reproduce the Austrian smoke test

```powershell
& C:\t\fv\venv\Scripts\python.exe scripts/fetch_austria_smoke.py --out C:\t\fv\data\timeseries\smoke_input
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.s2.extract --fields C:\t\fv\data\timeseries\smoke_input\fields.geojson --year 2024 --start 04-01 --end 11-30 --out C:\t\fv\data\timeseries\austria_2024_smoke --workers 16
```

The fetcher uses the public INVEKOS API with `snar_code='109'` and bbox
15.5,47.9,16.0,48.2. No imagery or result tables belong in the repository.

## Measured runs (2026-11-14, home connection, 16 workers)

| Run | Fields / blocks | Scenes (unique) | Failures | Usable dates per field min/med/max | Largest gap (days) min/med/max | Wall |
|---|---|---:|---:|---|---|---:|
| Austria 2024 smoke (sparse, 30 real silage parcels) | 30 / 8 | 52 | 0 | 13 / 18 / 38 | 18 / 28 / 45 | 119 s |
| Austria 2024 dense synthetic (100 squares, ~5.5 km) | 100 / 2 | 52 per block | 0 | 33 / 37 / 41 | 15 / 20 / 40 | 60 s |
| Beltsville 2021 (25 squares) | 25 / 2 | 73 (70 C1 + 3 PC) | 0 | 45 / 52 / 56 | 15 / 17 / 25 | 111 s |
| Beltsville 2022 | 25 / 2 | 61 (all PC: 36 b04.00, 25 b05.10) | 0 | 39 / 43 / 48 | 15 / 17 / 22 | 163 s |
| Beltsville 2023 | 25 / 2 | 72 (C1) | 0 | 39 / 50 / 55 | 13 / 15 / 27 | 85 s |

Cost is roughly 0.5–0.6 s per (scene, block) pair, nearly independent of the
number of fields in a block, so dense blocks are what make it cheap. The Austrian
smoke fields in the western blocks are covered by one MGRS tile/orbit (24 scenes);
the eastern ones by two tiles (52 scenes, duplicates resolved per date).
