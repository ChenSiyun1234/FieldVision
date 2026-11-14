# US county corn fields (Maryland 2022 backtest)

Unlabelled corn fields for one county and season, built from USDA NASS Crop Sequence
Boundaries (CSB), so the frozen models can be run on a whole county and the predicted acres
compared with the 2022 Census of Agriculture. No labels: the county check is the only test.

| Folder (under `<data>\labels\`) | County | Fields | Corn acres (polygons) | Census grain / silage harvested | Silage share |
|---|---|---:|---:|---:|---:|
| `us_24021_2022` | Frederick (24021) | 2,233 | 39,030 | 31,752 / 7,748 | 19.6 % |
| `us_24035_2022` | Queen Anne's (24035) | 1,449 | 51,533 | 53,815 / 715 | 1.3 % |

Each folder has `fields.geojson` (EPSG:4326), `fields.parquet`, `summary.json` and
`county_check.json`. Properties: the contract columns with `label` null, `field_id` =
`csb-<CSBID>`, `year` 2022, `country` US, `source` nass-csb, `crop_code` 1, `crop_name` Corn,
`area_ha` computed in UTM 18N (EPSG:32618), `block_id` on a 10 km grid of the same UTM
coordinates (`US-E<km>N<km>`), plus `acres_reported` (CSBACRES) and `county_fips`.

## Source

National file only: `NationalCSB_2018-2025_rev23.zip` (3,758,023,527 bytes, Last-Modified
2026-10-05), file geodatabase `CSB1825.gdb`, layer `national1825`, 15,973,095 polygons (the
metadata says 15,973,106). Checked against the
[2025 metadata](https://www.nass.usda.gov/research-and-science/crop-sequence-boundaries/metadata-crop-sequence-boundaries-2025):
layer name, attribute names (`CSBID`, `CSBACRES`, `CDL2018`..`CDL2025`, `STATEFIPS`,
`CNTYFIPS`, ...), CDL class 1 = Corn, CRS = NLCD Albers (NAD83, parallels 29.5/45.5, origin
23N 96W, read as EPSG:5070). Differences from the metadata: `CSBACRES` is a float (metadata:
integer); two extra attributes `CSBYEARS` and `ORIG_FID`.

The server sends an empty body to a plain GET but answers range requests; the archive was
fetched in 64 MB ranges and the zip CRC of the main table was checked while reading. The fiboa
GeoParquet on source.coop (`us-usda-cropland`) is **not** the same data: it is the 2016-2023
release with adjacent fields of equal 2023 crop dissolved and only the 2023 crop kept, so it
has no 2022 class. No per-state CSB downloads exist. The main table is a single 3.65 GB deflate
member, so the archive cannot be read partially; it was deleted after the state subset.

## Commands

From the repository root (PowerShell, `$env:PYTHONPATH='src'`):

```powershell
python scripts/csb_subset.py --zip <dir>\NationalCSB_2018-2025_rev23.zip --state 24 --out <data>\us\csb\csb1825_md.parquet
python -m fieldvision.labels.from_file <data>\us\csb\csb1825_md.parquet `
    --where STATEFIPS=24 --where CNTYFIPS=021 --where CDL2022=1 --unlabelled --year 2022 `
    --id-column CSBID --no-year-in-id --prefix csb --country US --source nass-csb `
    --crop-code-column CDL2022 --crop-name Corn --metric-crs EPSG:32618 `
    --keep CSBACRES=acres_reported --set county_fips=24021 --out <data>\labels\us_24021_2022
python scripts/county_corn_check.py --fields <data>\labels\us_24021_2022\fields.geojson --fips 24021 --year 2022 `
    --counties <data>\us\tiger\tl_2022_md_county.geojson `
    --quickstats <data>\eval\quickstats\corn_grain_silage_acres_2012_2025.csv `
    --csb <data>\us\csb\csb1825_md.parquet --out <data>\labels\us_24021_2022\county_check.json
```

The Maryland subset (86,840 polygons, 22,888 with CDL2022 = corn) and the Maryland counties of
TIGER/Line 2022 (`tl_2022_us_county.zip`) are kept under `<data>\us\`.

County membership is CSB's own `CNTYFIPS`. Against the TIGER 2022 outline: Frederick 1 field
whose representative point is outside the county (43 ha of polygon area outside in total),
Queen Anne's none (5 ha outside).

## What the acreage check can and cannot show

USDA numbers come from `fieldvision.model.quickstats.select` (one record per geography, year,
crop use and measurement; county rows from the Census). Neither county is suppressed in 2022.
The Census has no county planted series; the county SURVEY rows (Frederick planted 38,300,
grain harvested 28,600; Queen Anne's 48,100 / 43,400) are dropped by the rule and differ from
the Census by 10-20 %.

* Polygon corn area is 98.8 % (Frederick) and 94.5 % (Queen Anne's) of Census grain + silage
  harvested acres, so the field set covers the county's corn closely in total.
* The agreement is partly coincidental: CDL class 1 omits double-crop corn (Frederick: 3,363
  acres in classes 225/228/237, mostly triticale/corn, which in dairy areas is typically
  silage after a forage small grain; Queen Anne's: 225 acres) and sweet corn, while polygon
  edges, CDL errors and corn that was not harvested go the other way.
* The check is therefore of the silage share: predicted silage acres / polygon acres against
  19.6 % (Frederick) and 1.3 % (Queen Anne's). A difference of a few points is inside what the
  field set itself can explain; the contrast between the two counties is the stronger test.
* Census acres are harvested acres reported by farms by farm headquarters county, not by field
  location; CSB is located by field.

Fields under 0.3 ha: none in either county (CSB merges polygons under 1 ha into neighbours);
under 1 ha: 9 (Frederick), 3 (Queen Anne's). Median field 4.7 ha (Frederick), 8.0 ha (Queen
Anne's). Source polygons that were invalid and repaired with `make_valid`: 118 and 70.
