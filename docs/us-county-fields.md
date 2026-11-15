# US county corn fields (Maryland 2022 backtest)

Unlabelled corn fields for one county and season, built from USDA NASS Crop Sequence
Boundaries (CSB), so the frozen models can be run on a whole county and the predicted acres
compared with the 2022 Census of Agriculture. No labels: the county check is the only test.

| Folder (under `<data>\labels\`) | County | Fields | Corn acres (polygons) | Census grain / silage harvested | Silage share |
|---|---|---:|---:|---:|---:|
| `us_24021_2022` | Frederick (24021) | 2,233 | 39,030 | 31,752 / 7,748 | 19.6 % |
| `us_24035_2022` | Queen Anne's (24035) | 1,449 | 51,533 | 53,815 / 715 | 1.3 % |
| `us_24029_2022` | Kent (24029), test county | 1,363 | 44,034 | 38,642 / 2,264 | 5.5 % |
| `us_24013_2022` | Carroll (24013), second test county | 2,198 | 32,854 | 25,065 / 2,460 | 8.9 % |

Polygon acres over Census grain + silage: Frederick 0.988, Queen Anne's 0.945, Kent 1.076,
Carroll 1.194. Carroll's polygons exceed the Census by a fifth; it also has 884 acres of
double-crop corn not in the field set.

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

## Frederick 2022: county totals against the Census

`fieldvision.toolbox.Toolbox` from `main` at 12e5a4e, policy `standard` (`composite_v1`), on the
full county (2,233 fields, 39,030 acres; 207 fields and 3,059 acres abstained, so 35,971 acres
are counted). Census 2022: grain 31,752, silage 7,748 acres, share 19.6 %. Output kept in
`<data>\us\toolbox_results\`.

| Estimate | Silage share | Silage acres |
|---|---:|---:|
| Census 2022 | 19.6 % | 7,748 |
| counted share (headline rule logged 16:12) | 20.9 % | 7,525 |
| raw sum of probabilities | 36.5 % | 13,130 [12,155-14,066] |
| adjusted with European rates (TPR 0.70, FPR 0.16) | 8.6 % | 3,077 [908-5,139] |
| adjusted, other European designs | 0.0-9.1 % | |
| reference-calibrated, Queen Anne's as reference (local FPR 0.22) | 0.0 % [0-6.9 %] | 0 |
| prior-shift (EM) | 0.0 % | |

The counted share lands on the Census figure, but the same model calls 22.4 % of Queen Anne's
area silage where the Census says 1.3 %: the classifier calls about a fifth of the corn area
silage in both counties, so the Frederick agreement is not evidence that it separates silage
from grain. The corrected estimators, which remove that common false-positive level, put
Frederick at 0-9 %, below the Census. Frederick was seen in part before the rule was fixed.

## Choosing the untouched test county

Frederick's partial snapshot has been looked at and Queen Anne's serves as the grain reference,
so the test of the county procedure is a third Maryland county, chosen by this rule (written
before any field file for it was made), applied to the Census 2022 county rows after the
`quickstats.select` rule:

1. Maryland, year 2022, grain and silage harvested acres both published (neither withheld).
2. Grain + silage harvested at least 10,000 acres.
3. Silage share at least 3.0 % (more than twice Queen Anne's 1.3 %) and not within 2 points
   of Frederick's 19.6 % (that is, outside 17.6-21.6 %), so the test differs from both.
4. Not Frederick (24021) or Queen Anne's (24035).
5. Among those, the most grain + silage acres; the runner-up is the second test county.

`python scripts/pick_test_county.py --quickstats <csv>` applies the rule and prints the ranking.
No model is run on a test county until the county procedure is fixed and logged.

## Proposal: county checks in Iowa and five more states (not started)

Without field labels outside Beltsville, the only test there is the Maryland one: run the model on
a county's corn fields and compare the county total with the 2022 Census. Everything below comes
from the Census table on this machine; `python scripts/scope_states.py --quickstats <csv>`
reproduces the selections. Nothing has been downloaded or extracted for it.

**States.** The file has no 2022 state rows, so state totals are sums of the county Census rows
(withheld counties missing, so these are lower bounds). Rule: Iowa, plus the five states with the
most silage acres among states with at least 1,000,000 grain acres, so both classes matter.

| State | Grain acres | Silage acres | Silage share | Counties withheld (grain / silage) |
|---|---:|---:|---:|---|
| Iowa | 12,637,070 | 250,167 | 1.9 % | 0 / 3 |
| Wisconsin | 3,065,184 | 786,353 | 20.4 % | 4 / 2 |
| Texas | 1,626,900 | 372,510 | 18.6 % | 67 / 41 |
| South Dakota | 4,802,732 | 313,785 | 6.1 % | 2 / 2 |
| Minnesota | 7,732,455 | 312,984 | 3.9 % | 0 / 2 |
| Michigan | 2,099,511 | 307,637 | 12.8 % | 8 / 8 |

Next in line: Nebraska (242,277 silage), Kansas (241,676), North Dakota, Colorado, Ohio.

**Counties.** Per state: grain and silage both published, grain + silage at least 10,000 acres;
the eligible counties are split into terciles of silage share; "high" is the largest-corn county
of the top tercile, "low" the largest of the bottom tercile.

| State | | FIPS | County | Grain | Silage | Share |
|---|---|---|---|---:|---:|---:|
| Iowa | high | 19167 | Sioux | 221,894 | 40,668 | 15.5 % |
| Iowa | low | 19109 | Kossuth | 304,340 | 1,311 | 0.4 % |
| Wisconsin | high | 55073 | Marathon | 83,973 | 39,717 | 32.1 % |
| Wisconsin | low | 55105 | Rock | 119,270 | 6,159 | 4.9 % |
| Texas | high | 48205 | Hartley | 68,758 | 55,552 | 44.7 % |
| Texas | low | 48421 | Sherman | 92,815 | 7,253 | 7.2 % |
| South Dakota | high | 46059 | Hand | 184,807 | 18,937 | 9.3 % |
| South Dakota | low | 46013 | Brown | 273,340 | 3,846 | 1.4 % |
| Minnesota | high | 27145 | Stearns | 183,737 | 34,145 | 15.7 % |
| Minnesota | low | 27129 | Renville | 271,325 | 1,964 | 0.7 % |
| Michigan | high | 26057 | Gratiot | 87,321 | 27,878 | 24.2 % |
| Michigan | low | 26023 | Branch | 89,128 | 1,653 | 1.8 % |

**Cost.** These counties have 3 to 7 times Frederick's corn. Extraction cost is per read block,
and in Maryland a block covered about 18-25 km² of county land (Frederick 1,711 km² -> 70 blocks,
Queen Anne's 963 km² -> 52). With approximate land areas (from memory, about +/-10 %; the TIGER
file was deleted, a Census Gazetteer table would make this exact) and Midwest fields assumed at
about 40 acres (Maryland CSB fields average 15-36 acres; to be measured):

| County | Land km² (approx.) | Read blocks (~area/25) | Corn fields (~acres/40) | Colab time at 15-70 s/block |
|---|---:|---:|---:|---|
| Sioux IA | 1,990 | ~80 | ~6,600 | 20-95 min |
| Kossuth IA | 2,520 | ~100 | ~7,600 | 25-120 min |
| Marathon WI | 4,000 | ~160 (fewer: corn is sparse) | ~3,100 | 40-185 min |
| Rock WI | 1,860 | ~75 | ~3,100 | 20-90 min |
| Hartley TX | 3,790 | ~150 (fewer: pivots are sparse) | ~3,100 | 40-175 min |
| Sherman TX | 2,390 | ~95 | ~2,500 | 25-110 min |
| Hand SD | 3,720 | ~150 | ~5,100 | 40-175 min |
| Brown SD | 4,440 | ~180 | ~6,900 | 45-210 min |
| Stearns MN | 3,480 | ~140 | ~5,400 | 35-165 min |
| Renville MN | 2,550 | ~100 | ~6,800 | 25-120 min |
| Gratiot MI | 1,470 | ~60 | ~2,900 | 15-70 min |
| Branch MI | 1,310 | ~55 | ~2,300 | 15-65 min |

About 1,350 blocks in all: roughly 6 to 26 Colab hours on one 8-CPU session, depending on how
many scenes each block sees (one MGRS tile and orbit: ~30-60 scenes, ~15 s per block; tile and
UTM-zone overlaps: ~120 scenes, ~70 s). Sampling fields does not help; sampling counties does.
Imagery: Earth Search Collection 1 has 2021 and 2023-2025; for 2022 it has almost nothing in the
growing season, so 2022 comes from Planetary Computer (original baseline 04.00 plus ESA's
reprocessed 05.10), exactly as in Maryland. County silage numbers exist only for the Census years
2012, 2017 and 2022; Sentinel-2 L2A outside Europe is incomplete before late 2018, so 2022 is the
only county year we can check.

**Field boundaries without the slow link.** Do it all on Colab: range-download the national CSB
archive there (3.76 GB; the NASS server returns an empty body to a plain GET, so it needs the
64 MB range downloader used here, to be added to `scripts/`), run `scripts/csb_subset.py --state`
for each state, `from_file` with the same filters per county, `county_corn_check.py` against TIGER
(downloaded there too), then extract in the same session. Only the county field files (a few MB
each), the check JSONs and the finished blocks come back. Needed: a Colab session with this
branch's code, the Census CSV uploaded (1 MB), and about 15 GB of Colab disk.

**Caution.** What worked in Maryland may not carry over. Harvest calendars differ: Texas Panhandle
silage is cut in July-August and southern Texas grain is harvested from July, so the April-November
window and the model's peak window (DOY 160-280, `--peak-window`) need checking there. Irrigated
corn (Texas, and Nebraska or Kansas next in line) stays green longer and dries down differently from
the rain-fed Austrian training fields. Double-crop corn is not in CDL class 1 (small in the Corn
Belt, larger in Texas and the Mid-Atlantic), sweet corn is class 12 and excluded, but seed corn is
in class 1 and is harvested early as ears, which may look like silage; how the Census counts seed
corn needs checking. The Census tabulates by the farm's principal county, not by field location,
and in Maryland the CSB corn area was 0.95-1.19 times the Census total. Above all, the Maryland
counties showed the classifier calling about a fifth of the corn area silage in both a 1 % and a
20 % county; the low-share counties above (Kossuth 0.4 %, Renville 0.7 %) are where that would
show first.

## Time series and the toolbox copy

`<data>\timeseries\us_<fips>_2022`, extractor defaults (`--year 2022`, April 1 to
November 30, Collection 1 with the Planetary Computer fallback). Extraction cost is per read
block, not per field: Frederick's 2,233 fields make 70 blocks and a random 800 would still
make 63, so the whole county is extracted. Western Frederick blocks see about 120 scenes
(two MGRS tiles and the UTM 17/18 overlap).

`scripts/county_snapshot.py` copies fields and time series to `<root>/<fips>/<year>/` (the
layout the toolbox reads). During a run it takes only blocks listed as complete in
`manifest.json` and writes `PARTIAL.json` with the counts; `--final` copies the finished run.

Fields under 0.3 ha: none in either county (CSB merges polygons under 1 ha into neighbours);
under 1 ha: 9 (Frederick), 3 (Queen Anne's). Median field 4.7 ha (Frederick), 8.0 ha (Queen
Anne's). Source polygons that were invalid and repaired with `make_valid`: 118 and 70.
