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
branch's code, the Census CSV uploaded (6.5 MB), and about 15 GB of Colab disk.

**First pair if the answer is yes: Michigan, Gratiot (24.2 %) and Branch (1.8 %).** It gives the
most share contrast per block: 22.4 points for about 115 blocks (0.19 points per block), against
Wisconsin 27.2 points / ~235 blocks (0.12), Texas 37.5 / ~245 (0.15, and the riskiest calendar),
Iowa 15.1 / ~180 (0.08), Minnesota 15.0 / ~240 (0.06), South Dakota 7.9 / ~330. Both counties lie in
UTM zone 16 (EPSG:32616). The Colab bundle for the field files is ready, not started, in
`<work>\colab\fields_26_2022\` (notebook, code at 62d26e2, Census table); its code was checked by
rebuilding the Kent field file from the Maryland subset byte for byte.

## 2025 statewide samples (the organisers' target: NASS 2025 state figures)

Boundaries for 2025: the CSB 2018-2025 rev23 release is built from the 2018-2025 CDLs, so every
polygon carries `CDL2025`, its 2025 majority class. Frame = every polygon of the state with
`CDL2025` corn (1, stratum `corn`) or a small grain followed by corn (225, 226, 228, 237, stratum
`dblcrop_sg_corn`), made with `from_file` (cnty_fips, county_name kept) and completed by
`scripts/state_sample.py`, which adds `county_fips`, `stratum` and draws whole extractor read blocks
by simple random sampling with `numpy.random.default_rng(42)`, split into equal parts. Each drawn
read block's id becomes the fields' `block_id` (10 km id kept in `block10_id`), so the extractor
forms exactly the drawn blocks. Every read block has inclusion probability n/N, so a state total is
N/n times the sum over the drawn blocks. `draw.json` records the draw by stratum and county;
`state_acres.json` gives the frame acres by class and county beside the NASS target.

Maryland (`<data>\labels\us_24_2025\`, area and grid in UTM 18N): frame 22,273 fields,
494,947 acres (corn 21,058 / 477,891; double crop 1,215 / 17,056); 150 of 838 read blocks drawn
(probability 0.179): 3,929 fields, 84,195 acres (17.0 %); runs `us_24_2025_s1`-`s3`, 50 blocks each.
Class-1 acres are 1.062 times the NASS 2025 total (410,000 grain + 40,000 silage, 8.9 % silage).

Pennsylvania and Wisconsin: `<work>\colab\state_frames_2025\` makes both frames and draws on Colab
(area and grid in EPSG:5070, because each state spans two UTM zones) and returns only the part field
files and the two JSON files; extraction bundles follow on the laptop as for Maryland.

## 2023 test counties chosen by the team (Wisconsin, Pennsylvania)

The user asked for the teammates' counties (branch `dataset-analysis`, `src/config.py`) instead of
the Michigan pair above, in season 2023:

| Run | County | 2023 survey: planted / grain / silage (share) | 2022 Census: grain / silage (share) |
|---|---|---|---|
| `us_55073_2023` | Marathon, WI | 124,000 / 61,300 / 61,000 (49.9 %) | 83,973 / 39,717 (32.1 %) |
| `us_42071_2023` | Lancaster, PA | 69,000 / 38,200 / 30,600 (44.5 %) | 95,549 / 35,988 (27.4 %) |
| `us_55101_2023` | Racine, WI (mostly-grain reference) | 33,800 / 32,900 / 800 (2.4 %) | 31,327 / 2,154 (6.4 %) |

Survey values: `corn_grain_silage_acres_2021_2023.csv` from that branch, county SURVEY rows, domain
TOTAL, period YEAR (one record per series). The survey and the Census disagree badly for Lancaster
(grain 38,200 against 95,549); the CSB corn area is printed next to both by the notebook's cell B.

**Field set: our rule, not theirs.** Ours: every CSB 2018-2025 polygon in the county whose 2023 CDL
majority class is corn (`CDL2023 == 1`), no size limit. Theirs (`config.py`): CSB 2016-2023 rev23
polygons of at least 2 ha with at least 80 % of their CDL 2023 pixels corn. Theirs drops small and
mixed fields, so it covers less corn area; ours keeps any field where corn is the majority class,
so it can include fields with up to half their area in another class. The two CSB releases also
draw slightly different boundaries.

**Cap.** Marathon and Lancaster have about 120-130 thousand corn acres. Each run is capped at 70
read blocks (about Frederick's cost): whole 10 km blocks are taken in the order of
`numpy.random.default_rng(42).permutation` over the sorted block ids, and a block is kept if its
read blocks still fit under 70. `sample.json` in the county folder records the seed, label and read
blocks drawn of how many, fields and acres drawn of the total and the acre fraction; scale county
acres by that fraction, and never read a capped run as the whole county. Racine is expected to fit
whole (`is_sample` false).

Colab bundles: `<work>\colab\us_<fips>_2023\`, built by `scripts/build_colab_county.py`
(field file, survey check and draw on Colab, then the standard extraction with code 9cd0447b and
`--year 2023`; 2023 imagery is Earth Search Collection 1). Untouched test counties: no model is
run on them until the modelling session has logged the test.

## Corn outside the scored field set (Frederick, Queen Anne's)

From the Maryland CSB subset, county polygons by their 2022 CDL class (acres in UTM 18N):

| | Frederick | Queen Anne's |
|---|---:|---:|
| scored: class 1 corn | 39,030 | 51,533 |
| double crop triticale/corn (228) | 1,846 | 211 |
| double crop winter wheat/corn (225) | 895 | 0 |
| double crop barley/corn (237) | 623 | 12 |
| double crop oats/corn (226) | 0 | 3 |
| sweet corn (12) | 9 | 398 |
| dropped by a size filter | 0 (none applied; smallest CSB polygon 2.06 acres) | 0 (2.19 acres) |
| Census grain + silage | 31,752 + 7,748 = 39,500 | 53,815 + 715 = 54,530 |

Frederick has 3,364 acres of small-grain/corn double crops outside the scored set, 43 % of its
Census silage area. If all of it were silage, the silage left inside the scored 39,030 acres would
be 4,384 acres, an 11.2 % share, which is much closer to the corrected estimates (7-9 %) than
19.6 %. These polygons are fields whose 2022 majority class was the double crop, so this is an upper
bound on the silage they hold and says nothing about corn in polygons whose majority is another
crop. What would answer that: the 2022 CDL corn pixel acres for each county (CropScape county
statistics, a small table) set beside the class-1 and double-crop polygon acres. Queen Anne's has
225 acres of such double crops, too few to matter.

The CropScape table answers it (2022 CDL pixel acres per county, `<data>\us\cropscape\`):

| County | CSB class-1 polygons | CDL corn pixels | CSB small-grain/corn polygons | CDL small-grain/corn pixels | Census grain + silage |
|---|---:|---:|---:|---:|---:|
| Frederick | 39,030 | 36,426 | 3,364 | 2,807 | 39,500 |
| Queen Anne's | 51,533 | 50,705 | 226 | 218 | 54,530 |
| Kent | 44,034 | 43,045 | 0 | 40 | 40,906 |
| Carroll | 32,854 | 29,818 | 885 | 723 | 27,525 |

Corn pixels anywhere in the county are fewer than the class-1 polygon area in all four counties
(by 2-9 %), so the majority rule adds more non-corn area to the scored polygons than it leaves
corn outside them: there is no large stock of corn hidden in polygons of another majority crop.
In Frederick, corn pixels plus small-grain/corn pixels (39,233 acres) match the Census total
(39,500), which supports reading the double-crop area as corn that the Census counts. Per county,
`unscored_corn.json` (toolbox folder; Carroll's in its labels folder until it is finalised) holds
these numbers; `us_<fips>_2022_dblcrop\fields.geojson` in the labels folder holds the double-crop
polygons as field files (Frederick 181, Queen Anne's 10, Carroll 51; Kent none), ready to extract.

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

## State at hand-over (14 Nov, about 19:50 PT)

Finished, verified and in `<data>\timeseries\READY.txt`, with toolbox folders under
`<data>\counties\`: Frederick 24021, Queen Anne's 24035, Kent 24029, Carroll 24013 (2022; each with
fields.geojson, timeseries.parquet, fields_coverage.csv, unscored_corn.json, landscape.parquet), Branch 26023 and
Gratiot 26057 (2022; no landscape.parquet, their state CSB is not on this laptop), the Maryland 2025 sample
(`24_2025_sample`: fields, corn and double-crop series, draw.json, state_acres.json, sample_info.json,
landscape.parquet). Beltsville stays as 24033 (made by the modelling session).

Not finished:
- Racine `us_55101_2023` and Marathon `us_55073_2023`: results are in `<work>\colab_results\<run>\` (field zip +
  block parts, checksums OK) but not unpacked or merged. Next: unzip the fields zip as is to
  `<data>\labels\<run>\`, create the run folder with `build_colab_run.start_fresh_run` and template
  `<work>\colab\_templates\9cd0447b_2023` (README step 2 in `<work>\colab\<run>\`), then
  `scripts/merge_colab_blocks.py` and the finalise command. Marathon is a capped draw (sample.json in its zip).
  Lancaster `us_42071_2023` was still running on Colab. All three are untouched test counties.
- Pennsylvania and Wisconsin 2025 frames and draws: unzipped to `<data>\labels\us_42_2025\` and
  `us_55_2025\` (s1-s3 field files, draw.json, state_acres.json); no extraction bundles built. Next, per part:
  `python scripts/build_colab_run.py --run-dir <data>/timeseries/us_<fips>_2025_s<k> --fields <data>/labels/us_<fips>_2025/s<k>/fields.geojson --code <frozen 9cd0447b> --template-run <work>/colab/_templates/9cd0447b_2025 --check-external <as for Maryland> --dest <work>/colab/us_<fips>_2025_s<k>` (script on main).
- Landscape files for Michigan, Pennsylvania and Wisconsin: need their state CSB on Colab
  (`scripts/forage_landscape.py`).
- Six-state proposal (above): not started.
- County CDL statistics (CropScape) 2019-2025: `<data>\us\cropscape\county\`, about 2,885 counties per year,
  complete; county table and land-use prior results in `<data>\us\landuse\` (docs/landuse-prior.md).
