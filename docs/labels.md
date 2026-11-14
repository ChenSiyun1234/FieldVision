# Maize declaration labels

The files describe the crop farmers declared for a season, not an observed harvest.
A field declared as silage may eventually be harvested for grain, or vice versa.
Use the labels as supervised proxies and inspect harvest curves separately.

## Shared output and sampling

Run from the repository root in PowerShell with the supplied environment:

```powershell
$env:PYTHONPATH='C:\t\fv\wt-labels\src'
& C:\t\fv\venv\Scripts\python.exe -m pytest -q
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.fetch --country AT --year 2024 --out C:\t\fv\data\labels\at_2024 --target-per-class 1200
```

Each output directory contains `fields.geojson` (FeatureCollection, longitude/latitude
EPSG:4326), `fields.parquet` (GeoParquet, same CRS), `summary.json`, `quicklook.png`,
`country_boundary.geojson`, and a sanitized `candidates.geojson` for auditing.
All data stays outside git. Properties are explicitly whitelisted; no applicant or
farm identifiers are retained. `region` is an extra sampling-region property.

Area and centroids are computed in EPSG:3035. A parcel belongs to the 10 km grid cell
containing its metric centroid. `AT-E4680N2790` denotes a cell with lower-left corner
(4,680,000 m, 2,790,000 m). Invalid polygons are repaired with `make_valid`, retaining
polygonal parts only; unrecoverable geometry and parcels under 1 ha are dropped.
Whole polygons must be covered by the downloaded Eurostat GISCO 2024 country outline
at 1:1 million resolution. This conservative coarse-boundary check can discard valid
border parcels. It is a geographic sanity check, not cadastral boundary validation.

Complete paginated bbox results are collected region by region. Blocks are ranked by
the qualifying rarer-class count within each region, then selected by rotating
through regions to secure geographic coverage. The rarer class is silage in Austria
and grain in the Netherlands. Each block supplies equal numbers of the two labels,
up to 60 per class, constrained by whichever class is locally scarcer. Selection
uses seed 42 after sorting parcel identifiers. The final block may be truncated to
hit the requested total. Both classes use exactly the same blocks. Boundary blocks
are sampled only within the query window; this is a targeted sample, not a national
probability sample. Stop after the target and at least three regions are reached.

Split train/test by `block_id`, never by random field. Nearby blocks may still be
correlated; region holdouts are a stronger test. For cross-year evaluation also
hold out geography because registry IDs alone do not identify recurring land.
The quicklook contains actual field polygons in region panels and centroid dots
on the country overview for legibility. No satellite imagery is used in this map.

The exporter re-reads GeoJSON and checks unique IDs, labels, country/year, within-block
balance, valid geometry, recalculated areas >=1 ha, whole-polygon country containment,
and three regions. GeoParquet is re-read to check IDs and CRS. `summary.json` records
counts, area distributions, filtering/repair counts, parameters, timestamps and URLs.

## Austria: INVEKOS Schläge 2024-2

Provider: Agrarmarkt Austria (AMA). The
[collection](https://gis.lfrz.gv.at/api/geodata/i009501/ogc/features/v1/collections/i009501:invekos_schlaege_2024_2_polygon?f=json)
and [official metadata](https://www.data.gv.at/datasets/b037d48d-06f3-4af2-ac48-ae07e7ca1e00?locale=de)
describe farmer-declared management units for the vegetation period. The `-2` release
is the winter revision, with temporal reference **2024-12-31**, not a harvest date.
The metadata publication date is 2025-01-16; its revision timestamp is 2026-04-15.

Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), linked directly
from the collection and opened during verification. Attribute **Agrarmarkt Austria,
INVEKOS Schläge Österreich 2024-2**. Our derivative selects maize, filters/repairs
geometry, balances samples and computes area/grid properties.

| Code | Registry name | National parcels | Use |
|---|---|---:|---|
| 109 | SILOMAIS | 51,624 | Include as silage |
| 105 | KÖRNERMAIS | 107,687 | Include as grain |
| 106 | MAIS CORN-COB-MIX (CCM) | 8,358 | Exclude: cob mix |
| 107 | ZUCKERMAIS | 223 | Exclude: sweet corn |
| 135 | GRÜNMAIS | 209 | Exclude: ambiguous green maize use |
| 173 | SAATMAISVERMEHRUNG | 2,757 | Exclude: seed production |

Use the command above. Sampling windows cover Upper Austria, Lower Austria and Styria.
IDs use the registry's geometry/parcel `geo_id`, prefixed `at2024-`; farm-level
`fs_kennung` and all other administrative attributes are discarded.
Filtering requires `filter-lang=cql2-text` and `filter=snar_code IN ('109','105')`.
Plain property parameters can be silently ignored. Follow the server's `next` links,
which may choose a smaller page size than requested. The API currently lists annual
Schläge collections from 2015 through 2026-1; this adapter deliberately fixes the
verified 2024-2 collection. Other years need code/schema verification before enabling.

## Netherlands: BRP 2025

Provider: RVO, distributed through PDOK. The
[OGC API](https://api.pdok.nl/rvo/gewaspercelen/ogc/v1) describes annual farmer crop
declarations with reference date **15 May**. Live WFS records say `jaar=2025` and
`status=Definitief`. The OGC API exposes only the latest `brpgewas` collection and
rejects crop-code property filtering (HTTP 400); its queryables contain geometry only.
Use the same provider's [WFS](https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0?service=WFS&request=GetCapabilities)
with FES XML crop/year/bbox filters. `EPSG:4326` requests yield CRS84 longitude/latitude
GeoJSON; the FES envelope uses explicit EPSG URN latitude/longitude axis order.

The API links [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/deed.nl).
The [ATOM feed](https://service.pdok.nl/rvo/gewaspercelen/atom/basisregistratie_gewaspercelen_brp.xml)
instead states [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/deed.nl).
Both linked pages were opened; retain this distinction in provenance. Credit RVO/PDOK.

Codes were checked against the official RVO
[2025 crop list, version 22 April](https://www.rvo.nl/sites/default/files/2025-04/Tabel-2e-gewascodelijst-stikstofgebruiksnorm-2025-april.pdf).
National live WFS counts on 2026-11-14, before geometry/area filtering:

| Code | Dutch name | Parcels | Use |
|---|---|---:|---|
| 259 | Mais, snij- | 95,192 | Include: whole-plant silage |
| 316 | Mais, korrel- | 3,992 | Include: grain |
| 317 | Mais, corncob mix | 1,525 | Exclude: cob mix |
| 814 | Mais, suiker- | 395 | Exclude: sweet corn |
| 1935 | Maiskolvensilage | 812 | Exclude: cob silage, not whole plant |
| 2032 | Mais, energie | 24 | Exclude: energy use, not an unambiguous grain label |

The official list also mentions maize in the nutrient-norm description for code 3519,
but that crop is `Soedangras/Sorghum`, not maize, and is excluded.

```powershell
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.fetch --country NL --year 2025 --out C:\t\fv\data\labels\nl_2025 --target-per-class 800
```

The WFS serves 2025 only in the verified year checks: 2,331,084 total records for 2025,
zero for 2024 or 2026. The ATOM feed lists GeoPackages for 2020–2025 and concept 2026,
plus zipped archives for 2009–2019. The advertised full-country 2024 GeoPackage is
**3,106,852,864 bytes (3.11 GB)**; 2025 is 2,932,801,536 bytes. No bulk file was fetched.
2024 is not enabled because the verified distribution exceeds the 2 GB disk limit.

To refresh the national counts, collection inventory and advertised download sizes:

```powershell
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.labels.audit --out C:\t\fv\data\labels\source_audit.json
```
