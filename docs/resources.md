# Resources

Collected from the team's `#resources` channel. Add to it as you find things.

## Public data

| Source | What it gives us |
|---|---|
| [USDA Cropland Data Layer (CDL)](https://www.nass.usda.gov/research-and-science/cropland/sars1a) | Yearly crop-type map of the US. Tells us where corn is, not how it was harvested |
| [Sentinel-2](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2) | 10–20 m optical imagery, a revisit every few days |
| [Landsat Collection 2 Level-2](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) | 30 m optical imagery, long archive |
| [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/catalog) | STAC catalog and cloud access to Sentinel, Landsat and more |
| [Earth Search (Element 84)](https://element84.com/earth-search) | Open STAC API for Sentinel-2 and Landsat on AWS, no account needed |
| [gridMET](https://www.climatologylab.org/gridmet.html) | Daily weather grids, about 4 km |
| [Daymet](https://daymet.ornl.gov/) | Daily weather grids, 1 km |
| [USDA QuickStats](https://quickstats.nass.usda.gov/) | County-level statistics, including corn for silage and corn for grain |
| [Census TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | County and state boundaries |
| [USGS National Map](https://www.usgs.gov/the-national-map-data-delivery) | Elevation and base layers |

## Papers

- [Silage Maize Identification Using a Temporal Difference-Based Model with Sentinel-2 Data](https://www.mdpi.com/2073-4395/15/6/1438) — time-series classification built around the harvest signal.
- [Self-supervised pre-training for large-scale crop mapping using Sentinel-2 time series](https://www.sciencedirect.com/science/article/pii/S0924271623003386) — pre-training without labels. The downstream classifier still needs labels.

## Related projects

- [Terra](https://terra.gedik.tech/) — county-level corn yield from satellite, climate and soil data. Different question (yield, not harvest use), useful as an example of an end-to-end demo.

## Open questions

- Which region and year do we work on?
- Where do field-level silage/grain labels come from? CDL and QuickStats do not give them directly.
- How do we split train and evaluation so neighbouring fields do not leak into each other?
