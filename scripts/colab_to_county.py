"""Turn one Colab county extraction into the toolbox's county folder.

    python scripts/colab_to_county.py <extract_dir> <year> [--out data/counties]

<extract_dir> holds what the Colab notebooks write per county: fields.gpkg (field_id, fips, area_ha,
corn_frac, polygon) and s2_timeseries.csv (one row per field and clear date, bands b04 b05 b08 b8a b11 b12,
clear_frac, n_pixels). The toolbox reads data/counties/<fips>/<year>/fields.geojson and timeseries.parquet
with bands B02-B12 (see src/fieldvision/features/loading.py).

The Colab extraction kept six bands. B02, B03, B06 and B07 are filled from their neighbours (B04, B04, B05,
B8A) only so the loader's all-bands check passes. They feed EVI, PSRI and the brightness features, which
the final model (eu2_logistic_v1: harvest timing, NDVI and NDMI before the drop, NDVI slope, NDTI after)
does not use, and nothing in harvest detection, abstention or the evidence tiers reads them. Models that
do use them (boosted trees, composite) should not be run on these counties. PROVENANCE.md in the output
folder records this.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

FILL = {"B02": "B04", "B03": "B04", "B06": "B05", "B07": "B8A"}   # missing band -> band it is copied from


def convert(src, year, out_root):
    src, year = Path(src), int(year)
    fields = gpd.read_file(src / "fields.gpkg").to_crs(4326)
    fips = str(fields["fips"].iloc[0]).zfill(5)
    fields["field_id"] = "csb-" + fields["field_id"].astype(str)
    cent = fields.to_crs(5070).geometry.centroid
    feats = []
    for (_, r), c, c4326 in zip(fields.iterrows(), cent, fields.geometry.centroid):
        feats.append({"type": "Feature", "geometry": json.loads(gpd.GeoSeries([r.geometry]).to_json())["features"][0]["geometry"],
                      "properties": {"id": r["field_id"], "field_id": r["field_id"], "label": None, "year": year,
                                     "country": "US", "source": "nass-csb (Colab extraction)", "crop_code": 1, "crop_name": "Corn",
                                     "area_ha": float(r["area_ha"]), "corn_frac": float(r["corn_frac"]),
                                     "block_id": f"US-E{int(c.x // 10000)}N{int(c.y // 10000)}",
                                     "centroid_lon": float(c4326.x), "centroid_lat": float(c4326.y),
                                     "acres_reported": float(r["area_ha"]) * 2.47105, "county_fips": fips}})

    ts = pd.read_csv(src / "s2_timeseries.csv", dtype={"field_id": str})
    ts = ts[ts["year"] == year]
    out = pd.DataFrame({"field_id": "csb-" + ts["field_id"], "date": ts["date"].astype(str),
                        "scene_id": None, "collection": "colab:sentinel-2-l2a (6 bands)",
                        "n_pixels": ts["n_pixels"].astype("int64"),
                        "n_clear": np.round(ts["n_pixels"] * ts["clear_frac"]).astype("int64"),
                        "clear_fraction": ts["clear_frac"].astype(float)})
    for b in ("B04", "B05", "B08", "B8A", "B11", "B12"):
        out[b] = ts[b.lower()].astype(float).to_numpy()
    for missing, source in FILL.items():
        out[missing] = out[source]
    out = out[["field_id", "date", "scene_id", "collection", "n_pixels", "n_clear", "clear_fraction",
               "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]]

    folder = Path(out_root) / fips / str(year)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "fields.geojson").write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, "features": feats}))
    out.to_parquet(folder / "timeseries.parquet", index=False)
    (folder / "PROVENANCE.md").write_text(
        f"# {fips} {year}: converted from a Colab extraction\n\n"
        f"Source: `{src.name}` (fields.gpkg, s2_timeseries.csv), converted by scripts/colab_to_county.py.\n\n"
        f"- {len(feats):,} corn fields; {len(out):,} clear Sentinel-2 observations, {out['date'].min()} to {out['date'].max()}.\n"
        "- The extraction kept bands B04 B05 B08 B8A B11 B12. B02 and B03 are copies of B04, B06 of B05 and B07 of B8A, so the\n"
        "  loader accepts the table. They only feed EVI, PSRI and brightness features. The final model (eu2_logistic_v1)\n"
        "  uses none of them; harvest detection, abstention and the evidence tiers use NDVI. Do not run models that use\n"
        "  those features (boosted trees, composite) on this county.\n"
        "- Observations start in May, so the spring bare-soil baseline is thin.\n")
    return folder, len(feats), len(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("extract_dir")
    ap.add_argument("year", type=int)
    ap.add_argument("--out", default="data/counties")
    a = ap.parse_args()
    folder, n_fields, n_obs = convert(a.extract_dir, a.year, a.out)
    print(f"{folder}: {n_fields:,} fields, {n_obs:,} observations")
