"""Turn the organisers' Beltsville (BARC) GeoPackage into field-year rows.

    python -m fieldvision.model.barc --gpkg BARC_Corn.2021-2023.gpkg --out barc_field_years.geojson

The file has one row per field polygon and one column per year
(`Corn_2021`, `Corn_2022`, `Corn_2023`) holding `Silage`, `Grain` or
blank. A blank is "not labelled", never "not silage", so blank years are
kept as rows with label null (they can be predicted, e.g. for the map) and
are never scored. Each row becomes `barc-<OBJECTID>-<year>`; predictions
join back on OBJECTID plus year.

Output follows the field contract (EPSG:4326, field_id, label, year,
country, source, area_ha, block_id, centroid_lon/lat) plus the original
OBJECTID, GROWER, FARM, FIELD, ACRES.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd

from .splits import grid_block_id

LAYER = "barc_silage_grain_corn20212023"
YEAR_COL = re.compile(r"^corn_(\d{4})$", re.IGNORECASE)
ACRE_HA = 0.40468564224
KEEP = ["OBJECTID", "GROWER", "FARM", "FIELD", "ACRES"]


def field_years(gdf, id_col="OBJECTID", keep_blank=True):
    """GeoDataFrame in any CRS -> GeoDataFrame of field-years in EPSG:4326."""
    import geopandas as gpd
    from shapely import make_valid
    year_cols = {c: int(YEAR_COL.match(c).group(1)) for c in gdf.columns if YEAR_COL.match(str(c))}
    if not year_cols:
        raise ValueError(f"no Corn_<year> columns in {list(gdf.columns)}")
    if gdf.crs is None:
        raise ValueError("the field file has no CRS; refusing to guess")
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(lambda g: make_valid(g) if g is not None else None)
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    area_ha = metric.geometry.area.to_numpy() / 10000.0
    cent = gpd.GeoSeries(metric.geometry.centroid, crs=metric.crs).to_crs("EPSG:4326")
    wgs = gdf.to_crs("EPSG:4326")
    blocks = grid_block_id(cent.x.to_numpy(), cent.y.to_numpy(), prefix="US-G")
    rows = []
    odd = set()
    for i in range(len(gdf)):
        rec = gdf.iloc[i]
        for col, year in sorted(year_cols.items(), key=lambda kv: kv[1]):
            raw = rec[col]
            val = None if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw).strip()
            label = {"silage": "silage", "grain": "grain"}.get((val or "").lower())
            if val and label is None:
                odd.add(val)
            if label is None and not keep_blank:
                continue
            props = {
                "field_id": f"barc-{rec[id_col]}-{year}", "label": label, "year": year,
                "country": "US", "source": "BARC starter kit", "crop_code": None,
                "area_ha": float(area_ha[i]), "block_id": blocks[i],
                "centroid_lon": float(cent.x.iloc[i]), "centroid_lat": float(cent.y.iloc[i]),
            }
            for k in KEEP:
                if k in gdf.columns:
                    v = rec[k]
                    props[k] = v.item() if hasattr(v, "item") else v
            if "ACRES" in props and props["ACRES"] is not None:
                props["area_ha_from_acres"] = float(props["ACRES"]) * ACRE_HA
            props["geometry"] = wgs.geometry.iloc[i]
            rows.append(props)
    if odd:
        warnings.warn(f"label values other than Silage/Grain treated as unlabelled: {sorted(odd)}")
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if out["field_id"].duplicated().any():
        raise ValueError("duplicate OBJECTID-year combinations")
    return out


def counts_by_year(fy):
    lab = fy[fy["label"].notna()]
    t = pd.crosstab(lab["year"], lab["label"], margins=True, margins_name="total")
    t["unique_fields"] = lab.groupby("year")["OBJECTID"].nunique() if "OBJECTID" in lab else np.nan
    if "OBJECTID" in lab:
        t.loc["total", "unique_fields"] = lab["OBJECTID"].nunique()
    return t


def main(argv=None):
    import geopandas as gpd
    ap = argparse.ArgumentParser(description="BARC GeoPackage -> field-year GeoJSON.")
    ap.add_argument("--gpkg", required=True)
    ap.add_argument("--layer", default=LAYER)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labelled-only", action="store_true", help="drop blank field-years")
    args = ap.parse_args(argv)
    layers = None
    try:
        import pyogrio
        layers = [str(x[0]) for x in pyogrio.list_layers(args.gpkg)]
    except Exception:
        pass
    layer = args.layer if (layers is None or args.layer in layers) else layers[0]
    if layers is not None and layer != args.layer:
        warnings.warn(f"layer {args.layer!r} not found; using {layer!r} (have {layers})")
    gdf = gpd.read_file(args.gpkg, layer=layer)
    fy = field_years(gdf, keep_blank=not args.labelled_only)
    fy.to_file(args.out, driver="GeoJSON")
    t = counts_by_year(fy)
    print(f"{len(gdf)} polygons -> {len(fy)} field-years, {int(fy['label'].notna().sum())} labelled")
    print(t.to_string())
    with open(str(args.out) + ".counts.json", "w", encoding="utf-8") as fh:
        json.dump({str(k): {str(c): (None if pd.isna(v) else int(v)) for c, v in r.items()}
                   for k, r in t.to_dict("index").items()}, fh, indent=2)
    return fy


if __name__ == "__main__":
    main()
