"""Write Pennsylvania and Wisconsin corn polygons from the national CSB geodatabase.

The national layer has about 16 million polygons. This reads one state at a time
with an attribute filter and does not load the full file.

    python -m src.extract_csb
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio

from .collect_fields import tiger_counties
from .config import (
    CDL_PATHS,
    CORN_CDL_CLASS,
    CSB_EXTRACTS,
    CSB_LAYER,
    CSB_PATH,
    TEST_COUNTIES,
    TRANSFER_YEAR,
    resolve_data_path,
)


def _states() -> list[tuple[str, str, list[str]]]:
    """Unique states, with the two-digit FIPS and the county FIPS codes to keep."""
    order: list[str] = []
    grouped: dict[str, tuple[str, list[str]]] = {}
    for state, fips in TEST_COUNTIES:
        county_fips = str(fips).zfill(5)
        state_fips = county_fips[:2]
        if state not in grouped:
            order.append(state)
            grouped[state] = (state_fips, [])
        grouped[state][1].append(county_fips)
    return [(state, *grouped[state]) for state in order]


def _print_csb(gdb) -> tuple[str, list[str]]:
    layers = pyogrio.list_layers(gdb)
    print(f"CSB {gdb}")
    layer_name = CSB_LAYER
    names = [name for name, _geom in layers]
    if layer_name not in names:
        layer_name = names[0]
    fields: list[str] = []
    for name, geom in layers:
        info = pyogrio.read_info(gdb, layer=name)
        print(f"  layer {name!r} geometry={geom} features={info.get('features')}")
        print(f"  CRS: {info.get('crs')}")
        print("  columns:")
        for column, dtype in zip(info["fields"], info["dtypes"]):
            print(f"    {column}: {dtype}")
        if name == layer_name:
            fields = list(info["fields"])
    return layer_name, fields


def _print_cdl() -> None:
    cdl_path = resolve_data_path(CDL_PATHS.get(TRANSFER_YEAR))
    if cdl_path is None or not cdl_path.exists():
        print(f"CDL file is not on disk yet: {cdl_path}")
        return
    with rasterio.open(cdl_path) as source:
        print(
            f"CDL {cdl_path.name}: CRS {source.crs}, "
            f"resolution {source.res}, bounds {tuple(source.bounds)}"
        )


def _read_state(gdb, layer: str, fields: list[str], state: str, state_fips: str, county_fips: list[str]):
    crop_col = f"CDL{TRANSFER_YEAR}"
    if "STATEFIPS" in fields:
        where = f"STATEFIPS = '{state_fips}'"
        if crop_col in fields:
            where += f" AND {crop_col} = {CORN_CDL_CLASS}"
        print(f"{state}: reading {layer} where {where}")
        return pyogrio.read_dataframe(gdb, layer=layer, where=where)
    print(f"{state}: no STATEFIPS column, reading each county by bounding box")
    info = pyogrio.read_info(gdb, layer=layer)
    counties = tiger_counties()
    parts = []
    for fips in county_fips:
        county = counties.loc[counties["GEOID"] == fips]
        if county.empty:
            raise SystemExit(f"FIPS {fips} is not in the TIGER county file.")
        bounds = tuple(county.to_crs(info["crs"]).total_bounds)
        print(f"{state} {fips}: bbox {bounds}")
        part = pyogrio.read_dataframe(gdb, layer=layer, bbox=bounds)
        if part.empty:
            continue
        part = gpd.clip(part, county.to_crs(part.crs))
        parts.append(part)
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=info["crs"])
    frame = pd.concat(parts, ignore_index=True)
    if crop_col in frame.columns:
        frame = frame.loc[frame[crop_col] == CORN_CDL_CLASS].copy()
    return frame


def main() -> None:
    gdb = resolve_data_path(CSB_PATH)
    if gdb is None or not gdb.exists():
        raise SystemExit(f"Missing CSB geodatabase: {gdb}")
    _print_cdl()
    layer, fields = _print_csb(gdb)
    for state, state_fips, county_fips in _states():
        dest = resolve_data_path(CSB_EXTRACTS.get(state))
        if dest is None:
            raise SystemExit(f"No CSB_EXTRACTS path for {state}.")
        if dest.exists():
            print(f"{dest.name} already exists, skipping")
            continue
        frame = _read_state(gdb, layer, fields, state, state_fips, county_fips)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_file(dest, driver="GPKG")
        print(f"Wrote {len(frame):,} polygons to {dest}")


if __name__ == "__main__":
    main()
