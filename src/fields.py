"""Load BARC labels or collected county fields and assign each field to one tile."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .common import (
    assign_tiles,
    buffer_inward,
    find_barc_gpkg,
    padded_lonlat_bounds,
    region_dir,
    region_epsg,
)
from .config import BARC_LABELS, BARC_LAYER, N_FIELDS, RANDOM_SEED, TEST_COUNTIES, TRANSFER_YEAR, YEARS


def region_id(state: str, fips: str) -> str:
    return f"{state.lower()}_{str(fips).zfill(5)}"


def counties_for(region: str | None) -> list[tuple[str, str]]:
    """Resolve --region to (state, fips) pairs. barc is not a county."""
    if region in {None, "all"}:
        if not TEST_COUNTIES:
            raise SystemExit("TEST_COUNTIES is empty. Add counties in src/config.py, then rerun.")
        return [(state.upper(), str(fips).zfill(5)) for state, fips in TEST_COUNTIES]
    if region == "barc":
        return []
    for state, fips in TEST_COUNTIES:
        if region_id(state, fips) == region:
            return [(state.upper(), str(fips).zfill(5))]
    if "_" in str(region):
        state, fips = str(region).split("_", 1)
        return [(state.upper(), fips.zfill(5))]
    raise SystemExit(f"Unknown region {region}. Use barc, all, or a folder name like pa_42027.")


def load_barc_fields(years: list[int] | None = None) -> gpd.GeoDataFrame:
    """Labeled Grain/Silage fields for the requested years. Blank labels are dropped."""
    years = list(YEARS if years is None else years)
    source = gpd.read_file(find_barc_gpkg(), layer=BARC_LAYER)
    pieces = []
    print("BARC labeled fields:")
    for year in years:
        column = f"Corn_{year}"
        if column not in source.columns:
            raise KeyError(f"{column} is not in the BARC GeoPackage.")
        labeled = source.loc[source[column].isin(BARC_LABELS), ["OBJECTID", column, "geometry"]].copy()
        counts = labeled[column].value_counts()
        for label in BARC_LABELS:
            print(f"  {year} {label}: {int(counts.get(label, 0))}")
        labeled = labeled.rename(columns={column: "label", "OBJECTID": "field_id"})
        labeled["field_id"] = labeled["field_id"].astype(str)
        labeled["year"] = int(year)
        pieces.append(labeled)
    fields = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), geometry="geometry", crs=source.crs)
    return fields


def load_county_fields(region: str) -> gpd.GeoDataFrame:
    path = region_dir(region) / "fields.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"No fields at {path}. Run --step collect for this county first.")
    fields = gpd.read_file(path)
    fields["field_id"] = fields["field_id"].astype(str)
    fields["year"] = int(TRANSFER_YEAR)
    fields["label"] = ""
    return fields


def sample_fields(fields: gpd.GeoDataFrame, test: bool, barc: bool) -> gpd.GeoDataFrame:
    """--test keeps 4 fields. An integer N_FIELDS draws a random sample instead of every field."""
    if test and barc:
        chosen = []
        for year in sorted(fields["year"].unique()):
            pool = fields.loc[fields["year"] == year]
            for label in BARC_LABELS:
                part = pool.loc[pool["label"] == label]
                # Largest fields survive the inward buffer and make a faster test.
                part = part.assign(_area=part.to_crs("EPSG:5070").area).sort_values("_area", ascending=False)
                chosen.append(part.head(2).drop(columns="_area"))
        out = gpd.GeoDataFrame(pd.concat(chosen, ignore_index=True), geometry="geometry", crs=fields.crs)
        print(f"Test mode BARC fields: {out['field_id'].tolist()}")
        return out
    limit = 4 if test else (None if N_FIELDS == "all" else int(N_FIELDS))
    if limit is None or len(fields) <= limit:
        return fields
    rng = np.random.default_rng(RANDOM_SEED)
    take = rng.choice(len(fields), size=limit, replace=False)
    out = fields.iloc[np.sort(take)].copy()
    print(f"Sampled {len(out)} of {len(fields)} fields.")
    return out


def prepare_region(region: str, test: bool) -> tuple[gpd.GeoDataFrame, dict, str, list[int], list[str]]:
    """Buffered fields in the region UTM, plus one load box per tile."""
    notes: list[str] = []
    if region == "barc":
        fields = sample_fields(load_barc_fields(), test=test, barc=True)
        years = sorted(int(year) for year in fields["year"].unique())
    else:
        fields = load_county_fields(region)
        years = [TRANSFER_YEAR]
    epsg = region_epsg(fields)
    buffered, dropped = buffer_inward(fields, epsg)
    notes.extend(dropped)
    if buffered.empty:
        raise RuntimeError(f"{region} has no fields left after the inward buffer.")
    if region == "barc":
        # One search and one load for the whole farm. The runner still filters by year.
        buffered["tile_id"] = "all"
        return buffered, {"all": padded_lonlat_bounds(buffered)}, epsg, years, notes
    tiles = assign_tiles(buffered)
    if test:
        biggest = max(tiles, key=lambda tile: int((buffered["tile_id"] == tile).sum()))
        buffered = buffered.loc[buffered["tile_id"] == biggest].copy()
        if len(buffered) > 4:
            buffered = sample_fields(buffered, test=True, barc=False)
            buffered["tile_id"] = biggest
        tiles = {biggest: padded_lonlat_bounds(buffered)}
        print(f"Test mode is using tile {biggest} with {len(buffered)} fields.")
    return buffered, tiles, epsg, years, notes
