"""Clip Crop Sequence Boundaries to each test county and keep corn fields."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import box

from .common import acquisition_estimate, append_log, assign_tiles, buffer_inward, estimate_acquisition, region_epsg
from .config import (
    CDL_PATHS,
    CORN_CDL_CLASS,
    CSB_EXTRACTS,
    CSB_PATH,
    CSB_RELEASE,
    MIN_AREA_HA,
    MIN_CORN_FRAC,
    PROCESSED,
    RANDOM_SEED,
    RAW,
    TEST_COUNTIES,
    TIGER_URL,
    TRANSFER_YEAR,
    field_limit,
    resolve_data_path,
)
from .fields import counties_for, region_id

def tiger_counties() -> gpd.GeoDataFrame:
    """US Census county polygons, downloaded once into data/raw/tiger/."""
    dest = RAW / "tiger" / "tl_2023_us_county.zip"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading TIGER county boundaries to {dest}")
        urllib.request.urlretrieve(TIGER_URL, dest)
    counties = gpd.read_file(dest)
    counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)
    return counties


def corn_fraction(polys: gpd.GeoDataFrame, cdl_path) -> np.ndarray:
    """Share of CDL pixels inside each polygon that are corn (class 1)."""
    with rasterio.open(cdl_path) as source:
        raster_crs = source.crs
    projected = polys.to_crs(raster_crs)
    stats = zonal_stats(projected, str(cdl_path), categorical=True)
    fractions = np.zeros(len(projected), dtype="float64")
    for index, counts in enumerate(stats):
        if not counts:
            continue
        total = float(sum(counts.values()))
        corn = float(counts.get(CORN_CDL_CLASS, counts.get(str(CORN_CDL_CLASS), 0)))
        fractions[index] = corn / total if total else 0.0
    return fractions


def _cdl_is_2023(path: Path, source) -> bool:
    """The file name or raster tags must say 2023, and must not name a different year."""
    text = " ".join(
        [
            path.name,
            str(source.tags() or ""),
            " ".join(str(item) for item in (source.descriptions or []) if item),
        ]
    )
    return "2023" in text


def validate_inputs(selected: list[tuple[str, str]]) -> dict[str, dict]:
    """Stop before any field work if a file, the CDL coverage, or the NASS rows are wrong."""
    missing = []
    cdl_path = resolve_data_path(CDL_PATHS.get(TRANSFER_YEAR))
    csb_path = resolve_data_path(CSB_PATH)
    if csb_path is None or not csb_path.exists():
        missing.append(f"CSB geodatabase: {csb_path}")
    if cdl_path is None or not cdl_path.exists():
        missing.append(f"CDL {TRANSFER_YEAR}: {cdl_path}")
    for state, _fips in selected:
        extract_path = resolve_data_path(CSB_EXTRACTS.get(state))
        if extract_path is None or not extract_path.exists():
            missing.append(f"CSB extract {state}: {extract_path} (run python -m src.extract_csb)")
    if missing:
        print("Missing input files.")
        for item in missing:
            print(f"  missing: {item}")
        raise SystemExit(1)
    if TRANSFER_YEAR != 2023:
        raise SystemExit(f"TRANSFER_YEAR is {TRANSFER_YEAR}. This collect step expects the 2023 CDL.")

    counties = tiger_counties()
    with rasterio.open(cdl_path) as source:
        if not _cdl_is_2023(cdl_path, source):
            raise SystemExit(f"CDL file is not identified as 2023: {cdl_path}")
        print(f"CDL {cdl_path.name}: CRS {source.crs}, resolution {source.res}, bounds {tuple(source.bounds)}")
        pixel = max(abs(source.res[0]), abs(source.res[1]))
        raster_box = box(
            source.bounds.left - pixel,
            source.bounds.bottom - pixel,
            source.bounds.right + pixel,
            source.bounds.top + pixel,
        )
        cdl_crs = source.crs
    for state, fips in selected:
        county = counties.loc[counties["GEOID"] == fips]
        if county.empty:
            raise SystemExit(f"FIPS {fips} is not in the TIGER county file.")
        county_box = box(*county.to_crs(cdl_crs).total_bounds)
        if not raster_box.covers(county_box):
            raise SystemExit(f"CDL does not cover {state} {fips}. The county falls outside {cdl_path}.")

    nass_path = PROCESSED / "nass_county_2023.csv"
    if not nass_path.exists():
        raise SystemExit(f"Missing {nass_path}. Run --step select first.")
    nass = pd.read_csv(nass_path, dtype=str)
    checked = {}
    for state, fips in selected:
        hit = nass.loc[nass["fips"] == fips]
        if hit.empty:
            raise SystemExit(f"{fips} ({state}) is not in {nass_path}.")
        grain_text = hit["grain_acres"].iloc[0]
        silage_text = hit["silage_acres"].iloc[0]
        if grain_text in {"", "nan", "None"} or silage_text in {"", "nan", "None"} or pd.isna(grain_text) or pd.isna(silage_text):
            raise SystemExit(f"{fips} ({state}) is missing 2023 grain or silage acres in {nass_path}.")
        grain = float(grain_text)
        silage = float(silage_text)
        share = silage / (grain + silage)
        checked[fips] = {"grain": grain, "silage": silage, "share": share, "total": grain + silage}
        print(
            f"{state} {fips}: 2023 survey grain {grain:,.0f} acres, "
            f"silage {silage:,.0f} acres, silage share {share:.1%}."
        )
    return checked


def collect_county(state: str, fips: str, test: bool, nass_2023: dict) -> dict:
    cdl_path = resolve_data_path(CDL_PATHS[TRANSFER_YEAR])
    csb_path = resolve_data_path(CSB_EXTRACTS[state])
    county = tiger_counties()
    county = county.loc[county["GEOID"] == fips]
    if county.empty:
        raise RuntimeError(f"FIPS {fips} is not in the TIGER county file.")
    print(f"{state} {fips}: reading {csb_path.name}")
    # The bbox has to be in the file CRS, so read one row for that CRS first.
    sample_crs = gpd.read_file(csb_path, rows=1).crs
    polygons = gpd.read_file(csb_path, bbox=tuple(county.to_crs(sample_crs).total_bounds))
    polygons = gpd.clip(polygons, county.to_crs(polygons.crs))
    polygons = polygons.loc[~polygons.geometry.is_empty].copy()
    crop_col = f"CDL{TRANSFER_YEAR}"
    if crop_col in polygons.columns:
        polygons = polygons.loc[polygons[crop_col] == CORN_CDL_CLASS].copy()
    total_polygons = len(polygons)
    print(f"{state} {fips}: {total_polygons:,} corn polygons inside the county, counting CDL corn fraction")
    polygons["corn_frac"] = corn_fraction(polygons, cdl_path)
    epsg = region_epsg(county)
    polygons["area_ha"] = polygons.to_crs(epsg).area / 10_000
    qualified = polygons.loc[(polygons["corn_frac"] >= MIN_CORN_FRAC) & (polygons["area_ha"] >= MIN_AREA_HA)].copy()
    qualified = qualified.sort_values(["area_ha"], ascending=False)
    limit = field_limit(fips, test=test)
    if limit is not None and len(qualified) > limit:
        rng = np.random.default_rng(RANDOM_SEED)
        take = np.sort(rng.choice(len(qualified), size=limit, replace=False))
        kept = qualified.iloc[take].copy()
    else:
        kept = qualified
    kept = kept.reset_index(drop=True)
    kept["field_id"] = [f"{fips}_{index:05d}" for index in range(len(kept))]
    out = kept[["field_id", "area_ha", "corn_frac", "geometry"]].to_crs("EPSG:4326")
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    folder = PROCESSED / region_id(state, fips)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "fields.gpkg"
    if dest.exists():
        dest.unlink()
    out.to_file(dest, driver="GPKG")
    kept_ha = float(out.to_crs(epsg).area.sum() / 10_000) if len(out) else 0.0
    kept_acres = kept_ha * 2.47105
    nass_total = float(nass_2023["total"])
    print(
        f"{state} {fips}: {len(qualified):,} qualifying corn fields, {len(out):,} kept, "
        f"{kept_acres:,.0f} acres kept vs {nass_total:,.0f} NASS 2023 corn acres (grain + silage)."
    )
    # Tile count uses the same inward buffer the satellite steps use.
    if len(out):
        buffered, _notes = buffer_inward(out.assign(year=TRANSFER_YEAR, label=""), epsg)
        n_tiles = len(assign_tiles(buffered)) if len(buffered) else 0
    else:
        n_tiles = 0
    numbers = acquisition_estimate(len(out), n_tiles)
    estimate = estimate_acquisition(len(out), n_tiles)
    print(f"Estimate for s2 and s1: {estimate}")
    if numbers["hours"] > 8:
        suggested = max(1, int(len(out) * 8 / numbers["hours"])) if len(out) else 1
        print(
            f"Warning: estimated runtime for {state} {fips} is {numbers['hours']:.1f} hours. "
            f'Set N_FIELDS_BY_COUNTY["{fips}"] = {suggested} to bring it under 8 hours. '
            "The setting was not changed."
        )
    return {
        "state": state,
        "fips": fips,
        "polygons": total_polygons,
        "qualified": len(qualified),
        "kept": len(out),
        "kept_ha": kept_ha,
        "kept_acres": kept_acres,
        "nass_acres": nass_total,
        "nass_grain": nass_2023["grain"],
        "nass_silage": nass_2023["silage"],
        "nass_share": nass_2023["share"],
        "hours": numbers["hours"],
        "tiles": n_tiles,
        "estimate": estimate,
        "output": dest,
    }


def main(region: str | None = "all", test: bool = False) -> None:
    if region == "barc":
        raise SystemExit("BARC fields come from the labeled GeoPackage. collect is only for counties in TEST_COUNTIES.")
    if not TEST_COUNTIES and region in {None, "all"}:
        raise SystemExit("TEST_COUNTIES is empty. Fill src/config.py from candidate_counties.csv, then rerun.")
    selected = counties_for(region)
    nass_2023 = validate_inputs(selected)
    results = [collect_county(state, fips, test=test, nass_2023=nass_2023[fips]) for state, fips in selected]
    county_lines = [
        (
            f"{result['state']} {result['fips']}",
            (
                f"grain {result['nass_grain']:.0f}, silage {result['nass_silage']:.0f}, "
                f"share {result['nass_share']:.1%}; qualifying {result['qualified']}, kept {result['kept']}, "
                f"kept acres {result['kept_acres']:.0f} vs NASS {result['nass_acres']:.0f}; "
                f"extract {resolve_data_path(CSB_EXTRACTS[result['state']])}"
            ),
        )
        for result in results
    ]
    append_log(
        "Collect transfer counties",
        [
            ("year", TRANSFER_YEAR),
            ("csb", f"{Path(CSB_PATH).name} ({CSB_RELEASE})"),
            ("cdl", resolve_data_path(CDL_PATHS.get(TRANSFER_YEAR))),
            ("counties", ", ".join(f"{state} {fips}" for state, fips in selected)),
            *county_lines,
        ],
    )


if __name__ == "__main__":
    main()
