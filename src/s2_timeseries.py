"""Per-field Sentinel-2 indices and band medians. One load per BARC year, or per county tile."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import make_valid

from .common import (
    append_log,
    describe_scenes,
    field_date_medians,
    geobox_mid_lon,
    load_cube,
    normalized_difference,
    polygon_mask,
    region_dir,
    release_cube,
    run_tiles,
    scale_reflectance,
    search_items,
    season_dates,
    solar_day_groups,
)
from .config import (
    CATALOG,
    BUFFER_M,
    CLEAR_FRAC_MIN,
    CLOUD_COVER_LT,
    LABEL_COLORS,
    RANDOM_SEED,
    RESOLUTION_M,
    S2_BANDS,
    S2_COLLECTION,
    S2_COLUMNS,
    S2_REFLECTANCE,
    S2_RESAMPLING,
    SCL_CLASSES,
)
from .fields import prepare_region


def _one_mgrs_tile(items, fields):
    """Keep one MGRS tile when every field sits inside that tile. Otherwise keep all."""
    groups: dict[str, list] = {}
    for item in items:
        tile = str(item.properties.get("s2:mgrs_tile") or "")
        groups.setdefault(tile, []).append(item)
    if len(groups) <= 1:
        return items
    polygons = fields.to_crs("EPSG:4326").geometry

    def covers_all(group) -> bool:
        parts = [make_valid(shape(item.geometry)) for item in group if item.geometry]
        if not parts:
            return False
        cover = unary_union(parts)
        return all(cover.covers(geom) for geom in polygons)

    covering = [tile for tile, group in groups.items() if covers_all(group)]
    if len(covering) == 1:
        kept = groups[covering[0]]
        print(f"MGRS {covering[0]} covers every field: {len(kept)} of {len(items)} scenes")
        return kept
    print("Fields are not inside a single MGRS tile: keeping every scene")
    return items


def _clear_dates(cube, fields) -> np.ndarray:
    """True for a date when at least one field reaches the clear-fraction cutoff."""
    clear = np.isin(cube["SCL"].values, SCL_CLASSES)
    keep = np.zeros(cube.sizes["time"], dtype=bool)
    for field in fields.itertuples(index=False):
        inside = polygon_mask(cube, field.geometry)
        n_inside = int(inside.sum())
        if n_inside == 0:
            continue
        fraction = (clear & inside).sum(axis=(1, 2)) / n_inside
        keep |= fraction >= CLEAR_FRAC_MIN
    return keep


def _rows_from_cube(cube, fields, year: int):
    if cube.sizes.get("time", 0) == 0 or cube.sizes.get("x", 0) == 0:
        return [], [f"{year}: load returned an empty cube, skipped"]
    missing = [band for band in S2_BANDS if band not in cube]
    if missing:
        return [], [f"{year}: cube is missing {missing}, skipped"]
    reflectance = scale_reflectance(cube, S2_REFLECTANCE)
    # Indices are medians of the per-pixel values, not indices of the band medians.
    # NDMI is (B08 - B11) / (B08 + B11). Band columns are the same clear pixels.
    arrays = {
        "ndvi": normalized_difference(reflectance["B08"], reflectance["B04"]),
        "ndti": normalized_difference(reflectance["B11"], reflectance["B12"]),
        "ndmi": normalized_difference(reflectance["B08"], reflectance["B11"]),
        "b04": reflectance["B04"],
        "b08": reflectance["B08"],
        "b11": reflectance["B11"],
        "b12": reflectance["B12"],
    }
    clear = np.isin(cube["SCL"].values, SCL_CLASSES)
    rows = []
    notes = []
    for field in fields.itertuples(index=False):
        kept, note = field_date_medians(
            cube, field.geometry, clear, arrays, CLEAR_FRAC_MIN, require_finite="ndvi"
        )
        if note:
            notes.append(f"field {field.field_id} {year}: {note}")
        label = "" if field.label is None else field.label
        for item in kept:
            item["field_id"] = field.field_id
            item["year"] = int(year)
            item["label"] = label
            rows.append(item)
    return rows, notes


def _extract(fields, bbox, year: int, epsg: str, optimize: bool = True):
    started = time.perf_counter()
    found = search_items(S2_COLLECTION, bbox, year, cloud=True)
    print(
        f"Sentinel-2 {year}: {len(found)} scenes found, {describe_scenes(found)} "
        f"(search {time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    if not found:
        return [], 0, [f"{year}: zero scenes, skipped"]
    items = _one_mgrs_tile(found, fields) if optimize else found
    if not optimize:
        started = time.perf_counter()
        cube = load_cube(
            items, S2_BANDS, bbox, epsg, S2_RESAMPLING, groupby="solar_day", progress_label="Sentinel-2", optimize=False
        )
        print(f"Sentinel-2 {year}: load {time.perf_counter() - started:.1f}s")
        started = time.perf_counter()
        try:
            rows, notes = _rows_from_cube(cube, fields, year)
            return rows, len(found), notes
        finally:
            print(f"Sentinel-2 {year}: field extraction {time.perf_counter() - started:.1f}s")
            release_cube(cube)

    # SCL at native 20 m decides which solar days are worth a 10 m reflectance load.
    started = time.perf_counter()
    scl_cube = load_cube(
        items,
        ["SCL"],
        bbox,
        epsg,
        {"SCL": "nearest"},
        groupby="solar_day",
        progress_label="Sentinel-2 SCL",
        resolution=20,
        optimize=True,
    )
    scl_seconds = time.perf_counter() - started
    try:
        groups = solar_day_groups(items, geobox_mid_lon(scl_cube))
        n_dates = scl_cube.sizes.get("time", 0)
        print(f"Sentinel-2 {year}: {n_dates} dates after dedup", flush=True)
        if len(groups) != n_dates:
            print(
                f"Sentinel-2 {year}: solar-day groups ({len(groups)}) did not match the SCL cube. "
                "Loading every remaining scene."
            )
            chosen = items
            kept_dates = n_dates
        else:
            keep = _clear_dates(scl_cube, fields)
            kept_dates = int(keep.sum())
            chosen = [items[index] for group, flag in zip(groups, keep) if flag for index in group]
        print(
            f"Sentinel-2 {year}: {kept_dates} dates kept after the SCL pass "
            f"(SCL pass {scl_seconds:.1f}s)",
            flush=True,
        )
    finally:
        release_cube(scl_cube)
    if not chosen:
        return [], len(found), [f"{year}: no date passed the SCL clear-fraction pass, skipped"]
    started = time.perf_counter()
    cube = load_cube(
        chosen,
        S2_BANDS,
        bbox,
        epsg,
        S2_RESAMPLING,
        groupby="solar_day",
        progress_label="Sentinel-2 reflectance",
        optimize=True,
    )
    print(f"Sentinel-2 {year}: reflectance pass {time.perf_counter() - started:.1f}s")
    started = time.perf_counter()
    try:
        rows, notes = _rows_from_cube(cube, fields, year)
        notes.append(f"{year}: {len(found)} scenes, {n_dates} dates after dedup, {kept_dates} dates kept")
        return rows, len(found), notes
    finally:
        print(f"Sentinel-2 {year}: field extraction {time.perf_counter() - started:.1f}s")
        release_cube(cube)


def _plot(master: Path, years: list[int], region: str, test: bool) -> None:
    if not master.exists():
        return
    table = pd.read_csv(master)
    if table.empty:
        return
    for year in years:
        subset = table.loc[table["year"] == year]
        if subset.empty:
            continue
        ids = subset["field_id"].astype(str).unique()
        if region != "barc" and len(ids) > 50:
            rng = np.random.default_rng(RANDOM_SEED)
            ids = rng.choice(ids, size=50, replace=False)
            subset = subset.loc[subset["field_id"].astype(str).isin(ids)]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for (field_id, label), curve in subset.groupby(["field_id", "label"], dropna=False):
            curve = curve.sort_values("doy")
            color = LABEL_COLORS.get(label if isinstance(label, str) else "", "#4c78a8")
            ax.plot(curve["doy"], curve["ndvi"], color=color, marker="o", markersize=3, linewidth=1, alpha=0.85)
        ax.set_title(f"{region} {year} median NDVI")
        ax.set_xlabel("Day of year")
        ax.set_ylabel("NDVI")
        ax.set_xlim(121, 350)
        ax.grid(True, alpha=0.3)
        handles = [
            Line2D([0], [0], color=LABEL_COLORS["Silage"], marker="o", label="Silage"),
            Line2D([0], [0], color=LABEL_COLORS["Grain"], marker="o", label="Grain"),
        ]
        if region != "barc":
            handles = [Line2D([0], [0], color=LABEL_COLORS[""], marker="o", label="County field")]
        ax.legend(handles=handles, loc="lower right")
        fig.tight_layout()
        name = f"ndvi_curves_{year}_test.png" if test else f"ndvi_curves_{year}.png"
        fig.savefig(region_dir(region) / name, dpi=120)
        plt.close(fig)


def main(region: str, test: bool = False, optimize: bool = True) -> None:
    fields, tiles, epsg, years, notes = prepare_region(region, test)
    print(f"{region}: {len(fields)} field-years, {len(tiles)} tiles, {epsg}")

    def worker(subset, bbox, year):
        return _extract(subset, bbox, year, epsg, optimize=optimize)

    result = run_tiles(region, years, fields, tiles, "s2", S2_COLUMNS, "s2_timeseries", worker, test)
    notes.extend(result["notes"])
    _plot(result["master"], years, region, test)
    start, end = season_dates(years[0])
    shown = notes[:30]
    if len(notes) > 30:
        shown.append(f"... {len(notes) - 30} more skip notes")
    append_log(
        f"Sentinel-2 {region}" + (" test" if test else ""),
        [
            ("catalog", CATALOG),
            ("collection", S2_COLLECTION),
            ("date range", f"{start[5:]} to {end[5:]}"),
            ("years", years),
            ("cloud threshold", f"eo:cloud_cover < {CLOUD_COVER_LT}"),
            ("bands", ", ".join(S2_BANDS)),
            (
                "indices",
                "per-pixel then median: ndvi (B08, B04), ndti (B11, B12), ndmi (B08, B11); "
                "b04, b08, b11, b12 are median reflectance on the same mask",
            ),
            ("crs", epsg),
            ("resolution_m", RESOLUTION_M),
            ("buffer_m", BUFFER_M),
            ("scl classes", list(SCL_CLASSES)),
            ("clear_frac cutoff", CLEAR_FRAC_MIN),
            ("tiles", result["tiles"]),
            ("scene counts", result["scenes"]),
            ("rows written", result["rows"]),
            ("output", result["master"]),
            ("skip notes", "; ".join(shown) if shown else "none"),
        ],
    )
