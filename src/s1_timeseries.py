"""Per-field Sentinel-1 RTC backscatter. Medians are taken in linear power, then converted to dB."""

from __future__ import annotations

import time

import numpy as np

from .common import (
    append_log,
    describe_scenes,
    field_date_medians,
    load_cube,
    region_dir,
    release_cube,
    run_tiles,
    s1_group,
    s1_time_labels,
    search_items,
    season_dates,
)
from .config import (
    BUFFER_M,
    CATALOG,
    RESOLUTION_M,
    S1_BANDS,
    S1_COLLECTION,
    S1_COLUMNS,
    S1_RESAMPLING,
)
from .fields import prepare_region


def _to_db(linear: float) -> float:
    if not np.isfinite(linear) or linear <= 0:
        return float("nan")
    return float(10.0 * np.log10(linear))


def _extract(fields, bbox, year: int, epsg: str, optimize: bool = True):
    started = time.perf_counter()
    items = search_items(S1_COLLECTION, bbox, year, cloud=False)
    print(
        f"Sentinel-1 {year}: {len(items)} scenes, {describe_scenes(items)} "
        f"(search {time.perf_counter() - started:.1f}s)"
    )
    if not items:
        return [], 0, [f"{year}: zero Sentinel-1 scenes, skipped"]
    labels = s1_time_labels(items)
    started = time.perf_counter()
    cube = load_cube(
        items, S1_BANDS, bbox, epsg, S1_RESAMPLING, groupby=s1_group, progress_label="Sentinel-1", optimize=optimize
    )
    print(f"Sentinel-1 {year}: load {time.perf_counter() - started:.1f}s")
    started = time.perf_counter()
    try:
        if cube.sizes.get("time", 0) == 0 or cube.sizes.get("x", 0) == 0:
            return [], len(items), [f"{year}: Sentinel-1 load returned an empty cube, skipped"]
        missing = [band for band in S1_BANDS if band not in cube]
        if missing:
            return [], len(items), [f"{year}: cube is missing {missing}, skipped"]
        vv = cube["vv"].values.astype("float32")
        vh = cube["vh"].values.astype("float32")
        # RTC nodata is 0 or negative. Medians stay in linear power until after the pixel median.
        valid = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
        aligned = labels if len(labels) == cube.sizes["time"] else None
        notes = []
        if aligned is None:
            notes.append(
                f"{year}: {cube.sizes['time']} time steps and {len(labels)} orbit groups did not match; "
                "orbit columns left blank"
            )
        rows = []
        for field in fields.itertuples(index=False):
            kept, note = field_date_medians(
                cube,
                field.geometry,
                valid,
                {"vv": vv, "vh": vh},
                min_frac=None,
                time_labels=aligned,
            )
            if note:
                notes.append(f"field {field.field_id} {year}: {note}")
            label = "" if field.label is None else field.label
            for item in kept:
                vv_lin = item.pop("vv")
                vh_lin = item.pop("vh")
                item["vv_db"] = _to_db(vv_lin)
                item["vh_db"] = _to_db(vh_lin)
                item["vh_vv_db"] = _to_db(vh_lin / vv_lin) if np.isfinite(vv_lin) and vv_lin > 0 else float("nan")
                if not np.isfinite(item["vv_db"]):
                    continue
                item.setdefault("orbit_state", "")
                item.setdefault("relative_orbit", "")
                item["field_id"] = field.field_id
                item["year"] = int(year)
                item["label"] = label
                rows.append(item)
        return rows, len(items), notes
    finally:
        print(f"Sentinel-1 {year}: field extraction {time.perf_counter() - started:.1f}s")
        release_cube(cube)


def main(region: str, test: bool = False, optimize: bool = True) -> None:
    fields, tiles, epsg, years, notes = prepare_region(region, test)
    print(f"{region}: {len(fields)} field-years, {len(tiles)} tiles, {epsg}")

    def worker(subset, bbox, year):
        return _extract(subset, bbox, year, epsg, optimize=optimize)

    result = run_tiles(region, years, fields, tiles, "s1", S1_COLUMNS, "s1_timeseries", worker, test)
    notes.extend(result["notes"])
    start, end = season_dates(years[0])
    shown = notes[:30]
    if len(notes) > 30:
        shown.append(f"... {len(notes) - 30} more skip notes")
    append_log(
        f"Sentinel-1 {region}" + (" test" if test else ""),
        [
            ("catalog", CATALOG),
            ("collection", S1_COLLECTION),
            ("date range", f"{start[5:]} to {end[5:]}"),
            ("years", years),
            ("bands", ", ".join(S1_BANDS)),
            ("crs", epsg),
            ("resolution_m", RESOLUTION_M),
            ("buffer_m", BUFFER_M),
            ("units", "median vv and vh in linear power, then 10*log10; vh/vv ratio in dB"),
            ("tiles", result["tiles"]),
            ("scene counts", result["scenes"]),
            ("rows written", result["rows"]),
            ("output", result["master"]),
            ("skip notes", "; ".join(shown) if shown else "none"),
        ],
    )
    print(f"Sentinel-1 table: {region_dir(region) / result['master'].name}")
