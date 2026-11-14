"""Shared geometry, reflectance, masking, tiles, and resume bookkeeping.

Sentinel-2 and Sentinel-1 both call these functions so every region is processed
the same way. Raw imagery is never written.
"""

from __future__ import annotations

import os

from .config import GDAL_CLOUD_DEFAULTS

# Before geopandas or rasterio load GDAL, so remote COG reads see these options.
os.environ.update(GDAL_CLOUD_DEFAULTS)

import gc
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa: F401  registers the .odc geobox accessor
import pandas as pd
import planetary_computer
import rasterio
import rasterio.features
from odc.stac import load
from odc.stac._mdtools import parse_items
from odc.stac._stac_load import _group_items
from pystac_client import Client
from shapely import make_valid
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.ops import unary_union

from .config import (
    BUFFER_M,
    CATALOG,
    CLEAR_FRAC_MIN,
    CLOUD_COVER_LT,
    DN_OFFSET,
    LOAD_CHUNK_XY,
    LOAD_PAD_M,
    LOG_PATH,
    N_LOAD_THREADS,
    N_WORKERS,
    OFFSET_DATE,
    PROCESSED,
    RESOLUTION_M,
    ROOT,
    SCALE,
    SCL_CLASSES,
    SEASON_END,
    SEASON_START,
    TILE_SIZE_M,
)


def season_dates(year: int) -> tuple[str, str]:
    start_m, start_d = SEASON_START
    end_m, end_d = SEASON_END
    return f"{year}-{start_m:02d}-{start_d:02d}", f"{year}-{end_m:02d}-{end_d:02d}"


def utm_epsg(lon: float, lat: float) -> str:
    """UTM zone from a longitude. Northern hemisphere is 326xx."""
    zone = int((lon + 180) // 6) + 1
    zone = min(60, max(1, zone))
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def region_epsg(fields: gpd.GeoDataFrame) -> str:
    """UTM CRS of the field collection, from the centroid longitude."""
    ll = fields.to_crs("EPSG:4326")
    center = ll.geometry.union_all().centroid
    return utm_epsg(float(center.x), float(center.y))


def region_dir(region: str) -> Path:
    return PROCESSED / region


def find_barc_gpkg() -> Path:
    name = "BARC_Corn.2021-2023.gpkg"
    candidates = [
        ROOT / "04_Evaluation" / name,
        ROOT / "notebooks" / "dataset" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = [
        path
        for path in ROOT.rglob(name)
        if ".venv" not in path.parts and "site-packages" not in path.parts
    ]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {name}")


def polygonal(geom):
    """Keep polygon parts. An inward buffer can return a collection or nothing."""
    if geom is None or geom.is_empty:
        return Polygon()
    geom = make_valid(geom)
    if isinstance(geom, GeometryCollection):
        parts = [part for part in geom.geoms if part.geom_type in ("Polygon", "MultiPolygon")]
        geom = unary_union(parts) if parts else Polygon()
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        return Polygon()
    return geom


def buffer_inward(fields: gpd.GeoDataFrame, epsg: str) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Reproject and shrink every field by BUFFER_M. Drops empty results."""
    projected = fields.to_crs(epsg).copy()
    projected["geometry"] = projected.geometry.map(lambda geom: polygonal(geom.buffer(-BUFFER_M)))
    empty = projected.geometry.is_empty | projected.geometry.isna() | (projected.geometry.area <= 0)
    notes = []
    if empty.any():
        for field_id, year in projected.loc[empty, ["field_id", "year"]].itertuples(index=False):
            notes.append(f"field {field_id} {year}: empty after {BUFFER_M} m inward buffer, skipped")
        print(f"Dropped {int(empty.sum())} fields emptied by the {BUFFER_M} m buffer.")
    kept = projected.loc[~empty].copy()
    return kept, notes


def padded_lonlat_bounds(fields_utm: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Lon/lat bounds of the fields, padded so pixel snapping does not clip them."""
    minx, miny, maxx, maxy = fields_utm.total_bounds
    pad = LOAD_PAD_M
    cover = box(minx - pad, miny - pad, maxx + pad, maxy + pad)
    lonlat = gpd.GeoSeries([cover], crs=fields_utm.crs).to_crs("EPSG:4326").total_bounds
    return tuple(float(value) for value in lonlat)


def assign_tiles(fields_utm: gpd.GeoDataFrame) -> dict[str, tuple[float, float, float, float]]:
    """Give each field one tile by its centroid. Extend the tile to cover the whole field.

    Returns lon/lat load bounds for every tile that contains a field.
    """
    origin_x = np.floor(fields_utm.total_bounds[0] / TILE_SIZE_M) * TILE_SIZE_M
    origin_y = np.floor(fields_utm.total_bounds[1] / TILE_SIZE_M) * TILE_SIZE_M
    centers = fields_utm.geometry.centroid
    col = np.floor((centers.x.to_numpy() - origin_x) / TILE_SIZE_M).astype(int)
    row = np.floor((centers.y.to_numpy() - origin_y) / TILE_SIZE_M).astype(int)
    fields_utm["tile_id"] = [f"{c}_{r}" for c, r in zip(col, row)]
    tiles: dict[str, tuple[float, float, float, float]] = {}
    for tile_id, group in fields_utm.groupby("tile_id"):
        c, r = (int(part) for part in str(tile_id).split("_"))
        tile_box = box(
            origin_x + c * TILE_SIZE_M,
            origin_y + r * TILE_SIZE_M,
            origin_x + (c + 1) * TILE_SIZE_M,
            origin_y + (r + 1) * TILE_SIZE_M,
        )
        # The load window is the tile plus any part of a field that crosses its edge.
        cover = unary_union([tile_box, *list(group.geometry)]).buffer(LOAD_PAD_M)
        lonlat = gpd.GeoSeries([cover], crs=fields_utm.crs).to_crs("EPSG:4326").total_bounds
        tiles[str(tile_id)] = tuple(float(value) for value in lonlat)
    return tiles


# A hung scene read should fail and be retried, not block the tile forever.
SCENE_READ_TIMEOUT_S = 120
_READ_DEPTH = 0
_READ_DEPTH_LOCK = threading.Lock()
_READ_WRAP_INSTALLED = False


def describe_scenes(items) -> str:
    """First and last scene dates, for the line printed after a STAC search."""
    stamps = []
    for item in items:
        when = getattr(item, "datetime", None)
        if when is not None:
            stamps.append(pd.Timestamp(when))
    if not stamps:
        return "dates unavailable"
    stamps.sort()
    return f"{stamps[0].strftime('%Y-%m-%d')} to {stamps[-1].strftime('%Y-%m-%d')}"


def _install_scene_read_retry() -> None:
    """Retry one scene read after a timeout. Pixel math is unchanged."""
    global _READ_WRAP_INSTALLED
    if _READ_WRAP_INSTALLED:
        return
    import odc.loader._rio as rio_mod

    original = rio_mod._rio_read
    original_env = rio_mod.rio_env

    def _rio_env_cached(*args, **kwargs):
        # odc turns the cache off around each read. Keep the startup COG setting.
        if kwargs.get("VSI_CACHE") is False:
            kwargs["VSI_CACHE"] = True
        return original_env(*args, **kwargs)

    def _rio_read_with_retry(*args, **kwargs):
        if _READ_DEPTH <= 0:
            return original(*args, **kwargs)
        src = args[0] if args else kwargs.get("src")
        uri = getattr(src, "uri", "scene")
        delay = 2.0
        last_error = None
        for attempt in range(1, 4):
            try:
                return original(*args, **kwargs)
            except (rasterio.errors.RasterioIOError, TimeoutError, OSError) as exc:
                last_error = exc
                print(f"scene read attempt {attempt} failed ({uri}): {exc}")
                if attempt == 3:
                    raise
                time.sleep(delay)
                delay *= 2
        raise last_error

    rio_mod.rio_env = _rio_env_cached
    rio_mod._rio_read = _rio_read_with_retry
    _READ_WRAP_INSTALLED = True


@contextmanager
def _scene_read_limits():
    """Each scene HTTP read gives up after SCENE_READ_TIMEOUT_S and can be retried."""
    global _READ_DEPTH
    _install_scene_read_retry()
    with _READ_DEPTH_LOCK:
        _READ_DEPTH += 1
    try:
        with rasterio.Env(GDAL_HTTP_TIMEOUT=str(SCENE_READ_TIMEOUT_S), **GDAL_CLOUD_DEFAULTS):
            yield
    finally:
        with _READ_DEPTH_LOCK:
            _READ_DEPTH -= 1


def _scene_load_bar(n_items: int, label: str):
    """odc-stac calls this with the block iterator. The bar moves as each block is read."""
    from tqdm import tqdm

    def _bar(work):
        return tqdm(
            work,
            total=len(work),
            desc=f"loading {n_items} {label} scenes",
            unit="block",
            leave=False,
        )

    return _bar


def call_with_retry(action, label: str):
    """Retry a STAC search or read three times. HTTP 429 uses the same backoff."""
    delay = 2.0
    last_error = None
    for attempt in range(1, 4):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001  network and raster reads raise many types
            last_error = exc
            status = getattr(exc, "response", None)
            code = getattr(status, "status_code", None) or getattr(exc, "code", None)
            print(f"{label} attempt {attempt} failed ({code}): {exc}")
            if attempt == 3:
                break
            time.sleep(delay)
            delay *= 2
    raise last_error


def search_items(collection: str, bbox: tuple[float, float, float, float], year: int, cloud: bool) -> list:
    """One catalog search. Items are not signed here; signing happens immediately before load."""
    start, end = season_dates(year)

    def _search():
        catalog = Client.open(CATALOG)
        query = {"eo:cloud_cover": {"lt": CLOUD_COVER_LT}} if cloud else None
        search = catalog.search(
            collections=[collection],
            bbox=list(bbox),
            datetime=f"{start}/{end}",
            query=query,
        )
        return list(search.items())

    return call_with_retry(_search, f"{collection} {year} search")


def _compute_threaded(cube):
    """Run a dask-backed cube on a thread pool. A numpy cube is returned as-is."""
    lazy = [variable for variable in cube.data_vars.values() if hasattr(getattr(variable, "data", None), "compute")]
    if not lazy:
        return cube
    import dask

    with dask.config.set(scheduler="threads", num_workers=N_LOAD_THREADS):
        return cube.compute()


def geobox_mid_lon(cube) -> float:
    """Longitude odc uses when it groups scenes into solar days."""
    ((lon, _lat),) = cube.odc.geobox.extent.centroid.to_crs("EPSG:4326").points
    return float(lon)


def solar_day_groups(items, lon: float) -> list[list[int]]:
    """Item indexes grouped the same way as odc groupby='solar_day'."""
    parsed = list(parse_items(items))
    return _group_items(items, parsed, "solar_day", lon)


def load_cube(
    items,
    bands,
    bbox,
    epsg: str,
    resampling: dict,
    groupby,
    progress_label: str = "satellite",
    resolution: float | None = None,
    optimize: bool = True,
):
    """Sign, then load. Signing sits inside the retry so an expired URL is refreshed."""

    def _load():
        for item in items:
            planetary_computer.sign_inplace(item)
        kwargs = {
            "bands": list(bands),
            "bbox": bbox,
            "crs": epsg,
            "resolution": RESOLUTION_M if resolution is None else resolution,
            "groupby": groupby,
            "resampling": resampling,
            "fail_on_error": False,
            "progress": _scene_load_bar(len(items), progress_label),
        }
        if optimize:
            kwargs["chunks"] = {"x": LOAD_CHUNK_XY, "y": LOAD_CHUNK_XY}
            kwargs["pool"] = N_LOAD_THREADS
            print(f"{progress_label}: parallel load, {N_LOAD_THREADS} threads", flush=True)
        with _scene_read_limits():
            cube = load(items, **kwargs)
            if optimize:
                cube = _compute_threaded(cube)
            return cube

    return call_with_retry(_load, f"load {epsg}")


def scale_reflectance(cube, bands) -> dict[str, np.ndarray]:
    """Digital number to reflectance. Float32 before any arithmetic.

    Dates on or after 2022-01-25 use the baseline 04.00 offset of 1000.
    DN 0 is the nodata fill and is masked before that subtraction.
    """
    offset_start = np.datetime64(OFFSET_DATE)
    times = cube.time.values
    # String group keys are not dates; only a datetime axis can take the offset.
    if np.issubdtype(times.dtype, np.datetime64):
        offset_on = times >= offset_start
    else:
        offset_on = np.ones(cube.sizes["time"], dtype=bool)
    scaled = {}
    for band in bands:
        dn = cube[band].values.astype("float32")
        dn[dn == 0] = np.nan
        dn[offset_on] -= np.float32(DN_OFFSET)
        scaled[band] = np.clip(dn, 0, None) / np.float32(SCALE)
    return scaled


def normalized_difference(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """(high - low) / (high + low). Divide-by-zero becomes NaN."""
    denom = high + low
    with np.errstate(divide="ignore", invalid="ignore"):
        index = (high - low) / denom
    index[~np.isfinite(denom) | (denom == 0)] = np.nan
    return index


def _stamp(value) -> tuple[str, int]:
    ts = pd.Timestamp(value)
    return ts.strftime("%Y-%m-%d"), int(ts.day_of_year)


def polygon_mask(cube, geometry) -> np.ndarray:
    """Pixels whose centers fall inside one field. Same mask the medians use."""
    geobox = cube.odc.geobox
    out_shape = (geobox.height, geobox.width)
    if out_shape != (cube.sizes["y"], cube.sizes["x"]):
        raise RuntimeError("Geobox shape does not match the cube y, x dimensions.")
    return rasterio.features.geometry_mask(
        [geometry],
        out_shape=out_shape,
        transform=geobox.affine,
        invert=True,
        all_touched=False,
    )


def field_date_medians(
    cube,
    geometry,
    valid_stack: np.ndarray,
    arrays: dict[str, np.ndarray],
    min_frac: float | None,
    require_finite: str | None = None,
    time_labels: list[dict] | None = None,
) -> tuple[list[dict], str | None]:
    """Median of each array inside one polygon.

    valid_stack is true for usable pixels (clear SCL, or finite Sentinel-1 power)
    and is combined with the polygon mask. min_frac keeps a date only when
    valid / inside reaches that fraction. Returns (rows, skip note).
    """
    inside = polygon_mask(cube, geometry)
    n_inside = int(inside.sum())
    if n_inside == 0:
        return [], "zero pixels inside the polygon, skipped"
    valid = valid_stack & inside
    n_valid = valid.sum(axis=(1, 2))
    fraction = n_valid / n_inside
    if min_frac is None:
        keep = np.flatnonzero(n_valid > 0)
    else:
        keep = np.flatnonzero(fraction >= min_frac)
    rows = []
    for index in keep:
        pixels = valid[index]
        values = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for name, array in arrays.items():
                values[name] = float(np.nanmedian(array[index][pixels]))
        if require_finite and not np.isfinite(values.get(require_finite, np.nan)):
            continue
        if time_labels is not None:
            label = time_labels[index]
            date = label["date"]
            doy = label["doy"]
            extra = {key: label[key] for key in label if key not in {"date", "doy"}}
        else:
            date, doy = _stamp(cube.time.values[index])
            extra = {}
        rows.append(
            {
                "date": date,
                "doy": doy,
                "clear_frac": float(fraction[index]),
                "n_pixels": int(n_valid[index]),
                **extra,
                **values,
            }
        )
    if not rows:
        cutoff = f"clear_frac {min_frac}" if min_frac is not None else "no valid pixels"
        return [], f"no date passed {cutoff}, skipped"
    return rows, None


def release_cube(cube) -> None:
    del cube
    gc.collect()


def s1_group(item, parsed, index):
    """Keep each orbit separate so ascending and descending are not fused."""
    props = item.properties
    day = item.datetime.strftime("%Y-%m-%d") if item.datetime is not None else str(parsed.nominal_datetime.date())
    return (day, props.get("sat:orbit_state"), props.get("sat:relative_orbit"))


def s1_time_labels(items) -> list[dict]:
    """Orbit metadata in the same group order odc uses for the time axis."""
    parsed = list(parse_items(items))
    grouped = _group_items(list(items), parsed, s1_group)
    labels = []
    for group in grouped:
        item = items[group[0]]
        props = item.properties
        when = item.datetime or parsed[group[0]].nominal_datetime
        date, doy = _stamp(when)
        labels.append(
            {
                "date": date,
                "doy": doy,
                "orbit_state": props.get("sat:orbit_state") or "",
                "relative_orbit": props.get("sat:relative_orbit") if props.get("sat:relative_orbit") is not None else "",
            }
        )
    return labels


def progress_path(region: str, step: str, test: bool) -> Path:
    name = f"progress_{step}_test.csv" if test else f"progress_{step}.csv"
    return region_dir(region) / name


def load_progress(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    done = set()
    frame = pd.read_csv(path, dtype={"region": str, "tile": str, "year": int})
    for region, tile, year in frame.itertuples(index=False):
        done.add((str(region), str(tile), int(year)))
    return done


def mark_progress(path: Path, region: str, tile: str, year: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write("region,tile,year\n")
        handle.write(f"{region},{tile},{year}\n")


def part_path(region: str, step: str, year: int, tile: str, test: bool) -> Path:
    folder = region_dir(region) / ("parts_test" if test else "parts")
    return folder / f"{step}_{year}_{tile}.csv"


def write_frame(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.reindex(columns=columns) if len(frame) else pd.DataFrame(columns=columns)
    out.to_csv(path, index=False)


def rebuild_master(master: Path, parts: list[Path], columns: list[str]) -> int:
    frames = [pd.read_csv(path) for path in parts if path.exists() and path.stat().st_size > 0]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    write_frame(master, combined, columns)
    return len(combined)


def append_log(title: str, bullets: list[tuple[str, object]]) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [f"\n## {stamp} — {title}\n"]
    for key, value in bullets:
        lines.append(f"- {key}: {value}\n")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)
    print(f"Appended a run record to {LOG_PATH}")


def acquisition_estimate(n_fields: int, n_tiles: int) -> dict:
    """Rough wall time and CSV size. Imagery is not stored."""
    workers = max(1, N_WORKERS)
    s2_minutes = n_tiles * 4.0 / workers
    s1_minutes = n_tiles * 3.0 / workers
    minutes = s2_minutes + s1_minutes
    s2_rows = n_fields * 25
    s1_rows = n_fields * 35
    megabytes = (s2_rows * 140 + s1_rows * 160) / (1024 * 1024)
    return {
        "workers": workers,
        "s2_minutes": s2_minutes,
        "s1_minutes": s1_minutes,
        "minutes": minutes,
        "hours": minutes / 60.0,
        "megabytes": megabytes,
        "s2_rows": s2_rows,
        "s1_rows": s1_rows,
        "tiles": n_tiles,
    }


def estimate_acquisition(n_fields: int, n_tiles: int) -> str:
    estimate = acquisition_estimate(n_fields, n_tiles)
    return (
        f"{estimate['tiles']} tiles, about {estimate['minutes']:.0f} minutes "
        f"({estimate['s2_minutes']:.0f} Sentinel-2 + {estimate['s1_minutes']:.0f} Sentinel-1) "
        f"at {estimate['workers']} workers; "
        f"about {estimate['megabytes']:.1f} MB of CSV "
        f"(~{estimate['s2_rows']:,} Sentinel-2 rows and ~{estimate['s1_rows']:,} Sentinel-1 rows). "
        "Assumes ~70 Sentinel-2 scenes and ~40 Sentinel-1 scenes per tile. No imagery is saved."
    )


def output_name(stem: str, test: bool) -> str:
    return f"{stem}_test.csv" if test else f"{stem}.csv"


def run_tiles(region, years, fields_utm, tiles, step, columns, master_stem, worker, test: bool) -> dict:
    """Run one worker per tile and year. Finished tiles are skipped on the next run.

    The worker returns (rows, scene_count, notes) and must not write imagery.
    A tile that raises is logged and left unfinished so a restart tries it again.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    directory = region_dir(region)
    directory.mkdir(parents=True, exist_ok=True)
    progress = progress_path(region, step, test=False)
    done = set() if test else load_progress(progress)
    master = directory / output_name(master_stem, test)
    finished_parts = [
        part_path(region, step, year, tile, test)
        for year in years
        for tile in tiles
        if (region, str(tile), int(year)) in done
    ]
    if not test:
        rebuild_master(master, finished_parts, columns)

    jobs = []
    for year in years:
        for tile, bbox in tiles.items():
            if (region, str(tile), int(year)) in done:
                print(f"Skipping finished {region} tile {tile} {year}")
                continue
            subset = fields_utm.loc[fields_utm["tile_id"].astype(str) == str(tile)].copy()
            if "year" in subset.columns:
                subset = subset.loc[subset["year"] == year]
            if subset.empty:
                continue
            jobs.append((str(tile), subset, bbox, int(year)))

    notes: list[str] = []
    scene_counts = {int(year): 0 for year in years}
    fields_done = 0
    bar = tqdm(total=len(jobs), desc=f"{step} {region}", unit="tile")
    workers = 1 if len(jobs) <= 1 else max(1, N_WORKERS)

    def _submit(pool, job):
        tile, subset, bbox, year = job
        return pool.submit(worker, subset, bbox, year), job

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {_submit(pool, job)[0]: job for job in jobs}
        for future in as_completed(futures):
            tile, subset, bbox, year = futures[future]
            try:
                rows, n_scenes, job_notes = future.result()
            except Exception as exc:  # noqa: BLE001
                message = f"tile {tile} {year} failed and will be retried next run: {exc}"
                print(message)
                notes.append(message)
                bar.update(1)
                continue
            notes.extend(job_notes)
            scene_counts[year] = scene_counts.get(year, 0) + int(n_scenes)
            frame = pd.DataFrame(rows, columns=columns)
            write_frame(part_path(region, step, year, tile, test), frame, columns)
            if not test:
                mark_progress(progress, region, tile, year)
                finished_parts.append(part_path(region, step, year, tile, test))
            else:
                finished_parts.append(part_path(region, step, year, tile, test))
            fields_done += subset["field_id"].nunique()
            bar.set_postfix(fields=fields_done, scenes=sum(scene_counts.values()))
            bar.update(1)
    bar.close()
    n_rows = rebuild_master(master, [path for path in finished_parts if path.exists()], columns)
    print(f"Wrote {n_rows:,} rows to {master}")
    return {"rows": n_rows, "scenes": scene_counts, "notes": notes, "master": master, "tiles": len(tiles)}
