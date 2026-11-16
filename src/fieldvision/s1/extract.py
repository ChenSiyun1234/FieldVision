"""Per-field Sentinel-1 RTC backscatter (VV, VH) from Planetary Computer.

Run with ``python -m fieldvision.s1.extract --help``.

Same input contract, block layout, 5 m inward buffer, checkpoints and resume rules
as fieldvision.s2.extract. Per field and acquisition: mean and median of VV and VH
gamma0 in linear power over valid interior pixels (both polarisations > 0 and finite),
the dB of the means and the VH/VV ratio in dB, plus orbit direction, relative and
absolute orbit and platform. Slices of one pass (same absolute orbit) are merged by
keeping, per field, the slice with the most valid pixels. No cloud filter: radar.

Data access needs no account: an anonymous SAS token from the Planetary Computer
token endpoint is appended to each asset URL. RTC files end in ".tiff", so GDAL must
allow that extension.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape

from fieldvision.s2.catalog import http_session
from fieldvision.s2.core import coverage_table, interior_geometry, polygon_mask
from fieldvision.s2.geometry import load_fields, make_blocks, project

LOG = logging.getLogger("fieldvision.s1")
PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/" + COLLECTION
POLS = ("vv", "vh")
NODATA = -32768.0
COLUMNS = ["field_id", "date", "datetime", "scene_id", "collection", "platform", "orbit_state", "relative_orbit",
           "absolute_orbit", "n_pixels", "n_valid", "valid_fraction", "vv_mean", "vh_mean", "vv_median",
           "vh_median", "vv_db", "vh_db", "vh_vv_db"]
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_CONNECTTIMEOUT": "15",
    "GDAL_HTTP_TIMEOUT": "90",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
}
# Run fingerprint: this package plus the s2 pieces it reuses.
FINGERPRINT_FILES = ("s1/*.py", "s2/core.py", "s2/geometry.py", "s2/catalog.py")


def fingerprint(package_dir):
    files = sorted({p for pattern in FINGERPRINT_FILES for p in Path(package_dir).glob(pattern)},
                   key=lambda p: p.relative_to(package_dir).as_posix())
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def replace_with_retry(temporary, path, attempts=8):
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.25 * 2 ** attempt, 8))


def write_parquet(path, table):
    temporary = path.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    replace_with_retry(temporary, path)


def empty_table():
    ints = ("n_pixels", "n_valid")
    objects = ("field_id", "date", "datetime", "scene_id", "collection", "platform", "orbit_state")
    return pd.DataFrame({c: pd.Series(dtype="object" if c in objects else "int64" if c in ints else "float64")
                         for c in COLUMNS})


# ---------------------------------------------------------------- catalog and token

def search(bbox, start, end, session=None):
    session = session or http_session()
    params = {"collections": COLLECTION, "bbox": ",".join(map(str, bbox)),
              "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200}
    url, method, body, items, seen = PC_URL + "/search", "GET", None, {}, set()
    while url:
        response = session.request(method, url, params=params, json=body, timeout=(15, 90))
        response.raise_for_status()
        data = response.json()
        for item in data.get("features", []):
            item["collection"] = COLLECTION
            items[item["id"]] = item
        link = next((l for l in data.get("links", []) if l["rel"] == "next"), None)
        if not link or str(link) in seen:
            break
        seen.add(str(link))
        url, method, params, body = link["href"], link.get("method", "GET"), None, link.get("body")
    return sorted(items.values(), key=lambda i: (i["properties"]["datetime"], i["id"]))


_TOKEN = {"token": None, "expires": 0.0}
_LOCK = threading.Lock()


def token():
    with _LOCK:
        if _TOKEN["token"] is None or time.time() > _TOKEN["expires"] - 300:
            with http_session() as session:
                response = session.get(TOKEN_URL, timeout=(15, 60))
                response.raise_for_status()
                data = response.json()
            try:
                expires = datetime.fromisoformat(data["msft:expiry"].replace("Z", "+00:00")).timestamp()
            except (KeyError, ValueError):
                expires = time.time() + 1800
            _TOKEN.update(token=data["token"], expires=expires)
        return _TOKEN["token"]


def acquisition(item):
    p = item["properties"]
    return {"date": p["datetime"][:10], "datetime": p["datetime"][:19], "platform": p.get("platform", ""),
            "orbit_state": p.get("sat:orbit_state", ""), "relative_orbit": int(p.get("sat:relative_orbit", -1)),
            "absolute_orbit": int(p.get("sat:absolute_orbit", -1))}


# ---------------------------------------------------------------- grid and reads

def scene_grid(block, item):
    """Block window on the item's own 10 m grid (aligned to its proj:transform)."""
    props = item["properties"]
    crs = props.get("proj:epsg")
    affine = props.get("proj:transform")
    if crs is None or affine is None:
        raise ValueError("Item lacks proj:epsg/proj:transform")
    geometries = [project(field["geometry"], 4326, crs) for field in block["fields"]]
    bounds = np.array([g.bounds for g in geometries])
    west, south = bounds[:, :2].min(axis=0)
    east, north = bounds[:, 2:].max(axis=0)
    res, x0, y0 = affine[0], affine[2], affine[5]
    west = x0 + math.floor((west - x0) / res) * res
    north = y0 + math.ceil((north - y0) / res) * res
    width, height = max(1, math.ceil((east - west) / res)), max(1, math.ceil((north - south) / res))
    if width * height > 4_000_000:
        raise ValueError("Block window exceeds 4 million pixels")
    return crs, from_origin(west, north, res, res), (height, width), geometries


def read_pol(item, pol, grid, retries=3):
    crs, affine, dims = grid[:3]
    href = item["assets"][pol]["href"].split("?")[0]
    for attempt in range(retries):
        try:
            with rasterio.Env(**GDAL_OPTIONS):
                with rasterio.open(href + "?" + token()) as source:
                    with WarpedVRT(source, crs=crs, transform=affine, width=dims[1], height=dims[0],
                                   src_nodata=NODATA, nodata=NODATA, resampling=Resampling.nearest) as vrt:
                        return vrt.read(1, out_dtype="float32")
        except (rasterio.errors.RasterioError, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------- aggregation

def to_db(value):
    value = np.asarray(value, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(value) & (value > 0), 10 * np.log10(value), np.nan)


def field_stats(mask, vv, vh, valid):
    inside = mask & valid
    n_pixels, n_valid = int(mask.sum()), int(inside.sum())
    if n_valid == 0:
        return n_pixels, 0, np.nan, np.nan, np.nan, np.nan
    a, b = vv[inside].astype("float64"), vh[inside].astype("float64")
    return n_pixels, n_valid, float(a.mean()), float(b.mean()), float(np.median(a)), float(np.median(b))


def aggregate_scene(block, item, grid, arrays):
    _, affine, dims, geometries = grid
    selected = [interior_geometry(g, affine, dims) for g in geometries]
    labels = rasterize([(g, i + 1) for i, (g, _) in enumerate(selected)], out_shape=dims, transform=affine,
                       dtype="int32")
    vv, vh = arrays["vv"], arrays["vh"]
    valid = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0) & (vv != NODATA) & (vh != NODATA)
    rows = []
    for index, (geom, _) in enumerate(selected):
        mask = labels == index + 1
        exact = polygon_mask(geom, affine, dims)
        if int(mask.sum()) != int(exact.sum()):
            mask = exact            # overlapping parcels keep their own pixels
        rows.append(field_stats(mask, vv, vh, valid))
    table = pd.DataFrame(rows, columns=["n_pixels", "n_valid", "vv_mean", "vh_mean", "vv_median", "vh_median"])
    table["valid_fraction"] = np.divide(table.n_valid, table.n_pixels, out=np.zeros(len(table)),
                                        where=table.n_pixels > 0)
    table["vv_db"] = to_db(table.vv_mean)
    table["vh_db"] = to_db(table.vh_mean)
    table["vh_vv_db"] = to_db(table.vh_mean / table.vv_mean)
    table["field_id"] = [f["field_id"] for f in block["fields"]]
    for key, value in acquisition(item).items():
        table[key] = value
    table["scene_id"] = item["id"]
    table["collection"] = item["collection"]
    return table[COLUMNS], {"inset_fields": sum(inset for _, inset in selected),
                            "zero_valid_fields": int((table.n_valid == 0).sum())}


def resolve_duplicates(table):
    """One row per field and pass (absolute orbit): the slice with the most valid pixels."""
    if table.empty:
        return table.copy()
    return (table.sort_values(["field_id", "absolute_orbit", "n_valid", "scene_id"],
                              ascending=[True, True, False, True], kind="stable")
            .drop_duplicates(["field_id", "absolute_orbit"]).sort_values(["field_id", "datetime"])
            .reset_index(drop=True))


# ---------------------------------------------------------------- blocks and run

def process_block(block, args, pool):
    path = args.out / "blocks" / (block["id"] + ".parquet")
    status_path = path.with_suffix(".json")
    cache_path = args.out / "catalog" / (block["id"] + ".json")
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get("status") == "complete" and path.exists():
        LOG.info("Skipping complete block %s", block["id"])
        return previous
    started = time.monotonic()
    if cache_path.exists():
        items = json.loads(cache_path.read_text())["items"]
    else:
        items = search(block["bbox"], args.start_date, args.end_date)
        items = [i for i in items if any(shape(i["geometry"]).intersects(f["geometry"]) for f in block["fields"])]
        atomic_json(cache_path, {"items": items})
    skipped = {i["id"]: "missing vv/vh asset" for i in items if not all(p in i.get("assets", {}) for p in POLS)}
    items = [i for i in items if i["id"] not in skipped]
    completed = set(previous.get("completed_scene_ids", [])) if path.exists() else set()
    frames = [pd.read_parquet(path)] if path.exists() else []
    diagnostics = previous.get("diagnostics", {}) if path.exists() else {}
    errors = {}
    pending = [i for i in items if i["id"] not in completed]
    LOG.info("Block %s: %d fields, %d scenes, %d pending", block["id"], len(block["fields"]), len(items), len(pending))
    for offset in range(0, len(pending), args.scene_batch):
        batch = []
        for item in pending[offset:offset + args.scene_batch]:
            try:
                grid = scene_grid(block, item)
                batch.append((item, grid, {p: pool.submit(read_pol, item, p, grid) for p in POLS}))
            except Exception as exc:
                errors[item["id"]] = f"{type(exc).__name__}: {exc}"
        for item, grid, futures in batch:
            arrays, failure = {}, None
            for pol, future in futures.items():
                try:
                    arrays[pol] = future.result()
                except Exception as exc:
                    failure = exc
            try:
                if failure is not None:
                    raise failure
                rows, diagnostic = aggregate_scene(block, item, grid, arrays)
                frames.append(rows)
                diagnostics[item["id"]] = diagnostic
                completed.add(item["id"])
            except Exception as exc:
                errors[item["id"]] = f"{type(exc).__name__}: {exc}"
                LOG.warning("Scene %s failed: %s", item["id"], exc)
        LOG.info("Block %s: processed %d/%d scenes (%d failed)", block["id"],
                 min(offset + args.scene_batch, len(pending)), len(pending), len(errors))
        table = resolve_duplicates(pd.concat(frames, ignore_index=True)) if frames else empty_table()
        write_parquet(path, table)
        frames = [table]
        atomic_json(status_path, {"status": "running", "completed_scene_ids": sorted(completed),
                                  "diagnostics": diagnostics, "scene_errors": errors})
    table = resolve_duplicates(pd.concat(frames, ignore_index=True)) if frames else empty_table()
    write_parquet(path, table)
    status = {"status": "partial" if errors else "complete", "field_count": len(block["fields"]),
              "field_ids": [f["field_id"] for f in block["fields"]], "bbox": block["bbox"],
              "scenes_found": len(items), "scenes_succeeded": len(completed),
              "completed_scene_ids": sorted(completed), "scene_errors": errors, "skipped_items": skipped,
              "diagnostics": diagnostics, "rows": len(table),
              "wall_seconds": previous.get("wall_seconds", 0) + time.monotonic() - started}
    atomic_json(status_path, status)
    return status


def run(args):
    start = time.monotonic()
    args.start_date = date.fromisoformat(f"{args.year}-{args.start}").isoformat()
    args.end_date = date.fromisoformat(f"{args.year}-{args.end}").isoformat()
    if args.start_date > args.end_date:
        raise ValueError("Start must precede end within the selected year")
    if args.workers < 1 or args.scene_batch < 1 or not 0 < args.block_size <= 6000:
        raise ValueError("Workers/batch must be positive; block size must be in (0, 6000]")
    if not 0 <= args.min_valid_fraction <= 1:
        raise ValueError("Invalid min_valid_fraction")
    args.out.mkdir(parents=True, exist_ok=True)
    for sub in ("blocks", "catalog"):
        (args.out / sub).mkdir(exist_ok=True)
    fields = load_fields(args.fields, args.year)
    # --blocks only limits which blocks this invocation processes; it never changes a row.
    parameters = {k: v for k, v in vars(args).items() if k not in ("fields", "out", "workers", "scene_batch", "blocks")}
    source_hash = fingerprint(Path(__file__).parent.parent)
    signature = hashlib.sha256(args.fields.read_bytes() + json.dumps(parameters, sort_keys=True).encode()
                               + source_hash.encode()).hexdigest()
    manifest_path = args.out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous.get("signature") != signature:
        raise ValueError("Output belongs to different inputs/options/code; use a new --out directory")
    blocks = make_blocks(fields, args.block_size)
    selected = [b for b in blocks if not args.blocks or b["id"] in args.blocks]
    manifest = {"signature": signature, "source_sha256": source_hash, "parameters": parameters,
                "product": "sentinel-1-rtc per-field backscatter", "fields_file": str(args.fields.resolve()),
                "workers": args.workers, "scene_batch": args.scene_batch,
                "started_at": previous["started_at"] if previous else utc_now(), "attempt_started_at": utc_now(),
                "ended_at": None, "status": "running", "input_fields": len(fields), "block_count": len(blocks),
                "invalid_fields": {f["field_id"]: f["error"] for f in fields if f.get("error")},
                "blocks": dict(previous.get("blocks", {})) if previous else {}}
    atomic_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for block in selected:
            try:
                status = process_block(block, args, pool)
            except Exception as exc:
                LOG.exception("Block %s failed", block["id"])
                status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "field_ids": [f["field_id"] for f in block["fields"]]}
            manifest["blocks"][block["id"]] = status
            atomic_json(manifest_path, manifest)
    tables = [pd.read_parquet(args.out / "blocks" / (b["id"] + ".parquet")) for b in blocks
              if (args.out / "blocks" / (b["id"] + ".parquet")).exists()]
    table = resolve_duplicates(pd.concat(tables, ignore_index=True)) if tables else empty_table()
    write_parquet(args.out / "timeseries.parquet", table)
    usable = table.rename(columns={"n_valid": "n_clear", "valid_fraction": "clear_fraction"})
    coverage = coverage_table([f for f in fields if not args.blocks or any(f in b["fields"] for b in selected)],
                              usable, args.start_date, args.end_date, 1, args.min_valid_fraction)
    coverage = coverage.rename(columns={"usable_dates": "usable_passes"})
    coverage.to_csv(args.out / "fields_coverage.csv", index=False)
    done = [manifest["blocks"].get(b["id"], {}).get("status") for b in blocks]
    incomplete = bool(manifest["invalid_fields"]) or any(s != "complete" for s in done)
    manifest.update({"status": "partial" if incomplete else "complete", "ended_at": utc_now(),
                     "attempt_wall_seconds": time.monotonic() - start, "rows": len(table),
                     "unique_scenes": int(table.scene_id.nunique()),
                     "fields_with_usable_passes": int((coverage.usable_passes > 0).sum()),
                     "scene_failures": sum(len(b.get("scene_errors", {})) for b in manifest["blocks"].values())})
    atomic_json(manifest_path, manifest)
    LOG.info("%s: %d rows, %d/%d fields with passes, %.1f seconds", manifest["status"], len(table),
             manifest["fields_with_usable_passes"], len(coverage), manifest["attempt_wall_seconds"])
    return 2 if incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start", default="04-01", help="Inclusive MM-DD")
    parser.add_argument("--end", default="11-30", help="Inclusive MM-DD")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scene-batch", type=int, default=6)
    parser.add_argument("--block-size", type=float, default=6000)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--blocks", nargs="*", default=[], help="Only these block ids (rows unaffected)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
