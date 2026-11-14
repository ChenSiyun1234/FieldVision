"""Read the field GeoJSON files and the per-field time-series Parquet files."""
from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .indices import BANDS

FIELD_PROPS = ["field_id", "label", "year", "country", "source", "crop_code", "area_ha",
               "block_id", "centroid_lon", "centroid_lat"]
TS_REQUIRED = ["field_id", "date", "clear_fraction"] + BANDS
TS_OPTIONAL = ["scene_id", "collection", "n_pixels", "n_clear"]
LABEL_TO_Y = {"silage": 1.0, "grain": 0.0}      # positive class = silage


def expand_paths(paths, suffixes):
    """Files from a mix of files, directories and glob patterns."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for suf in suffixes:
                out.extend(sorted(p.rglob(f"*{suf}")))
        elif p.exists():
            out.append(p)
        else:
            hits = sorted(Path(h) for h in glob.glob(str(p)))
            if not hits:
                raise FileNotFoundError(f"no file matches {p}")
            out.extend(hits)
    if not out:
        raise FileNotFoundError(f"nothing found under {list(map(str, paths))}")
    return out


VECTOR_SUFFIXES = (".geojson", ".json", ".gpkg", ".shp", ".fgb")


def _read_vector(path):
    """Properties of one vector file as a frame, plus centroid and area when missing."""
    suffix = Path(path).suffix.lower()
    if suffix in (".geojson", ".json"):
        with open(path, encoding="utf-8") as fh:
            gj = json.load(fh)
        rows = []
        for feat in gj.get("features", []):
            props = dict(feat.get("properties") or {})
            props["__geometry"] = feat.get("geometry")
            rows.append(props)
        df = pd.DataFrame(rows)
        need_geom = any(c not in df.columns or df[c].isna().any()
                        for c in ("centroid_lon", "centroid_lat", "area_ha"))
        if len(df) and need_geom and df["__geometry"].notna().any():
            import geopandas as gpd
            from shapely.geometry import shape
            geoms = [shape(g) if g else None for g in df["__geometry"]]
            gdf = gpd.GeoDataFrame(df.drop(columns="__geometry"), geometry=geoms, crs="EPSG:4326")
            return _fill_geometry_props(gdf)
        return df.drop(columns="__geometry", errors="ignore")
    import geopandas as gpd
    return _fill_geometry_props(gpd.read_file(path))


def _fill_geometry_props(gdf):
    """centroid_lon/lat and area_ha from geometry where the attributes are missing."""
    import geopandas as gpd
    if gdf.crs is None:
        warnings.warn("field file has no CRS; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    valid = (gdf.geometry.notna() & ~gdf.geometry.is_empty).to_numpy()
    for col in ("centroid_lon", "centroid_lat", "area_ha"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    if valid.any():
        sub = gdf.loc[valid]
        metric = sub.to_crs(sub.estimate_utm_crs())
        cent = gpd.GeoSeries(metric.geometry.centroid, crs=metric.crs).to_crs("EPSG:4326")
        idx = df.index[valid]
        for col, vals in (("centroid_lon", cent.x.to_numpy()), ("centroid_lat", cent.y.to_numpy()),
                          ("area_ha", metric.geometry.area.to_numpy() / 10000.0)):
            df.loc[idx, col] = df.loc[idx, col].fillna(pd.Series(vals, index=idx))
    return df


def load_fields(paths, rename=None, label_map=None, keep_columns=()):
    """Field properties from one or more vector files (GeoJSON, GeoPackage, ...).

    rename        {our_name: their_name}, e.g. {"label": "harvest", "field_id": "FID"},
                  for files that do not follow the contract
    label_map     {their_value: "silage" | "grain"} applied before validation
    keep_columns  extra columns to carry along (e.g. a county code)

    Missing centroid/area are computed from geometry. A missing block_id
    becomes a rough 10 km grid cell so grouped evaluation still works.
    Adds `y`: 1.0 silage, 0.0 grain, NaN for unlabelled or unknown labels.
    """
    from ..model.splits import grid_block_id
    frames = []
    for path in expand_paths(paths, VECTOR_SUFFIXES):
        part = _read_vector(path)
        if rename:
            part = part.rename(columns={v: k for k, v in rename.items() if v in part.columns})
        if label_map and "label" in part.columns:
            part["label"] = part["label"].map(lambda v: label_map.get(str(v), v))
        sub = pd.DataFrame({k: part[k] if k in part.columns else None for k in FIELD_PROPS},
                           index=part.index)
        for c in keep_columns:
            sub[c] = part[c] if c in part.columns else None
        sub["fields_file"] = Path(path).name
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise ValueError("the field files contain no features")
    no_block = df["block_id"].isna() | (df["block_id"].astype(str).str.len() == 0)
    if no_block.any():
        warnings.warn(f"{int(no_block.sum())} fields have no block_id; using a 10 km grid cell "
                      "from the centroid")
        df.loc[no_block, "block_id"] = grid_block_id(
            pd.to_numeric(df.loc[no_block, "centroid_lon"], errors="coerce"),
            pd.to_numeric(df.loc[no_block, "centroid_lat"], errors="coerce"))
    if df["field_id"].isna().any():
        raise ValueError("some features have no field_id")
    df["field_id"] = df["field_id"].astype(str)
    n_dup = int(df["field_id"].duplicated().sum())
    if n_dup:
        warnings.warn(f"{n_dup} duplicated field_id values in the field files; keeping the first")
        df = df.drop_duplicates("field_id", keep="first")
    label = df["label"].astype("string").str.strip().str.lower()
    df["label"] = label.where(label.isin(list(LABEL_TO_Y)), None).astype(object)
    unknown = sorted(set(label.dropna().unique()) - set(LABEL_TO_Y))
    if unknown:
        warnings.warn(f"labels {unknown} are neither 'silage' nor 'grain'; treated as unlabelled")
    df["y"] = df["label"].map(LABEL_TO_Y).astype(float)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for c in ("country", "block_id"):
        df[c] = df[c].astype("string").fillna("unknown").astype(object)
    return df.reset_index(drop=True)


def load_timeseries(paths, field_ids=None):
    """Long-format time series from one or more Parquet files or a directory of them.

    Returns (frame, notes). Rows for the same field and day (overlapping
    tiles or orbits) are reduced to the one with the highest clear fraction.
    """
    notes = []
    wanted = None if field_ids is None else set(map(str, field_ids))
    frames = []
    for path in expand_paths(paths, (".parquet",)):
        import pyarrow.parquet as pq
        available = pq.read_schema(path).names
        missing = [c for c in TS_REQUIRED if c not in available]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        cols = TS_REQUIRED + [c for c in TS_OPTIONAL if c in available]
        df = pd.read_parquet(path, columns=cols)
        df["field_id"] = df["field_id"].astype(str)
        if wanted is not None:
            df = df[df["field_id"].isin(wanted)]
        frames.append(df)
    ts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TS_REQUIRED)

    date = pd.to_datetime(ts["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    n_bad = int(date.isna().sum())
    if n_bad:
        notes.append(f"{n_bad} rows with unreadable dates dropped")
    ts["date"] = date
    ts = ts[date.notna()]
    for c in ["clear_fraction"] + BANDS:
        ts[c] = pd.to_numeric(ts[c], errors="coerce").astype(float)

    # reflectance should be 0-1; stored as 0-10000 is the usual mistake
    vals = ts[BANDS].to_numpy()
    if np.isfinite(vals).any():
        p99 = float(np.nanpercentile(vals, 99))
        if p99 > 2.0:
            msg = (f"band values look like scaled integers (99th percentile {p99:.0f}); "
                   "divided by 10000. Check the extractor, the contract says 0-1.")
            warnings.warn(msg)
            notes.append(msg)
            ts[BANDS] = ts[BANDS] / 10000.0

    n_before = len(ts)
    ts = ts.sort_values(["field_id", "date", "clear_fraction"], ascending=[True, True, False],
                        na_position="last", kind="stable")
    ts = ts.drop_duplicates(["field_id", "date"], keep="first").reset_index(drop=True)
    if len(ts) < n_before:
        notes.append(f"{n_before - len(ts)} same-day duplicate rows removed "
                     "(kept the one with the highest clear fraction)")
    return ts, notes
