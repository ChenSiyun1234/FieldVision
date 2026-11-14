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


def load_fields(paths):
    """Field properties from one or more GeoJSON files (geometry is not needed here).

    Adds `y`: 1.0 silage, 0.0 grain, NaN for unlabelled or unknown labels.
    """
    rows = []
    for path in expand_paths(paths, (".geojson", ".json")):
        with open(path, encoding="utf-8") as fh:
            gj = json.load(fh)
        for feat in gj.get("features", []):
            props = feat.get("properties") or {}
            row = {k: props.get(k) for k in FIELD_PROPS}
            row["fields_file"] = Path(path).name
            rows.append(row)
    if not rows:
        raise ValueError("the field files contain no features")
    df = pd.DataFrame(rows)
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
