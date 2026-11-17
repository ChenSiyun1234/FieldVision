"""One feature row per field-year, from the Sentinel time series.

Harvest is the day NDVI falls through 0.5 after the summer peak. Moisture and
greenness are read on the last clear date before that crossing, and again over
the 30 days after it. NDMI and NDTI are recomputed from B08, B11 and B12 so
BARC and the counties use the same bands (county files store an NDMI from B8A).

The model uses the plant and within-field features. Regrowth, absolute residue,
peak greenness, the calendar harvest day and the radar drop are saved so they
can be inspected, and are not passed to the classifier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Summer peak is searched between these days of year, then the harvest drop after it.
PEAK_START = 150
PEAK_END = 270
# NDVI at the cut, and NDVI that still counts as a full canopy.
CUT_NDVI = 0.5
FULL_CANOPY = 0.8
POST_DAYS = 30
# Bare soil in late spring, the field's own residue baseline.
SPRING_START = 121
SPRING_END = 165
SPRING_NDVI_MAX = 0.35

# Passed to the classifier. Order is the coefficient order.
MODEL_FEATURES = [
    "ndmi_pre",
    "ndmi_drop",
    "pre_ndvi",
    "ndvi_drop",
    "n_decline_obs",
    "days_below08_to_harvest",
    "ndti_delta",
]
# Computed and written, not used for training.
EXTRA_FEATURES = [
    "ndti_post",
    "regrowth",
    "peak_ndvi",
    "harvest_doy",
    "vh_drop",
]
FEATURE_COLUMNS = MODEL_FEATURES + EXTRA_FEATURES


def _index(high: pd.Series, low: pd.Series) -> pd.Series:
    denom = high + low
    out = (high - low) / denom
    return out.where(np.isfinite(denom) & (denom != 0))


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """NDMI from B08 and B11, NDTI from B11 and B12, when the bands are present."""
    out = frame.copy()
    if {"b08", "b11"}.issubset(out.columns):
        out["ndmi"] = _index(out["b08"], out["b11"])
    if {"b11", "b12"}.issubset(out.columns):
        out["ndti"] = _index(out["b11"], out["b12"])
    return out


def _radar_drop(radar: pd.DataFrame, harvest: float) -> float:
    """VH change across harvest on the orbit that visits the field most often."""
    if radar is None or radar.empty or "vh_db" not in radar.columns:
        return np.nan
    counts = radar.groupby("relative_orbit").size()
    if counts.empty:
        return np.nan
    orbit = counts.idxmax()
    passes = radar.loc[radar["relative_orbit"] == orbit].sort_values("doy")
    before = passes.loc[passes["doy"] < harvest, "vh_db"]
    after = passes.loc[passes["doy"] >= harvest, "vh_db"]
    if before.empty or after.empty:
        return np.nan
    return float(after.iloc[0] - before.iloc[-1])


def _one_field(group: pd.DataFrame, radar: pd.DataFrame | None) -> dict | None:
    group = group.sort_values("doy")
    needed = {"doy", "ndvi", "ndmi", "ndti"}
    if not needed.issubset(group.columns):
        return None
    day = group["doy"].to_numpy(dtype="float64")
    ndvi = group["ndvi"].to_numpy(dtype="float64")
    ndmi = group["ndmi"].to_numpy(dtype="float64")
    season = np.flatnonzero((day >= PEAK_START) & (day <= PEAK_END) & np.isfinite(ndvi))
    if season.size == 0:
        return None
    peak_at = season[int(np.argmax(ndvi[season]))]
    later = [i for i in range(peak_at + 1, len(day)) if np.isfinite(ndvi[i]) and ndvi[i] < CUT_NDVI]
    if not later:
        return None
    cut_at = later[0]
    pre_at = cut_at - 1
    if not np.isfinite(ndvi[pre_at]) or ndvi[pre_at] <= ndvi[cut_at]:
        harvest = float(day[cut_at])
    else:
        span = ndvi[pre_at] - ndvi[cut_at]
        harvest = float(day[pre_at] + (ndvi[pre_at] - CUT_NDVI) / span * (day[cut_at] - day[pre_at]))

    after = group[(group["doy"] >= day[cut_at]) & (group["doy"] <= day[cut_at] + POST_DAYS)]
    post_ndvi = float(after["ndvi"].median()) if len(after) else np.nan
    post_ndmi = float(after["ndmi"].median()) if len(after) else np.nan
    post_ndti = float(after["ndti"].median()) if len(after) else np.nan

    left_canopy = [i for i in range(peak_at, cut_at + 1) if np.isfinite(ndvi[i]) and ndvi[i] < FULL_CANOPY]
    if left_canopy:
        leave_day = float(day[left_canopy[0]])
        decline = [i for i in range(left_canopy[0] + 1, cut_at) if np.isfinite(ndvi[i])]
        days_below = harvest - leave_day
        n_decline = len(decline)
    else:
        days_below = np.nan
        n_decline = np.nan

    spring = group[
        (group["doy"] >= SPRING_START) & (group["doy"] <= SPRING_END) & (group["ndvi"] < SPRING_NDVI_MAX)
    ]
    spring_ndti = float(spring["ndti"].median()) if len(spring) else np.nan
    regrowth_window = group[(group["doy"] >= harvest + 14) & (group["doy"] <= harvest + 60)]
    if len(regrowth_window) and np.isfinite(post_ndvi):
        regrowth = max(0.0, float(regrowth_window["ndvi"].max()) - post_ndvi)
    else:
        regrowth = np.nan

    return {
        "peak_ndvi": float(ndvi[peak_at]),
        "harvest_doy": harvest,
        "pre_ndvi": float(ndvi[pre_at]),
        "ndmi_pre": float(ndmi[pre_at]),
        "ndvi_drop": float(ndvi[pre_at]) - post_ndvi,
        "ndmi_drop": float(ndmi[pre_at]) - post_ndmi,
        "ndti_post": post_ndti,
        "ndti_delta": post_ndti - spring_ndti,
        "n_decline_obs": n_decline,
        "days_below08_to_harvest": days_below,
        "regrowth": regrowth,
        "vh_drop": _radar_drop(radar, harvest),
    }


def field_features(sentinel2: pd.DataFrame, sentinel1: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features for every field-year. A field with no harvest drop is left out."""
    optical = _prepare(sentinel2)
    optical["field_id"] = optical["field_id"].astype(str)
    radar = None
    if sentinel1 is not None and len(sentinel1):
        radar = sentinel1.copy()
        radar["field_id"] = radar["field_id"].astype(str)
    rows = []
    for (field_id, year), group in optical.groupby(["field_id", "year"], sort=False):
        passes = None
        if radar is not None:
            passes = radar[(radar["field_id"] == field_id) & (radar["year"] == year)]
        features = _one_field(group, passes)
        if features is None:
            continue
        label = group["label"].iloc[0] if "label" in group.columns else ""
        if pd.isna(label):
            label = ""
        features["field_id"] = field_id
        features["year"] = int(year)
        features["label"] = str(label)
        rows.append(features)
    columns = ["field_id", "year", "label", *FEATURE_COLUMNS]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def load_region(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    optical = pd.read_csv(folder / "s2_timeseries.csv", dtype={"field_id": str})
    radar_path = folder / "s1_timeseries.csv"
    radar = pd.read_csv(radar_path, dtype={"field_id": str}) if radar_path.exists() else None
    return optical, radar


def region_areas(folder: Path) -> pd.DataFrame:
    """Acres per field from field_context.csv. Empty when the file is absent (BARC)."""
    path = folder / "field_context.csv"
    if not path.exists():
        return pd.DataFrame(columns=["field_id", "fips", "area_ha"])
    context = pd.read_csv(path, dtype={"field_id": str, "fips": str})
    keep = [name for name in ("field_id", "fips", "area_ha") if name in context.columns]
    return context[keep].drop_duplicates("field_id")
