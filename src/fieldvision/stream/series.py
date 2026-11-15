"""Per-field sequences: one step per usable acquisition, padded to a common length.

Numpy only. "Usable" is the feature code's definition
(`features.season.usable_mask`), so every model in the comparison sees the
same dates of the same fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.indices import BANDS
from ..features.season import FeatureConfig, usable_mask

RED, NIR = BANDS.index("B04"), BANDS.index("B08")
GRID_START, GRID_STEP, GRID_BINS = 91, 10, 25          # calendar grid: 1 April + 25 x 10 days
REL_START = -60                                         # relative grid: 60 days before green-up


def build_sequences(ts, field_ids, cfg=FeatureConfig(), max_len=96):
    """Arrays for the given fields, in their order.

    x      (N, L, 10) band medians, 0 where padded
    doy    (N, L) day of year
    valid  (N, L) True for a real acquisition
    date   (N, L) 'YYYY-MM-DD' or ''
    """
    field_ids = np.asarray(field_ids).astype(str)
    index = pd.Series(np.arange(len(field_ids)), index=field_ids)
    t = ts[ts["field_id"].astype(str).isin(index.index)]
    ok = usable_mask(t["clear_fraction"], t[BANDS].to_numpy(), cfg,
                     t["n_clear"] if "n_clear" in t.columns else None)
    t = t[ok].sort_values(["field_id", "date"], kind="stable")
    row = index.loc[t["field_id"].astype(str)].to_numpy()
    pos = t.groupby("field_id", sort=False).cumcount().to_numpy()
    keep = pos < max_len
    row, pos = row[keep], pos[keep]
    length = int(pos.max()) + 1 if len(pos) else 1
    n = len(field_ids)
    x = np.zeros((n, length, len(BANDS)), np.float32)
    doy = np.zeros((n, length), np.float32)
    valid = np.zeros((n, length), bool)
    date = np.full((n, length), "", dtype="<U10")
    when = pd.to_datetime(t["date"])
    x[row, pos] = t[BANDS].to_numpy(np.float32)[keep]
    doy[row, pos] = when.dt.day_of_year.to_numpy()[keep]
    date[row, pos] = when.dt.strftime("%Y-%m-%d").to_numpy()[keep]
    valid[row, pos] = True
    return {"field_id": field_ids, "x": x, "doy": doy, "valid": valid, "date": date}


def greenup_doy(x, doy, valid, window=(160, 280), lookback=120.0, fallback=170.0):
    """Day of year the crop greens up: NDVI rising through half-way between its low
    before the summer peak and that peak.

    Measuring time from this day instead of from 1 January removes the
    planting calendar of the region and year (a field's age, not the date).
    The peak is searched in `window`; the low within `lookback` days before
    it, so a spring cover crop does not count. Uses a 3-point running
    median against single cloudy dates. Fields without data get `fallback`.
    """
    out = np.full(len(x), fallback, dtype=np.float32)
    for i in range(len(x)):
        m = valid[i]
        if m.sum() < 3:
            continue
        d = doy[i, m]
        red, nir = x[i, m, RED], x[i, m, NIR]
        v = (nir - red) / np.maximum(nir + red, 1e-6)
        s = v.copy()
        s[1:-1] = np.median(np.stack([v[:-2], v[1:-1], v[2:]]), axis=0)
        inside = (d >= window[0]) & (d <= window[1])
        peak = int(np.argmax(np.where(inside, s, -np.inf))) if inside.any() else int(np.argmax(s))
        before = (np.arange(len(d)) < peak) & (d >= d[peak] - lookback)
        if not before.any():
            out[i] = d[0]
            continue
        level = 0.5 * (s[before].min() + s[peak])
        below = np.flatnonzero(before & (s < level))
        if not len(below):
            out[i] = d[before][0]
            continue
        j = int(below[-1])
        rise = s[j + 1] - s[j]
        frac = (level - s[j]) / rise if rise > 1e-6 else 0.5
        out[i] = d[j] + float(np.clip(frac, 0.0, 1.0)) * (d[j + 1] - d[j])
    return out


def time_axis(doy, valid, anchor=None):
    """Days fed to the models: from 1 July (calendar) or from each field's green-up."""
    if anchor is None:
        return np.where(valid, doy - 182.0, 0.0).astype(np.float32)
    return np.where(valid, doy - np.asarray(anchor)[:, None], 0.0).astype(np.float32)


def regular_grid(days, valid, start, step=GRID_STEP, bins=GRID_BINS):
    """Index of the acquisition nearest each bin centre, -1 for an empty bin.

    days  (N, L) the time axis (day of year, or days since green-up)
    Returns (N, bins) positions along L. A 3D convolution needs evenly
    spaced frames; this puts the irregular acquisitions on such a grid
    without inventing values for the gaps.
    """
    n = len(days)
    out = np.full((n, bins), -1, dtype=np.int64)
    centre = start + step * (np.arange(bins) + 0.5)
    for i in range(n):
        pos = np.flatnonzero(valid[i])
        if not len(pos):
            continue
        d = days[i, pos]
        b = np.floor((d - start) / step).astype(int)
        for k in np.unique(b[(b >= 0) & (b < bins)]):
            cand = pos[b == k]
            out[i, k] = cand[np.argmin(np.abs(days[i, cand] - centre[k]))]
    return out


def attach_chips(seq, chips):
    """Row of the chip store for every step of every sequence, -1 where padded.

    chips  dict with string arrays `field_id` and `date` (one entry per stored chip).
    Steps without a chip are switched off in `seq["valid"]`, so the models
    that read medians and the models that read pixels use the same dates.
    Returns (obs, n_dropped).
    """
    key = pd.Series(np.arange(len(chips["field_id"])),
                    index=pd.MultiIndex.from_arrays([np.asarray(chips["field_id"]).astype(str),
                                                     np.asarray(chips["date"]).astype(str)]))
    key = key[~key.index.duplicated()]
    n, length = seq["valid"].shape
    fid = np.repeat(seq["field_id"], length)
    want = pd.MultiIndex.from_arrays([fid, seq["date"].ravel()])
    obs = key.reindex(want).to_numpy(dtype=float).reshape(n, length)
    found = np.isfinite(obs) & seq["valid"]
    dropped = int((seq["valid"] & ~found).sum())
    seq["valid"] = found
    return np.where(found, obs, -1).astype(np.int64), dropped
