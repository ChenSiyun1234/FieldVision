"""Time-series augmentation for training fields, applied before features are computed.

Three kinds, each drawn once per (field, copy), magnitudes fixed in advance:
  shift  move the whole season by a whole number of days, uniform in [-21, +21],
         so a model cannot lean on the absolute day of year
  thin   keep each observation with probability 0.7, then drop every observation in
         one random window of 10-20 days inside the field's own observed span
         (cloud gaps, a different revisit between imagery sources)
  gain   per band, reflectance * g + o with g uniform in [0.95, 1.05] and o uniform in
         [-0.01, +0.01], clipped to [0, 1] (processing differences between sources;
         the indices computed from the bands move accordingly)

A copy gets field_id "<original>~a<k>" and the original's labels, block and country,
so it always lands on the same side of every split as its original. Only the
training part of a split may use copies; test fields are scored on their real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KINDS = ("shift", "thin", "gain")
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SHIFT_DAYS = 21
KEEP_PROB = 0.7
GAP_DAYS = (10, 20)
GAIN = (0.95, 1.05)
OFFSET = 0.01
SEP = "~a"


def copy_id(field_id, k):
    return f"{field_id}{SEP}{k}"


def parent_id(copy):
    return str(copy).split(SEP)[0]


def augment_timeseries(ts, kinds, copies=3, seed=0):
    """`copies` augmented copies of every field in `ts`, each with every kind in `kinds`."""
    kinds = tuple(kinds)
    bad = set(kinds) - set(KINDS)
    if bad:
        raise ValueError(f"unknown augmentation {bad}")
    rng = np.random.default_rng(seed)
    ts = ts.reset_index(drop=True)
    date = pd.to_datetime(ts["date"])
    fid = ts["field_id"].astype(str).to_numpy()
    fields, inv = np.unique(fid, return_inverse=True)
    out = []
    for k in range(copies):
        part = ts.copy()
        d = date.copy()
        keep = np.ones(len(ts), bool)
        if "shift" in kinds:
            shift = rng.integers(-SHIFT_DAYS, SHIFT_DAYS + 1, size=len(fields))
            d = d + pd.to_timedelta(shift[inv], unit="D")
        if "thin" in kinds:
            keep &= rng.random(len(ts)) < KEEP_PROB
            dn = date.to_numpy().astype("datetime64[D]").astype(np.int64)
            lo = pd.Series(dn).groupby(inv).min().to_numpy()
            hi = pd.Series(dn).groupby(inv).max().to_numpy()
            length = rng.integers(GAP_DAYS[0], GAP_DAYS[1] + 1, size=len(fields))
            start = lo + (rng.random(len(fields)) * np.maximum(hi - lo - length, 0)).astype(np.int64)
            keep &= ~((dn >= start[inv]) & (dn < start[inv] + length[inv]))
        if "gain" in kinds:
            g = rng.uniform(*GAIN, size=(len(fields), len(BANDS)))
            o = rng.uniform(-OFFSET, OFFSET, size=(len(fields), len(BANDS)))
            vals = part[BANDS].to_numpy(float)
            part[BANDS] = np.clip(vals * g[inv] + o[inv], 0.0, 1.0)
        part["date"] = d.dt.strftime("%Y-%m-%d").astype(ts["date"].dtype)
        part["field_id"] = [copy_id(f, k) for f in fid]
        out.append(part[keep])
    return pd.concat(out, ignore_index=True)


def augment_fields(fields, copies=3):
    """Field rows for the copies: same attributes, new field_id."""
    parts = []
    for k in range(copies):
        f = fields.copy()
        f["field_id"] = [copy_id(x, k) for x in fields["field_id"].astype(str)]
        parts.append(f)
    return pd.concat(parts, ignore_index=True)
