"""Landsat Collection 2 rows beside the Sentinel-2 series (an option, off by default).

The Landsat extractor writes the same columns as the Sentinel-2 one, with the bands under their
Sentinel-2 names (B02, B03, B04, B08 = B8A = nir08, B11, B12) and the red-edge bands empty. The
module's classifier uses NDVI, NDMI and NDTI only, but the usable-look check wants every band, so
the red-edge bands of Landsat rows are filled by linear interpolation in wavelength between red
(665 nm) and near infrared (865 nm); only red-edge features (NDRE, PSRI) see these values.

`pair_check` compares NDVI and NDMI of the same field seen by both sensors within `max_days`;
`fit_adjustment` fits one line per band (Sentinel-2 = a + b x Landsat) on those pairs, and
`merge` applies it when asked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RED_EDGE = {"B05": 705.0, "B06": 740.0, "B07": 783.0}
ADJUST_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")


def _ndvi(t):
    return (t["B08"] - t["B04"]) / (t["B08"] + t["B04"])


def _ndmi(t):
    return (t["B8A"] - t["B11"]) / (t["B8A"] + t["B11"])


def fill_red_edge(ls):
    ls = ls.copy()
    for band, wl in RED_EDGE.items():
        w = (wl - 665.0) / (865.0 - 665.0)
        ls[band] = ls["B04"] + w * (ls["B8A"] - ls["B04"])
    return ls


def pairs(s2, ls, max_days=2, min_clear=0.9):
    """Same field seen clear by both sensors within `max_days`: one row per Landsat look, with
    the nearest Sentinel-2 look."""
    a = s2[s2["clear_fraction"] >= min_clear].copy()
    b = ls[ls["clear_fraction"] >= min_clear].copy()
    a["t"] = pd.to_datetime(a["date"])
    b["t"] = pd.to_datetime(b["date"])
    a = a.sort_values("t")
    b = b.sort_values("t")
    m = pd.merge_asof(b, a, on="t", by="field_id", tolerance=pd.Timedelta(days=max_days),
                      direction="nearest", suffixes=("_ls", "_s2"))
    return m.dropna(subset=[f"B04_s2", "B04_ls"])


def pair_check(s2, ls, max_days=2):
    m = pairs(s2, ls, max_days)
    out = {"pairs": int(len(m)), "fields": int(m["field_id"].nunique()) if len(m) else 0}
    if not len(m):
        return out, m
    for name, f in (("ndvi", _ndvi), ("ndmi", _ndmi)):
        s2v = f(m.rename(columns=lambda c: c[:-3] if c.endswith("_s2") else c))
        lsv = f(m.rename(columns=lambda c: c[:-3] if c.endswith("_ls") else c))
        d = lsv - s2v
        out[f"{name}_median_diff"] = float(np.nanmedian(d))
        out[f"{name}_mad"] = float(np.nanmedian(np.abs(d - np.nanmedian(d))))
        out[f"{name}_corr"] = float(pd.Series(s2v).corr(pd.Series(lsv)))
    return out, m


def fit_adjustment(m):
    """Sentinel-2 = a + b x Landsat per band, least squares on the pairs."""
    coef = {}
    for band in ADJUST_BANDS:
        x, y = m[f"{band}_ls"].to_numpy(float), m[f"{band}_s2"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 30:
            b, a = np.polyfit(x[ok], y[ok], 1)
            coef[band] = (float(a), float(b))
    return coef


def merge(s2, ls, adjust=None):
    """Union of the two series; Landsat rows get filled red edge and, optionally, the per-band
    adjustment; `collection` keeps the source of each row."""
    ls = ls.copy()
    if adjust:
        for band, (a, b) in adjust.items():
            ls[band] = a + b * ls[band]
    ls = fill_red_edge(ls)
    cols = [c for c in s2.columns if c in ls.columns]
    out = pd.concat([s2, ls[cols]], ignore_index=True)
    return out.sort_values(["field_id", "date"]).reset_index(drop=True)
