"""Harvest-window weighted silage share (headline for Michigan and the 2025 state step, logged 18:54).

A field gets a narrow harvest bracket (clear tier) only when the sky was clear around its cut, so
the clear tier over-represents fields cut in clear spells. Instead of carrying the clear tier's
share to the whole county, every scored field is put in a two-week window of its drop day (the
first image after the cut), whatever its tier; each window's silage share comes from its own
clear-tier fields, and is weighted by all corn acres in the window. A window with fewer than
`min_clear` clear fields is BLIND: the headline counts it as no silage (we count silage only
where the harvest was seen) and the upper figure gives it the previous window's share. Fields with
no drop detected count as grain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WIDTH = 14
MIN_CLEAR = 10


def window_table(drop_doy, clear, call, acres, width=WIDTH, min_clear=MIN_CLEAR):
    drop_doy = np.asarray(drop_doy, float)
    clear, call = np.asarray(clear, bool), np.asarray(call, bool)
    acres = np.asarray(acres, float)
    has = np.isfinite(drop_doy)
    w = np.full(len(drop_doy), np.nan)
    w[has] = np.floor(drop_doy[has] / width)
    rows = []
    for win in sorted(set(w[has])):
        m = w == win
        mc = m & clear
        share = (float(np.sum(acres[mc] * call[mc]) / np.sum(acres[mc]))
                 if mc.sum() >= min_clear and np.sum(acres[mc]) > 0 else np.nan)
        rows.append({"doy_from": int(win * width), "acres_all": float(np.sum(acres[m])),
                     "n_clear": int(mc.sum()), "share": share})
    return pd.DataFrame(rows, columns=["doy_from", "acres_all", "n_clear", "share"])


def window_share(drop_doy, clear, call, acres, width=WIDTH, min_clear=MIN_CLEAR):
    """Returns zero-fill share (headline), previous-window fill (upper figure), next-window fill,
    the blind share of corn area and the no-drop share; all as shares of all scored acres."""
    acres = np.asarray(acres, float)
    total = float(np.sum(acres))
    det = window_table(drop_doy, clear, call, acres, width, min_clear)
    out = {"window_zero": np.nan, "window_upper": np.nan, "window_next": np.nan,
           "blind_share": np.nan, "silage_acres_zero": np.nan, "silage_acres_upper": np.nan}
    if total <= 0:
        return out, det
    s = det["share"]
    zero = s.fillna(0.0)
    upper = s.ffill().bfill().fillna(0.0)
    nxt = s.bfill().fillna(0.0)
    out["silage_acres_zero"] = float((det["acres_all"] * zero).sum())
    out["silage_acres_upper"] = float((det["acres_all"] * upper).sum())
    out["window_zero"] = out["silage_acres_zero"] / total
    out["window_upper"] = out["silage_acres_upper"] / total
    out["window_next"] = float((det["acres_all"] * nxt).sum()) / total
    out["blind_share"] = float(det.loc[s.isna(), "acres_all"].sum()) / total
    out["no_drop_share"] = float(np.sum(acres[~np.isfinite(np.asarray(drop_doy, float))])) / total
    return out, det
