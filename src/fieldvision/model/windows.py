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


def seen_bounds(mid_doy, clear, call, acres, min_clear_area=0.3, min_clear_fields=10, width=WIDTH):
    """Seen / unseen corn and the bounds on the county's silage share (amendment logged 23:1x).

    Every field goes to the two-week window of its harvest-bracket midpoint (fields with no
    harvest found have no window and count as unseen). A window is SEEN when its clear-tier
    fields hold at least `min_clear_area` of its corn area and number at least
    `min_clear_fields`; its silage is then its clear-tier share times all its corn. Lower bound
    = silage in seen windows over all corn; upper bound = lower bound + unseen corn over all corn.
    """
    mid = np.asarray(mid_doy, float)
    clear, call = np.asarray(clear, bool), np.asarray(call, bool)
    acres = np.asarray(acres, float)
    total = float(np.sum(acres))
    has = np.isfinite(mid)
    w = np.full(len(mid), np.nan)
    w[has] = np.floor(mid[has] / width)
    seen_acres = silage = 0.0
    rows = []
    for win in sorted(set(w[has])):
        m = w == win
        mc = m & clear
        a_all, a_clear = float(np.sum(acres[m])), float(np.sum(acres[mc]))
        seen = a_all > 0 and a_clear / a_all >= min_clear_area and mc.sum() >= min_clear_fields
        share = float(np.sum(acres[mc] * call[mc]) / a_clear) if a_clear > 0 else np.nan
        if seen:
            seen_acres += a_all
            silage += share * a_all
        rows.append({"doy_from": int(win * width), "acres_all": a_all, "clear_area_share": a_clear / a_all
                     if a_all else np.nan, "n_clear": int(mc.sum()), "share": share, "seen": bool(seen)})
    out = {"seen_share": seen_acres / total if total else np.nan,
           "lower": silage / total if total else np.nan}
    out["upper"] = out["lower"] + (1 - out["seen_share"]) if total else np.nan
    out["seen_window_share"] = silage / seen_acres if seen_acres else np.nan
    return out, pd.DataFrame(rows)


def seen_field_mask(mid_doy, clear, acres, min_clear_area=0.5, min_clear_fields=10, width=WIDTH):
    """Fields seen clearly: clear tier and in a seen window of the bracket midpoint (23:17)."""
    mid = np.asarray(mid_doy, float)
    clear, acres = np.asarray(clear, bool), np.asarray(acres, float)
    w = np.full(len(mid), np.nan)
    ok = np.isfinite(mid)
    w[ok] = np.floor(mid[ok] / width)
    seen = np.zeros(len(mid), bool)
    for win in np.unique(w[ok]):
        m = w == win
        mc = m & clear
        if acres[m].sum() > 0 and acres[mc].sum() / acres[m].sum() >= min_clear_area \
                and mc.sum() >= min_clear_fields:
            seen |= m
    return seen & clear


def prior_updated_share(p, prior, seen, acres, pi_train=0.5):
    """Version A of the field-level prior update: seen fields take
    sigmoid(logit p + logit prior - logit pi_train), unseen fields take the prior; area mean.
    pi_train = 0.5 for eu2_logistic_v1 (balanced training, Platt-calibrated on that balance)."""
    def logit(x):
        x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    p, prior = np.asarray(p, float), np.broadcast_to(np.asarray(prior, float), np.shape(p))
    seen, acres = np.asarray(seen, bool), np.asarray(acres, float)
    pp = 1 / (1 + np.exp(-(logit(p) + logit(prior) - logit(pi_train))))
    val = np.where(seen, pp, prior)
    ok = np.isfinite(val) & np.isfinite(acres)
    return float(np.sum(acres[ok] * val[ok]) / np.sum(acres[ok])) if ok.any() else float("nan")
