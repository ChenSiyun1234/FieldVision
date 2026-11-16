"""Feature table for many fields at once."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .indices import BANDS
from .season import (FeatureConfig, REASON_NAMES, evidence_flags, field_features,
                     numeric_feature_names)

MAX_SEASON_DOY = 400     # allow a little spill into January for very late harvests


def build_features(ts, fields, cfg=FeatureConfig()):
    """One row of features per field in `fields`, in the same order.

    ts      long time series (see loading.load_timeseries)
    fields  frame with at least field_id; `year` is used to define the season
            and falls back to the most common year in the field's own dates

    Fields without any acquisition still get a row: all NaN, reason
    'no_timeseries', flagged low evidence.

    Returns (features, info). info has the timing used for the response-time
    number in the report.
    """
    t0 = time.perf_counter()
    names = numeric_feature_names(cfg)
    field_ids = fields["field_id"].astype(str).to_numpy()
    pos = {fid: i for i, fid in enumerate(field_ids)}
    values = np.full((len(field_ids), len(names)), np.nan)
    j_rows, j_use, j_use_season = (names.index(k) for k in
                                   ("q_n_rows", "q_n_usable", "q_n_usable_season"))
    values[:, [j_rows, j_use, j_use_season]] = 0.0
    reasons = np.full((len(field_ids), len(REASON_NAMES)), "no_timeseries", dtype=object)

    n_unknown = 0
    n_other_season = 0
    if len(ts):
        ts = ts[ts["field_id"].isin(pos)]
        ts = ts.sort_values(["field_id", "date"], kind="stable")
        fid = ts["field_id"].to_numpy()
        date = ts["date"]
        years = date.dt.year.to_numpy()

        year_of = {}
        if "year" in fields.columns:
            fy = pd.to_numeric(fields["year"], errors="coerce").to_numpy()
            year_of = {f: int(y) for f, y in zip(field_ids, fy) if np.isfinite(y)}
        season_year = np.array([year_of.get(f, -1) for f in fid])
        if (season_year < 0).any():
            mode_year = pd.Series(years).groupby(fid).agg(lambda s: int(s.mode().iloc[0]))
            fill = pd.Series(fid).map(mode_year).to_numpy()
            n_unknown = int(pd.Series(fid[season_year < 0]).nunique())
            season_year = np.where(season_year < 0, fill, season_year)

        jan1 = pd.to_datetime(pd.Series(season_year).astype(str) + "-01-01").to_numpy()
        doy = (date.to_numpy() - jan1) / np.timedelta64(1, "D") + 1.0
        keep = (doy >= 1) & (doy <= MAX_SEASON_DOY)
        n_other_season = int((~keep).sum())

        month = date.dt.month.to_numpy()[keep]
        day = date.dt.day.to_numpy()[keep]
        cf = ts["clear_fraction"].to_numpy(dtype=float)[keep]
        bands = ts[BANDS].to_numpy(dtype=float)[keep]
        n_clear = ts["n_clear"].to_numpy(dtype=float)[keep] if "n_clear" in ts.columns else None
        fid = fid[keep]
        doy = doy[keep]

        starts = np.flatnonzero(np.r_[True, fid[1:] != fid[:-1]])
        ends = np.r_[starts[1:], len(fid)]
        for a, b in zip(starts, ends):
            vals, rs = field_features(doy[a:b], month[a:b], day[a:b], cf[a:b], bands[a:b], cfg,
                                      None if n_clear is None else n_clear[a:b])
            i = pos[fid[a]]
            values[i] = list(vals.values())
            reasons[i] = [rs[k] for k in REASON_NAMES]

    out = pd.DataFrame(values, columns=names)
    out.insert(0, "field_id", field_ids)
    for j, k in enumerate(REASON_NAMES):
        out[k] = reasons[:, j]
    low, why = evidence_flags(out["q_n_usable_season"], out["harv_mid_doy"],
                              out["q_harv_width_days"], out["q_harv_confirmed"], cfg)
    out["low_evidence"] = low
    out["low_evidence_reason"] = why
    out = out.copy()     # defragment

    seconds = time.perf_counter() - t0
    info = {
        "n_fields": int(len(field_ids)),
        "n_fields_without_timeseries": int((out["q_n_rows"] == 0).sum()),
        "n_fields_year_inferred": n_unknown,
        "n_rows_outside_season": n_other_season,
        "seconds": seconds,
        "seconds_per_1000_fields": seconds / max(len(field_ids), 1) * 1000.0,
    }
    return out, info
