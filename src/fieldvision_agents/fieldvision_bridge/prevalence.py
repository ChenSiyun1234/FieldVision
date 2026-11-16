"""County silage-share prior and the probability shift that uses it.

A model calibrated on 50/50 samples says p for a field. If the county's silage share is pi,
Bayes' rule gives the field's probability there by scaling the odds:
    odds' = odds(p) * [pi / (1 - pi)] / [0.5 / 0.5]
The prior is taken from years BEFORE the target year, so the check against that year's
USDA figures stays independent.
"""

import math

import numpy as np
import pandas as pd

from .config import FALLBACK_PRIOR, RANGE_DRAWS, RANGE_SEED, TRAINING_PRIOR


def shift_probability(p, prior, training_prior=TRAINING_PRIOR):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    ratio = (prior / (1 - prior)) / (training_prior / (1 - training_prior))
    odds = p / (1 - p) * ratio
    return odds / (1 + odds)


def _number(v):
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return math.nan            # "(D)" and other withheld codes stay missing, never zero


def _shares_by_year(rows):
    """Silage share of grain + silage harvested acres, per year where both are published."""
    wide = rows.pivot_table(index="year", columns="crop_use", values="v", aggfunc="first")
    if not {"grain", "silage"} <= set(wide.columns):
        return pd.Series(dtype=float)
    wide = wide.dropna(subset=["grain", "silage"])
    wide = wide[(wide["grain"] + wide["silage"]) > 0]
    return (wide["silage"] / (wide["grain"] + wide["silage"])).sort_index()


def _harvested(csv_path, year):
    """Quick Stats harvested grain and silage acres for years before the target year."""
    df = pd.read_csv(csv_path, dtype=str)
    df = df[(df["measurement"] == "harvested") & (df["domain"] == "TOTAL") &
            (df["reference_period"] == "YEAR") & df["crop_use"].isin(["grain", "silage"])].copy()
    df["v"] = df["value"].map(_number)
    df["year"] = df["year"].astype(int)
    return df[df["year"] < int(year)]


def _county_rows(df, fips):
    fips = str(fips).zfill(5)
    return df[(df["state_ansi"].fillna("").str.zfill(2) == fips[:2]) & (df["aggregation_level"] == "COUNTY") &
              (df["county_ansi"].fillna("").str.zfill(3) == fips[2:])]


def county_corn_acres(csv_path, fips, year):
    """The county's harvested corn (grain + silage) in its latest published year before the target year.

    A withheld silage figure leaves grain alone, a lower bound. None when the county has no figure at all.
    """
    c = _county_rows(_harvested(csv_path, year), fips).dropna(subset=["v"])
    if c.empty:
        return None
    y = int(c["year"].max())
    latest = c[c["year"] == y].groupby("crop_use")["v"].first()
    parts = " + ".join(latest.index)
    return {"acres": float(latest.sum()), "year": y,
            "source": f"USDA NASS {y}, county {str(fips).zfill(5)} harvested corn ({parts})"}


def silage_share_prior(csv_path, fips, year, recent_state_years=3):
    """Expected silage share for a county-year, from Quick Stats years before the target year.

    1. the county's latest Census share (2012/2017/2022) before the target year;
    2. else the state's mean SURVEY share over the last recent_state_years years before it;
    3. else the national fallback.
    """
    df = _harvested(csv_path, year)
    fips = str(fips).zfill(5)
    state = fips[:2]
    same_state = df["state_ansi"].fillna("").str.zfill(2) == state

    c = _county_rows(df, fips)
    c = c[c["source"] == "CENSUS"]
    shares = _shares_by_year(c)
    if len(shares):
        y = int(shares.index[-1])
        return {"share": float(shares.iloc[-1]), "level": "county",
                "source": f"USDA NASS Census {y}, county {fips} silage share of harvested corn"}

    s = df[same_state & (df["aggregation_level"] == "STATE") & (df["source"] == "SURVEY")]
    shares = _shares_by_year(s).tail(recent_state_years)
    if len(shares):
        years = [int(y) for y in shares.index]
        return {"share": float(shares.mean()), "level": "state",
                "source": f"USDA NASS survey, state {state} mean silage share {years[0]}-{years[-1]} "
                          f"(county figure withheld or missing)"}

    share, source = FALLBACK_PRIOR
    return {"share": share, "level": "national", "source": source}


def simulate_range(area, p, draws=RANGE_DRAWS, seed=RANGE_SEED, q=(5, 95)):
    """5th-95th percentile of silage acres when each field is silage with its own probability."""
    area, p = np.asarray(area, float), np.asarray(p, float)
    if not len(p):
        return 0.0, 0.0
    sims = (np.random.default_rng(seed).random((draws, len(p))) < p) @ area
    lo, hi = np.percentile(sims, q)
    return float(lo), float(hi)
