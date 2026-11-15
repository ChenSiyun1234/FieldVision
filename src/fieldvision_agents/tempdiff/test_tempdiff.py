"""TempDiff-US on synthetic seasons (no data files, no network)."""

import numpy as np
import pandas as pd

from .tempdiff import RULE, metrics, score_field, whittaker


def season(drop_doy, year=2022, step=5):
    doy = np.arange(95, 330, step)
    dates = [pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(d) - 1) for d in doy]
    lswi = np.where(doy < drop_doy, 0.35, -0.10)
    ndvi = np.where(doy < drop_doy, 0.85, 0.25)
    return dates, lswi, ndvi


def test_whittaker_keeps_a_straight_line():
    y = np.linspace(0, 1, 15)
    assert np.allclose(whittaker(y, np.ones(15)), y, atol=1e-6)


def test_early_harvest_is_silage_late_harvest_is_grain():
    silage = score_field(*season(drop_doy=245), year=2022)
    grain = score_field(*season(drop_doy=295), year=2022)
    assert silage["call"] == "silage" and silage["lswi_on_date"] < RULE["lswi_threshold"], silage
    assert grain["call"] == "grain" and grain["lswi_on_date"] > RULE["lswi_threshold"], grain
    assert not silage["low_evidence"] and "harvested" in silage["reason"]


def test_too_few_observations_make_no_call():
    dates, lswi, ndvi = season(245)
    lswi = np.full(len(lswi), np.nan)
    lswi[:2] = 0.3
    out = score_field(dates, lswi, ndvi, 2022)
    assert out["call"] is None and out["low_evidence"]


def test_metrics_count_no_call_as_grain():
    df = pd.DataFrame({"label": ["silage", "silage", "grain", "grain"],
                       "call": ["silage", None, "grain", "silage"],
                       "low_evidence": [False, True, False, False], "score": [0.2, np.nan, -0.3, 0.1]})
    m = metrics(df)
    assert (m["tp"], m["fn"], m["tn"], m["fp"], m["correct"]) == (1, 1, 1, 1, 2)


def run_all():
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS ", name)
            n += 1
    print(f"\n{n} tests passed")
    return n
