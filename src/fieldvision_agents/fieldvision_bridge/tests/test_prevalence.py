"""Prior silage share and the probability shift."""

import tempfile
from pathlib import Path

import numpy as np

from ..prevalence import county_corn_acres, shift_probability, silage_share_prior, simulate_range
from .fakes import make_quickstats


def test_shift_math():
    assert np.allclose(shift_probability([0.3, 0.9], 0.5), [0.3, 0.9])     # same balance: unchanged
    assert np.isclose(shift_probability([0.9], 0.1)[0], 0.5)               # odds 9 x 1/9 = 1
    assert shift_probability([0.2], 0.1)[0] < 0.2


def test_prior_uses_county_census_before_target_year():
    with tempfile.TemporaryDirectory() as tmp:
        csv = make_quickstats(Path(tmp) / "qs.csv", county_rows=[(2017, 900, 100), (2022, 500, 500)])
        prior = silage_share_prior(csv, "24033", 2022)
    assert prior["level"] == "county" and np.isclose(prior["share"], 0.1) and "2017" in prior["source"]


def test_prior_falls_back_to_state_then_national():
    with tempfile.TemporaryDirectory() as tmp:
        withheld = make_quickstats(Path(tmp) / "a.csv", county_rows=[(2017, 900, "(D)")],
                                   state_rows=[(2019, 800, 200), (2020, 900, 100), (2021, 700, 300), (2022, 1, 1)])
        state = silage_share_prior(withheld, "24033", 2022)
        empty = silage_share_prior(make_quickstats(Path(tmp) / "b.csv"), "24033", 2022)
    assert state["level"] == "state" and np.isclose(state["share"], 0.2), state
    assert empty["level"] == "national"


def test_range_brackets_the_expected_value():
    area, p = np.array([10.0, 20.0, 30.0]), np.array([0.9, 0.5, 0.1])
    lo, hi = simulate_range(area, p)
    assert lo <= float((area * p).sum()) <= hi


def test_county_corn_acres_before_target_year():
    with tempfile.TemporaryDirectory() as tmp:
        both = make_quickstats(Path(tmp) / "a.csv", county_rows=[(2017, 900, 100), (2022, 5, 5)])
        withheld = make_quickstats(Path(tmp) / "b.csv", county_rows=[(2017, 900, "(D)")])
        corn = county_corn_acres(both, "24033", 2022)
        grain_only = county_corn_acres(withheld, "24033", 2022)
        none = county_corn_acres(make_quickstats(Path(tmp) / "c.csv"), "24033", 2022)
    assert corn["acres"] == 1000 and corn["year"] == 2017
    assert grain_only["acres"] == 900 and "grain" in grain_only["source"] and none is None
