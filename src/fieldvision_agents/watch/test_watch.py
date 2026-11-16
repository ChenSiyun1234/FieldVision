"""Offline tests for the watch loop: no network, no key. Run: python -m fieldvision_agents.watch.test_watch"""
import tempfile
from pathlib import Path

import pandas as pd

from fieldvision_agents.gap.gap import gap_table
from fieldvision_agents.gap.test_gap import AREA, ROWS
from fieldvision_agents.watch.watch import (chart, cutoffs, decide, first_within, new_look_share,
                                            step_metrics, truncate_county)


def test_cutoffs():
    assert cutoffs("2022-08-01", "2022-08-16") == ["2022-08-01", "2022-08-06", "2022-08-11", "2022-08-16"]
    assert cutoffs("2022-08-01", "2022-08-13")[-1] == "2022-08-13"      # the end date is always a step


def _ts():
    return pd.DataFrame({"field_id": ["a", "a", "b", "b", "c"],
                         "date": ["2022-08-02", "2022-08-09", "2022-08-02", "2022-08-09", "2022-08-09"],
                         "clear_fraction": [1.0, 0.2, 1.0, 1.0, 0.9]})


def test_new_looks_and_decision():
    ts = _ts()
    assert new_look_share(ts, None, "2022-08-05", 4) == 0.5             # a and b seen clearly, of 4 fields
    assert new_look_share(ts, "2022-08-05", "2022-08-10", 4) == 0.5     # b and c; a was cloudy
    assert new_look_share(ts, "2022-08-10", "2022-08-15", 4) == 0.0
    assert decide(0.5) == "classify" and decide(0.0) == "skip" and decide(0.05) == "classify"


def test_step_metrics():
    m = step_metrics(gap_table(ROWS, AREA))       # judged: c1 silage, c2 grain, w1 grain; three not judged
    assert m["silage_at_least_pct"] == 16.7 and m["grain_found_pct"] == 33.3
    assert m["not_judged_pct"] == 50.0 and m["silage_at_most_pct"] == 66.7 and m["fields_judged"] == 3


def test_first_within():
    rows = [{"date": d, "silage_at_least_pct": lo, "silage_at_most_pct": hi}
            for d, lo, hi in (("a", 0, 100), ("b", 10, 25), ("c", 10, 40), ("d", 12, 30), ("e", 14, 20))]
    assert first_within(rows, 20) == "d"          # b was narrow but c widened again
    assert first_within(rows, 10) == "e" and first_within(rows, 5) is None


def test_truncate_and_chart():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        _ts().to_parquet(src / "timeseries.parquet", index=False)
        (src / "fields.geojson").write_text("{}")
        dst = truncate_county(src, Path(d) / "dst", "2022-08-05")
        assert len(pd.read_parquet(dst / "timeseries.parquet")) == 2 and (dst / "fields.geojson").exists()
        rows = [{"date": f"2022-09-{i:02d}", "silage_at_least_pct": i, "silage_at_most_pct": 100 - 3 * i,
                 "call_agent": i % 2 == 0} for i in range(1, 21, 5)]
        s = {"fips": "00000", "year": 2022, "range_within_20_points_from": None, "range_within_10_points_from": None}
        p = chart(rows, s, Path(d) / "c.png", reference=("Census", 50.0))
        assert Path(p).stat().st_size > 5000


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} passed")
