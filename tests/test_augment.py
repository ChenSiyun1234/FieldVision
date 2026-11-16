import numpy as np
import pandas as pd
import pytest

from fieldvision.foundation.augment import (BANDS, augment_fields, augment_timeseries, copy_id,
                                            parent_id)


def _ts(n_fields=20, n_dates=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fields):
        dates = pd.date_range("2024-04-01", periods=n_dates, freq="5D")
        for d in dates:
            rows.append({"field_id": f"f{f}", "date": d.strftime("%Y-%m-%d"), "clear_fraction": 1.0,
                         **{b: rng.uniform(0.05, 0.5) for b in BANDS}})
    return pd.DataFrame(rows)


def test_copies_are_named_after_their_original_and_original_is_untouched():
    ts = _ts()
    before = ts.copy()
    a = augment_timeseries(ts, ("shift", "thin", "gain"), copies=3, seed=1)
    pd.testing.assert_frame_equal(ts, before)
    assert set(a["field_id"].map(parent_id)) == set(ts["field_id"])
    assert a["field_id"].nunique() == 3 * ts["field_id"].nunique()
    assert parent_id(copy_id("abc~x", 2)) == "abc~x".split("~a")[0]


def test_shift_moves_a_whole_field_by_one_offset_within_three_weeks():
    ts = _ts()
    a = augment_timeseries(ts, ("shift",), copies=2, seed=3)
    assert len(a) == 2 * len(ts)
    for cid, g in a.groupby("field_id"):
        orig = ts[ts["field_id"] == parent_id(cid)]
        d = (pd.to_datetime(g["date"]).to_numpy() - pd.to_datetime(orig["date"]).to_numpy()) / np.timedelta64(1, "D")
        assert len(set(d)) == 1 and abs(d[0]) <= 21
        assert np.allclose(g[BANDS].to_numpy(), orig[BANDS].to_numpy())


def test_thin_drops_dates_and_leaves_a_gap_of_at_least_ten_days():
    ts = _ts(n_fields=30)
    a = augment_timeseries(ts, ("thin",), copies=1, seed=5)
    assert 0.4 * len(ts) < len(a) < 0.75 * len(ts)
    gaps = a.assign(d=pd.to_datetime(a["date"])).groupby("field_id")["d"].apply(
        lambda s: s.sort_values().diff().dt.days.max())
    assert (gaps >= 10).mean() > 0.9


def test_gain_changes_bands_mildly_and_keeps_them_in_range():
    ts = _ts()
    a = augment_timeseries(ts, ("gain",), copies=1, seed=7)
    ratio = a[BANDS].to_numpy() / ts[BANDS].to_numpy()
    assert (a[BANDS].to_numpy() >= 0).all() and (a[BANDS].to_numpy() <= 1).all()
    assert np.abs(ratio - 1).max() < 0.05 + 0.01 / 0.05 + 1e-9
    assert not np.allclose(ratio, 1)


def test_same_seed_same_copies_and_unknown_kind_rejected():
    ts = _ts()
    a = augment_timeseries(ts, ("shift", "thin", "gain"), 2, seed=9)
    b = augment_timeseries(ts, ("shift", "thin", "gain"), 2, seed=9)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        augment_timeseries(ts, ("rotate",))


def test_augment_fields_keeps_labels_and_blocks():
    f = pd.DataFrame({"field_id": ["a", "b"], "y": [0, 1], "block_id": ["B1", "B2"]})
    af = augment_fields(f, copies=3)
    assert len(af) == 6
    assert (af.groupby(af["field_id"].map(parent_id))["block_id"].nunique() == 1).all()
    assert af.set_index("field_id").loc[copy_id("b", 2), "y"] == 1
