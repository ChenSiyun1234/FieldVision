import numpy as np
import pandas as pd
import pytest

from fieldvision.features import indices
from fieldvision.features.build import build_features
from fieldvision.features.loading import load_timeseries
from fieldvision.features.season import (FeatureConfig, REASON_NAMES, field_features,
                                         is_feature_column, signal_features)


def test_indices_nan_not_zero():
    assert np.isnan(indices.ndvi(0.0, 0.0))
    assert np.isnan(indices.ndvi(np.nan, 0.1))
    assert indices.ndvi(0.5, 0.1) == pytest.approx(0.4 / 0.6)
    assert np.isnan(indices.evi(0.0, 0.0, 1.0))     # denominator 1 - 7.5 < 0


def test_harvest_interval_contains_true_date(synth, synth_features):
    fields, _ = synth
    X, _ = synth_features
    found = np.isfinite(X["harv_lo_doy"].to_numpy())
    assert found.mean() > 0.8
    t = fields["true_harvest_doy"].to_numpy()[found]
    lo = X["harv_lo_doy"].to_numpy()[found]
    hi = X["harv_hi_doy"].to_numpy()[found]
    inside = (lo <= t + 0.5) & (t <= hi + 0.5)
    assert inside.mean() > 0.85, inside.mean()
    # silage detected earlier than grain on average
    cls = fields["true_class"].to_numpy()[found]
    mid = X["harv_mid_doy"].to_numpy()[found]
    assert np.median(mid[cls == "silage"]) < np.median(mid[cls == "grain"]) - 15


def test_empty_and_all_cloudy_fields_give_nan_with_reasons():
    cfg = FeatureConfig()
    vals, reasons = field_features(np.array([]), np.array([]), np.array([]), np.array([]),
                                   np.zeros((0, 10)), cfg)
    assert vals["q_n_rows"] == 0 and np.isnan(vals["harv_mid_doy"])
    assert set(reasons.values()) == {"no_timeseries"}
    doy = np.arange(150, 300, 10.0)
    n = len(doy)
    bands = np.full((n, 10), np.nan)
    vals, reasons = field_features(doy, np.full(n, 8), np.full(n, 1), np.zeros(n), bands, cfg)
    assert vals["q_n_rows"] == n and vals["q_n_usable"] == 0
    assert np.isnan(vals["pre_ndmi"])
    assert all(reasons[k] == "no_usable_obs" for k in REASON_NAMES)


def test_build_features_keeps_fields_without_timeseries(synth):
    fields, ts = synth
    f = fields.head(5).copy()
    extra = f.iloc[[0]].copy()
    extra["field_id"] = "nothing_here"
    f = pd.concat([f, extra], ignore_index=True)
    X, info = build_features(ts[ts.field_id.isin(f.field_id)], f)
    assert list(X["field_id"]) == list(f["field_id"])
    row = X.iloc[-1]
    assert row["q_n_rows"] == 0 and bool(row["low_evidence"])
    assert row["reason_harvest"] == "no_timeseries"
    assert info["n_fields_without_timeseries"] == 1


def test_scaled_integer_bands_are_detected(tmp_path, synth):
    fields, ts = synth
    sub = ts[ts.field_id.isin(fields.field_id.head(3))].copy()
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    sub[bands] = sub[bands] * 10000
    sub.to_parquet(tmp_path / "x.parquet", index=False)
    with pytest.warns(UserWarning):
        back, notes = load_timeseries(tmp_path / "x.parquet")
    assert np.nanmax(back[bands].to_numpy()) < 2
    assert any("10000" in n for n in notes)


def test_feature_selection_excludes_ids_and_flags(synth_features):
    X, _ = synth_features
    cols = signal_features(X.columns, "all")
    for bad in ("field_id", "low_evidence", "low_evidence_reason"):
        assert bad not in cols
        assert not is_feature_column(bad)
    assert not any(c.startswith(("q_", "reason_")) for c in cols)
    rel = signal_features(X.columns, "relative")
    assert "harv_days_since_peak" in rel and "harv_mid_doy" not in rel
    cal = signal_features(X.columns, "calendar")
    assert "harv_mid_doy" in cal and "harv_days_since_peak" not in cal


def test_single_bare_soil_drop_followed_by_catch_crop_is_unconfirmed_harvest():
    from fieldvision.features.season import detect_harvest
    t = np.array([150, 180, 200, 220, 230, 265, 290, 310.0])
    ndvi = np.array([0.3, 0.7, 0.85, 0.86, 0.8, 0.30, 0.75, 0.8])
    base, amp = 0.2, 0.66
    # bare soil after the cut: NDMI below zero -> accepted, unconfirmed
    ndmi = np.array([0.0, 0.3, 0.4, 0.4, 0.35, -0.05, 0.25, 0.3])
    i_lo, i_hi, confirmed, reason = detect_harvest(t, ndvi, 3, base, amp, FeatureConfig(), ndmi)
    assert (t[i_lo], t[i_hi], confirmed, reason) == (230, 265, False, "single_low_obs")
    # same NDVI dip but wet canopy (e.g. a shadow): rejected
    ndmi_wet = ndmi.copy()
    ndmi_wet[5] = 0.3
    assert detect_harvest(t, ndvi, 3, base, amp, FeatureConfig(), ndmi_wet)[3] == "no_sustained_drop"


def test_double_crop_picks_the_longer_green_cycle():
    from fieldvision.features.season import find_peak
    t = np.arange(150, 300, 5.0)
    # a short early forage crop (green 160-180), bare soil, then a long corn season (205-270)
    ndvi = np.where((t >= 160) & (t <= 180), 0.88, 0.15)
    ndvi = np.where((t >= 205) & (t <= 270), 0.85, ndvi)
    ndmi = np.where(ndvi > 0.5, 0.35, -0.05)
    vals, i, reason = find_peak(t, ndvi, FeatureConfig(), ndmi)
    assert reason == "ok" and 205 <= vals["peak_doy"] <= 270
    assert vals["q_n_green_cycles"] == 2
