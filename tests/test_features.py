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
