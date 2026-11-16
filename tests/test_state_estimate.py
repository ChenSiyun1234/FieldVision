import numpy as np
import pandas as pd

from fieldvision.model.state_estimate import block_bootstrap, estimates, to_acres


def _sample(seed=0, n_blocks=40, per_block=30, silage=0.15, ref_blocks=10, fpr=0.05, tpr=0.7):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        is_ref = b < ref_blocks
        s = 0.01 if is_ref else silage
        y = rng.random(per_block) < s
        p = np.where(y, rng.beta(5, 2, per_block), rng.beta(2, 5, per_block))
        call = np.where(y, rng.random(per_block) < tpr, rng.random(per_block) < fpr)
        tier = rng.choice(["clear", "wide", "abstain"], size=per_block, p=[0.6, 0.3, 0.1])
        rows.append(pd.DataFrame({"block_id": f"b{b}", "county_fips": "R" if is_ref else "T",
                                  "acres": rng.uniform(5, 30, per_block), "p": p, "call": call,
                                  "tier": tier, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_estimates_and_block_bootstrap():
    df = _sample()
    ref = (df["county_fips"] == "R").to_numpy()
    e = estimates(df, ref, 0.01, 0.7, {"clear": 0.6, "wide": 0.5})
    assert 0 <= e["counted_clear"] <= 1 and 0 <= e["reference_counted_clear"] <= 0.3
    for k in ("reference_calibrated", "local_threshold", "local_flagged", "local_per_tier"):
        assert np.isfinite(e[k]), k
    point, ranges, n = block_bootstrap(df, ref, 0.01, 0.7, {"clear": 0.6, "wide": 0.5}, n_boot=200)
    assert n == 40
    lo, hi = ranges["counted_clear"]
    assert lo <= point["counted_clear"] <= hi and hi - lo < 0.2


def test_reference_too_small_gives_nan_not_error():
    df = _sample(ref_blocks=0)
    ref = (df["county_fips"] == "R").to_numpy()
    e = estimates(df, ref, 0.01, 0.7, {"clear": 0.6, "wide": 0.5})
    assert np.isfinite(e["counted_clear"]) and np.isnan(e["local_threshold"])


def test_to_acres_range_with_double_crops():
    a = to_acres(0.1, 0.08, 0.12, 400_000, 20_000)
    assert a["silage_acres"] == 40_000 and a["grain_acres"] == 360_000
    assert a["share_all_corn_if_double_crop_grain"] < 0.1 < a["share_all_corn_if_double_crop_silage"]


def test_window_share_blind_windows_count_zero_and_upper_borrows():
    from fieldvision.model.windows import window_share
    rng = np.random.default_rng(1)
    n = 300
    drop = np.r_[np.full(100, 240.0), np.full(100, 256.0), np.full(100, 270.0)]
    clear = np.r_[np.ones(100, bool), np.zeros(100, bool), np.ones(100, bool)]   # middle window blind
    call = np.r_[rng.random(100) < 0.5, np.zeros(100, bool), np.zeros(100, bool)]
    acres = np.ones(n)
    out, det = window_share(drop, clear, call, acres)
    assert abs(out["blind_share"] - 1 / 3) < 1e-9
    assert out["window_zero"] < out["window_upper"]
    assert abs(out["window_zero"] - call[:100].mean() / 3) < 1e-9
