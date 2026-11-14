"""Presto input adapter, encoder and comparison helpers."""
import numpy as np
import pandas as pd
import pytest

from fieldvision.features.indices import BANDS
from fieldvision.foundation.adapter import (OUR_TO_PRESTO, build_presto_inputs, masked_groups,
                                            monthly_composites)
from fieldvision.foundation.presto_bands import BAND_GROUPS, DW_MISSING, PRESTO_BANDS


def _row(fid, date, value, cf=1.0, **over):
    r = {"field_id": fid, "date": pd.Timestamp(date), "clear_fraction": cf, "n_clear": 50}
    for i, b in enumerate(BANDS):
        r[b] = value + 0.01 * i
    r.update(over)
    return r


@pytest.fixture
def small():
    fields = pd.DataFrame({"field_id": ["a", "b", "c"], "year": [2024, 2024, 2024],
                           "centroid_lat": [48.1, 47.5, 46.0], "centroid_lon": [15.2, 14.0, 13.0]})
    rows = [
        _row("a", "2024-04-05", 0.10), _row("a", "2024-04-20", 0.20), _row("a", "2024-04-25", 0.60),
        _row("a", "2024-06-10", 0.30),
        _row("a", "2024-07-01", 0.50, cf=0.3),                 # too cloudy: not usable
        _row("a", "2024-08-01", 0.40, B04=np.nan),             # NaN band: not usable
        _row("a", "2023-09-01", 0.90),                         # other year: ignored
        _row("a", "2024-12-03", 0.90),                         # after the window: ignored
        _row("b", "2024-11-30", 0.05),
        # field c has no rows at all
    ]
    return fields, pd.DataFrame(rows)


def test_monthly_median_and_counts(small):
    fields, ts = small
    vals, n = monthly_composites(ts, fields, 4, 11)
    assert vals.shape == (3, 8, 10) and n.shape == (3, 8)
    # April for field a: median of 0.10, 0.20, 0.60 -> 0.20 (+0.01*i per band)
    np.testing.assert_allclose(vals[0, 0], 0.20 + 0.01 * np.arange(10))
    assert n[0].tolist() == [3, 0, 1, 0, 0, 0, 0, 0]
    assert np.isnan(vals[0, 3]).all() and np.isnan(vals[0, 4]).all()   # Jul cloudy, Aug NaN band
    assert n[1].tolist() == [0] * 7 + [1]                               # Nov 30 is the last step
    assert n[2].sum() == 0 and np.isnan(vals[2]).all()


def test_presto_tensor_layout_scaling_and_masks(small):
    fields, ts = small
    b = build_presto_inputs(ts, fields, 4, 11)
    assert b.x.shape == (3, 8, 17) and b.mask.shape == (3, 8, 17) and len(PRESTO_BANDS) == 17
    assert b.month == 3                                     # April, 0-based
    assert (b.dynamic_world == DW_MISSING).all()
    np.testing.assert_allclose(b.latlons[0], [48.1, 15.2], rtol=1e-6)   # [lat, lon]
    # reflectance goes in unchanged (Presto's DN / 10000 == our 0-1 values)
    for i, ours in enumerate(BANDS):
        k = PRESTO_BANDS.index(OUR_TO_PRESTO[ours])
        assert b.x[0, 0, k] == pytest.approx(0.20 + 0.01 * i, abs=1e-6)
    b8, b4 = b.x[0, 0, PRESTO_BANDS.index("B8")], b.x[0, 0, PRESTO_BANDS.index("B4")]
    assert b.x[0, 0, PRESTO_BANDS.index("NDVI")] == pytest.approx((b8 - b4) / (b8 + b4), abs=1e-6)
    groups = masked_groups(b)
    for g in ("S1", "ERA5", "SRTM"):                        # modalities we do not have
        assert (groups[g] == 1).all()
    s2 = ["S2_RGB", "S2_Red_Edge", "S2_NIR_10m", "S2_NIR_20m", "S2_SWIR", "NDVI"]
    for g in s2:
        assert groups[g][0].tolist() == [0, 1, 0, 1, 1, 1, 1, 1]
        assert (groups[g][2] == 1).all()
    # masked values are zero, never invented
    assert (b.x[b.mask == 1] == 0).all()
    assert b.n_months_observed.tolist() == [2, 1, 0]


def test_band_groups_cover_presto_channels():
    idx = sorted(i for g in BAND_GROUPS.values() for i in g)
    assert idx == list(range(17))


def test_missing_centroid_is_an_error(small):
    fields, ts = small
    fields = fields.copy()
    fields.loc[0, "centroid_lat"] = np.nan
    with pytest.raises(ValueError):
        build_presto_inputs(ts, fields)


def test_encoder_masking_ignores_values_under_mask(small):
    torch = pytest.importorskip("torch")
    from fieldvision.foundation.presto_encoder import Encoder
    torch.manual_seed(0)
    enc = Encoder().eval()
    fields, ts = small
    b = build_presto_inputs(ts, fields, 4, 11)
    x = torch.from_numpy(b.x)
    m = torch.from_numpy(b.mask)
    dw = torch.from_numpy(b.dynamic_world)
    ll = torch.from_numpy(b.latlons)
    with torch.no_grad():
        e1 = enc(x, dw, ll, mask=m, month=b.month)
        x2 = x.clone()
        x2[m.bool()] = 123.0                          # garbage under the mask
        e2 = enc(x2, dw, ll, mask=m, month=b.month)
        x3 = x.clone()
        x3[0, 0, PRESTO_BANDS.index("B8")] += 0.2     # a real change in an observed month
        e3 = enc(x3, dw, ll, mask=m, month=b.month)
    assert e1.shape == (3, 128)
    torch.testing.assert_close(e1, e2)
    assert not torch.allclose(e1[0], e3[0])
    torch.testing.assert_close(e1[1:], e3[1:])        # other fields untouched


def test_pretrained_weights_load_if_present():
    pytest.importorskip("torch")
    from fieldvision.foundation.embed import DEFAULT_WEIGHTS, embed_batch
    if not DEFAULT_WEIGHTS.exists():
        pytest.skip("Presto weights not downloaded")
    from fieldvision.foundation.presto_encoder import load_encoder
    enc = load_encoder(DEFAULT_WEIGHTS)
    assert sum(p.numel() for p in enc.parameters()) > 100_000


def test_embed_batch_nan_for_fields_without_data(small):
    torch = pytest.importorskip("torch")
    from fieldvision.foundation.embed import EMB_COLS, embed_batch
    from fieldvision.foundation.presto_encoder import Encoder
    torch.manual_seed(0)
    fields, ts = small
    b = build_presto_inputs(ts, fields, 4, 11)
    df = embed_batch(b, encoder=Encoder())
    assert df["field_id"].tolist() == ["a", "b", "c"]
    assert np.isfinite(df.loc[:1, EMB_COLS].to_numpy()).all()
    assert df.loc[2, EMB_COLS].isna().all()


def test_restrict_and_verdict():
    from fieldvision.foundation.compare import restrict, verdict
    splits = [("f0", np.array([0, 1, 2]), np.array([3, 4])), ("f1", np.array([3, 4]), np.array([0, 1, 2]))]
    keep = np.array([True, False, True, True, False])
    r = restrict(splits, keep)
    assert r[0][1].tolist() == [0, 2] and r[0][2].tolist() == [3]
    assert r[1][1].tolist() == [3] and r[1][2].tolist() == [0, 2]
    assert verdict({"a": 0.01, "b": -0.01, "c": 0.02}, {"ci": [-0.01, 0.03]}) == "no clear difference"
    assert verdict({"a": 0.05, "b": 0.06, "c": 0.04}, {"ci": [0.03, 0.07]}) == "better"
    assert verdict({"a": -0.05, "b": -0.06, "c": -0.04}, {"ci": [-0.07, -0.03]}) == "worse"
