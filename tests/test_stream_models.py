"""Sequence and stream models: inputs, networks, training loop and the hold-out table."""
import numpy as np
import pandas as pd
import pytest

from fieldvision.features.indices import BANDS
from fieldvision.stream import series
from fieldvision.stream.folds import folds_from_reference


def _table(rows):
    """rows: (field_id, date, ndvi, clear_fraction) -> a time-series frame with all ten bands."""
    out = []
    for fid, date, ndvi, cf in rows:
        r = {"field_id": fid, "date": pd.Timestamp(date), "clear_fraction": cf, "n_clear": 50}
        r.update({b: 0.1 for b in BANDS})
        r["B04"] = 0.05
        r["B08"] = 0.05 * (1 + ndvi) / (1 - ndvi)
        out.append(r)
    return pd.DataFrame(out)


def test_sequences_keep_usable_dates_in_order():
    ts = _table([("a", "2024-06-01", 0.3, 1.0), ("a", "2024-05-01", 0.2, 0.9), ("a", "2024-07-01", 0.8, 0.3),
                 ("b", "2024-06-10", 0.5, 0.7), ("c", "2024-06-10", 0.5, 0.1)])
    ts.loc[3, "B11"] = np.nan
    s = series.build_sequences(ts, ["a", "b", "zz", "c"])
    assert s["x"].shape == (4, 2, 10)
    assert s["valid"].sum(1).tolist() == [2, 0, 0, 0]          # cloudy, NaN and unknown rows are gone
    assert s["date"][0].tolist() == ["2024-05-01", "2024-06-01"]
    assert s["doy"][0].tolist() == [122.0, 153.0]
    assert s["x"][0, 0, BANDS.index("B04")] == pytest.approx(0.05)


def _curve(shift=0.0, cover_crop=True):
    days = np.arange(95, 330, 5.0)
    ndvi = 0.15 + 0.7 / (1 + np.exp(-(days - 170 - shift) / 6.0)) - 0.7 / (1 + np.exp(-(days - 280 - shift) / 5.0))
    if cover_crop:
        ndvi += 0.5 * np.exp(-((days - 110) / 12.0) ** 2)
    x = np.full((1, len(days), 10), 0.1, np.float32)
    x[0, :, series.RED] = 0.05
    x[0, :, series.NIR] = 0.05 * (1 + ndvi) / (1 - ndvi)
    return x, days[None].astype(np.float32), np.ones((1, len(days)), bool)


def test_greenup_follows_the_crop_not_the_cover_crop():
    g0 = series.greenup_doy(*_curve())[0]
    assert abs(g0 - 170) < 6
    assert abs(series.greenup_doy(*_curve(shift=25))[0] - g0 - 25) < 6
    assert series.greenup_doy(np.zeros((1, 3, 10), np.float32), np.zeros((1, 3), np.float32),
                              np.zeros((1, 3), bool))[0] == 170.0


def test_time_axis_and_regular_grid():
    doy = np.array([[92.0, 97.0, 104.0, 131.0, 0.0]], np.float32)
    valid = np.array([[True, True, True, True, False]])
    assert series.time_axis(doy, valid)[0].tolist() == [-90.0, -85.0, -78.0, -51.0, 0.0]
    assert series.time_axis(doy, valid, np.array([110.0]))[0].tolist() == [-18.0, -13.0, -6.0, 21.0, 0.0]
    grid = series.regular_grid(doy, valid, start=91, step=10, bins=5)
    # bins of 10 days from day 91; in the first bin day 97 is nearer the centre (96) than day 92
    assert grid[0].tolist() == [1, 2, -1, -1, 3]


def test_attach_chips_drops_steps_without_a_chip():
    ts = _table([("a", "2024-05-01", 0.2, 1.0), ("a", "2024-06-01", 0.3, 1.0), ("b", "2024-06-10", 0.5, 1.0)])
    s = series.build_sequences(ts, ["a", "b"])
    chips = {"field_id": np.array(["b", "a"]), "date": np.array(["2024-06-10", "2024-06-01"])}
    obs, dropped = series.attach_chips(s, chips)
    assert obs.tolist() == [[-1, 1], [0, -1]] and dropped == 1
    assert s["valid"].tolist() == [[False, True], [True, False]]


def test_folds_from_reference():
    preds = pd.DataFrame({"field_id": ["a", "b", "a"], "block_id": ["x", "y", "x"], "country": ["AT", "FR", "AT"],
                          "year": ["2024"] * 3, "low_evidence": False, "fold": ["fold0", "fold1", "fold0"],
                          "y": [1, 0, 1], "scheme": ["spatial", "spatial", "loco:AT"],
                          "model": ["phys-logistic"] * 3})
    f = folds_from_reference(preds, field_ids=["a"])
    assert f["field_id"].tolist() == ["a"] and f["fold"].tolist() == ["fold0"]


# ---------------------------------------------------------------- networks and training

def _toy(n=96, length=14, chip=6, seed=0):
    """Fields whose class shows only late in the season, in the medians and in the chips."""
    from fieldvision.stream.train import StreamData
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    rows, cf, cd, cb = [], [], [], []
    for i in range(n):
        for k in range(length):
            day = 100 + 15 * k
            green = 0.75 if 150 < day < (235 if y[i] else 275) else 0.2
            green += rng.normal(0, 0.03)
            date = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=day - 1)).strftime("%Y-%m-%d")
            rows.append((f"f{i}", date, green, 1.0))
            nir = 0.05 * (1 + green) / (1 - green)
            px = np.full((10, chip, chip), 0.1) + rng.normal(0, 0.0002, (10, chip, chip))
            px[series.RED], px[series.NIR] = 0.05, nir
            cf.append(f"f{i}")
            cd.append(date)
            cb.append(np.round(px * 10000).astype(np.uint16))
    chips = {"field_id": np.array(cf), "date": np.array(cd), "bands": np.stack(cb),
             "scl": np.full((len(cb), chip, chip), 4, np.uint8),
             "inside": np.ones((len(cb), chip, chip), bool)}
    chips["bands"][0, :, 0, 0] = 65535
    data = StreamData(_table(rows), np.array([f"f{i}" for i in range(n)]), chips)
    return data, y


def test_networks_ignore_padding_and_pixel_order():
    torch = pytest.importorskip("torch")
    from fieldvision.stream import nets
    torch.manual_seed(0)
    x, days = torch.randn(3, 7, 10), torch.randn(3, 7)
    valid = torch.ones(3, 7, dtype=torch.bool)
    valid[1, 4:] = False
    net = nets.SeqNet().eval()
    a = net(x, days, valid)
    x2 = x.clone()
    x2[1, 4:] = 99.0
    longer = net(torch.cat([x2, torch.randn(3, 2, 10)], 1), torch.cat([days, torch.randn(3, 2)], 1),
                 torch.cat([valid, torch.zeros(3, 2, dtype=torch.bool)], 1))
    assert torch.allclose(a, longer, atol=1e-5)
    assert torch.isfinite(net(x, days, torch.zeros(3, 7, dtype=torch.bool))).all()

    ps = nets.PixelSetNet().eval()
    px, use = torch.randn(2, 5, 9, 10), torch.rand(2, 5, 9) > 0.3
    perm = torch.randperm(9)
    v = torch.ones(2, 5, dtype=torch.bool)
    assert torch.allclose(ps(px, use, days[:2, :5], v), ps(px[:, :, perm], use[:, :, perm], days[:2, :5], v),
                          atol=1e-5)
    hidden = px.clone()
    hidden[~use] = 7.0                                       # values under the mask must not matter
    assert torch.allclose(ps(px, use, days[:2, :5], v), ps(hidden, use, days[:2, :5], v), atol=1e-5)
    assert nets.ChipSeqNet().eval()(torch.randn(2, 5, 12, 6, 6), days[:2, :5], v).shape == (2,)
    assert nets.Conv3dNet().eval()(torch.randn(2, 13, 25, 12, 12)).shape == (2,)
    assert nets.count_parameters(net) < 200_000


def test_batches_have_the_layout_each_model_expects():
    pytest.importorskip("torch")
    from fieldvision.stream.train import Settings, make_batch
    data, _ = _toy(n=8)
    cfg, rows = Settings(set_size=5), np.arange(8)
    stats = data.band_stats(rows)
    seq = make_batch(data, rows, "seq", "greenup", stats, cfg)
    assert seq["x"].shape == (8, 14, 10) and seq["valid"].all()
    nir = seq["x"][seq["valid"]][:, series.NIR]               # scaled with the training statistics
    assert abs(float(nir.mean())) < 1e-3 and abs(float(nir.std()) - 1) < 1e-3
    ps = make_batch(data, rows, "pixelset", "calendar", stats, cfg)
    assert ps["px"].shape == (8, 14, 36, 10) and ps["use"].sum() == 8 * 14 * 36 - 1   # one missing pixel
    rng = np.random.default_rng(0)
    tr = make_batch(data, rows, "pixelset", "calendar", stats, cfg, rng)
    assert tr["px"].shape[2] == 5 and tr["use"][tr["valid"]].all()
    assert (tr["valid"].sum(1) >= cfg.min_dates).all() and not tr["valid"].all()
    ch = make_batch(data, rows, "chipseq", "greenup", stats, cfg)
    assert ch["chips"].shape == (8, 14, 12, 6, 6) and ch["chips"][0, 0, 10, 0, 0] == 0   # clear mask off
    fr = make_batch(data, rows, "conv3d", "calendar", stats, cfg)
    assert fr["frames"].shape == (8, 13, series.GRID_BINS, 6, 6)
    exists = fr["frames"][:, 12, :, 0, 0]
    assert 0 < exists.sum() < exists.size                    # some bins of the even grid are empty


@pytest.mark.parametrize("kind", ["seq", "pixelset", "chipseq", "conv3d"])
def test_models_learn_a_late_season_difference(kind):
    pytest.importorskip("torch")
    from sklearn.metrics import roc_auc_score
    from fieldvision.stream.train import Settings, fit_predict
    data, y = _toy()
    train, test = np.arange(0, 64), np.arange(64, 96)
    p, losses = fit_predict(data, y, kind, "calendar", train, test, Settings(epochs=20, batch=16), seed=0)
    assert p.shape == (32,) and np.isfinite(p).all()
    assert losses[-1] < losses[0]
    assert roc_auc_score(y[test], p) > 0.9


def test_train_cli_and_hold_out_table(tmp_path):
    pytest.importorskip("torch")
    from fieldvision.stream import report, train
    data, y = _toy(n=48)
    ids = data.seq["field_id"]
    country = np.array(["AT", "FR", "NL"])[np.arange(48) % 3]
    folds = pd.DataFrame({"field_id": ids, "block_id": [f"{c}-{i % 4}" for i, c in enumerate(country)],
                          "country": country, "year": "2024", "low_evidence": False,
                          "fold": [f"fold{i % 2}" for i in range(48)], "y": y})
    folds.to_parquet(tmp_path / "folds.parquet")
    rows = [(f, d, 0.0, 1.0) for f, d in zip(data.chips["field_id"], data.chips["date"])]
    ts = _table(rows)
    nir = data.chips["bands"][:, series.NIR, 1, 1] / 10000.0
    ts["B08"] = nir
    ts.to_parquet(tmp_path / "ts.parquet")
    np.savez_compressed(tmp_path / "chips.npz", **data.chips)
    args = ["--folds", str(tmp_path / "folds.parquet"), "--timeseries", str(tmp_path / "ts.parquet"),
            "--chips", str(tmp_path / "chips.npz"), "--models", "seq", "pixelset", "--time", "calendar",
            "--seeds", "2", "--epochs", "2", "--threads", "1", "--out", str(tmp_path / "run")]
    preds = train.main(args)
    assert set(preds["scheme"]) == {"spatial", "loco:AT", "loco:FR", "loco:NL"}
    assert set(preds["model"]) == {"seq-cal", "pixelset-cal"}
    assert len(preds[preds["scheme"] == "loco:AT"]) == 2 * 16
    assert {"p_seed0", "p_seed1", "p", "pred", "threshold", "block_id", "low_evidence"} <= set(preds.columns)
    again = train.main(args)                                  # finished parts are not trained twice
    assert np.allclose(again["p"], preds["p"])

    ref = preds[preds["model"] == "seq-cal"].copy()
    ref["model"] = "phys-logistic"
    ref["p"] = np.where(ref["y"] == 1, 0.8, 0.3)
    stream = report.load_stream([tmp_path / "run"])
    both = pd.concat([report.align_reference(stream, ref), stream], ignore_index=True)
    table = report.transfer_table(both, ["phys-logistic", "seq-cal", "pixelset-cal"], n_boot=50)
    assert len(table) == 9 and (table[table["model"] == "phys-logistic"]["diff"] == 0).all()
    assert np.isfinite(table["auc_loss"]).all() and (table["n_seeds"] >= 0).all()
    assert "(ref)" in report.markdown(table, ["phys-logistic", "seq-cal"])
    report.figure(table, ["phys-logistic", "seq-cal", "pixelset-cal"], tmp_path / "f.png", "t")
    assert (tmp_path / "f.png").stat().st_size > 2000
