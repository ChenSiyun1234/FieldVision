import json

import numpy as np
import pandas as pd
import pytest

from fieldvision.model.artifacts import PREDICTION_COLUMNS
from fieldvision.model.harness import run_splits
from fieldvision.model.models import (GBMModel, LogisticModel, MajorityModel, RuleModel,
                                      make_factories)
from fieldvision.model.splits import spatial_folds


def _labelled(synth, synth_features):
    fields, _ = synth
    X, _ = synth_features
    y = (fields["label"] == "silage").astype(int).to_numpy()
    meta = fields.copy()
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    return X, meta, y


def test_models_learn_something_and_handle_nan(synth, synth_features):
    X, meta, y = _labelled(synth, synth_features)
    tr = meta["country"] != "AT"
    for name, make in make_factories(X.columns).items():
        m = make().fit(X[tr], y[tr])
        p = m.predict_proba(X[~tr])
        assert p.shape == (int((~tr).sum()),)
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        assert isinstance(m.describe(), str)
    rule = RuleModel().fit(X[tr], y[tr])
    Xn = X[~tr].copy()
    Xn[["harv_days_since_peak", "pre_ndmi"]] = np.nan
    assert np.allclose(rule.predict_proba(Xn), rule.p_missing_)


def test_single_class_training_gives_constant():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    for m in (LogisticModel(["a"]), GBMModel(["a"])):
        m.fit(X, np.array([1, 1, 1]))
        assert np.allclose(m.predict_proba(X), 1.0)
    maj = MajorityModel().fit(X, np.array([0, 0, 1]))
    assert (maj.predict(X) == 0).all()


class _Spy:
    """A model that remembers which fields it was trained on."""
    seen = []

    def __init__(self):
        self.threshold = 0.5
        self.feature_names = []

    def fit(self, X, y):
        _Spy.seen.append(set(X["field_id"]))
        return self

    def predict_proba(self, X):
        return np.full(len(X), 0.5)

    def describe(self):
        return "spy"


def test_cv_fits_only_on_training_blocks(synth, synth_features):
    X, meta, y = _labelled(synth, synth_features)
    Xs = X.copy()
    folds = spatial_folds(meta["block_id"], y, 4, 0)
    _Spy.seen = []
    preds, info, _ = run_splits(Xs, meta, y, {"spy": _Spy},
                                [(f"f{i}", tr, te) for i, (tr, te) in enumerate(folds)],
                                "spatial", "relative")
    block_of = dict(zip(meta["field_id"], meta["block_id"]))
    for seen, (_, te) in zip(_Spy.seen, folds):
        test_ids = set(meta["field_id"].iloc[te])
        assert not seen & test_ids
        assert not {block_of[f] for f in seen} & {block_of[f] for f in test_ids}
    assert preds["field_id"].is_unique and len(preds) == len(meta)


def test_rule_thresholds_depend_only_on_training_rows(synth, synth_features):
    X, meta, y = _labelled(synth, synth_features)
    tr = (meta["country"] != "NL").to_numpy()
    a = RuleModel().fit(X[tr], y[tr])
    y2 = y.copy()
    y2[~tr] = 1 - y2[~tr]                  # scramble the test labels
    b = RuleModel().fit(X[tr], y2[tr])
    assert (a.t_time_, a.t_moist_, a.mode_) == (b.t_time_, b.t_moist_, b.mode_)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from synthetic import write_dataset
    from fieldvision.model import train
    root = tmp_path_factory.mktemp("pipe")
    fdir, tdir = write_dataset(root / "data", n_fields=300, seed=5, blocks_per_country=6)
    out = root / "run"
    import os
    old = os.environ.get("FIELDVISION_LEDGER")
    os.environ["FIELDVISION_LEDGER"] = str(root / "ledger.jsonl")
    metrics = train.main(["--timeseries", str(tdir), "--fields", str(fdir), "--out", str(out),
                          "--n-boot", "50", "--n-splits", "3", "--holdout-blocks", "AT-002-*"])
    yield root, out, fdir, tdir, metrics
    if old is None:
        os.environ.pop("FIELDVISION_LEDGER", None)
    else:
        os.environ["FIELDVISION_LEDGER"] = old


def test_train_writes_outputs(run_dir):
    root, out, *_ , metrics = run_dir
    for name in ("metrics.json", "cv_predictions.parquet", "features.parquet", "REPORT.md",
                 "run.json", "models/gbm.pkl", "models/majority.pkl"):
        assert (out / name).exists(), name
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in report and "majority" in report
    assert metrics["synthetic_data_software_check_only"] is True
    s = metrics["summary"]
    assert set(s["spatial"]["relative"]) >= {"majority", "rule", "logistic", "gbm"}
    assert any(k.startswith("loco:") for k in s)
    assert metrics["holdout"]["n_fields"] > 0
    cv = pd.read_parquet(out / "cv_predictions.parquet")
    hold_ids = set(cv.loc[cv.scheme == "holdout", "field_id"])
    assert hold_ids and not hold_ids & set(cv.loc[cv.scheme != "holdout", "field_id"])
    assert metrics["response_time"]["gbm"]["total_seconds_per_1000"] > 0
    assert (out / "figures" / "example_fields.png").exists()


def test_predict_and_external_evaluation(run_dir):
    from fieldvision.model import evaluate, evaluate_adapted, predict
    root, out, fdir, tdir, _ = run_dir
    pred_path = root / "pred.parquet"
    p = predict.main(["--model", str(out), "--timeseries", str(tdir), "--fields", str(fdir),
                      "--out", str(pred_path)])
    back = pd.read_parquet(pred_path)
    for c in PREDICTION_COLUMNS:
        assert c in back.columns, c
    assert back["low_evidence"].dtype == bool
    ok = back["harvest_start"].notna()
    assert (back.loc[ok, "harvest_start"] <= back.loc[ok, "harvest_end"]).all()
    assert set(back["predicted_label"]) <= {"silage", "grain"}

    args = ["--model", str(out), "--timeseries", str(tdir), "--fields", str(fdir)]
    m = evaluate.main(args + ["--out", str(root / "zs"), "--n-boot", "50"])
    assert m["mode"] == "zero_shot" and m["results"]["gbm"]["all"]["n"] == 300
    assert "per_year" in m["results"]["gbm"]
    assert (root / "zs" / "misclassified.csv").exists()
    with pytest.raises(RuntimeError, match="single-use"):
        evaluate.main(args + ["--out", str(root / "zs2"), "--n-boot", "50"])
    a = evaluate_adapted.main(args + ["--out", str(root / "ad"), "--n-boot", "50",
                                      "--n-calib", "12"])
    assert a["NOT_ZERO_SHOT"] and a["n_calibration"] == 12 and a["n_reported"] == 288
    cal = set(pd.read_csv(root / "ad" / "calibration_fields.csv")["field_id"])
    assert len(cal) == 12
    rep = (root / "ad" / "REPORT.md").read_text(encoding="utf-8")
    assert "NOT the zero-shot" in rep
    ledger = [json.loads(line) for line in open(root / "ledger.jsonl", encoding="utf-8")]
    assert {r["kind"] for r in ledger} >= {"holdout", "external_zero_shot"}


def test_calibrated_wrapper_and_explanations(synth, synth_features):
    import pickle
    from fieldvision.model.explain import explanation_columns, reference_values
    from fieldvision.model.artifacts import predictions_frame
    X, meta, y = _labelled(synth, synth_features)
    X = X.copy()
    X["block_id"] = meta["block_id"].to_numpy()
    tr = (meta["country"] != "AT").to_numpy()
    fac = make_factories(X.columns)
    for name in ("rule", "logistic", "gbm"):
        m = fac[name]().fit(X[tr], y[tr])
        assert m.platt_ is not None
        p = m.predict_proba(X[~tr])
        assert np.isfinite(p).all() and 0 <= p.min() and p.max() <= 1
        pickle.loads(pickle.dumps(m))            # run dirs must be loadable
        ex = m.explain(X[~tr])
        assert ex is not None and len(ex) == int((~tr).sum())
    fields = synth[0]
    ref = reference_values(X[tr], y[tr])
    assert ref["pre_ndmi"]["silage"] > ref["pre_ndmi"]["grain"]
    pf = predictions_frame(X[~tr].reset_index(drop=True), fields, p, 0.5, "gbm", model=m,
                           reference=ref, prevalence=0.07, train_prevalence=float(y[tr].mean()))
    assert pf["reason"].str.len().min() > 10
    assert (pf["p_silage_at_prevalence"] <= pf["p_silage"] + 1e-9).all()
    assert pf["top_model_factors"].str.len().max() > 0
    assert ((pf["n_signals_silage_like"] <= pf["n_signals_observed"])).all()


def test_asof_truncation_and_cli(tmp_path):
    from synthetic import write_dataset
    from fieldvision.features.loading import load_fields, load_timeseries
    from fieldvision.model import asof
    fdir, tdir = write_dataset(tmp_path / "d", n_fields=150, seed=9, blocks_per_country=4)
    f = load_fields(fdir)
    ts, _ = load_timeseries(tdir)
    cut = asof.truncate(ts, f, "09-01")
    assert cut["date"].max() <= pd.Timestamp("2023-09-01") and len(cut) < len(ts)
    tab = asof.main(["--timeseries", str(tdir), "--fields", str(fdir), "--out", str(tmp_path / "a"),
                     "--cutoffs", "08-15", "12-01", "--models", "majority", "gbm",
                     "--n-splits", "3", "--n-boot", "30"])
    g = tab[tab.model == "gbm"].set_index("cutoff")
    assert g.loc["12-01", "harvest_seen_share"] > g.loc["08-15", "harvest_seen_share"]
    assert (tmp_path / "a" / "asof.png").exists()


def test_barc_field_years_and_external_report(tmp_path, run_dir):
    import geopandas as gpd
    from shapely.geometry import box
    from fieldvision.model.barc import counts_by_year, field_years
    rows = []
    for i in range(4):
        rows.append({"OBJECTID": i + 1, "ACRES": 20.0,
                     "Corn_2021": ["Silage", None, "Grain", ""][i],
                     "Corn_2022": ["Grain", "Silage", " ", "grain"][i],
                     "geometry": box(1335000 + i * 1500, 470000, 1335900 + i * 1500, 470900)})
    g = gpd.GeoDataFrame(rows, crs="EPSG:2248")
    fy = field_years(g)
    assert len(fy) == 8 and fy["field_id"].iloc[0] == "barc-1-2021"
    assert int(fy["label"].notna().sum()) == 5
    assert fy.crs.to_epsg() == 4326 and -77.5 < fy["centroid_lon"].iloc[0] < -76.5
    t = counts_by_year(fy).set_index(counts_by_year(fy).index.astype(str))
    assert t.loc["total", "total"] == 5
    # the external report from the pipeline fixture has the required sections
    root = run_dir[0]
    rep = (root / "zs" / "REPORT.md").read_text(encoding="utf-8")
    for needle in ("Claim and validation design", "by year and class", "timing_rule",
                   "grain precision", "Confusion matrix", "Abstention", "Limits"):
        assert needle in rep, needle


def test_training_refuses_evaluation_fields(tmp_path):
    import json
    from synthetic import fields_to_geojson, make_dataset
    from fieldvision.model import asof, train
    f, ts = make_dataset(n_fields=60, seed=4, blocks_per_country=3)
    ts.to_parquet(tmp_path / "ts.parquet", index=False)
    for variant in ("us", "split"):
        g = f.copy()
        if variant == "us":
            g.loc[g.index[:3], "country"] = "US"
        else:
            g["split"] = None
            g.loc[g.index[:1], "split"] = "external_eval"
        path = tmp_path / f"fields_{variant}.geojson"
        path.write_text(json.dumps(fields_to_geojson(g)), encoding="utf-8")
        for main in (train.main, asof.main):
            with pytest.raises(SystemExit, match="reserved for external evaluation"):
                main(["--timeseries", str(tmp_path / "ts.parquet"), "--fields", str(path),
                      "--out", str(tmp_path / f"out_{variant}")])


def test_frozen_model_id_recorded_and_reported(run_dir):
    import json
    from fieldvision.model.artifacts import frozen_model_id
    root, out, *_ = run_dir
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert meta["frozen_model_ids"]["gbm"] == frozen_model_id(out, "gbm")
    rep = (root / "zs" / "REPORT.md").read_text(encoding="utf-8")
    assert meta["frozen_model_ids"]["gbm"] in rep
