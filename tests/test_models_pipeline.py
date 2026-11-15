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


def test_access_log_quoted_and_appended(tmp_path):
    from fieldvision.model.evaluate import access_log_rows, append_access_log
    log = tmp_path / "ACCESS_LOG.md"
    log.write_text("# log\n\n| Time | Who | What | Changed? |\n|---|---|---|---|\n"
                   "| 11:29 | coordinator | counts | No |\n\nRule text.\n", encoding="utf-8")
    rows = access_log_rows(log)
    assert len(rows) == 3 and "coordinator" in rows[2]
    append_access_log(log, "abc123", "zero-shot", tmp_path / "out")
    lines = log.read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if "abc123" in ln)
    assert lines[i - 1].startswith("| 11:29") and lines[-1] == "Rule text."
    assert access_log_rows(None) is None


def test_registry_export_classify_roundtrip(run_dir, tmp_path):
    import json
    from fieldvision.model import classify, registry
    root, out, fdir, tdir, _ = run_dir
    reg = tmp_path / "registry"
    files = sorted(str(p) for p in fdir.iterdir())
    folder = registry.export(out, "gbm", "ts_model_test_v1", reg, files, "abc1234",
                             eval_dir=root / "zs", description="test")
    for f in ("model.json", "model.joblib", "feature_schema.json", "applicability.json",
              "model_card.md", "provenance.json"):
        assert (folder / f).exists(), f
    with pytest.raises(SystemExit):
        registry.export(out, "gbm", "ts_model_test_v1", reg, files, "abc1234")
    registry.export(out, "timing_rule", "rule_test_v1", reg, files, "abc1234")
    mj = json.loads((reg / "rule_test_v1" / "model.json").read_text(encoding="utf-8"))
    from fieldvision.model.models import RuleModel
    r = RuleModel.from_dict(mj["rule"])
    assert r.t_time_ == mj["rule"]["t_time_"] and r.mode_ == "timing_only"
    res = classify.main(["--registry", str(reg), "--model", "ts_model_test_v1", "--timeseries",
                         str(tdir), "--fields", str(fdir), "--state", "XX", "--out",
                         str(tmp_path / "res")])
    classify.main(["--registry", str(reg), "--model", "rule_test_v1", "--timeseries",
                   str(tdir), "--fields", str(fdir), "--state", "XX", "--out", str(tmp_path / "res")])
    t = pd.read_parquet(next((tmp_path / "res").glob("XX_*.parquet")))
    assert set(t["model"]) == {"ts_model_test_v1", "rule_test_v1"}
    rec = json.loads((tmp_path / "res").joinpath(f"XX_{int(t['year'].iloc[0])}.jsonl")
                     .read_text(encoding="utf-8").splitlines()[0])
    assert set(rec) >= {"field_id", "year", "model", "p_silage", "label", "abstain", "evidence"}
    assert {"drop_date", "ndvi_before", "ndvi_after", "days_to_drop"} <= set(rec["evidence"])
    assert classify.team_field_id("barc-112-2023") == "BARC-112"


def test_beltsville_timing_rule_loyo_on_synthetic():
    from synthetic import make_dataset
    from fieldvision.beltsville.timing_rule import loyo
    from fieldvision.features.build import build_features
    frames = [make_dataset(n_fields=60, seed=s, countries=("NL",), year=yr, blocks_per_country=2)
              for s, yr in ((1, 2021), (2, 2022), (3, 2023))]
    f = pd.concat([a for a, _ in frames], ignore_index=True)
    f["field_id"] = f["field_id"] + "-" + f["year"].astype(str)
    ts = pd.concat([b.assign(field_id=b["field_id"] + "-" + str(a["year"].iloc[0]))
                    for a, b in frames], ignore_index=True)
    X, _ = build_features(ts, f)
    y = (f["label"] == "silage").astype(int).to_numpy()
    oof, folds = loyo(X, y, f["year"].to_numpy())
    assert len(folds) == 3 and oof["p"].notna().all()
    assert (oof["pred"] == oof["y"]).mean() > 0.7


def test_app_predictions_format(run_dir, tmp_path):
    from fieldvision.model import app_predictions, registry
    root, out, fdir, tdir, _ = run_dir
    reg = tmp_path / "reg"
    registry.export(out, "logistic", "logit_test_v1", reg, sorted(str(p) for p in fdir.iterdir()),
                    "abc1234", note="a note")
    assert "**a note**" in (reg / "logit_test_v1" / "model_card.md").read_text(encoding="utf-8")
    assert not any(p.name.endswith(".partial") for p in reg.iterdir())
    app = app_predictions.main(["--registry", str(reg), "--model", "logit_test_v1", "--timeseries",
                                str(tdir), "--fields", str(fdir), "--region-prefix", "syn",
                                "--out", str(tmp_path / "app")])
    need = {"field_id", "p_silage", "harvest_start", "harvest_end", "pre_ndvi", "pre_ndmi",
            "low_evidence"}
    f = next((tmp_path / "app").glob("syn-*/predictions.parquet"))
    assert need <= set(pd.read_parquet(f).columns)
    assert app["flip_reason"].str.len().min() > 0


STUB_KEYS = {
    "scout": {"fips", "year", "n_fields", "fields_with_clear_series", "fields_with_cloud_gap_radar_ok",
              "fields_with_little_data", "median_clear_obs_jul_nov", "largest_optical_gap_days_p90"},
    "route_options": {"fips", "year", "routing_table", "options"},
    "extract_features": {"fips", "year", "rows", "path", "quality"},
    "classify": {"fips", "year", "policy", "n_fields", "models_used", "n_abstain", "results_path"},
    "aggregate": {"fips", "year", "silage_acres", "silage_acres_low", "silage_acres_high",
                  "grain_acres", "abstained_acres", "n_fields", "models_used"},
}


def test_toolbox_matches_stub_interface(run_dir, tmp_path):
    import json
    from fieldvision.model import registry
    from fieldvision.toolbox import Toolbox, make_pseudo_county
    root, out, fdir, tdir, _ = run_dir
    reg = tmp_path / "reg"
    files = sorted(str(p) for p in fdir.iterdir())
    registry.export(out, "timing_rule", "eu_timing_rule_v1", reg, files, "abc1234")
    registry.export_composite(out, "composite_v1", reg, files, "abc1234")
    registry.write_rates(reg / "composite_v1", "composite", {"syn": (out, "composite")}, "syn:spatial")
    registry.export(out, "logistic", "eu2_logistic_v1", reg, files, "abc1234")
    registry.write_rates(reg / "eu2_logistic_v1", "logistic", {"syn": (out, "logistic")}, "syn:spatial",
                         max_bracket=12, primary_clear="clear:syn:spatial",
                         local_run=(out, "logistic"))
    fpath = next(fdir.glob("*AT*.geojson"))
    tpath = next(tdir.glob("*AT*.parquet"))
    make_pseudo_county(fpath, tpath, tmp_path / "counties", "24033", 2023)
    gj = json.loads((tmp_path / "counties" / "24033" / "2023" / "fields.geojson").read_text())
    assert "label" not in gj["features"][0]["properties"]
    tb = Toolbox(tmp_path / "counties", reg, tmp_path / "out")
    s = tb.scout("24033", 2023)
    assert set(s) == STUB_KEYS["scout"] | {"notes"} and "stub" not in s
    assert s["fields_with_cloud_gap_radar_ok"] == 0
    r = tb.route_options("24033", 2023)
    assert set(r) == STUB_KEYS["route_options"] | {"notes"}
    assert set(r["options"]) == {"standard", "radar_first", "regional_threshold", "in_site",
                                 "module_v1", "module_v2"}
    for o in r["options"].values():
        assert {"models_used", "n_abstain"} <= set(o)
    assert r["options"]["radar_first"]["models_used"] == r["options"]["standard"]["models_used"]
    e = tb.extract_features("24033", 2023)
    assert set(e) == STUB_KEYS["extract_features"] and set(e["quality"]) == {"rows_with_gap_filled"}
    c = tb.classify("24033", 2023, "standard")
    assert set(c) == STUB_KEYS["classify"]
    rec = json.loads(open(c["results_path"].replace(".parquet", ".jsonl")).readline())
    assert {"field_id", "year", "model", "p_silage", "label", "abstain", "evidence"} <= set(rec)
    a1 = tb.aggregate("24033", 2023)
    a2 = Toolbox(tmp_path / "counties", reg, tmp_path / "out2").aggregate("24033", 2023)
    assert STUB_KEYS["aggregate"] <= set(a1) and a1 == a2
    extra = {"silage_acres_adjusted", "silage_acres_adjusted_low", "silage_acres_adjusted_high",
             "grain_acres_adjusted", "silage_share_counted", "silage_share_adjusted",
             "adjustment_method", "adjustment_clipped", "adjusted_share_by_rates", "notes",
             "silage_share_headline", "headline_method", "silage_share_prior_shift",
             "silage_share_prior_shift_low", "silage_share_prior_shift_high",
             "prior_shift_converged", "prior_shift_iterations", "silage_share_calibrated",
             "silage_share_calibrated_low", "silage_share_calibrated_high",
             "silage_acres_calibrated", "calibration_reference", "module_headline_share",
             "module_headline_method", "calibration_note"}
    assert set(a1) == STUB_KEYS["aggregate"] | extra      # 'standard' unchanged for the team
    assert rec["model"] == "composite_v1"
    total = gj and sum(f["properties"]["area_ha"] for f in gj["features"]) * 2.47105
    assert abs(a1["silage_acres"] + a1["grain_acres"] + a1["abstained_acres"] - total) < 1.0
    assert a1["silage_acres_low"] <= a1["silage_acres"] <= a1["silage_acres_high"]
    assert a1["silage_acres_adjusted"] is not None
    assert abs(a1["silage_acres_adjusted"] + a1["grain_acres_adjusted"] + a1["abstained_acres"]
               - total) < 1.0
    assert a1["silage_acres_adjusted_low"] <= a1["silage_acres_adjusted"] <= a1["silage_acres_adjusted_high"]
    assert 0.0 <= a1["silage_share_adjusted"] <= 1.0
    assert "half the fields are silage" in a1["notes"]
    # the module under its own policy name
    c2 = tb.classify("24033", 2023, "module_v2")
    am = tb.aggregate("24033", 2023)
    clear_keys = {"clear_tier_share_of_area", "clear_tier_acres", "wide_tier_acres",
                  "wide_tier_share_called_silage", "clear_silage_share_counted",
                  "clear_silage_share_europe_adjusted", "clear_silage_share_reference_calibrated",
                  "clear_silage_share_local_threshold", "clear_reference",
                  "clear_silage_share_local_flagged", "local_per_tier_share", "local_per_tier_detail",
                  "unscored_corn_acres", "county_share_with_unscored_low",
                  "county_share_with_unscored_high"}
    assert set(am) == STUB_KEYS["aggregate"] | extra | clear_keys
    rows = [json.loads(l) for l in open(c2["results_path"].replace(".parquet", ".jsonl"))]
    assert {r["model"] for r in rows} == {"eu2_logistic_v1"}
    assert {r["evidence"]["evidence_tier"] for r in rows} <= {"clear", "wide", "abstain"}
    assert all((r["evidence"]["evidence_tier"] == "abstain") == r["abstain"] for r in rows)
    assert am["clear_tier_acres"] + am["wide_tier_acres"] + am["abstained_acres"] - total < 1.0
    ab = tb.abstain_region("24033", 2023, "test")
    assert ab["abstained"] and ab["abstained_acres"] > 0


def test_adjusted_share_is_clipped_and_says_so(tmp_path):
    import numpy as np
    from fieldvision.toolbox import Toolbox
    reg = tmp_path / "reg" / "m_v1"
    reg.mkdir(parents=True)
    (reg / "rates.json").write_text(json.dumps({"primary": "x", "sets": {
        "x": {"tp": 70, "fp": 30, "fn": 30, "tn": 70, "tpr": 0.7, "fpr": 0.3, "n": 200}}}))
    tb = Toolbox(data_root=tmp_path, registry=tmp_path / "reg", out_root=tmp_path / "o", n_resample=200)
    a = np.array([10.0, 10.0, 10.0, 10.0])
    called = np.array([False, False, False, True])        # counted share 0.25 < FPR 0.3
    ok = np.ones(4, bool)
    r = tb._adjusted("m_v1", a, called, ok, 0.0)
    assert r["silage_share_adjusted"] == 0.0 and r["adjustment_clipped"] is True
    assert "clipped" in r["notes"]


MACHINE_PATH = r"[A-Za-z]:[\\/]|\bUsers\b|worktrees"


def test_registry_export_writes_no_machine_paths(run_dir, tmp_path):
    import re
    import sys
    from fieldvision.beltsville import timing_rule as tr
    from fieldvision.model import registry
    from fieldvision.model.registry import portable_command, portable_text
    root, out, fdir, tdir, _ = run_dir
    reg = tmp_path / "reg"
    files = sorted(str(p) for p in fdir.iterdir())
    registry.export(out, "gbm", "gbm_t_v1", reg, files, "abc1234")
    registry.export_composite(out, "comp_t_v1", reg, files, "abc1234")
    bad = re.compile(MACHINE_PATH)
    for p in reg.rglob("*"):
        if p.suffix in {".json", ".md", ".txt", ".jsonl"}:
            for line in p.read_text(encoding="utf-8").splitlines():
                assert not bad.search(line), (p.name, line)
    fake = r"C:\Users\someone\proj\.work\worktrees\x\src\fieldvision\model\train.py"
    cmd = portable_command([fake, "--fields", r"C:\t\fv\data\labels\a\fields.geojson",
                            "--out", r"D:\scratch\run"])
    assert cmd == ("python -m fieldvision.model.train --fields data/labels/a/fields.geojson "
                   "--out <external>/run")
    assert not bad.search(portable_text(fake + " /home/me/x/y.parquet"))
    assert portable_text("0.766/0.783 and https://example.org/a") == "0.766/0.783 and https://example.org/a"


def test_committed_registry_and_results_have_no_machine_paths():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bad = re.compile(MACHINE_PATH)
    for folder in ("registry", "results", "docs"):
        for p in (root / folder).rglob("*"):
            if p.is_file() and p.suffix in {".json", ".md", ".jsonl", ".txt"}:
                for line in p.read_text(encoding="utf-8").splitlines():
                    hit = bad.search(line.replace("https://", "").replace("http://", ""))
                    assert not hit, (str(p.relative_to(root)), line[:120])


def test_toolbox_says_how_to_install_a_parquet_engine(monkeypatch):
    import importlib.util
    from fieldvision import toolbox
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name in ("pyarrow", "fastparquet") else real(name, *a))
    with pytest.raises(ImportError, match="pip install --target /workspace/FieldVision/.deps pyarrow"):
        toolbox.Toolbox()


def test_in_site_cross_validation_and_policy_scope(tmp_path):
    from synthetic import make_dataset
    from fieldvision.beltsville import in_site
    from fieldvision.features.build import build_features
    from fieldvision.model.models import LogisticModel
    from fieldvision.toolbox import Toolbox
    frames = [make_dataset(n_fields=16, seed=s, countries=("NL",), year=yr, blocks_per_country=2)
              for s, yr in ((1, 2021), (2, 2022), (3, 2023))]
    f = pd.concat([a.assign(field_id=a["field_id"].str[-3:].radd("barc-") + "-" + str(yr))
                   for (a, _), yr in zip(frames, (2021, 2022, 2023))], ignore_index=True)
    ts = pd.concat([b.assign(field_id=b["field_id"].str[-3:].radd("barc-") + "-" + str(yr))
                    for (_, b), yr in zip(frames, (2021, 2022, 2023))], ignore_index=True)
    X, _ = build_features(ts, f)
    y = (f["label"] == "silage").astype(int).to_numpy()
    at_X, at_ts = make_dataset(n_fields=90, seed=9, countries=("AT",), blocks_per_country=3)
    at, _ = build_features(at_ts, at_X)
    at["y"] = (at_X["label"] == "silage").astype(int).to_numpy()
    eu = LogisticModel(in_site.FEATURES).fit(at, at["y"].to_numpy())
    makers = in_site.candidates(eu, at, at["y"].to_numpy())
    unit = f["field_id"].str.replace(r"-(19|20)\d\d$", "", regex=True).to_numpy()
    for groups in (f["year"].to_numpy(), unit):
        oof = in_site.cross_validate(X, y, groups, makers)
        assert set(oof["model"]) == set(makers) and oof.groupby("model").size().eq(len(f)).all()
        for g in np.unique(groups):        # no field is predicted by a model that saw it
            assert (oof[oof["fold"] == str(g)]["field_id"].isin(f["field_id"][groups == g])).all()
    tab = in_site.summary_table(oof, "leave-one-field-out")
    assert {"correct", "silage_found", "grain_called_silage", "abstain_rate"} <= set(tab.columns)
    tb = Toolbox(data_root=tmp_path, registry=tmp_path / "noreg", out_root=tmp_path / "o")
    with pytest.raises(ValueError, match="only for FIPS"):
        tb._policy_predictions("24021", 2022, "in_site")


def test_prior_shift_recovers_a_known_share(tmp_path):
    import numpy as np
    from fieldvision.toolbox import Toolbox
    rng = np.random.default_rng(0)
    n, true_share = 4000, 0.1
    y = rng.random(n) < true_share
    # a score drawn per class, turned into the probability that is calibrated for 50/50
    from scipy.stats import beta
    s = np.where(y, rng.beta(8, 2, n), rng.beta(2, 8, n))
    f1, f0 = beta.pdf(s, 8, 2), beta.pdf(s, 2, 8)
    p = f1 / (f1 + f0)
    tb = Toolbox(data_root=tmp_path, registry=tmp_path, out_root=tmp_path / "o", n_resample=200)
    r = tb._prior_shift(p, np.ones(n), np.ones(n, bool))
    assert r["prior_shift_converged"]
    assert abs(r["silage_share_prior_shift"] - true_share) < 0.05
    assert r["silage_share_prior_shift_low"] <= r["silage_share_prior_shift"] <= r["silage_share_prior_shift_high"]


def test_module_v1_tiers_and_reference_calibration(run_dir, tmp_path):
    import joblib
    from fieldvision.model import registry
    from fieldvision.toolbox import Toolbox, make_pseudo_county
    root, out, fdir, tdir, _ = run_dir
    reg = tmp_path / "reg"
    files = sorted(str(p) for p in fdir.iterdir())
    registry.export(out, "timing_rule", "eu_timing_rule_v1", reg, files, "abc1234")
    registry.export_composite(out, "composite_v1", reg, files, "abc1234")
    registry.write_rates(reg / "composite_v1", "composite", {"syn": (out, "composite")}, "syn:spatial")
    make_pseudo_county(next(fdir.glob("*AT*.geojson")), next(tdir.glob("*AT*.parquet")),
                       tmp_path / "c", "24033", 2023)
    make_pseudo_county(next(fdir.glob("*FR*.geojson")), next(tdir.glob("*FR*.parquet")),
                       tmp_path / "c", "24099", 2023)
    tb = Toolbox(tmp_path / "c", reg, tmp_path / "o")
    c = tb.classify("24033", 2023, "module_v1")
    rows = [json.loads(l) for l in open(c["results_path"].replace(".parquet", ".jsonl"))]
    assert {r["evidence"]["tier"] for r in rows} <= {"classifier_only", "abstained"}
    assert all((r["evidence"]["tier"] == "abstained") == r["abstain"] for r in rows)
    a = tb.aggregate("24033", 2023)
    assert STUB_KEYS["aggregate"] <= set(a) and a["silage_share_calibrated"] is None
    assert a["module_headline_method"].startswith("Europe-rate adjusted")
    tb.set_reference("24033", "24099", 2023, 0.02)
    tb.classify("24033", 2023, "module_v1")
    a = tb.aggregate("24033", 2023)
    assert 0.0 <= a["silage_share_calibrated"] <= 1.0
    assert a["silage_share_calibrated_low"] <= a["silage_share_calibrated"] <= a["silage_share_calibrated_high"]
    assert a["module_headline_share"] == a["silage_share_calibrated"]
    assert a["calibration_reference"]["fips"] == "24099"
    # a registered anomaly scorer turns the tiers into agree / disagree
    (reg / "anomaly_v1").mkdir()
    joblib.dump(_ConstScorer(), reg / "anomaly_v1" / "model.joblib")
    (reg / "anomaly_v1" / "model.json").write_text(json.dumps({"threshold": 0.5}))
    c = Toolbox(tmp_path / "c", reg, tmp_path / "o2").classify("24033", 2023, "module_v1")
    rows = [json.loads(l) for l in open(c["results_path"].replace(".parquet", ".jsonl"))]
    for r in rows:                       # scorer says "not grain-like" for every field
        want = "abstained" if r["abstain"] else ("agree" if r["label"] == "silage" else "disagree")
        assert r["evidence"]["tier"] == want


class _ConstScorer:
    def score(self, X):
        import numpy as np
        return np.ones(len(X))
