"""Tests for the demo app export and the synthetic sample generator.

Run from the repository root:
    python -m pytest tests/test_app_export.py --basetemp C:/t/fv/pytest-app
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldvision.app import export, sample_data  # noqa: E402


def strict_load(path):
    """json.loads that fails on NaN / Infinity, which browsers cannot parse."""
    def bad(token):
        raise ValueError(f"{path}: non-standard JSON constant {token}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


def square(lon, lat, d=0.002):
    return {"type": "Polygon", "coordinates": [[[lon, lat], [lon + d, lat], [lon + d, lat + d], [lon, lat + d], [lon, lat]]]}


def write_fields(path, rows):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": props, "geometry": square(10 + i * 0.01, 48)} for i, props in enumerate(rows)
    ]}
    path.write_text(json.dumps(fc), encoding="utf-8")
    return path


# ---------------------------------------------------------------- metrics

def test_confusion_hand_made():
    y_true = ["silage"] * 4 + ["grain"] * 6
    y_pred = ["silage", "silage", "silage", "grain", "silage", "grain", "grain", "grain", "grain", "grain"]
    m = export.confusion(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(1 / 6, abs=1e-4)
    assert m["accuracy"] == pytest.approx(0.8)


def test_confusion_undefined_ratios_are_none():
    m = export.confusion(["grain", "grain"], ["grain", "grain"])
    assert m["precision"] is None and m["recall"] is None
    assert m["fpr"] == 0.0


# ---------------------------------------------------------------- series

def test_build_series_quality_codes_and_duplicates():
    nan = np.nan
    ts = pd.DataFrame([
        # usable: NDVI = (0.4 - 0.1) / 0.5 = 0.6
        dict(field_id="a", date="2024-06-01", n_pixels=100, n_clear=95, clear_fraction=0.95, B04=0.1, B08=0.4, B8A=0.4, B11=0.2),
        # same day from an overlapping tile with fewer clear pixels: dropped
        dict(field_id="a", date="2024-06-01", n_pixels=100, n_clear=10, clear_fraction=0.10, B04=0.3, B08=0.3, B8A=0.3, B11=0.3),
        # partly cloudy: excluded but kept
        dict(field_id="a", date="2024-06-06", n_pixels=100, n_clear=30, clear_fraction=0.30, B04=0.2, B08=0.3, B8A=0.3, B11=0.25),
        # nothing clear
        dict(field_id="a", date="2024-06-11", n_pixels=100, n_clear=0, clear_fraction=0.0, B04=nan, B08=nan, B8A=nan, B11=nan),
        dict(field_id="a", date="2024-07-11", n_pixels=100, n_clear=90, clear_fraction=0.90, B04=0.1, B08=0.3, B8A=0.3, B11=0.2),
    ])
    ts["date"] = pd.to_datetime(ts["date"])
    series, cov = export.build_series(ts, 2024, min_clear=0.7)
    s = series["a"]
    assert s["q"] == [2, 1, 0, 2]
    assert s["t"][0] == (pd.Timestamp("2024-06-01") - pd.Timestamp("2024-01-01")).days
    assert s["ndvi"][0] == pytest.approx(0.6)
    assert s["ndvi"][2] is None and s["ndmi"][2] is None
    # no B12 column in this input: the residue index is missing, not zero
    assert s["ndti"] == [None, None, None, None]
    peak_day = (pd.Timestamp("2024-06-01") - pd.Timestamp("2024-01-01")).days
    assert cov["a"] == {"n_obs": 4, "n_usable": 2, "max_gap": 40, "peak_ndvi": 0.6, "peak_day": peak_day}


def test_residue_index_and_post_harvest_values():
    ts = pd.DataFrame([
        dict(field_id="a", date="2024-09-01", n_pixels=50, n_clear=50, clear_fraction=1.0, B04=0.03, B08=0.40, B8A=0.40, B11=0.20, B12=0.12),
        # cut between Sep 1 and Sep 6; bare soil afterwards: NDTI = (0.30 - 0.27) / 0.57
        dict(field_id="a", date="2024-09-06", n_pixels=50, n_clear=50, clear_fraction=1.0, B04=0.20, B08=0.25, B8A=0.25, B11=0.30, B12=0.27),
        dict(field_id="a", date="2024-09-11", n_pixels=50, n_clear=10, clear_fraction=0.2, B04=0.20, B08=0.25, B8A=0.25, B11=0.30, B12=0.10),
        dict(field_id="a", date="2024-09-16", n_pixels=50, n_clear=50, clear_fraction=1.0, B04=0.20, B08=0.26, B8A=0.26, B11=0.30, B12=0.26),
        dict(field_id="a", date="2024-11-30", n_pixels=50, n_clear=50, clear_fraction=1.0, B04=0.20, B08=0.26, B8A=0.26, B11=0.30, B12=0.10),
    ])
    ts["date"] = pd.to_datetime(ts["date"])
    series, _ = export.build_series(ts, 2024, min_clear=0.7)
    s = series["a"]
    assert s["ndti"][0] == pytest.approx(0.25, abs=1e-3)
    he = (pd.Timestamp("2024-09-06") - pd.Timestamp("2024-01-01")).days
    post_ndvi, post_ndti = export.post_harvest(s, he)
    assert post_ndvi == pytest.approx(0.111, abs=1e-3)
    # median over usable views within 30 days: Sep 6 and Sep 16; the hazy Sep 11 and late Nov 30 are left out
    assert post_ndti == pytest.approx((0.053 + 0.071) / 2, abs=2e-3)
    assert export.post_harvest(s, None) == (None, None)
    assert export.post_harvest(None, he) == (None, None)


# ---------------------------------------------------------------- whole export

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("sample")
    metas = sample_data.build(["at-2024", "barc-2023", "us-2022"], work=root / "work", app_data=root / "app", n=30)
    return root / "app", {m["region_id"]: m for m in metas}


def test_sample_regions_written_and_strict_json(built):
    app, metas = built
    index = strict_load(app / "regions.json")
    assert [r["id"] for r in index["regions"]] == ["at-2024", "barc-2023", "us-2022"]
    for rid in metas:
        for name in ("meta.json", "fields.geojson", "series.json"):
            strict_load(app / rid / name)
        assert strict_load(app / rid / "meta.json")["synthetic"] is True


def test_labelled_region_has_metrics_and_edge_cases(built):
    app, metas = built
    meta = metas["at-2024"]
    m = meta["metrics"]
    assert meta["has_predictions"] and meta["has_labels"]
    assert m["all"]["n"] == m["n_labelled"] <= m["n_fields"]
    assert 0 <= m["low_evidence_share"] <= 1
    props = [f["properties"] for f in strict_load(app / "at-2024" / "fields.geojson")["features"]]
    # the generator plants fields with nothing usable and fields with no declared label
    blind = [p for p in props if p["n_usable"] == 0]
    assert blind and all(p["low"] is True and p["hs"] is None and p["pre_ndvi"] is None for p in blind)
    assert any(p["label"] is None and p["correct"] is None for p in props)
    for p in props:
        if p["p"] is not None:
            assert 0 <= p["p"] <= 1
            assert p["pred"] == ("silage" if p["p"] >= 0.5 else "grain")
        if p["hs"] and p["he"]:
            assert p["hs"] <= p["he"]
    series = strict_load(app / "at-2024" / "series.json")
    for s in series.values():
        assert set(s["q"]) <= {0, 1, 2}
        assert s["t"] == sorted(s["t"])
        assert len(s["t"]) == len(s["ndvi"]) == len(s["ndmi"]) == len(s["cf"]) == len(s["q"])


def test_region_without_predictions(built):
    app, metas = built
    meta = metas["barc-2023"]
    assert meta["has_predictions"] is False
    assert meta["metrics"]["all"] is None and meta["metrics"]["low_evidence_share"] is None
    props = [f["properties"] for f in strict_load(app / "barc-2023" / "fields.geojson")["features"]]
    assert all(p["p"] is None and p["pred"] is None and p["correct"] is None for p in props)


def test_us_counties(built):
    app, metas = built
    meta = metas["us-2022"]
    assert meta["has_counties"] and meta["has_county_shapes"] and not meta["has_labels"]
    c = strict_load(app / "us-2022" / "counties.json")
    assert c["summary"]["n_counties"] == 10
    assigned = sum(r["n_fields"] for r in c["rows"])
    assert assigned == meta["metrics"]["n_fields"]
    for r in c["rows"]:
        for k in ("pred_silage_share", "usda_silage_share"):
            assert r[k] is None or 0 <= r[k] <= 1
    shapes = strict_load(app / "us-2022" / "counties.geojson")
    assert len(shapes["features"]) == 10

    # acres: silage + grain = area of the predicted fields; the state row sums the counties
    props = [f["properties"] for f in strict_load(app / "us-2022" / "fields.geojson")["features"]]
    by_fips = {}
    for p in props:
        if p["pred"]:
            by_fips[p["county_fips"]] = by_fips.get(p["county_fips"], 0) + p["area_ha"] * export.ACRES_PER_HA
    for r in c["rows"]:
        assert r["pred_silage_acres"] + r["pred_grain_acres"] == pytest.approx(by_fips.get(r["fips"], 0), abs=0.5)
    (st,) = c["states"]
    assert st["state"] == "WI" and st["n_counties"] == 10
    for k in ("pred_silage_acres", "pred_grain_acres", "usda_silage_acres", "usda_grain_acres"):
        assert st[k] == pytest.approx(sum(r[k] for r in c["rows"]), abs=1)


def test_asof_tables(built):
    app, metas = built
    for rid in ("at-2024", "us-2022"):
        assert metas[rid]["has_asof"]
        a = strict_load(app / rid / "asof.json")
        n = len(a["cutoffs"])
        assert n >= 3 and a["cutoffs"] == sorted(a["cutoffs"])
        assert len(a["metrics"]) == n
        assert [m["cutoff"] for m in a["metrics"]] == a["cutoffs"]
        assert set(a["fields"]) == {f["properties"]["field_id"] for f in strict_load(app / rid / "fields.geojson")["features"]}
        for f in a["fields"].values():
            assert len(f["p"]) == len(f["low"]) == n
    # labelled region: accuracy per cutoff exists; unlabelled: no accuracy, only coverage
    at = strict_load(app / "at-2024" / "asof.json")
    assert all(m["all"] is not None for m in at["metrics"])
    us = strict_load(app / "us-2022" / "asof.json")
    assert all(m["all"] is None and m["low_evidence_share"] is not None for m in us["metrics"])
    assert metas["barc-2023"]["has_asof"] is False
    assert not (app / "barc-2023" / "asof.json").exists()


def test_harvest_reference_and_evidence_fields(built):
    app, metas = built
    ref = metas["at-2024"]["harvest_ref"]
    assert ref["silage_mid"] < ref["grain_mid"]
    props = [f["properties"] for f in strict_load(app / "at-2024" / "fields.geojson")["features"]]
    cut = [p for p in props if p["he"]]
    # the generator leaves stalks after grain and bare soil after silage
    assert any(p["post_ndti"] is not None for p in cut)
    sil = [p["post_ndti"] for p in cut if p["label"] == "silage" and p["post_ndti"] is not None]
    grn = [p["post_ndti"] for p in cut if p["label"] == "grain" and p["post_ndti"] is not None]
    assert np.median(sil) < 0.1 < np.median(grn)
    assert any(p["explanation"] for p in props) and any(p["explanation"] is None and p["pred"] for p in props)
    for p in props:
        assert set(p) >= {"peak_ndvi", "peak_day", "post_ndvi", "post_ndti", "explanation", "flip"}


# ---------------------------------------------------------------- small hand-made inputs

def test_fields_only_export(tmp_path):
    fields = write_fields(tmp_path / "f.geojson", [
        {"field_id": "x1", "label": "silage", "year": 2022, "country": "FR", "area_ha": 3.0},
        {"field_id": "x2", "label": None, "year": 2022, "country": "FR"},
    ])
    meta = export.export_region(fields, tmp_path / "app" / "fr-2022", title="France 2022")
    assert meta["has_series"] is False and meta["has_predictions"] is False
    assert meta["season"] is None
    props = {f["properties"]["field_id"]: f["properties"] for f in strict_load(tmp_path / "app" / "fr-2022" / "fields.geojson")["features"]}
    assert props["x2"]["area_ha"] > 0          # filled from the geometry
    assert props["x1"]["n_usable"] == 0
    assert strict_load(tmp_path / "app" / "fr-2022" / "series.json") == {}


def test_predictions_csv_with_nulls_and_validation(tmp_path):
    fields = write_fields(tmp_path / "f.geojson", [
        {"field_id": "a", "label": "silage", "year": 2024},
        {"field_id": "b", "label": "grain", "year": 2024},
        {"field_id": "c", "label": "grain", "year": 2024},
    ])
    pred = tmp_path / "p.csv"
    pred.write_text(
        "field_id,p_silage,harvest_start,harvest_end,pre_ndvi,pre_ndmi,low_evidence\n"
        "a,0.9,2024-09-01,2024-09-06,0.8,0.3,False\n"
        "b,0.7,,,,,True\n"
        "c,,,,,,\n", encoding="utf-8")
    meta = export.export_region(fields, tmp_path / "out" / "r", predictions_path=pred)
    m = meta["metrics"]
    assert m["n_predicted"] == 2                      # c has no probability
    assert (m["all"]["tp"], m["all"]["fp"]) == (1, 1)
    assert m["confident"]["n"] == 1                   # b is low evidence
    props = {f["properties"]["field_id"]: f["properties"] for f in strict_load(tmp_path / "out" / "r" / "fields.geojson")["features"]}
    assert props["a"]["hs"] == "2024-09-01" and props["b"]["hs"] is None and props["b"]["pre_ndvi"] is None
    assert props["c"]["pred"] is None and props["c"]["low"] is None

    bad = tmp_path / "bad.csv"
    bad.write_text("field_id,p_silage\na,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        export.export_region(fields, tmp_path / "out" / "r2", predictions_path=bad)
    bad.write_text("field_id,p_silage,harvest_start,harvest_end,pre_ndvi,pre_ndmi,low_evidence\na,1.5,,,,,False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="between 0 and 1"):
        export.export_region(fields, tmp_path / "out" / "r3", predictions_path=bad)


def test_optional_prediction_columns_and_asof_input(tmp_path):
    fields = write_fields(tmp_path / "f.geojson", [
        {"field_id": "a", "label": "silage", "year": 2024},
        {"field_id": "b", "label": "grain", "year": 2024},
    ])
    pd.DataFrame({
        "field_id": ["a", "b"], "p_silage": [0.9, 0.2],
        "harvest_start": ["2024-09-01", "2024-10-10"], "harvest_end": ["2024-09-06", "2024-10-15"],
        "pre_ndvi": [0.8, 0.4], "pre_ndmi": [0.3, 0.0], "low_evidence": [False, False],
        "explanation": ["green and wet at the cut", None], "flip_reason": [None, "would be silage if cut earlier"],
        "post_ndti": [0.04, None],
    }).to_parquet(tmp_path / "p.parquet", index=False)
    asof = tmp_path / "asof.csv"
    asof.write_text(
        "field_id,cutoff_date,p_silage,low_evidence\n"
        "a,2024-08-15,0.5,True\nb,2024-08-15,0.6,True\n"
        "a,2024-09-15,0.85,False\nb,2024-09-15,,\n", encoding="utf-8")
    meta = export.export_region(fields, tmp_path / "app" / "r", predictions_path=tmp_path / "p.parquet", asof_path=asof)
    props = {f["properties"]["field_id"]: f["properties"] for f in strict_load(tmp_path / "app" / "r" / "fields.geojson")["features"]}
    assert props["a"]["explanation"] == "green and wet at the cut" and props["a"]["flip"] is None
    assert props["b"]["flip"] == "would be silage if cut earlier"
    assert props["a"]["post_ndti"] == 0.04          # the model's own number wins
    assert props["b"]["post_ndti"] is None          # no series to read it from: stays missing
    a = strict_load(tmp_path / "app" / "r" / "asof.json")
    assert a["cutoffs"] == ["2024-08-15", "2024-09-15"]
    assert a["fields"]["b"]["p"] == [0.6, None]
    m0, m1 = a["metrics"]
    assert (m0["all"]["tp"], m0["all"]["fp"]) == (1, 1) and m0["low_evidence_share"] == 1.0
    assert m1["n_predicted"] == 1 and m1["all"]["n"] == 1
    assert meta["has_asof"] is True

    bad = tmp_path / "bad.csv"
    bad.write_text("field_id,cutoff_date,p_silage\na,2024-08-15,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="as-of predictions are missing"):
        export.export_region(fields, tmp_path / "app" / "r2", asof_path=bad)


def test_max_fields_keeps_metrics_on_all_fields(tmp_path):
    rows = [{"field_id": f"f{i}", "label": "silage" if i % 2 else "grain", "year": 2024} for i in range(20)]
    fields = write_fields(tmp_path / "f.geojson", rows)
    pred = pd.DataFrame({
        "field_id": [r["field_id"] for r in rows], "p_silage": [0.8 if i % 2 else 0.2 for i in range(20)],
        "harvest_start": None, "harvest_end": None, "pre_ndvi": math.nan, "pre_ndmi": math.nan, "low_evidence": False,
    })
    pred.to_parquet(tmp_path / "p.parquet", index=False)
    meta = export.export_region(fields, tmp_path / "app" / "r", predictions_path=tmp_path / "p.parquet", max_fields=5)
    assert meta["metrics"]["n_fields"] == 20 and meta["metrics"]["all"]["n"] == 20
    assert meta["n_fields_shown"] == 5
    assert len(strict_load(tmp_path / "app" / "r" / "fields.geojson")["features"]) == 5


def test_index_upsert_does_not_duplicate(tmp_path):
    fields = write_fields(tmp_path / "f.geojson", [{"field_id": "a", "label": None, "year": 2024}])
    export.export_region(fields, tmp_path / "app" / "r", title="One")
    export.export_region(fields, tmp_path / "app" / "r", title="Two")
    export.export_region(fields, tmp_path / "app" / "s", title="Other")
    regions = strict_load(tmp_path / "app" / "regions.json")["regions"]
    assert [(r["id"], r["title"]) for r in regions] == [("r", "Two"), ("s", "Other")]


def test_regreen_after_harvest():
    s = {"t": [250, 255, 262, 270, 290, 300], "q": [2, 2, 2, 1, 2, 2],
         "ndvi": [0.8, 0.2, 0.25, 0.9, 0.62, None], "ndmi": [None] * 6, "ndti": [None] * 6, "cf": [1] * 6}
    # day 270 is excluded (cloud), day 300 has no value: the green-up is read at day 290
    assert export.regreen(s, 255) == (0.62, 290)
    assert export.regreen(s, 295) == (None, None)
    assert export.regreen(None, 255) == (None, None)


def test_always_grain_yardstick(tmp_path):
    fields = write_fields(tmp_path / "f.geojson", [
        {"field_id": "a", "label": "grain", "year": 2024},
        {"field_id": "b", "label": "grain", "year": 2024},
        {"field_id": "c", "label": "grain", "year": 2024},
        {"field_id": "d", "label": "silage", "year": 2024},
        {"field_id": "e", "label": None, "year": 2024},
    ])
    pd.DataFrame({
        "field_id": list("abcde"), "p_silage": [0.1, 0.2, 0.7, 0.9, 0.5],
        "harvest_start": None, "harvest_end": None, "pre_ndvi": math.nan, "pre_ndmi": math.nan, "low_evidence": False,
    }).to_parquet(tmp_path / "p.parquet", index=False)
    m = export.export_region(fields, tmp_path / "app" / "r", predictions_path=tmp_path / "p.parquet")["metrics"]
    assert m["always_grain"] == {"accuracy": 0.75, "recall": 0.0}
    assert m["all"]["accuracy"] == 0.75


def test_model_evidence_and_display_options(tmp_path):
    """--evidence-reference: the export recomputes the evidence numbers with the project's
    feature code and marks which training-class median each value is closer to."""
    sample_data.build(["at-2024"], work=tmp_path / "work", app_data=tmp_path / "app0", n=12)
    src = tmp_path / "work" / "at-2024"
    ref = {
        "harv_days_since_peak": {"silage": 46.5, "grain": 55.5},
        "sen_ndvi_slope_rel": {"silage": -0.005, "grain": -0.014},
        "pre_ndvi_frac": {"silage": 0.83, "grain": 0.52},
        "pre_ndmi": {"silage": 0.32, "grain": 0.12},
        "post_ndti": {"silage": 0.14, "grain": 0.16},
    }
    (tmp_path / "run.json").write_text(json.dumps({"evidence_reference": ref}), encoding="utf-8")
    meta = export.export_region(
        src / "fields.geojson", tmp_path / "app" / "x", src / "timeseries.parquet", src / "predictions.parquet",
        evidence_reference=tmp_path / "run.json", label_name="Organisers' truth",
        model_name="toy_v1", trained_on="nothing real", min_clear=0.6,
    )
    assert meta["label_name"] == "Organisers' truth"
    assert meta["model"] == {"name": "toy_v1", "trained_on": "nothing real"}
    assert meta["evidence_ref"]["post_ndti"] == {"silage": 0.14, "grain": 0.16, "label": "residue after the cut"}
    props = [f["properties"] for f in strict_load(tmp_path / "app" / "x" / "fields.geojson")["features"]]
    with_ev = [p for p in props if p.get("ev")]
    assert len(with_ev) == len(props)
    for p in with_ev:
        assert set(p["ev"]) == set(ref)
        for k, e in p["ev"].items():
            assert e["s"] in (-1, 0, 1)
            if e["v"] is None:
                assert e["s"] == 0
            else:   # the side is the nearer class median, as in the model's explanation
                r = ref[k]
                assert e["s"] == (1 if abs(e["v"] - r["silage"]) < abs(e["v"] - r["grain"]) else -1)
    # a field that was cut shows a measured greenness fraction
    assert any(p["ev"]["pre_ndvi_frac"]["v"] is not None for p in with_ev if p["he"])

    # without the option nothing changes
    meta2 = export.export_region(src / "fields.geojson", tmp_path / "app" / "y", src / "timeseries.parquet",
                                 src / "predictions.parquet")
    assert meta2["evidence_ref"] is None and meta2["label_name"] == "Declared" and meta2["model"] is None


def test_display_name():
    assert export._display_name({"barc_farm": "Central Farm", "barc_field": "1-11"}) == "Central Farm, field 1-11"
    assert export._display_name({"name": "North 40", "barc_farm": "X", "barc_field": "1"}) == "North 40"
    assert export._display_name({}) is None


def test_run_demo_builds_from_a_data_root(tmp_path):
    """app/run_demo.py --no-serve with a data root laid out like the team's folder."""
    import importlib.util
    import shutil

    root = tmp_path / "root"
    sample_data.build(["barc-2021", "barc-2022", "barc-2023"], work=tmp_path / "work", app_data=tmp_path / "app0", n=8)
    for y in (2021, 2022, 2023):
        w = tmp_path / "work" / f"barc-{y}"
        (root / "labels" / f"barc_{y}").mkdir(parents=True)
        shutil.copy(w / "fields.geojson", root / "labels" / f"barc_{y}" / "fields.geojson")
        (root / "timeseries" / f"barc_{y}").mkdir(parents=True)
        shutil.copy(w / "timeseries.parquet", root / "timeseries" / f"barc_{y}" / "timeseries.parquet")
        d = root / "app" / "predictions_toy" / f"barc-{y}"
        d.mkdir(parents=True)
        if (w / "predictions.parquet").exists():
            shutil.copy(w / "predictions.parquet", d / "predictions.parquet")
        else:   # the 2023 sample has no predictions: write an all-null file in the right shape
            fids = [f["properties"]["field_id"] for f in json.loads((w / "fields.geojson").read_text())["features"]]
            pd.DataFrame({"field_id": fids, "p_silage": np.nan, "harvest_start": None, "harvest_end": None,
                          "pre_ndvi": np.nan, "pre_ndmi": np.nan, "low_evidence": None}).to_parquet(d / "predictions.parquet")

    spec = importlib.util.spec_from_file_location("run_demo", Path(__file__).resolve().parents[1] / "app" / "run_demo.py")
    run_demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_demo)
    out = tmp_path / "app_data"
    run_demo.build_beltsville(root, "toy", out)
    regions = strict_load(out / "regions.json")["regions"]
    assert [r["id"] for r in regions] == ["barc-2021", "barc-2022", "barc-2023"]
    assert all(r["group"] == "Beltsville, Maryland" and r["synthetic"] is False for r in regions)
    meta = strict_load(out / "barc-2021" / "meta.json")
    assert meta["label_name"] == "Organisers' truth" and meta["min_clear"] == 0.6


def test_cli_runs(tmp_path):
    fields = write_fields(tmp_path / "f.geojson", [{"field_id": "a", "label": "grain", "year": 2023}])
    export.main(["--fields", str(fields), "--out", str(tmp_path / "app" / "x"), "--title", "X 2023", "--synthetic"])
    assert strict_load(tmp_path / "app" / "x" / "meta.json")["synthetic"] is True
