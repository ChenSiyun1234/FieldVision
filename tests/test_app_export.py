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
    assert cov["a"] == {"n_obs": 4, "n_usable": 2, "max_gap": 40}


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


def test_cli_runs(tmp_path):
    fields = write_fields(tmp_path / "f.geojson", [{"field_id": "a", "label": "grain", "year": 2023}])
    export.main(["--fields", str(fields), "--out", str(tmp_path / "app" / "x"), "--title", "X 2023", "--synthetic"])
    assert strict_load(tmp_path / "app" / "x" / "meta.json")["synthetic"] is True
