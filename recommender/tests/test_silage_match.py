"""Offline tests for silage_match with a tiny fake county. Run: python3 recommender/tests/test_silage_match.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import silage_match as sm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIPS, YEAR = "24999", 2022
LAT0, LON0 = 39.50, -77.40
MILE_LAT = 1 / 69.0  # degrees latitude per mile


def make_county(root: Path, fields: list[dict]):
    """fields: id, dlat_miles, acres, p, label, abstain, drop, ndvi(list of (date, value))"""
    folder = root / "counties" / FIPS / str(YEAR)
    folder.mkdir(parents=True)
    feats, preds, ts = [], [], []
    for f in fields:
        lat = LAT0 + f["miles"] * MILE_LAT
        feats.append({"type": "Feature", "geometry": None,
                      "properties": {"field_id": f["id"], "centroid_lat": lat, "centroid_lon": LON0,
                                     "acres_reported": f["acres"]}})
        preds.append({"field_id": f["id"], "year": YEAR, "source_field_id": f["id"], "model": "fake",
                      "p_silage": f["p"], "label": "silage" if f["p"] >= 0.5 else "grain",
                      "abstain": f.get("abstain", False),
                      "evidence": json.dumps({"drop_date": f.get("drop")})})
        for date, nd in f["ndvi"]:
            # pick B04/B08 so (B08-B04)/(B08+B04) == nd
            b04 = 0.05
            ts.append({"field_id": f["id"], "date": date, "clear_fraction": 1.0, "B04": b04,
                       "B08": b04 * (1 + nd) / (1 - nd)})
    (folder / "fields.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    pd.DataFrame(ts).to_parquet(folder / "timeseries.parquet")
    res = root / "results"
    res.mkdir()
    pd.DataFrame(preds).to_parquet(res / f"XX_{YEAR}_{FIPS}_standard.parquet")
    return root / "counties", res


GREEN = [("2022-08-20", 0.80), ("2022-09-20", 0.30)]  # the Sep 20 value must be ignored for as_of Sep 1
FIELDS = [
    {"id": "grain_near", "miles": 2, "acres": 50, "p": 0.30, "drop": "2022-10-10", "ndvi": GREEN},
    {"id": "grain_mid", "miles": 8, "acres": 40, "p": 0.20, "drop": "2022-10-20", "ndvi": GREEN},
    {"id": "silage_bound", "miles": 1, "acres": 60, "p": 0.95, "drop": "2022-09-10", "ndvi": GREEN},
    {"id": "harvested", "miles": 1, "acres": 60, "p": 0.20, "drop": "2022-08-25", "ndvi": GREEN},
    {"id": "tiny", "miles": 1, "acres": 2, "p": 0.20, "drop": "2022-10-10", "ndvi": GREEN},
    {"id": "brown", "miles": 1, "acres": 50, "p": 0.20, "drop": "2022-10-10", "ndvi": [("2022-08-28", 0.35)]},
    {"id": "abstained", "miles": 1, "acres": 50, "p": 0.40, "abstain": True, "ndvi": GREEN},
    {"id": "grain_far", "miles": 40, "acres": 100, "p": 0.10, "drop": "2022-10-10", "ndvi": GREEN},
]


def dairies(*rows):
    return pd.DataFrame([{"dairy_id": i, "county": "Nowhere", "lat": LAT0 + m * MILE_LAT, "lon": LON0, "head": h}
                         for i, m, h in rows])


def run(dairy_df, **kw):
    tmp = Path(tempfile.mkdtemp())
    data_root, results = make_county(tmp, FIELDS)
    P = sm.Params(prior=0.2, use_prior=kw.pop("use_prior", False), **kw)
    return sm.recommend(FIPS, YEAR, "2022-09-01", P, dairies=dairy_df, data_root=data_root, results=results)


def test_funnel():
    r = run(dairies(("A", 0, 1000)))
    fu = r["funnel"]
    assert fu["fields"] == 8
    assert fu["abstained"] == 1 and fu["harvested_by_date"] == 1 and fu["silage_bound"] == 1
    assert fu["too_small"] == 1 and fu["browned_down_or_no_view"] == 1
    assert fu["candidate"] == 3, fu  # near, mid, far


def test_no_future_ndvi():
    tmp = Path(tempfile.mkdtemp())
    data_root, _ = make_county(tmp, FIELDS)
    nd = sm.ndvi_as_of(FIPS, YEAR, pd.Timestamp("2022-09-01"), data_root=data_root)
    assert abs(nd.loc["grain_near", "ndvi"] - 0.80) < 1e-6
    assert str(nd.loc["grain_near", "ndvi_date"].date()) == "2022-08-20"


def test_capacity_and_requests_respected():
    r = run(dairies(("A", 0, 1000), ("B", 3, 500)))
    by_field, by_dairy = {}, {}
    for a in r["assignments"]:
        by_field[a["field_id"]] = by_field.get(a["field_id"], 0) + a["tons"]
        by_dairy[a["dairy_id"]] = by_dairy.get(a["dairy_id"], 0) + a["tons"]
        assert a["road_miles"] <= 20
        assert a["field_id"] != "grain_far"
    cap = {"grain_near": int(0.7 * 50 * 20), "grain_mid": int(0.8 * 40 * 20)}
    for f, t in by_field.items():
        assert t <= cap[f], (f, t)
    for d in r["dairies"]:
        assert d["matched_tons"] <= d["request_tons"]


def test_contested_field_goes_to_closer_dairy():
    # both dairies want more than all supply; the near field should serve the dairy right next to it
    r = run(dairies(("A", 2, 5000), ("B", 9, 5000)), shortfall_frac=1.0)
    near = [a for a in r["assignments"] if a["field_id"] == "grain_near"]
    assert near and all(a["dairy_id"] == "A" for a in near), near
    total = sum(a["tons"] for a in r["assignments"])
    assert total == int(0.7 * 50 * 20) + int(0.8 * 40 * 20)
    assert any(c["check"] == "requests_covered" and c["status"] == "warn" for c in r["checks"])


def test_prior_shift():
    assert abs(sm.adjust_prior([0.5], 0.2)[0] - 0.2) < 1e-9
    p = sm.adjust_prior([0.1, 0.5, 0.9], 0.1)
    assert np.all(np.diff(p) > 0) and p[2] < 0.9
    r = run(dairies(("A", 0, 1000)), use_prior=True)
    assert r["county_silage_share_pct"]["after_prior"] < r["county_silage_share_pct"]["raw"]


def test_dairy_out_of_reach_dropped():
    r = run(dairies(("A", 0, 1000), ("FAR", 100, 1000)))
    assert [d["dairy_id"] for d in r["dairies"]] == ["A"]


def test_geojson_and_hindsight():
    r = run(dairies(("A", 0, 1000)))
    g = sm.to_geojson(r)
    kinds = {f["properties"]["kind"] for f in g["features"]}
    assert kinds == {"dairy", "source_field", "haul"}
    assert r["hindsight"]["held_pct"] == 100.0  # both candidates dropped in October


def test_frs_dairy_file():
    desc = ('<tr><td>REGISTRY ID</td><td><A HREF="x">110070115749</A></td></tr>'
            '<tr><td>LOCATION ADDRESS</td><td>1 ROAD, TOWN, MD 21701</td></tr>'
            '<tr><td>COUNTY NAME</td><td>FREDERICK COUNTY</td></tr>')
    raw = pd.DataFrame({"Name": ["SOME DAIRY"], "Description": [desc], "kml_layer": ["TOWN"],
                        "longitude": [LON0], "latitude": [LAT0]})
    d = sm.normalize_dairies(raw)
    assert d.loc[0, "dairy_id"] == "FRS-110070115749" and d.loc[0, "state"] == "MD"
    assert d.loc[0, "county"] == "Frederick" and "Name" not in d.columns
    r = run(raw)
    x = r["dairies"][0]
    assert x["head_assumed"] and x["head"] == 300 and x["request_tons"] == 450


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
