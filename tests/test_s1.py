import json

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from fieldvision.s1 import extract as s1
from fieldvision.s2.geometry import load_fields, make_blocks, project


def item(item_id="S1A_X", when="2022-07-26T22:59:15.750364Z", abs_orbit=44278, epsg=32618, x0=300000, y0=4400000):
    return {"id": item_id, "collection": s1.COLLECTION,
            "properties": {"datetime": when, "platform": "SENTINEL-1A", "sat:orbit_state": "ascending",
                           "sat:relative_orbit": 106, "sat:absolute_orbit": abs_orbit, "proj:epsg": epsg,
                           "proj:transform": [10.0, 0.0, x0, 0.0, -10.0, y0, 0, 0, 1]},
            "assets": {"vv": {"href": "x"}, "vh": {"href": "y"}}}


def test_db_of_linear_and_invalid():
    assert s1.to_db([0.1, 1.0])[0] == pytest.approx(-10.0)
    assert np.isnan(s1.to_db([0.0, -1.0, np.nan])).all()


def test_field_stats_uses_only_pixels_valid_in_both_polarisations():
    mask = np.ones((2, 2), bool)
    vv = np.array([[0.2, 0.4], [s1.NODATA, 0.1]], dtype="float32")
    vh = np.array([[0.02, 0.04], [0.03, 0.0]], dtype="float32")
    valid = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
    n_pixels, n_valid, vv_mean, vh_mean, vv_med, vh_med = s1.field_stats(mask, vv, vh, valid)
    assert (n_pixels, n_valid) == (4, 2)
    assert vv_mean == pytest.approx(0.3) and vh_mean == pytest.approx(0.03)
    assert vv_med == pytest.approx(0.3)


def test_aggregate_scene_means_dbs_and_orbit_columns(tmp_path):
    epsg, x0, y0 = 32618, 300000, 4400000
    field = project(box(x0 + 20, y0 - 120, x0 + 120, y0 - 20), epsg, 4326)
    path = tmp_path / "f.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(field), "properties": {"field_id": "a", "year": 2022}}]}))
    block = make_blocks(load_fields(path, 2022))[0]
    it = item(x0=x0, y0=y0)
    grid = s1.scene_grid(block, it)
    dims = grid[2]
    arrays = {"vv": np.full(dims, 0.2, "float32"), "vh": np.full(dims, 0.02, "float32")}
    rows, diag = s1.aggregate_scene(block, it, grid, arrays)
    row = rows.iloc[0]
    assert row.n_valid == row.n_pixels > 0 and row.valid_fraction == 1
    assert row.vv_mean == pytest.approx(0.2) and row.vv_db == pytest.approx(10 * np.log10(0.2))
    assert row.vh_vv_db == pytest.approx(-10.0)
    assert (row.orbit_state, row.relative_orbit, row.absolute_orbit) == ("ascending", 106, 44278)
    assert row.date == "2022-07-26" and list(rows.columns) == s1.COLUMNS
    # Grid is snapped to the item's 10 m grid.
    affine = grid[1]
    assert (affine.c - x0) % 10 == 0 and (affine.f - y0) % 10 == 0


def test_slices_of_one_pass_keep_the_one_with_most_valid_pixels():
    table = pd.DataFrame([
        dict(field_id="a", absolute_orbit=1, n_valid=5, scene_id="s1", datetime="2022-07-01T23:00:00"),
        dict(field_id="a", absolute_orbit=1, n_valid=9, scene_id="s2", datetime="2022-07-01T23:00:20"),
        dict(field_id="a", absolute_orbit=2, n_valid=3, scene_id="s3", datetime="2022-07-13T23:00:00")])
    out = s1.resolve_duplicates(table)
    assert out.scene_id.tolist() == ["s2", "s3"]


def test_fingerprint_covers_reused_s2_files(tmp_path):
    for rel in ("s1/a.py", "s2/core.py", "s2/geometry.py", "s2/catalog.py", "s2/extract.py"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text(rel)
    before = s1.fingerprint(tmp_path)
    (tmp_path / "s2" / "extract.py").write_text("changed")      # not used by s1
    assert s1.fingerprint(tmp_path) == before
    (tmp_path / "s2" / "core.py").write_text("changed")         # used by s1
    assert s1.fingerprint(tmp_path) != before
