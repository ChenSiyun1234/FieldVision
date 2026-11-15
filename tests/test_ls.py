import json

import numpy as np
import pytest
from shapely.geometry import box, mapping

from fieldvision.ls import extract as ls
from fieldvision.s2.core import BANDS, COLUMNS
from fieldvision.s2.geometry import load_fields, make_blocks, project

SR = {"raster:bands": [{"scale": 2.75e-05, "offset": -0.2, "nodata": 0}]}


def item(platform="landsat-9", tier="T1", epsg=32616, x0=230985, y0=5060715):
    assets = {a: dict(SR, href="x", **{"proj:transform": [30, 0, x0, 0, -30, y0]}) for a in ls.READ_ASSETS}
    assets["qa_pixel"]["raster:bands"] = [{"nodata": 1}]
    return {"id": "LC09_X", "collection": ls.COLLECTION, "assets": assets,
            "properties": {"datetime": "2023-10-01T16:40:32Z", "platform": platform, "proj:epsg": epsg,
                           "landsat:collection_category": tier}}


def test_scale_offset_and_nodata():
    dn = np.array([0, 10000, 20000], dtype="float32")
    out = ls.reflectance(dn, SR)
    assert np.isnan(out[0]) and out[1] == pytest.approx(10000 * 2.75e-05 - 0.2)


def test_qa_bits_0_to_5_mask_out_water_kept():
    clear_land, water = 1 << 6, (1 << 6) | (1 << 7)
    cloud, shadow, snow, cirrus, dilated, fill = 1 << 3, 1 << 4, 1 << 5, 1 << 2, 1 << 1, 1
    qa = np.array([clear_land, water, cloud, shadow, snow, cirrus, dilated, fill, np.nan])
    assert ls.qa_clear(qa).tolist() == [True, True] + [False] * 7


def test_only_landsat_8_9_tier_1():
    assert ls.unusable_reason(item()) is None
    assert "platform" in ls.unusable_reason(item(platform="landsat-7"))
    assert "tier" in ls.unusable_reason(item(tier="T2"))


def test_aggregate_maps_bands_to_sentinel_columns(tmp_path):
    epsg, x0, y0 = 32616, 230985, 5060715
    field = project(box(x0 + 60, y0 - 300, x0 + 300, y0 - 60), epsg, 4326)
    path = tmp_path / "f.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(field), "properties": {"field_id": "a", "year": 2023}}]}))
    block = make_blocks(load_fields(path, 2023))[0]
    it = item(x0=x0, y0=y0)
    grid = ls.scene_grid(block, it)
    dims = grid[2]
    dn = lambda r: np.full(dims, (r + 0.2) / 2.75e-05, "float32")
    arrays = {"blue": dn(0.03), "green": dn(0.06), "red": dn(0.04), "nir08": dn(0.40), "swir16": dn(0.20),
              "swir22": dn(0.10), "qa_pixel": np.full(dims, float(1 << 6))}
    rows, diag = ls.aggregate_scene(block, it, grid, arrays)
    row = rows.iloc[0]
    assert list(rows.columns) == COLUMNS
    assert row.B04 == pytest.approx(0.04, abs=1e-4) and row.B08 == pytest.approx(0.40, abs=1e-4)
    assert row.B8A == pytest.approx(row.B08) and row.B12 == pytest.approx(0.10, abs=1e-4)
    assert np.isnan(row.B05) and np.isnan(row.B06) and np.isnan(row.B07)
    assert row.n_clear == row.n_pixels > 0 and row.clear_fraction == 1
    assert row.collection == "landsat-c2-l2" and row.date == "2023-10-01"
    affine = grid[1]
    assert (affine.c - x0) % 30 == 0 and affine.a == 30          # counted on the 30 m grid


def test_implausible_scene_is_rejected():
    bands = {"B04": np.full((6, 6), -0.05), "B08": np.full((6, 6), 0.5)}
    with pytest.raises(ValueError, match="sanity check"):
        ls.check_plausible(bands, np.ones((6, 6), bool), np.ones((6, 6)))


def test_fingerprint_is_separate_from_sentinel2(tmp_path):
    for rel in ("ls/a.py", "s2/core.py", "s2/geometry.py", "s2/catalog.py", "s2/extract.py"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text(rel)
    before = ls.fingerprint(tmp_path)
    (tmp_path / "s2" / "extract.py").write_text("changed")
    assert ls.fingerprint(tmp_path) == before
