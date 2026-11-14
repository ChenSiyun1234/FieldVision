import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from fieldvision.s2 import indices
from fieldvision.s2.catalog import LEGACY, PRIMARY, calibration, check_vegetation, scale_reflectance
from fieldvision.s2.core import BANDS, COLUMNS, coverage_table, interior_geometry, resolve_duplicates, scl_clear, zonal_medians
from fieldvision.s2.geometry import load_fields, make_blocks, scene_grid


def test_scl_keeps_bare_soil_water_unclassified():
    assert np.flatnonzero(scl_clear(np.arange(12))).tolist() == [4, 5, 6, 7]


def test_zonal_medians_and_missing_footprint():
    labels = np.array([[1, 1, 1], [2, 2, 0]])
    bands = {"B04": np.array([[0.1, 0.3, 0.9], [np.nan, np.nan, 0.7]])}
    clear = np.array([[True, True, False], [True, True, True]])
    result = zonal_medians(labels, bands, clear, 3)
    assert result.n_pixels.tolist() == [3, 2, 0]
    assert result.n_clear.tolist() == [2, 0, 0]
    assert result.B04.iloc[0] == pytest.approx(0.2)
    assert result.clear_fraction.iloc[0] == pytest.approx(2 / 3)
    assert result.B04.iloc[1:].isna().all()


def test_duplicate_dates_prefer_clear_pixel_count_then_collection():
    table = pd.DataFrame([
        ["a", "2024-06-01", "s1", LEGACY, 20, 8, 0.4],
        ["a", "2024-06-01", "s2", PRIMARY, 10, 7, 0.7],
        ["b", "2024-06-01", "s3", LEGACY, 10, 5, 0.5],
        ["b", "2024-06-01", "s4", PRIMARY, 10, 5, 0.5]], columns=COLUMNS[:7])
    assert resolve_duplicates(table).scene_id.tolist() == ["s1", "s4"]


def test_small_field_fallback_counts_actual_grid_centers():
    affine = from_origin(0, 100, 10, 10)
    small = box(1, 1, 19, 19)
    selected, inset = interior_geometry(small, affine, (10, 10))
    assert not inset and selected.equals(small)
    large = box(0, 0, 80, 80)
    selected, inset = interior_geometry(large, affine, (10, 10))
    assert inset and selected.area < large.area


def test_indices_and_no_mutation():
    table = pd.DataFrame({"B02": [0.04, 0], "B04": [0.1, 0], "B06": [0.3, 0],
                          "B08": [0.5, 0], "B8A": [0.48, 0], "B11": [0.2, 0], "B12": [0.15, 0]})
    before = table.copy(deep=True)
    result = indices.add_indices(table)
    assert result.NDVI.iloc[0] == pytest.approx(2 / 3)
    assert result.NDMI.iloc[0] == pytest.approx(0.28 / 0.68)
    assert result.EVI.iloc[0] == pytest.approx(1 / 1.8)
    assert result.NBR.iloc[0] == pytest.approx(0.35 / 0.65)
    assert result.PSRI.iloc[0] == pytest.approx(0.2)
    assert np.isnan(result.NDVI.iloc[1])
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("collection,flag,expected", [(PRIMARY, None, 0.05), (LEGACY, True, 0.15), (LEGACY, False, 0.05)])
def test_scale_and_offset(collection, flag, expected):
    item = {"collection": collection, "properties": {"earthsearch:boa_offset_applied": flag}}
    asset = {"raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}]}
    values = scale_reflectance(np.array([1500, 0]), item, asset)
    assert values[0] == pytest.approx(expected)
    assert np.isnan(values[1])


def test_ambiguous_legacy_metadata_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        calibration({"collection": LEGACY, "properties": {}}, {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]})


def test_negative_red_guard():
    with pytest.raises(ValueError, match="sanity"):
        check_vegetation({"B04": np.full((5, 5), -0.03), "B08": np.full((5, 5), 0.4)},
                         np.full((5, 5), 4), np.ones((5, 5)))


def test_coverage_keeps_every_field_and_window_edges():
    fields = [{"field_id": "a"}, {"field_id": "b", "error": "invalid_geometry"}, {"field_id": "c"}]
    table = pd.DataFrame([dict(field_id="a", date="2024-06-01", n_pixels=10, n_clear=5, clear_fraction=0.5),
                          dict(field_id="a", date="2024-08-01", n_pixels=10, n_clear=5, clear_fraction=0.5)])
    coverage = coverage_table(fields, table, "2024-04-01", "2024-11-30").set_index("field_id")
    assert coverage.loc["a", "usable_dates"] == 2
    assert coverage.loc["a", "largest_gap_days"] == 121
    assert coverage.loc["b", "status"] == "invalid_geometry"
    assert coverage.loc["c", "status"] == "no_scenes"
    assert coverage.loc["c", "largest_gap_days"] == 243


def test_field_validation_and_multiple_utm_zones(tmp_path):
    features = [{"type": "Feature", "geometry": mapping(box(x, 48, x + 0.001, 48.001)),
                 "properties": {"field_id": str(x), "year": 2024}} for x in (5.99, 6.01)]
    path = tmp_path / "fields.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    fields = load_fields(path, 2024)
    blocks = make_blocks(fields)
    assert {block["epsg"] for block in blocks} == {32631, 32632}
    # A block can be read in a different scene UTM zone.
    item = {"properties": {"proj:epsg": 32632}, "assets": {"red": {"proj:transform": [10, 0, 0, 0, -10, 6000000]}}}
    grid = scene_grid(blocks[0], item)
    assert grid[0] == 32632 and min(grid[2]) > 0
