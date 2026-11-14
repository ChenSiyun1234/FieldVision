import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from fieldvision.s2 import indices
from fieldvision.s2.catalog import LEGACY, PLANETARY, PRIMARY, calibration, check_vegetation, scale_reflectance
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


@pytest.mark.parametrize("collection,flag,expected", [(PRIMARY, None, 0.05), (LEGACY, True, 0.15)])
def test_scale_and_offset(collection, flag, expected):
    item = {"collection": collection, "properties": {"earthsearch:boa_offset_applied": flag}}
    asset = {"raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}]}
    values = scale_reflectance(np.array([1500, 0]), item, asset)
    assert values[0] == pytest.approx(expected)
    assert np.isnan(values[1])


def test_ambiguous_legacy_metadata_rejected():
    asset = {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}
    with pytest.raises(ValueError, match="ambiguous"):
        calibration({"collection": LEGACY, "properties": {}}, asset)
    # flag=False was measured to be wrong on real 2022 scenes; never trust it.
    with pytest.raises(ValueError, match="unreliable"):
        calibration({"collection": LEGACY, "properties": {"earthsearch:boa_offset_applied": False}}, asset)


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


def test_label_block_id_keeps_fields_together_within_size(tmp_path):
    # Two fields 1 km apart straddle a 6 km grid line but share a label block.
    features = [{"type": "Feature", "geometry": mapping(box(x, 48.0, x + 0.002, 48.002)),
                 "properties": {"field_id": f"f{i}", "year": 2024, "block_id": "AT-E1/N2"}}
                for i, x in enumerate((15.0, 15.013))]
    features.append({"type": "Feature", "geometry": mapping(box(15.3, 48.0, 15.302, 48.002)),
                     "properties": {"field_id": "far", "year": 2024, "block_id": "AT-E1/N2"}})
    path = tmp_path / "fields.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    blocks = make_blocks(load_fields(path, 2024), 6000)
    assert sorted(len(b["fields"]) for b in blocks) == [1, 2]
    assert all("/" not in b["id"] and "AT-E1-N2" in b["id"] for b in blocks)


def test_2022_fallback_fills_partial_c1_catalog(monkeypatch):
    from fieldvision.s2 import catalog
    def item(scene_id, day, tile="33UWP"):
        return {"id": scene_id, "properties": {"datetime": f"2022-07-{day}T10:00:00Z",
                                               "s2:mgrs_tile": tile, "platform": "sentinel-2a"}}
    current = item("current", "01")
    old_duplicate = item("old-duplicate", "01")
    old_missing = item("old-missing", "06")
    old_other_tile = item("old-tile", "01", "33UWN")
    monkeypatch.setattr(catalog, "search", lambda collection, *args, **kwargs: [current] if collection == PRIMARY
                        else [old_duplicate, old_missing, old_other_tile])
    items, counts = catalog.find_scenes([15, 48, 16, 49], "2022-04-01", "2022-11-30", fallback=LEGACY)
    assert {item["id"] for item in items} == {"current", "old-missing", "old-tile"}
    assert counts == {"primary_scenes": 1, "fallback_scenes": 2, "fallback_source": LEGACY}
    assert {item["collection"] for item in items} == {PRIMARY, LEGACY}


def test_planetary_keeps_newest_baseline_and_platform_case(monkeypatch):
    from fieldvision.s2 import catalog
    def item(scene_id, baseline_value, platform="Sentinel-2A"):
        return {"id": scene_id, "properties": {"datetime": "2022-07-01T15:58:31Z", "s2:mgrs_tile": "18SUJ",
                                               "platform": platform, "s2:processing_baseline": baseline_value}}
    monkeypatch.setattr(catalog, "search", lambda *args, **kwargs: [item("n0400", "04.00"), item("n0510", "05.10")])
    items = catalog.search_source(PLANETARY, [0, 0, 1, 1], "2022-04-01", "2022-11-30")
    assert [i["id"] for i in items] == ["n0510"] and items[0]["collection"] == PLANETARY
    # Same acquisition in Earth Search and Planetary Computer must share a key.
    earth = {"id": "S2A_18SUJ_20220701_0_L2A", "properties": {"datetime": "2022-07-01T16:12:22Z",
             "mgrs:utm_zone": 18, "mgrs:latitude_band": "S", "mgrs:grid_square": "UJ", "platform": "sentinel-2a"}}
    assert catalog.acquisition_key(earth) == catalog.acquisition_key(items[0])


@pytest.mark.parametrize("baseline_value,expected", [("05.10", 0.05), ("04.00", 0.05), ("03.01", 0.15)])
def test_planetary_offset_follows_processing_baseline(baseline_value, expected):
    item = {"collection": PLANETARY, "properties": {"s2:processing_baseline": baseline_value}}
    assert scale_reflectance(np.array([1500.0]), item, {})[0] == pytest.approx(expected)
    with pytest.raises(ValueError, match="baseline"):
        calibration({"collection": PLANETARY, "properties": {}}, {})


def test_local_raster_pipeline_partial_resume_and_overlap(tmp_path, monkeypatch):
    """Exercise real raster reads, checkpoint recovery and final table assembly."""
    import rasterio
    from fieldvision.s2 import extract
    from fieldvision.s2.catalog import ASSETS
    from fieldvision.s2.geometry import project

    affine = from_origin(500000, 5300000, 10, 10)
    geometry = project(box(500010, 5299910, 500090, 5299990), 32633, 4326)
    features = [{"type": "Feature", "geometry": mapping(geometry),
                 "properties": {"field_id": name, "year": 2024}} for name in ("a", "b")]
    fields = tmp_path / "fields.geojson"
    fields.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    assets = {}
    for name in (*BANDS, "SCL"):
        scl = name == "SCL"
        asset_key = "scl" if scl else ASSETS[name]
        # Native 20 m SCL and red-edge/SWIR grids test nearest resampling.
        resolution = 10 if name in ("B02", "B03", "B04", "B08") else 20
        size = 100 // resolution
        path = tmp_path / f"{name}.tif"
        values = np.full((size, size), 4 if scl else 6000 if name == "B08" else 1500, dtype="uint16")
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1,
                           dtype="uint16", crs="EPSG:32633", transform=from_origin(500000, 5300000, resolution, resolution),
                           nodata=0) as dataset:
            dataset.write(values, 1)
        assets[asset_key] = {"href": str(path), "proj:transform": list(affine)[:6],
                             "raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}]}
    items = [{"id": f"scene-{day}", "collection": PRIMARY, "geometry": mapping(geometry), "assets": assets,
              "properties": {"datetime": f"2024-06-{day}T10:00:00Z", "proj:epsg": 32633}} for day in ("01", "11")]
    monkeypatch.setattr(extract, "find_scenes", lambda *args: (items, {"primary_scenes": 2, "fallback_scenes": 0}))
    real_read = extract.read_band
    calls = []
    fail_second = [True]
    def read(item, name, grid):
        calls.append(item["id"])
        if item["id"] == "scene-11" and fail_second[0]:
            raise OSError("temporary read failure")
        return real_read(item, name, grid)
    monkeypatch.setattr(extract, "read_band", read)
    args = Namespace(fields=fields, out=tmp_path / "run", year=2024, start="04-01", end="11-30",
                     workers=4, scene_batch=2, block_size=6000, max_cloud=80, collection=PRIMARY,
                     no_fallback=False, fallback_source=PLANETARY, min_clear_pixels=1, min_clear_fraction=0.2)
    assert extract.run(args) == 2
    first = pd.read_parquet(args.out / "timeseries.parquet")
    assert len(first) == 2 and set(first.field_id) == {"a", "b"}
    assert first.B04.tolist() == pytest.approx([0.05, 0.05])
    assert first.n_clear.iloc[0] == first.n_clear.iloc[1] > 0
    assert first.clear_fraction.eq(1).all()
    fail_second[0] = False
    calls.clear()
    assert extract.run(args) == 0
    assert set(calls) == {"scene-11"}
    result = pd.read_parquet(args.out / "timeseries.parquet")
    assert len(result) == 4
    calls.clear()
    assert extract.run(args) == 0
    assert not calls
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert manifest["scene_failures"] == 0
    assert manifest["fields_with_usable_dates"] == 2
    args.max_cloud = 70
    with pytest.raises(ValueError, match="different inputs/options/code"):
        extract.run(args)
