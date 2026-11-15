import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import Affine, from_origin
from shapely.geometry import box, mapping

from fieldvision.s2 import extract
from fieldvision.s2.catalog import ASSETS, LEGACY, PRIMARY
from fieldvision.s2.core import BANDS, COLUMNS, interior_geometry, polygon_mask, resolve_duplicates, scl_clear
from fieldvision.s2.geometry import project
from fieldvision.stream import pixels


def _arrays(dimensions, offset=0.0):
    rows, cols = np.indices(dimensions)
    values = 0.05 + rows * 0.002 + cols * 0.001 + offset
    result = {band: values.astype("float32").copy() for band in BANDS}
    result["B08"] += 0.4
    result["SCL"] = np.full(dimensions, 4, dtype="uint8")
    return result


def _row(field_id, scene_id, n_pixels=10, n_clear=6, collection=PRIMARY, day="01"):
    return dict(field_id=field_id, date=f"2024-06-{day}", scene_id=scene_id,
                collection=collection, n_pixels=n_pixels, n_clear=n_clear,
                clear_fraction=n_clear / n_pixels if n_pixels else 0.0,
                **{band: 0.2 for band in BANDS})


@pytest.mark.parametrize("size", [3, 4, 12])
def test_chip_window_uses_centroid_and_frozen_inset(size):
    affine = from_origin(1000, 2000, 10, 10)
    dimensions = (20, 30)
    geometry = box(1057, 1853, 1131, 1929)
    selected, _ = interior_geometry(geometry, affine, dimensions)
    row, col, transform, inset = pixels.chip_window(geometry, affine, dimensions, size)
    center_col, center_row = ~affine * (geometry.centroid.x, geometry.centroid.y)
    assert row == int(np.floor(center_row)) - size // 2
    assert col == int(np.floor(center_col)) - size // 2
    assert transform == affine * Affine.translation(col, row)
    assert inset.equals(selected)


@pytest.mark.parametrize("row,col", [(0, 0), (0, 4), (0, 8), (3, 0),
                                         (3, 8), (6, 0), (6, 4), (6, 8)])
def test_cut_chip_pads_every_edge_and_preserves_grid_alignment(row, col):
    dimensions, size = (7, 9), 4
    affine = from_origin(0, 70, 10, 10)
    x, y = affine * (col + 0.5, row + 0.5)
    geometry = box(x - 2, y - 2, x + 2, y + 2)
    arrays = _arrays(dimensions)
    arrays["SCL"] = (np.arange(np.prod(dimensions)).reshape(dimensions) % 12).astype("uint8")
    chip = pixels.cut_chip(geometry, (32633, affine, dimensions, [geometry]), arrays, size)
    assert chip["bands"].shape == (len(BANDS), size, size)
    assert chip["bands"].dtype == np.uint16
    assert chip["scl"].shape == chip["inside"].shape == (size, size)
    decoded = pixels.decode_bands(chip["bands"])
    expected_mask = polygon_mask(geometry, affine, dimensions)
    for chip_row in range(size):
        for chip_col in range(size):
            source_row, source_col = row - size // 2 + chip_row, col - size // 2 + chip_col
            if 0 <= source_row < dimensions[0] and 0 <= source_col < dimensions[1]:
                expected = [arrays[band][source_row, source_col] for band in BANDS]
                np.testing.assert_allclose(decoded[:, chip_row, chip_col], expected, atol=5.1e-5)
                assert chip["scl"][chip_row, chip_col] == arrays["SCL"][source_row, source_col]
                assert bool(chip["inside"][chip_row, chip_col]) == expected_mask[source_row, source_col]
            else:
                assert np.all(chip["bands"][:, chip_row, chip_col] == 65535)
                assert np.isnan(decoded[:, chip_row, chip_col]).all()
                assert chip["scl"][chip_row, chip_col] == 0
                assert chip["inside"][chip_row, chip_col] == 0


def test_encoding_roundtrip_preserves_zero_and_missing_sentinel():
    values = np.array([0.0, 0.0001, 0.12346, 0.5, 1.0, np.nan, np.inf, -np.inf, -0.01, 1.01])
    original = values.copy()
    encoded = pixels.encode_bands(values)
    assert encoded.dtype == np.uint16
    np.testing.assert_array_equal(encoded[:5], [0, 1, 1235, 5000, 10000])
    np.testing.assert_array_equal(encoded[5:], np.full(5, 65535))
    decoded = pixels.decode_bands(encoded)
    np.testing.assert_allclose(decoded[:5], values[:5], atol=5.1e-5)
    assert np.isnan(decoded[5:]).all()
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("geometry", [box(20, 20, 160, 160), box(21, 21, 39, 39)])
def test_inside_is_cropped_frozen_polygon_mask(geometry):
    affine, dimensions, size = from_origin(0, 200, 10, 10), (20, 20), 6
    selected, _ = interior_geometry(geometry, affine, dimensions)
    full_mask = polygon_mask(selected, affine, dimensions)
    row, col, _, _ = pixels.chip_window(geometry, affine, dimensions, size)
    expected = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if 0 <= row + r < dimensions[0] and 0 <= col + c < dimensions[1]:
                expected[r, c] = full_mask[row + r, col + c]
    chip = pixels.cut_chip(geometry, (32633, affine, dimensions, [geometry]), _arrays(dimensions), size)
    np.testing.assert_array_equal(chip["inside"], expected)


def test_chip_clear_inside_medians_equal_frozen_full_field_when_it_fits():
    affine, dimensions = from_origin(0, 100, 10, 10), (10, 10)
    geometry = box(10, 10, 90, 90)
    arrays = _arrays(dimensions)
    arrays["SCL"][2, 2] = 9
    arrays["B03"][3, 3] = np.nan
    arrays["B02"][4, 4] = -0.1
    arrays["B12"][5, 5] = 1.1
    item = {"id": "scene", "collection": PRIMARY,
            "properties": {"datetime": "2024-06-01T10:00:00Z"}}
    grid = (32633, affine, dimensions, [geometry])
    table, _ = extract.aggregate_scene({"fields": [{"field_id": "field"}]}, item, grid, arrays)
    chip = pixels.cut_chip(geometry, grid, arrays, 12)
    decoded = pixels.decode_bands(chip["bands"])
    valid = chip["inside"].astype(bool) & scl_clear(chip["scl"]) & np.isfinite(decoded).all(axis=0)
    assert valid.sum() == table.n_clear.iloc[0]
    assert np.count_nonzero(chip["inside"]) == table.n_pixels.iloc[0]
    np.testing.assert_allclose(np.median(decoded[:, valid], axis=1),
                               table.loc[0, list(BANDS)].to_numpy(dtype=float), atol=5.1e-5)


def test_select_chips_resolves_duplicates_before_applying_full_field_clear_rule():
    rows = [
        _row("a", "loser-clear", 10, 9),
        _row("a", "winner-cloudy", 20, 10),
        _row("b", "legacy", collection=LEGACY),
        _row("b", "primary"),
        _row("c", "below", 100, 59),
        _row("d", "above", 10, 7),
        _row("e", "empty", 0, 0),
    ]
    table = pd.DataFrame(rows, columns=COLUMNS)
    chips = {(row["field_id"], row["date"], row["scene_id"]):
             {"bands": np.full((len(BANDS), 4, 4), 1000 + index, dtype="uint16"),
              "scl": np.full((4, 4), 9, dtype="uint8"), "inside": np.ones((4, 4), dtype="uint8")}
             for index, row in enumerate(rows)}
    result = pixels.select_chips(table, chips, min_clear_fraction=0.6, chip_size=4)
    assert result["field_id"].tolist() == ["b", "d"]
    assert result["scene_id"].tolist() == ["primary", "above"]
    assert result["date"].tolist() == ["2024-06-01", "2024-06-01"]
    np.testing.assert_array_equal(result["bands"][:, 0, 0, 0], [1003, 1005])
    assert result["bands"].shape == (2, len(BANDS), 4, 4)
    assert result["scl"].shape == result["inside"].shape == (2, 4, 4)
    empty = pixels.select_chips(table.iloc[:0], {}, min_clear_fraction=0.6, chip_size=4)
    assert empty["bands"].shape == (0, len(BANDS), 4, 4)
    assert empty["scl"].shape == empty["inside"].shape == (0, 4, 4)


def test_check_against_reports_band_tolerance_and_field_coverage():
    reference = pd.DataFrame([_row("a", "s1"), _row("b", "s2"), _row("reference-only", "s3")])
    current = pd.DataFrame([_row("a", "s1"), _row("b", "s2"), _row("new-only", "s4")])
    current.loc[current.field_id == "a", "B04"] += 5e-7
    current.loc[current.field_id == "b", "B04"] += 2e-6
    report = pixels.check_against(current, reference)
    assert report["rows_compared"] == 2
    assert report["same_scene_id"] == 2
    assert report["rows_agree"] == 1
    assert report["agreement_fraction"] == pytest.approx(0.5)
    assert report["fields_compared"] == 2
    assert report["fields_only_new"] == 1
    assert report["fields_only_reference"] == 1
    assert report["fields_only_one"] == 2
    assert pixels.check_against(current.iloc[:0], reference)["agreement_fraction"] == 0


def test_check_command_enforces_agreement_and_rejects_empty_comparisons(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    reference_path = tmp_path / "reference.parquet"
    reference = pd.DataFrame([_row("a", "scene-a"), _row("b", "scene-b")])
    reference.to_parquet(reference_path, index=False)
    current = reference.copy()
    current.loc[1, "B04"] += 0.01
    current.to_parquet(run / "timeseries.parquet", index=False)
    command = ["check", "--run", str(run), "--reference", str(reference_path)]
    assert pixels.main(command) == 1
    assert json.loads(capsys.readouterr().out)["agreement_fraction"] == 0.5
    assert pixels.main([*command, "--min-agree", "0.5"]) == 0
    assert json.loads(capsys.readouterr().out)["rows_agree"] == 1
    current["field_id"] = ["other-a", "other-b"]
    current.to_parquet(run / "timeseries.parquet", index=False)
    assert pixels.main([*command, "--min-agree", "0"]) == 1
    assert json.loads(capsys.readouterr().out)["rows_compared"] == 0


def test_synthetic_pipeline_rebuilds_partial_blocks_and_skips_complete(tmp_path, monkeypatch):
    affine = from_origin(500000, 5300000, 10, 10)
    geometry = project(box(500010, 5299910, 500090, 5299990), 32633, 4326)
    features = [{"type": "Feature", "geometry": mapping(geometry),
                 "properties": {"field_id": name, "year": 2024}}
                for name in ("a", "b")]
    fields = tmp_path / "fields.geojson"
    fields.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    assets = {"scl" if name == "SCL" else ASSETS[name]:
              {"href": f"unused-{name}.tif", "proj:transform": list(affine)[:6],
               "raster:bands": [{"scale": 0.0001, "offset": 0.0, "nodata": 0}]}
              for name in (*BANDS, "SCL")}
    items = [{"id": scene_id, "collection": PRIMARY, "geometry": mapping(geometry), "assets": assets,
              "properties": {"datetime": f"2024-06-{day}T10:00:00Z", "proj:epsg": 32633}}
             for scene_id, day in (("cloudier", "01"), ("winner", "01"), ("retry", "11"))]
    searches, reads, events, reference_frames = [], [], [], []
    state = {"fail": True, "offset": 0.0}
    aggregated_arrays = set()
    real_aggregate, real_cut = extract.aggregate_scene, pixels.cut_chip

    def find(*args):
        searches.append(args)
        return items, {"primary_scenes": 3, "fallback_scenes": 0}

    def read(item, name, grid):
        reads.append((item["id"], name))
        if item["id"] == "retry" and name == "B03" and state["fail"]:
            raise OSError("temporary read failure")
        offset = state["offset"] + {"cloudier": 0.0, "winner": 0.01, "retry": 0.03}[item["id"]]
        arrays = _arrays(grid[2], offset)
        arrays["B02"][0, 0] = -0.1
        if item["id"] == "cloudier":
            arrays["SCL"][::2] = 9
        return arrays[name]

    def aggregate(block, item, grid, arrays, max_blue=None):
        table, diagnostic = real_aggregate(block, item, grid, arrays, max_blue)
        aggregated_arrays.add(id(arrays))
        events.append(("aggregate", item["id"]))
        reference_frames.append(table.copy(deep=True))
        return table, diagnostic

    def cut(geometry, grid, arrays, chip_size):
        assert id(arrays) in aggregated_arrays
        assert np.isnan(arrays["B02"][0, 0])
        events.append(("cut", None))
        return real_cut(geometry, grid, arrays, chip_size)

    monkeypatch.setattr(pixels, "find_scenes", find)
    monkeypatch.setattr(pixels, "read_band", read)
    monkeypatch.setattr(extract, "aggregate_scene", aggregate)
    monkeypatch.setattr(pixels, "cut_chip", cut)
    args = pixels.parser().parse_args(["--fields", str(fields), "--year", "2024", "--out", str(tmp_path / "run"),
                                      "--workers", "2", "--scene-batch", "2", "--chip", "12"])
    assert args.min_clear_fraction == 0.6
    assert pixels.run(args) == 2
    first_table = pd.read_parquet(args.out / "timeseries.parquet")
    assert first_table.scene_id.tolist() == ["winner", "winner"]
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["scene_failures"] == 1
    assert manifest["n_obs"] == 2
    block_paths = list((args.out / "blocks").glob("*.parquet"))
    assert len(block_paths) == 1
    block_path = block_paths[0]
    assert block_path.with_suffix(".npz").is_file()
    block_status = json.loads(block_path.with_suffix(".json").read_text())
    assert block_status["status"] == "partial"
    assert set(block_status["scene_errors"]) == {"retry"}
    pd.testing.assert_frame_equal(pd.read_parquet(block_path), first_table)
    first_chips = pixels.load_chips(args.out / "chips.npz")
    assert first_chips["scene_id"].tolist() == ["winner", "winner"]
    assert Counter(scene for scene, _ in reads) == {item["id"]: len(BANDS) + 1 for item in items}
    assert len(searches) == 1

    state.update(fail=False, offset=0.02)
    reads.clear()
    reference_frames.clear()
    aggregated_arrays.clear()
    events.clear()
    assert pixels.run(args) == 0
    assert Counter(scene for scene, _ in reads) == {item["id"]: len(BANDS) + 1 for item in items}
    assert len(searches) == 1
    assert events[0] == ("aggregate", "cloudier")
    result = pd.read_parquet(args.out / "timeseries.parquet")
    expected = resolve_duplicates(pd.concat(reference_frames, ignore_index=True))
    pd.testing.assert_frame_equal(result, expected)
    assert len(result) == 4
    assert result.query("date == '2024-06-01'").B04.to_numpy() == pytest.approx(first_table.B04.to_numpy() + 0.02)
    report = pixels.check_against(result, expected)
    assert report["rows_compared"] == report["same_scene_id"] == report["rows_agree"] == 4
    assert report["agreement_fraction"] == 1.0
    pd.testing.assert_frame_equal(pd.read_parquet(block_path), result)
    block_status = json.loads(block_path.with_suffix(".json").read_text())
    assert block_status["status"] == "complete" and not block_status["scene_errors"]
    merged = pixels.load_chips(args.out / "chips.npz")
    block_chips = pixels.load_chips(block_path.with_suffix(".npz"))
    assert merged["band_names"].tolist() == list(BANDS)
    assert merged["chip_size"] == 12
    assert merged["scale"] == 10000
    assert merged["missing"] == 65535
    assert merged["bands"].dtype == np.uint16
    assert merged["scl"].dtype == np.uint8
    assert merged["inside"].dtype == np.bool_
    assert merged["bands"].shape == (4, len(BANDS), 12, 12)
    assert merged["scl"].shape == merged["inside"].shape == (4, 12, 12)
    keys = list(zip(merged["field_id"], merged["date"], merged["scene_id"]))
    assert keys == list(result[["field_id", "date", "scene_id"]].itertuples(index=False, name=None))
    assert len(set(keys)) == len(keys)
    for key in ("field_id", "date", "scene_id", "bands", "scl", "inside"):
        np.testing.assert_array_equal(merged[key], block_chips[key])
    decoded = pixels.decode_bands(merged["bands"])
    for index, row in result.iterrows():
        clear = merged["inside"][index] & scl_clear(merged["scl"][index]) & np.isfinite(decoded[index]).all(axis=0)
        assert int(clear.sum()) == row.n_clear
        np.testing.assert_allclose(np.median(decoded[index][:, clear], axis=1),
                                   row[list(BANDS)].to_numpy(dtype=float), atol=5.1e-5)
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["scene_failures"] == 0
    assert manifest["fields_with_chips"] == 2 and manifest["n_obs"] == 4

    reads.clear()
    events.clear()
    assert pixels.run(args) == 0
    assert not reads and not events
    assert len(searches) == 1
    pd.testing.assert_frame_equal(pd.read_parquet(args.out / "timeseries.parquet"), result)
    resumed = pixels.load_chips(args.out / "chips.npz")
    for key in merged:
        np.testing.assert_array_equal(resumed[key], merged[key])
    args.max_cloud = 70
    with pytest.raises(ValueError, match="different inputs/options/code"):
        pixels.run(args)

