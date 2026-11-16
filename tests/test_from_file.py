import json

import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from fieldvision.labels.from_file import InputError, convert, main, parse_map, parse_year_columns

MAP = parse_map(['Silage=silage', 'Grain=grain'])
FT = 200 / 0.3048006096012192          # 200 m in US survey feet, so a 200 m square is 4 ha


def square_ft(x, y, side=FT):
    return box(x, y, x + side, y + side)


def run(frame, **kw):
    base = dict(label_map=MAP, country='US', source='test', prefix='t', id_column='ID')
    base.update(kw)
    return convert(frame, **base)


def test_wide_layout_in_projected_feet():
    frame = gpd.GeoDataFrame({
        'ID': [1, 2, 3],
        'Corn_2021': ['Silage', 'Grain', None],
        'Corn_2022': ['Grain', '', 'Silage'],
        'Corn_2023': [None, 'Other', 'Grain'],
    }, geometry=[square_ft(1300000 + 2000 * i, 500000) for i in range(3)], crs='EPSG:2248')
    features, report = run(frame, year_columns=parse_year_columns(['Corn_2021=2021', 'Corn_2022=2022', 'Corn_2023=2023']),
                           split='external_eval')
    ids = {f['properties']['field_id']: f['properties'] for f in features}
    assert set(ids) == {'t-1-2021', 't-2-2021', 't-1-2022', 't-3-2022', 't-3-2023'}
    assert ids['t-1-2021']['label'] == 'silage' and ids['t-1-2022']['label'] == 'grain'
    assert ids['t-3-2023']['year'] == 2023 and ids['t-3-2023']['split'] == 'external_eval'
    for p in ids.values():
        assert p['area_ha'] == pytest.approx(4.0, rel=2e-3)
        assert -78 < p['centroid_lon'] < -76 and 38 < p['centroid_lat'] < 40
        assert p['block_id'].startswith('US-E')
    # one polygon labelled in several years is reused as-is
    geoms = {f['id']: f['geometry'] for f in features}
    assert geoms['t-1-2021'] == geoms['t-1-2022']
    assert report['dropped']['blank_label_2022'] == 1 and report['dropped']['blank_label_2023'] == 1
    assert report['dropped']['blank_label_2021'] == 1
    assert report['dropped']['unmapped_label_2023'] == 1 and report['unmapped_values'] == {'Other': 1}
    assert report['input_crs'] == 'EPSG:2248'


def test_long_layout_and_constant_year():
    frame = gpd.GeoDataFrame({'ID': ['a', 'a', 'b'], 'use': ['silage', ' GRAIN ', 'Grain'], 'yr': [2022, 2023.0, 2022]},
                             geometry=[box(15.0, 48.0, 15.003, 48.003)] * 2 + [box(15.01, 48.0, 15.013, 48.003)],
                             crs='EPSG:4326')
    features, report = run(frame, label_column='use', year_column='yr')
    assert [f['properties']['field_id'] for f in features] == ['t-a-2022', 't-a-2023', 't-b-2022']
    assert [f['properties']['label'] for f in features] == ['silage', 'grain', 'grain']
    assert features[0]['properties']['block_id'].startswith('US-E4')       # Europe: EPSG:3035 grid
    assert report['grid_crs'] == 'EPSG:3035'
    constant, _ = run(frame.iloc[[0, 2]], label_column='use', year=2024)
    assert {f['properties']['year'] for f in constant} == {2024}


def test_invalid_polygon_and_geometry_collection_are_repaired():
    bow_tie = Polygon([(15.0, 48.0), (15.004, 48.004), (15.0, 48.004), (15.004, 48.0)])
    collection = GeometryCollection([box(15.02, 48.0, 15.024, 48.004), LineString([(15.03, 48.0), (15.04, 48.0)])])
    multi = MultiPolygon([box(15.05, 48.0, 15.054, 48.004), box(15.06, 48.0, 15.064, 48.004)])
    frame = gpd.GeoDataFrame({'ID': [1, 2, 3], 'L': ['Silage'] * 3}, geometry=[bow_tie, collection, multi], crs=4326)
    features, report = run(frame, label_column='L', year=2024)
    assert len(features) == 3
    for f in features:
        assert f['geometry']['type'] in ('Polygon', 'MultiPolygon')
        assert f['properties']['area_ha'] > 1
    assert report['repair']['source_crs']['invalid_input'] == 1
    assert report['repair']['source_crs']['geometry_collection'] == 1
    assert report['repair']['source_crs']['repaired'] == 2


def test_stray_line_after_reprojection_does_not_survive():
    # a sliver polygon with a spike: repair in feet may leave a line; the output must be polygonal only
    spike = Polygon([(1300000, 500000), (1300000 + FT, 500000), (1300000 + FT, 500000 + FT),
                     (1300000, 500000 + FT), (1300000, 500000), (1300000 + 3 * FT, 500000)])
    frame = gpd.GeoDataFrame({'ID': [1], 'L': ['Grain']}, geometry=[spike], crs='EPSG:2248')
    features, _ = run(frame, label_column='L', year=2021)
    assert features[0]['geometry']['type'] in ('Polygon', 'MultiPolygon')
    assert features[0]['properties']['area_ha'] == pytest.approx(4.0, rel=2e-3)


def test_duplicate_ids_are_refused():
    frame = gpd.GeoDataFrame({'ID': [7, 7], 'L': ['Silage', 'Grain']},
                             geometry=[box(15, 48, 15.003, 48.003), box(15.01, 48, 15.013, 48.003)], crs=4326)
    with pytest.raises(InputError, match='Duplicate field_id'):
        run(frame, label_column='L', year=2024)
    # the same id in two different years is fine
    frame['Y'] = [2022, 2023]
    features, _ = run(frame, label_column='L', year_column='Y')
    assert len(features) == 2


def test_blank_and_unmapped_labels_are_dropped_and_counted():
    frame = gpd.GeoDataFrame({'ID': [1, 2, 3, 4], 'L': ['Silage', None, '  ', 'Fallow']},
                             geometry=[box(15 + 0.01 * i, 48, 15.003 + 0.01 * i, 48.003) for i in range(4)], crs=4326)
    features, report = run(frame, label_column='L', year=2024)
    assert len(features) == 1
    assert report['dropped'] == {'blank_label_2024': 2, 'unmapped_label_2024': 1}
    with pytest.raises(InputError, match='No row has a label'):
        run(frame.iloc[1:], label_column='L', year=2024)


def test_points_are_rejected_with_a_clear_message():
    frame = gpd.GeoDataFrame({'ID': [1, 2], 'L': ['Silage', 'Grain']}, geometry=[Point(15, 48), Point(15.1, 48)], crs=4326)
    with pytest.raises(InputError, match='only contains points.*polygon'):
        run(frame, label_column='L', year=2024)


def test_argument_errors():
    frame = gpd.GeoDataFrame({'ID': [1], 'L': ['Silage']}, geometry=[box(15, 48, 15.003, 48.003)], crs=4326)
    with pytest.raises(InputError, match='exactly one'):
        run(frame, label_column='L')
    with pytest.raises(InputError, match='not found'):
        run(frame, label_column='Missing', year=2024)
    with pytest.raises(InputError, match='map target'):
        parse_map(['Silage=corn'])
    with pytest.raises(InputError, match='no coordinate system'):
        run(gpd.GeoDataFrame({'ID': [1], 'L': ['Silage']}, geometry=[box(15, 48, 15.003, 48.003)]), label_column='L', year=2024)


def test_keep_min_area_and_constant_block():
    frame = gpd.GeoDataFrame({'ID': [1, 2], 'L': ['Silage', 'Grain'], 'farm': ['x', 'y']},
                             geometry=[box(15, 48, 15.003, 48.003), box(15.01, 48, 15.0102, 48.0002)], crs=4326)
    features, report = run(frame, label_column='L', year=2024, keep=['farm'], block_const='US-ONE', min_area_ha=1.0)
    assert len(features) == 1 and report['dropped']['below_min_area'] == 1
    assert features[0]['properties']['farm'] == 'x' and features[0]['properties']['block_id'] == 'US-ONE'
    with pytest.raises(InputError, match='contract names'):
        run(frame, label_column='L', year=2024, keep=['label'])


def test_command_line_end_to_end_and_no_overwrite(tmp_path, capsys):
    frame = gpd.GeoDataFrame({'OBJECTID': [1, 2], 'Corn_2021': ['Silage', 'Grain'], 'Corn_2022': ['Grain', None]},
                             geometry=[square_ft(1300000, 500000), square_ft(1303000, 500000)], crs='EPSG:2248')
    path = tmp_path / 'in.gpkg'
    frame.to_file(path, layer='fields', driver='GPKG')
    out = tmp_path / 'out'
    argv = [str(path), '--layer', 'fields', '--id-column', 'OBJECTID', '--year-columns', 'Corn_2021=2021', 'Corn_2022=2022',
            '--map', 'Silage=silage', '--map', 'Grain=grain', '--country', 'US', '--prefix', 'x', '--out', str(out),
            '--by-year-dirs']
    main(argv)
    data = json.loads((out / 'fields.geojson').read_text(encoding='utf-8'))
    assert [f['properties']['field_id'] for f in data['features']] == ['x-1-2021', 'x-1-2022', 'x-2-2021']
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['counts_per_class'] == {'silage': 1, 'grain': 2}
    assert summary['counts_per_year'] == {'2021': {'grain': 1, 'silage': 1}, '2022': {'grain': 1, 'silage': 0}}
    assert summary['dropped'] == {'blank_label_2022': 1}
    assert (out / 'fields.parquet').exists() and (out / '2022' / 'fields.geojson').exists()
    with pytest.raises(SystemExit):
        main(argv)                                            # refuses to overwrite
    point_file = tmp_path / 'pts.geojson'
    gpd.GeoDataFrame({'OBJECTID': [1]}, geometry=[Point(1, 1)], crs=4326).to_file(point_file, driver='GeoJSON')
    with pytest.raises(SystemExit, match='only contains points'):
        main([str(point_file), '--label-column', 'OBJECTID', '--year', '2024', '--map', 'a=silage',
              '--country', 'US', '--out', str(tmp_path / 'p')])


def utm_square(x, y, side=200.0):
    return box(x, y, x + side, y + side)


def test_unlabelled_where_rename_and_metric_crs():
    # CSB-like table in UTM 18N: filter to county 021 and corn (CDL2022 == 1), no labels, id without year
    frame = gpd.GeoDataFrame({
        'CSBID': ['241825000000001', '241825000000002', '241825000000003', '241825000000004'],
        'CNTYFIPS': ['021', '021', '021', '013'],
        'CDL2022': [1, 5, 1.0, 1],
        'CSBACRES': [10, 10, 20, 10],
    }, geometry=[utm_square(290000 + 1000 * i, 4360000) for i in range(4)], crs='EPSG:32618')
    features, report = convert(frame, label_map={}, country='US', source='nass-csb', prefix='csb', id_column='CSBID',
                               year=2022, unlabelled=True, year_in_id=False, metric_crs='EPSG:32618',
                               where={'CNTYFIPS': ['021'], 'CDL2022': ['1']}, keep=['CSBACRES=acres_reported'],
                               set_props={'county_fips': '24021'})
    assert [f['properties']['field_id'] for f in features] == ['csb-241825000000001', 'csb-241825000000003']
    assert report['where'] == {'conditions': {'CNTYFIPS': ['021'], 'CDL2022': ['1']},
                               'rows_matching': 2, 'rows_not_matching': 2}
    assert report['grid_crs'] == 'EPSG:32618'
    for f in features:
        p = f['properties']
        assert p['label'] is None and p['year'] == 2022 and p['county_fips'] == '24021'
        assert p['area_ha'] == pytest.approx(4.0, rel=1e-9)            # exact in the metric CRS itself
        assert p['block_id'] == 'US-E290N4360'                          # 10 km cells of the metric CRS
        assert 'CSBACRES' not in p
    assert [f['properties']['acres_reported'] for f in features] == [10, 20]


def test_unlabelled_argument_checks():
    frame = gpd.GeoDataFrame({'ID': [1, 1], 'L': ['a', 'b'], 'K': [1, 2]},
                             geometry=[box(15, 48, 15.003, 48.003), box(15.01, 48, 15.013, 48.003)], crs=4326)
    with pytest.raises(InputError, match='needs one --year'):
        run(frame, label_map={}, unlabelled=True, year_column='K')
    with pytest.raises(InputError, match='cannot be combined'):
        run(frame, unlabelled=True, year=2022)                          # MAP given
    with pytest.raises(InputError, match='No row matches'):
        run(frame, label_map={}, unlabelled=True, year=2022, where={'K': ['3']})
    with pytest.raises(InputError, match='contract names'):
        run(frame, label_map={}, unlabelled=True, year=2022, keep=['K=label'])
    with pytest.raises(InputError, match='Duplicate field_id'):
        run(frame, label_map={}, unlabelled=True, year=2022, year_in_id=False)


def test_command_line_unlabelled(tmp_path):
    frame = gpd.GeoDataFrame({'CSBID': ['a1', 'a2'], 'CDL2022': [1, 1], 'CSBACRES': [10, 10]},
                             geometry=[utm_square(290000, 4360000), utm_square(291000, 4360000)], crs='EPSG:32618')
    path = tmp_path / 'csb.parquet'
    frame.to_parquet(path)
    out = tmp_path / 'out'
    main([str(path), '--where', 'CDL2022=1', '--unlabelled', '--year', '2022', '--id-column', 'CSBID', '--no-year-in-id',
          '--prefix', 'csb', '--country', 'US', '--source', 'nass-csb', '--metric-crs', 'EPSG:32618',
          '--keep', 'CSBACRES=acres_reported', '--set', 'county_fips=24021', '--out', str(out)])
    data = json.loads((out / 'fields.geojson').read_text(encoding='utf-8'))
    props = [f['properties'] for f in data['features']]
    assert [p['field_id'] for p in props] == ['csb-a1', 'csb-a2']
    assert all(p['label'] is None and p['county_fips'] == '24021' and p['acres_reported'] == 10 for p in props)
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['counts_per_class'] == {'unlabelled': 2}
    assert summary['area_ha_total'] == pytest.approx(8.0)
    assert summary['input']['where']['rows_matching'] == 2
