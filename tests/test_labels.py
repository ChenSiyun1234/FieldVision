from collections import Counter

import pytest

from fieldvision.labels.core import balance, block_id, field_id
from fieldvision.labels.core import normalize, TO_WGS84
from fieldvision.labels.adapters import get_adapter
from shapely.geometry import box, mapping, Polygon
from shapely.ops import transform


def test_grid_edges_and_negative_coordinates():
    assert block_id('AT', 4680999, 2799999) == 'AT-E4680N2790'
    assert block_id('AT', 4690000, 2800000) == 'AT-E4690N2800'
    assert block_id('NL', -1, -10001) == 'NL-E-10N-20'


def test_id_namespace_and_stability():
    assert field_id('AT', 2024, 106323636) == 'at2024-106323636'
    assert field_id('NL', 2025, 'abc/def') == 'nl2025-abc%2Fdef'
    assert field_id('AT', 2024, 1) != field_id('AT', 2025, 1)
    assert field_id('AT', 2024, 1) != field_id('NL', 2024, 1)
    with pytest.raises(ValueError):
        field_id('AT', 2024, None)


def test_balance_regions_cap_duplicates_and_reproducibility():
    features = []
    for region in ['west', 'east', 'south']:
        for block, sizes in [('dense', (70, 90)), ('sparse', (4, 2)), ('single', (9, 0))]:
            for label, size in zip(['silage', 'grain'], sizes):
                for i in range(size):
                    features.append({'properties': {'region': region, 'block_id': region + block,
                                     'label': label, 'field_id': f'{region}-{block}-{label}-{i}'}})
    result = balance(features + features[:10], target=182)
    assert len(result) == 364
    assert result == balance(list(reversed(features)), target=182)
    assert len({f['properties']['field_id'] for f in result}) == 364
    table = Counter((f['properties']['block_id'], f['properties']['label']) for f in result)
    for block, label in table:
        assert table[block, 'silage'] == table[block, 'grain'] <= 60
    assert {f['properties']['region'] for f in result} == {'west', 'east', 'south'}
    assert not any('single' in f['properties']['block_id'] for f in result)
    assert len(balance(features, target=999)) == 372


def test_normalization_area_privacy_and_geometry_repair():
    adapter = get_adapter('AT', 2024)
    boundary = box(9, 46, 18, 50)
    counters = Counter()
    geom = transform(TO_WGS84, box(4650000, 2800000, 4650200, 2800200))
    raw = {'geometry': mapping(geom), 'properties': {
        'geo_id': 99, 'snar_code': '109', 'snar_bezeichnung': 'SILOMAIS',
        'fs_kennung': 'must-not-leak', 'applicant_id': 'must-not-leak'}}
    result = normalize(raw, adapter, 'test', counters, boundary)
    assert result['properties']['area_ha'] == pytest.approx(4, rel=1e-6)
    assert 'fs_kennung' not in result['properties'] and 'applicant_id' not in result['properties']
    raw['geometry'] = mapping(transform(TO_WGS84, box(4650000, 2800000, 4650050, 2800050)))
    assert normalize(raw, adapter, 'test', counters, boundary) is None
    assert counters['below_1ha'] == 1
    raw['geometry'] = mapping(transform(TO_WGS84, Polygon([
        (4650000, 2800000), (4650400, 2800400), (4650000, 2800400),
        (4650400, 2800000), (4650000, 2800000)])))
    assert normalize(raw, adapter, 'test', counters, boundary) is not None
    assert counters['invalid_input'] == counters['repaired'] == 1
    assert normalize(raw, adapter, 'test', counters, box(-1, -1, 1, 1)) is None


def test_dutch_filter_year_and_axis_order():
    query = get_adapter('NL', 2025).query((5, 51, 6, 52), ['silage', 'grain'])
    assert '<gml:lowerCorner>51 5</gml:lowerCorner>' in query['filter']
    assert '<Literal>2025</Literal>' in query['filter']
    assert '<Literal>259</Literal>' in query['filter']
    assert '<Literal>316</Literal>' in query['filter']
    with pytest.raises(ValueError):
        get_adapter('NL', 2024)
