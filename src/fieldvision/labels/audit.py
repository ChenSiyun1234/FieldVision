"""Record national crop counts, available collections and download sizes."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .fetch import Client, now
from .adapters import get_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    client = Client()
    result = {'checked_at': now(), 'AT': {}, 'NL': {}}
    at = get_adapter('AT', 2024)
    collections = client.get(at.url.split('/collections/')[0] + '/collections', {'f': 'json'})
    result['AT']['collections'] = [c['id'] for c in collections['collections'] if 'schlaege' in c['id'] and 'point' not in c['id']]
    for code in ['105', '106', '107', '109', '135', '173']:
        data = client.get(at.url, {'f': 'json', 'limit': 1, 'filter-lang': 'cql2-text', 'filter': f"snar_code='{code}'"})
        result['AT'][code] = {'count': data['numberMatched'], 'name': data['features'][0]['properties']['snar_bezeichnung']}
    nl = get_adapter('NL', 2025)
    for prop, values in [('gewascode', [259, 316, 317, 814, 1935, 2032]), ('jaar', [2024, 2025, 2026])]:
        for value in values:
            xml = f'<Filter xmlns="http://www.opengis.net/fes/2.0"><PropertyIsEqualTo><ValueReference>{prop}</ValueReference><Literal>{value}</Literal></PropertyIsEqualTo></Filter>'
            params = {'service': 'WFS', 'version': '2.0.0', 'request': 'GetFeature',
                      'typenames': 'brpgewaspercelen:BrpGewas', 'resultType': 'hits', 'filter': xml}
            response = client.session.get(nl.url, params=params, timeout=90)
            response.raise_for_status()
            result['NL'][f'{prop}={value}'] = {'count': int(ET.fromstring(response.content).attrib['numberMatched']),
                                             'url': response.url, 'fetched_at': now()}
    url = 'https://service.pdok.nl/rvo/gewaspercelen/atom/basisregistratie_gewaspercelen_brp.xml'
    response = client.session.get(url, timeout=90)
    response.raise_for_status()
    result['NL']['downloads'] = [dict(e.attrib) for e in ET.fromstring(response.content).iter()
                                 if e.tag.endswith('link') and e.attrib.get('rel') == 'section']
    result['api_requests'] = client.log
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
