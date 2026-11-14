"""Public registry endpoints and country-specific property mappings."""

from dataclasses import dataclass


@dataclass
class Adapter:
    country: str
    year: int
    url: str
    code_field: str
    name_field: str
    codes: dict
    regions: list
    rare_label: str
    source: str

    def parcel_id(self, feature):
        if self.country == 'AT':
            return feature['properties']['geo_id']
        return feature['id']

    def query(self, bbox):
        if self.country == 'AT':
            return {'f': 'json', 'limit': 1000, 'filter-lang': 'cql2-text',
                    'filter': "snar_code IN ('109','105')", 'bbox': ','.join(map(str, bbox))}
        if self.country == 'NL':
            west, south, east, north = bbox
            code_tests = ''.join(f'<PropertyIsEqualTo><ValueReference>gewascode</ValueReference><Literal>{code}</Literal></PropertyIsEqualTo>' for code in self.codes)
            xml = ('<Filter xmlns="http://www.opengis.net/fes/2.0" xmlns:gml="http://www.opengis.net/gml/3.2"><And>'
                   f'<Or>{code_tests}</Or><PropertyIsEqualTo><ValueReference>jaar</ValueReference><Literal>{self.year}</Literal></PropertyIsEqualTo>'
                   '<BBOX><ValueReference>geom</ValueReference><gml:Envelope srsName="urn:ogc:def:crs:EPSG::4326">'
                   f'<gml:lowerCorner>{south} {west}</gml:lowerCorner><gml:upperCorner>{north} {east}</gml:upperCorner>'
                   '</gml:Envelope></BBOX></And></Filter>')
            return {'service': 'WFS', 'version': '2.0.0', 'request': 'GetFeature',
                    'typenames': 'brpgewaspercelen:BrpGewas', 'outputformat': 'application/json',
                    'srsName': 'EPSG:4326', 'count': 1000, 'startIndex': 0,
                    'sortBy': 'fuuid', 'filter': xml}
        raise NotImplementedError(self.country)


def get_adapter(country, year):
    if country == 'AT' and year == 2024:
        url = 'https://gis.lfrz.gv.at/api/geodata/i009501/ogc/features/v1/collections/i009501:invekos_schlaege_2024_2_polygon/items'
        return Adapter(country, year, url, 'snar_code', 'snar_bezeichnung',
                       {'109': ('silage', 'SILOMAIS'), '105': ('grain', 'KÖRNERMAIS')},
                       [('Upper Austria', (13.45, 48.0, 14.45, 48.5)),
                        ('Lower Austria', (14.9, 48.0, 15.9, 48.5)),
                        ('Styria', (15.2, 46.75, 16.05, 47.2))],
                       'silage', url.split('/items')[0])
    if country == 'NL' and year == 2025:
        url = 'https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0'
        return Adapter(country, year, url, 'gewascode', 'gewas',
                       {'259': ('silage', 'Mais, snij-'), '316': ('grain', 'Mais, korrel-')},
                       [('Limburg', (5.65, 51.0, 6.15, 51.6)),
                        ('North Brabant', (5.0, 51.35, 5.65, 51.8)),
                        ('Eastern Gelderland', (6.1, 51.85, 6.75, 52.25)),
                        ('Drenthe', (6.3, 52.7, 7.05, 53.05))],
                       'grain', 'https://api.pdok.nl/rvo/gewaspercelen/ogc/v1/collections/brpgewas')
    raise ValueError(f'Unsupported country/year: {country} {year}')
