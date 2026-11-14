"""Public registry endpoints and country-specific property mappings."""

from dataclasses import dataclass, field

AT_BASE = 'https://gis.lfrz.gv.at/api/geodata/i009501/ogc/features/v1/collections/'
# Release used for each Austrian year: the "-2" (winter) revision, which exists for
# every year we use and is the later, corrected state of the declarations.
AT_RELEASES = {2022: '2022_2', 2023: '2023_2', 2024: '2024_2'}
FR_BASE = 'https://data.geopf.fr/wfs/ows'
NL_WFS = 'https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0'


@dataclass
class Adapter:
    country: str
    year: int
    url: str
    code_field: str
    name_field: str
    codes: dict            # crop code -> (label, fallback crop name)
    windows: list          # (region, (west, south, east, north), rare label in that window)
    source: str            # collection / service identifier written into every feature
    keep: tuple            # raw properties that may be cached or kept; everything else is dropped
    page_size: int = 100
    workers: int = 4
    geom_field: str = ''
    info: dict = field(default_factory=dict)

    def parcel_id(self, feature):
        if self.country == 'AT':
            return feature['properties']['geo_id']
        if self.country == 'FR':
            return feature['properties']['id_parcel']
        return feature['id']

    def code_list(self, labels):
        return [c for c, (label, _) in self.codes.items() if label in labels]

    def query(self, bbox, labels, start=0):
        """Request parameters for one page of the given classes inside a lon/lat box."""
        west, south, east, north = bbox
        codes = self.code_list(labels)
        if self.country == 'AT':
            quoted = ','.join(f"'{c}'" for c in codes)
            return {'f': 'json', 'limit': self.page_size, 'startIndex': start,
                    'filter-lang': 'cql2-text', 'filter': f'snar_code IN ({quoted})',
                    'bbox': f'{west},{south},{east},{north}'}
        if self.country == 'NL':
            tests = ''.join(f'<PropertyIsEqualTo><ValueReference>gewascode</ValueReference><Literal>{c}</Literal></PropertyIsEqualTo>' for c in codes)
            xml = ('<Filter xmlns="http://www.opengis.net/fes/2.0" xmlns:gml="http://www.opengis.net/gml/3.2"><And>'
                   f'<Or>{tests}</Or><PropertyIsEqualTo><ValueReference>jaar</ValueReference><Literal>{self.year}</Literal></PropertyIsEqualTo>'
                   '<BBOX><ValueReference>geom</ValueReference><gml:Envelope srsName="urn:ogc:def:crs:EPSG::4326">'
                   f'<gml:lowerCorner>{south} {west}</gml:lowerCorner><gml:upperCorner>{north} {east}</gml:upperCorner>'
                   '</gml:Envelope></BBOX></And></Filter>')
            return {'service': 'WFS', 'version': '2.0.0', 'request': 'GetFeature',
                    'typenames': 'brpgewaspercelen:BrpGewas', 'outputformat': 'application/json',
                    'srsName': 'EPSG:4326', 'count': self.page_size, 'startIndex': start,
                    'sortBy': 'fuuid', 'filter': xml}
        if self.country == 'FR':
            quoted = ','.join(f"'{c}'" for c in codes)
            cql = (f"code_cultu IN ({quoted}) AND BBOX({self.geom_field},{south},{west},{north},{east},"
                   "'urn:ogc:def:crs:EPSG::4326')")
            return {'SERVICE': 'WFS', 'VERSION': '2.0.0', 'REQUEST': 'GetFeature',
                    'TYPENAMES': f'RPG.{self.year}:parcelles_graphiques', 'OUTPUTFORMAT': 'application/json',
                    'SRSNAME': 'EPSG:4326', 'COUNT': self.page_size, 'STARTINDEX': start,
                    'SORTBY': 'id_parcel', 'CQL_FILTER': cql}
        raise NotImplementedError(self.country)


def get_adapter(country, year):
    if country == 'AT' and year in AT_RELEASES:
        release = AT_RELEASES[year]
        name = f'i009501:invekos_schlaege_{release}_polygon'
        return Adapter(
            'AT', year, f'{AT_BASE}{name}/items', 'snar_code', 'snar_bezeichnung',
            {'109': ('silage', 'SILOMAIS'), '105': ('grain', 'KÖRNERMAIS')},
            [('Upper Austria', (13.45, 48.0, 14.45, 48.5), 'silage'),
             ('Lower Austria', (14.9, 48.0, 15.9, 48.5), 'silage'),
             ('Styria', (15.2, 46.75, 16.05, 47.2), 'silage')],
            f'{AT_BASE}{name}', ('geo_id', 'snar_code', 'snar_bezeichnung'),
            page_size=100, workers=4, info={'release': release})
    if country == 'NL' and year == 2025:
        return Adapter(
            'NL', year, NL_WFS, 'gewascode', 'gewas',
            {'259': ('silage', 'Mais, snij-'), '316': ('grain', 'Mais, korrel-')},
            [('Limburg', (5.65, 51.0, 6.15, 51.6), 'grain'),
             ('North Brabant', (5.0, 51.35, 5.65, 51.8), 'grain'),
             ('Eastern Gelderland', (6.1, 51.85, 6.75, 52.25), 'grain'),
             ('Drenthe', (6.3, 52.7, 7.05, 53.05), 'grain')],
            'https://api.pdok.nl/rvo/gewaspercelen/ogc/v1/collections/brpgewas',
            ('gewascode', 'gewas', 'jaar'), page_size=1000, workers=3)
    if country == 'FR' and year == 2022:
        return Adapter(
            'FR', year, FR_BASE, 'code_cultu', 'culture_d1',
            {'MIE': ('silage', 'Maïs ensilage'), 'MIS': ('grain', 'Maïs')},
            [], f'{FR_BASE}?TYPENAMES=RPG.2022:parcelles_graphiques',
            ('id_parcel', 'code_cultu'), page_size=1000, workers=3, geom_field='the_geom')
    raise ValueError(f'Unsupported country/year: {country} {year}')
