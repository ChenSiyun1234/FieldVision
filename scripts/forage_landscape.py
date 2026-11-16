"""Forage landscape around each scored corn field, from CSB polygons (boundaries only, no imagery, no model).

    python scripts/forage_landscape.py --fields C:/t/fv/data/counties/24021/2022/fields.geojson \\
        --csb C:/t/fv/data/us/csb/csb1825_md.parquet --year 2022 --out C:/t/fv/data/counties/24021/2022/landscape.parquet

For every field: the CSB polygons whose representative point lies within 2 km and 5 km of the field's
centroid (the field's own polygon excluded) are summed by area in EPSG:5070 (the CSB's equal-area CRS),
per CDL year 2018-2025. Written per field and radius:

  forage_share_<r>_year    forage acres / cropland acres in the field's own year
  forage_share_<r>_mean8   the same share averaged over the eight years of the CSB sequence
  forage_ext_share_<r>_*   forage plus triticale (205) and triticale/corn double crop (228)
  pasture_share_<r>_*      grass/pasture (176) / (cropland + grass/pasture)
  forage_pasture_share_<r>_*  (forage + grass/pasture) / (cropland + grass/pasture)
  forage_share_<r>_mean1823   forage share averaged over 2018-2023 only

CDL 2024 and 2025 put much of what earlier years called other hay (37) into grass/pasture (176): in the
Maryland CSB, other hay falls from about 100,000 acres (2022-2023) to about 38,000 (2024-2025) while
grass/pasture rises from about 60,000 to 140,000-150,000. For 2024+ fields use forage_pasture_share or
forage_share_*_mean1823, not forage_share_*_year.
  cropland_acres_<r>_year  cropland acres in the neighbourhood in the field's own year

Forage = alfalfa 36, other hay 37, clover/wildflowers 58, vetch 224. Cropland = CDL crop classes
(1-61, 66-77, 204-254, which include the forage classes) and excludes grass/pasture, forest, water,
developed and other non-crop classes that some CSB polygons carry in some years.
Own-field flags from the CSB sequence (other years than --year): hay_or_alfalfa_history, wheat_history,
soybean_years, corn_years (of 8, class 1). Neighbourhoods stop at the state edge of the CSB file given.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

YEARS = list(range(2018, 2026))
FORAGE = {36, 37, 58, 224}
FORAGE_EXT = FORAGE | {205, 228}
PASTURE = {176}
HAY = {36, 37}
WHEAT = {22, 23, 24, 26, 225, 236, 238}
SOY = {5, 26, 239, 240, 241, 254}
ACRES_PER_M2 = 1 / 4046.8564224


def is_cropland(code):
    return ((code >= 1) & (code <= 61)) | ((code >= 66) & (code <= 77)) | ((code >= 204) & (code <= 254))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fields', type=Path, required=True, help='field file with field_id csb-<CSBID> and centroid_lon/lat')
    ap.add_argument('--csb', type=Path, required=True, help='state CSB GeoParquet (scripts/csb_subset.py)')
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--radii', type=float, nargs='+', default=[2000, 5000])
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)

    fields = json.loads(a.fields.read_text(encoding='utf-8'))['features']
    fid = [f['properties']['field_id'] for f in fields]
    to5070 = Transformer.from_crs(4326, 5070, always_xy=True).transform
    fx, fy = to5070(np.array([f['properties']['centroid_lon'] for f in fields]),
                    np.array([f['properties']['centroid_lat'] for f in fields]))

    csb = gpd.read_parquet(a.csb).to_crs(5070)
    pts = csb.geometry.representative_point()
    px, py = pts.x.to_numpy(), pts.y.to_numpy()
    area = csb.geometry.area.to_numpy() * ACRES_PER_M2
    csb_id = ('csb-' + csb['CSBID'].astype(str)).to_numpy()
    codes = np.stack([csb[f'CDL{y}'].to_numpy().astype(int) for y in YEARS], axis=1)      # polygons x years
    w_forage = np.isin(codes, list(FORAGE)) * area[:, None]
    w_ext = np.isin(codes, list(FORAGE_EXT)) * area[:, None]
    w_past = np.isin(codes, list(PASTURE)) * area[:, None]
    w_crop = is_cropland(codes) * area[:, None]
    yi = YEARS.index(a.year)
    own_index = {k: i for i, k in enumerate(csb_id)}
    missing = [f for f in fid if f not in own_index]
    if missing:
        raise SystemExit(f'{len(missing)} fields not in the CSB file, e.g. {missing[:3]}')
    own = np.array([own_index[f] for f in fid])

    tree = cKDTree(np.column_stack([px, py]))
    out = pd.DataFrame({'field_id': fid})
    for r in a.radii:
        tag = f'{int(r / 1000)}km'
        lists = tree.query_ball_point(np.column_stack([fx, fy]), r)
        F, E, P, C = (np.zeros((len(fid), len(YEARS))) for _ in range(4))
        for i, idx in enumerate(lists):
            idx = [j for j in idx if j != own[i]]
            if idx:
                F[i], E[i], P[i], C[i] = (w[idx].sum(axis=0) for w in (w_forage, w_ext, w_past, w_crop))
        with np.errstate(invalid='ignore', divide='ignore'):
            sf, se, sp = F / C, E / C, P / (C + P)
        out[f'forage_share_{tag}_year'] = sf[:, yi]
        out[f'forage_share_{tag}_mean8'] = np.nanmean(sf, axis=1)
        out[f'forage_ext_share_{tag}_year'] = se[:, yi]
        out[f'forage_ext_share_{tag}_mean8'] = np.nanmean(se, axis=1)
        out[f'pasture_share_{tag}_year'] = sp[:, yi]
        out[f'pasture_share_{tag}_mean8'] = np.nanmean(sp, axis=1)
        # CDL 2024+ moved much "other hay" (37) into grass/pasture (176): forage + pasture is comparable across years,
        # and the 2018-2023 mean of the forage share avoids the break
        with np.errstate(invalid='ignore', divide='ignore'):
            sfp = (F + P) / (C + P)
        out[f'forage_pasture_share_{tag}_year'] = sfp[:, yi]
        out[f'forage_pasture_share_{tag}_mean8'] = np.nanmean(sfp, axis=1)
        out[f'forage_share_{tag}_mean1823'] = np.nanmean(sf[:, :6], axis=1)
        out[f'cropland_acres_{tag}_year'] = C[:, yi]
    own_codes = codes[own]
    others = [i for i in range(len(YEARS)) if i != yi]
    out['hay_or_alfalfa_history'] = np.isin(own_codes[:, others], list(HAY)).any(axis=1)
    out['wheat_history'] = np.isin(own_codes[:, others], list(WHEAT)).any(axis=1)
    out['soybean_years'] = np.isin(own_codes[:, others], list(SOY)).sum(axis=1)
    out['corn_years'] = (own_codes == 1).sum(axis=1)
    out['year'] = a.year
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    desc = out[[c for c in out.columns if c.startswith('forage_share')]].describe().round(3)
    print(len(out), 'fields ->', a.out)
    print(desc.to_string())


if __name__ == '__main__':
    main()
