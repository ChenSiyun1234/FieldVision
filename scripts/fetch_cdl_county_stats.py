"""Fetch CropScape county statistics of the Cropland Data Layer (pixel acres by class) for many counties and years.

    python scripts/fetch_cdl_county_stats.py --quickstats corn_grain_silage_acres_2012_2025.csv \\
        --years 2019 2020 2021 2022 2023 2024 2025 --out C:/t/fv/data/us/cropscape/county --workers 24

Counties: every county FIPS with any corn record in the Quick Stats table (all years). Files are the
CropScape cache, https://nassgeodata.gmu.edu/webservice/nass_data_cache/byfips/CDL_<year>_<fips>.csv
(the URL the GetCDLStat service returns), saved as is; existing files are skipped, so a rerun resumes.
failures.json lists what could not be fetched. Nothing here is parsed or modelled.
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

URL = 'https://nassgeodata.gmu.edu/webservice/nass_data_cache/byfips/CDL_{year}_{fips}.csv'
SERVICE = 'https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat?year={year}&fips={fips}&format=csv'


def fetch(job, out):
    year, fips = job
    path = out / f'CDL_{year}_{fips}.csv'
    if path.exists() and path.stat().st_size > 0:
        return None
    last = None
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(URL.format(year=year, fips=fips), timeout=60).read()
            if not data.startswith(b'Value'):
                # not cached yet: ask the service once, which creates the cache file
                urllib.request.urlopen(SERVICE.format(year=year, fips=fips), timeout=120).read()
                data = urllib.request.urlopen(URL.format(year=year, fips=fips), timeout=60).read()
            if not data.startswith(b'Value'):
                raise IOError(f'unexpected content {data[:60]!r}')
            path.write_bytes(data)
            return None
        except Exception as e:
            last = repr(e)
            time.sleep(3 * (attempt + 1))
    return f'{year}_{fips}', last


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quickstats', type=Path, required=True)
    ap.add_argument('--years', type=int, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args(argv)
    q = pd.read_csv(a.quickstats, dtype=str)
    q = q[(q.aggregation_level == 'COUNTY') & q.county_ansi.notna() & (q.county_ansi.str.strip() != '')]
    fips = sorted(set(q.state_ansi.str.zfill(2) + q.county_ansi.str.zfill(3)))
    jobs = [(y, f) for y in a.years for f in fips]
    a.out.mkdir(parents=True, exist_ok=True)
    print(len(fips), 'counties,', len(jobs), 'files', flush=True)
    t0, failures = time.time(), {}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(lambda j: fetch(j, a.out), jobs), 1):
            if r:
                failures[r[0]] = r[1]
            if i % 1000 == 0:
                print(f'{i}/{len(jobs)} {time.time() - t0:.0f} s, {len(failures)} failed', flush=True)
    (a.out / 'failures.json').write_text(json.dumps(failures, indent=1))
    print('done', len(jobs), 'files,', len(failures), 'failed,', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
