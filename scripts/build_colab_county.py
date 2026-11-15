"""Colab bundle for a county with no field file yet: make the field file on Colab, then extract there.

    python scripts/build_colab_county.py --fips 55101 --year 2023 --utm EPSG:32616 --code <dir with fieldvision/> \\
        --template-run colab/_templates/9cd0447b_2023 \\
        --check-external "data/timeseries/us_24035_2022|data/labels/us_24035_2022/fields.geojson|32618_US-E380N4310_1" \\
        --quickstats corn_grain_silage_acres_2012_2025.csv --census-year 2022 \\
        --survey-csv corn_grain_silage_acres_2021_2023.csv --max-blocks 70 --seed 42 --dest colab/us_55101_2023

Builds the standard continue-a-run bundle with scripts/build_colab_run.py (same-result check,
extraction, 50 MB result parts and <run>_SHA256SUMS.txt, unchanged), using a temporary stub run
folder and a placeholder field file, then inserts cells after the code check:

  A. range-download the national CSB archive (2018-2025) and cut the state (scripts/csb_subset.py);
  B. make this county's field file with the Maryland rule (CDL<year> == 1, unlabelled, ids
     csb-<CSBID>, area and grid in --utm) and check it against TIGER 2022, the Census
     (--census-year) and the county survey (--survey-csv, year --year);
  S. if the county has more than --max-blocks read blocks, draw whole 10 km blocks in a random
     order (numpy default_rng(--seed)) and keep each block whose read blocks still fit under the
     cap; the kept fields are what is extracted; sample.json records the draw;
  C. zip fields.geojson (extracted set), fields_all.geojson (whole county), sample.json, summary
     and checks as us_<fips>_<year>_fields.zip (+ SHA256) for download.

The laptop run folder is created after the fields come back (README), with the signature the
extractor computes, so scripts/merge_colab_blocks.py works unchanged.
"""
import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ['csb_subset.py', 'county_corn_check.py', 'range_download.py']
CSB_URL = ('https://www.nass.usda.gov/sites/default/files/Research_and_Science/Crop-Sequence-Boundaries/'
           'datasets/NationalCSB_2018-2025_rev23.zip')
CSB_SIZE = 3758023527
TIGER_URL = 'https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip'


def cell(kind, source):
    c = {'cell_type': kind, 'metadata': {}, 'source': source.strip('\n').splitlines(keepends=True)}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


def field_cells(fips, year, utm, qs_name, census_year, survey_name, max_blocks, seed, block_size):
    state, county = fips[:2], fips[2:]
    survey_args = f'"--survey-csv", UPLOAD / "{survey_name}", "--survey-year", "{year}", ' if survey_name else ''
    a = f'''
# A. National CSB archive by range requests (a plain GET returns an empty body) and the state subset.
#    Needs geopandas/pyogrio (installed without dependencies, so the pinned versions above stay).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:", "--no-deps",
                "geopandas==1.2.0", "pyogrio==0.13.0", "packaging", "certifi"], check=True)
FIPS, YEAR, UTM, QS = "{fips}", {year}, "{utm}", UPLOAD / "{qs_name}"
FIELDS_OUT = Path("/content/out")          # outside the code tree: from_file refuses folders inside it
FIELDS_OUT.mkdir(parents=True, exist_ok=True)
CSB_ZIP = Path("/content/csb/NationalCSB_2018-2025_rev23.zip")
STATE_PQ = FIELDS_OUT / "csb1825_{state}.parquet"

def run_py(*args):
    print("$", " ".join(map(str, args)), flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore", *map(str, args)], env=env, capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("failed: " + str(args[0]))

if not STATE_PQ.exists():
    CSB_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not CSB_ZIP.exists() or CSB_ZIP.stat().st_size != {CSB_SIZE} or Path(str(CSB_ZIP) + ".done").exists():
        run_py(CODE / "scripts" / "range_download.py", "{CSB_URL}", CSB_ZIP, "--workers", "8")
    assert CSB_ZIP.stat().st_size == {CSB_SIZE}
    run_py(CODE / "scripts" / "csb_subset.py", "--zip", CSB_ZIP, "--state", "{state}", "--out", STATE_PQ)
    CSB_ZIP.unlink()
print(json.loads(STATE_PQ.with_suffix(".json").read_text())["rows"], "CSB polygons in state {state}")
'''
    b = f'''
# B. The county's corn fields (CSB polygons whose CDL{year} majority class is corn) and the corn-area check
TIGER = Path("/content/tl_2022_us_county.zip")
if not TIGER.exists():
    subprocess.run(["curl", "-sS", "-L", "--retry", "3", "-o", str(TIGER), "{TIGER_URL}"], check=True)
COUNTY_DIR = FIELDS_OUT / RUN
if not (COUNTY_DIR / "fields.geojson").exists():
    run_py("-m", "fieldvision.labels.from_file", STATE_PQ, "--where", "STATEFIPS={state}", "--where", "CNTYFIPS={county}",
           "--where", "CDL{year}=1", "--unlabelled", "--year", "{year}", "--id-column", "CSBID", "--no-year-in-id",
           "--prefix", "csb", "--country", "US", "--source", "nass-csb", "--crop-code-column", "CDL{year}",
           "--crop-name", "Corn", "--metric-crs", UTM, "--keep", "CSBACRES=acres_reported",
           "--set", "county_fips={fips}", "--out", COUNTY_DIR)
run_py(CODE / "scripts" / "county_corn_check.py", "--fields", COUNTY_DIR / "fields.geojson", "--fips", FIPS, "--year", "{year}",
       "--census-year", "{census_year}", {survey_args}"--counties", TIGER, "--quickstats", QS, "--csb", STATE_PQ,
       "--utm", UTM, "--out", COUNTY_DIR / "county_check.json")
c = json.loads((COUNTY_DIR / "county_check.json").read_text())
s = c.get("survey") or {{}}
print(f"{{RUN}}: {{c['fields']}} fields, {{c['area_acres_total']:,.0f}} corn acres in CSB polygons")
print(f"  Census {census_year}: grain {{c['usda']['grain_harvested_acres']}}, silage {{c['usda']['silage_harvested_acres']}} "
      f"-> polygons / (grain+silage) = {{c.get('polygon_acres_over_census_harvested')}}")
print(f"  Survey {year}: planted {{s.get('planted_acres')}}, grain {{s.get('grain_harvested_acres')}}, silage {{s.get('silage_harvested_acres')}} "
      f"-> polygons / (grain+silage) = {{c.get('polygon_acres_over_survey_harvested')}}, / planted = {{c.get('polygon_acres_over_survey_planted')}}")
'''
    s = f'''
# S. Cap the extraction at about {max_blocks} read blocks by drawing whole 10 km blocks (fixed seed); record the draw
import numpy as np
sys.path.insert(0, str(CODE))
from fieldvision.s2.geometry import load_fields, make_blocks
MAX_BLOCKS, SEED, ACRES_PER_HA = {max_blocks}, {seed}, 2.471053814671653
full = json.loads((COUNTY_DIR / "fields.geojson").read_text(encoding="utf-8"))
read_blocks = make_blocks(load_fields(COUNTY_DIR / "fields.geojson", YEAR), {block_size})
per_label = {{}}
for blk in read_blocks:
    lb = blk["fields"][0]["properties"]["block_id"]
    per_label[lb] = per_label.get(lb, 0) + 1
labels = sorted(per_label)
order = [labels[i] for i in np.random.default_rng(SEED).permutation(len(labels))]
drawn, used = [], 0
for lb in order:
    if used + per_label[lb] <= MAX_BLOCKS:
        drawn.append(lb)
        used += per_label[lb]
keep = set(drawn)
subset = dict(full, features=[f for f in full["features"] if f["properties"]["block_id"] in keep])
area = lambda feats: sum(f["properties"]["area_ha"] for f in feats) * ACRES_PER_HA
sample = {{"run": RUN, "rule": "whole 10 km blocks (block_id) in numpy default_rng(seed).permutation order of the sorted ids; "
                          "a block is kept if its read blocks still fit under max_read_blocks",
          "seed": SEED, "max_read_blocks": MAX_BLOCKS, "is_sample": len(drawn) < len(labels),
          "label_blocks_drawn": len(drawn), "label_blocks_total": len(labels),
          "read_blocks_drawn": used, "read_blocks_total": len(read_blocks),
          "fields_drawn": len(subset["features"]), "fields_total": len(full["features"]),
          "acres_drawn": round(area(subset["features"]), 1), "acres_total": round(area(full["features"]), 1),
          "drawn_label_blocks": sorted(drawn)}}
sample["fraction_acres"] = round(sample["acres_drawn"] / sample["acres_total"], 4)
(COUNTY_DIR / "sample.json").write_text(json.dumps(sample, indent=2))
(UPLOAD / bundle["fields"]).write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")   # what the extraction reads
bundle["block_count"] = len(make_blocks(load_fields(UPLOAD / bundle["fields"], YEAR), {block_size}))
assert bundle["block_count"] == used, (bundle["block_count"], used)
(UPLOAD / "bundle.json").write_text(json.dumps(bundle, indent=2))
print({{k: v for k, v in sample.items() if k != "drawn_label_blocks"}})
'''
    c = '''
# C. Field files + checks for the laptop, small: download now (also copied to Drive if mounted)
FZIP = Path(f"/content/{RUN}_fields.zip")
with zipfile.ZipFile(FZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(UPLOAD / bundle["fields"], "fields.geojson")                 # the extracted set (all fields unless sampled)
    z.write(COUNTY_DIR / "fields.geojson", "fields_all.geojson")          # every corn field of the county
    for name in ("sample.json", "summary.json", "county_check.json"):
        z.write(COUNTY_DIR / name, name)
    z.write(STATE_PQ.with_suffix(".json"), STATE_PQ.with_suffix(".json").name)
FSUM = Path(f"/content/{RUN}_fields_SHA256SUMS.txt")
FSUM.write_text(f"{hashlib.sha256(FZIP.read_bytes()).hexdigest()}  {FZIP.name}\\n")
print(FZIP.name, FZIP.stat().st_size // 1024, "KB")
if Path("/content/drive/MyDrive").exists():
    for p in (FZIP, FSUM):
        shutil.copy2(p, Path("/content/drive/MyDrive") / p.name)
from google.colab import files
files.download(str(FSUM)); files.download(str(FZIP))
'''
    return [cell('code', a), cell('code', b), cell('code', s), cell('code', c)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fips', required=True)
    ap.add_argument('--year', type=int, default=2022)
    ap.add_argument('--utm', required=True)
    ap.add_argument('--code', type=Path, required=True, help='folder containing fieldvision/ (s2 fingerprint must match the template run)')
    ap.add_argument('--template-run', type=Path, required=True, help='run folder whose manifest parameters are used (year must match)')
    ap.add_argument('--check-external', nargs='+', required=True)
    ap.add_argument('--quickstats', type=Path, required=True)
    ap.add_argument('--census-year', type=int, default=2022)
    ap.add_argument('--survey-csv', type=Path)
    ap.add_argument('--max-blocks', type=int, default=70)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--builder-dir', type=Path, default=HERE, help='folder holding build_colab_run.py (main)')
    ap.add_argument('--frozen-dir', default=r'C:\t\fv\frozen_src_154133e')
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.builder_dir))
    import build_colab_run as bcr

    template = json.loads((a.template_run / 'manifest.json').read_text())
    if template['parameters']['year'] != a.year:
        raise SystemExit(f"template year {template['parameters']['year']} differs from --year {a.year}")
    run = f'us_{a.fips}_{a.year}'
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir = tmp / run
        placeholder = tmp / 'placeholder.geojson'
        # one tiny field so the builder can compute a stub; replaced on Colab by cell S
        placeholder.write_text(json.dumps({'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {
            'field_id': 'placeholder', 'year': a.year}, 'geometry': {'type': 'Polygon', 'coordinates': [[
                [-84.6, 43.3], [-84.599, 43.3], [-84.599, 43.301], [-84.6, 43.301], [-84.6, 43.3]]]}}]}))
        bcr.start_fresh_run(run_dir, placeholder, a.code, a.template_run)
        argv_b = ['build_colab_run', '--run-dir', str(run_dir), '--fields', str(placeholder), '--code', str(a.code),
                  '--check-external', *a.check_external, '--dest', str(a.dest)]
        old, sys.argv = sys.argv, argv_b
        try:
            bcr.main()
        finally:
            sys.argv = old
    bundle_path = a.dest / 'bundle.json'
    bundle = json.loads(bundle_path.read_text())
    (a.dest / bundle['fields']).unlink()                    # the real file is made on Colab
    bundle['block_count'] = None
    bundle['fields_made_on_colab'] = True
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    with zipfile.ZipFile(a.dest / bundle['code_zip'], 'a', zipfile.ZIP_DEFLATED) as z:
        for name in SCRIPTS:
            z.write(HERE / name, f'scripts/{name}')
    uploads_extra = [a.quickstats.name]
    shutil.copyfile(a.quickstats, a.dest / a.quickstats.name)
    if a.survey_csv:
        shutil.copyfile(a.survey_csv, a.dest / a.survey_csv.name)
        uploads_extra.append(a.survey_csv.name)

    nb_path = a.dest / f'extract_{run}.ipynb'
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    cells = nb['cells']
    src1 = ''.join(cells[1]['source']).replace('needed = [bundle["code_zip"], bundle["fields"],',
                                               'needed = [bundle["code_zip"], ' + ', '.join(f'"{n}"' for n in uploads_extra) + ',')
    assert 'bundle["fields"]' not in src1.split('needed =')[1].split('\n')[0]
    cells[1]['source'] = src1.splitlines(keepends=True)
    cells[0]['source'] = (f'# `{run}` on Colab: corn fields from CSB, then Sentinel-2 extraction\n\n'
                          f'Cells A, B, S, C (after step 4) make the county field file here, cap it at about {a.max_blocks} read blocks '
                          f'by drawing whole blocks (seed {a.seed}), and offer the field files for download; then steps 5-7 extract '
                          f'exactly as for the Maryland runs (frozen code, `{" ".join(bundle["cli"])}`).\n'
                          f'Upload every file of this folder except README.md into `/content`, run all cells top to bottom.\n'
                          f'Untouched test county: nothing here classifies.\n').splitlines(keepends=True)
    idx = next(i for i, c in enumerate(cells) if ''.join(c['source']).lstrip().startswith('# 5.'))
    nb['cells'] = cells[:idx] + field_cells(a.fips, a.year, a.utm, a.quickstats.name, a.census_year,
                                            a.survey_csv.name if a.survey_csv else None, a.max_blocks, a.seed,
                                            int(template['parameters']['block_size'])) + cells[idx:]
    nb_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
    uploads = sorted(p.name for p in a.dest.iterdir() if p.name != 'README.md')
    (a.dest / 'README.md').write_text(f'''# {run}: upload {", ".join(uploads)} to /content; run all cells top to bottom (1-4, A, B, S, C, 5-7).

Cell C downloads `{run}_fields.zip` + `{run}_fields_SHA256SUMS.txt` (fields.geojson = the extracted set,
fields_all.geojson = every corn field of the county, sample.json = the block draw, summary, corn-area check);
cell 7 downloads the result parts `{run}_colab_blocks.zip.NNN` + `{run}_SHA256SUMS.txt`, as for the other runs.
Untouched test county: no model on it until the modelling track has logged the test.

Back on the laptop (repository root, PYTHONPATH=src):
1. check the fields zip's SHA256, unzip it to C:\\t\\fv\\data\\labels\\{run}\\;
2. create the run folder with the extractor's signature for that file:
   python -c "import sys; sys.path.insert(0, 'scripts'); from pathlib import Path; import build_colab_run as b; b.start_fresh_run(Path(r'C:\\t\\fv\\data\\timeseries\\{run}'), Path(r'C:\\t\\fv\\data\\labels\\{run}\\fields.geojson'), Path(r'{a.frozen_dir}'), Path(r'{a.template_run}'))"
3. python scripts/merge_colab_blocks.py --parts-dir colab_results/{run} --run-dir data/timeseries/{run}
4. finalise with the frozen code: PYTHONPATH={a.frozen_dir.replace(chr(92), "/")} python -m fieldvision.s2.extract --fields data/labels/{run}/fields.geojson {" ".join(bundle["cli"])} --out data/timeseries/{run} --workers 16
''', encoding='utf-8')
    print(json.dumps({'dest': str(a.dest), 'uploads': uploads, 'check_blocks': bundle['check_blocks'],
                      'code_zip': bundle['code_zip'], 'cli': bundle['cli']}, indent=1))


if __name__ == '__main__':
    main()
