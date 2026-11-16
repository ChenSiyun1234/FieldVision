"""Colab notebook that builds statewide corn frames and block draws from CSB (boundaries only, no imagery).

    python scripts/build_colab_state_frames.py --states 42:PENNSYLVANIA 55:WISCONSIN --year 2025 \\
        --targets C:/t/fv/data/eval/nass_2025_state_targets.csv --dest C:/t/fv/colab/state_frames_2025

On Colab: range-download the national CSB 2018-2025 archive, cut each state (scripts/csb_subset.py),
make the frame with from_file (CDL<year> in 1, 225, 226, 228, 237; area and 10 km grid in EPSG:5070,
the CSB's own equal-area CRS, because these states span two UTM zones), then scripts/state_sample.py
(--blocks read blocks in --parts parts, --seed). Returns one small zip per state with the part field
files, draw.json, state_acres.json and summaries; the full frames go to Drive if it is mounted.
The extraction bundles are then built on the laptop from the part files with build_colab_run.py.
"""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ['csb_subset.py', 'range_download.py', 'state_sample.py']
CSB_URL = ('https://www.nass.usda.gov/sites/default/files/Research_and_Science/Crop-Sequence-Boundaries/'
           'datasets/NationalCSB_2018-2025_rev23.zip')
CSB_SIZE = 3758023527


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--states', nargs='+', required=True, help='FIPS:NAME, e.g. 42:PENNSYLVANIA')
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--blocks', type=int, default=150)
    ap.add_argument('--parts', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--crs', default='EPSG:5070')
    ap.add_argument('--targets', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    a = ap.parse_args(argv)
    states = [s.split(':', 1) for s in a.states]
    a.dest.mkdir(parents=True, exist_ok=True)

    rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip()
    names = [n for n in subprocess.check_output(['git', 'ls-files', 'src/fieldvision'], cwd=REPO, text=True).split()
             if n.endswith('.py') and '/app/' not in n]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n[len('src/'):], subprocess.check_output(['git', 'show', f'HEAD:{n}'], cwd=REPO))
        for n in SCRIPTS:
            z.writestr(f'scripts/{n}', subprocess.check_output(['git', 'show', f'HEAD:scripts/{n}'], cwd=REPO))
    code_zip = f'code_frames_{rev}.zip'
    (a.dest / code_zip).write_bytes(buf.getvalue())
    code_sha = hashlib.sha256(buf.getvalue()).hexdigest()
    shutil.copyfile(a.targets, a.dest / a.targets.name)

    settings = f'''# 1. Settings
import os, sys, json, hashlib, shutil, subprocess, zipfile
from pathlib import Path
UPLOAD, WORK, OUTD = Path("/content"), Path("/content/work"), Path("/content/out")
CODE = WORK / "code"
STATES = {json.dumps(states)}
YEAR, BLOCKS, PARTS, SEED, CRS = {a.year}, {a.blocks}, {a.parts}, {a.seed}, "{a.crs}"
CODE_ZIP, CODE_SHA, TARGETS = "{code_zip}", "{code_sha}", "{a.targets.name}"
print("missing uploads:", [n for n in (CODE_ZIP, TARGETS) if not (UPLOAD / n).exists()] or "none")
'''
    cells = [
        ('markdown', f'# Statewide corn frames and block draws, {a.year} (boundaries only)\n\n'
                     f'States: {", ".join(n for _, n in states)}. Upload `{code_zip}` and `{a.targets.name}`, run all cells. '
                     f'Downloads the national CSB archive (3.76 GB) here, never on the laptop. Returns `frames_<fips>_{a.year}.zip` '
                     f'per state (part field files, draw.json, state_acres.json) and its SHA256. Nothing is scored.'),
        ('code', settings),
        ('code', '# 2. Optional: Google Drive keeps the full frames\nfrom google.colab import drive\ndrive.mount("/content/drive")'),
        ('code', '''# 3. Installs (laptop versions, binary wheels), code check
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:", *"numpy==2.5.4 pandas==3.1.1 pyarrow==26.0.1 shapely==2.2.0 pyproj==3.8.0 geopandas==1.2.0 pyogrio==0.13.0".split()], check=True)
assert hashlib.sha256((UPLOAD / CODE_ZIP).read_bytes()).hexdigest() == CODE_SHA, "code zip differs from the bundle"
shutil.rmtree(CODE, ignore_errors=True)
zipfile.ZipFile(UPLOAD / CODE_ZIP).extractall(CODE)
env = dict(os.environ, PYTHONPATH=str(CODE))
def run_py(*args):
    print("$", " ".join(map(str, args)), flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore", *map(str, args)], env=env, capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:]); raise SystemExit("failed: " + str(args[0]))
OUTD.mkdir(parents=True, exist_ok=True)'''),
        ('code', f'''# 4. National CSB archive by range requests, then one subset per state
ZIP = WORK / "csb" / "NationalCSB_2018-2025_rev23.zip"
todo = [f for f, _ in STATES if not (OUTD / f"csb1825_{{f}}.parquet").exists()]
if todo:
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists() or ZIP.stat().st_size != {CSB_SIZE} or Path(str(ZIP) + ".done").exists():
        run_py(CODE / "scripts" / "range_download.py", "{CSB_URL}", ZIP, "--workers", "8")
    assert ZIP.stat().st_size == {CSB_SIZE}
    for f in todo:
        run_py(CODE / "scripts" / "csb_subset.py", "--zip", ZIP, "--state", f, "--out", OUTD / f"csb1825_{{f}}.parquet")
    ZIP.unlink()'''),
        ('code', '''# 5. Frame per state (corn + small-grain/corn double crops) and the seeded draw of read blocks
for f, name in STATES:
    frame_dir = OUTD / f"us_{f}_{YEAR}" / "frame"
    if not (frame_dir / "fields.geojson").exists():
        run_py("-m", "fieldvision.labels.from_file", OUTD / f"csb1825_{f}.parquet", "--where", f"STATEFIPS={f}",
               "--where", f"CDL{YEAR}=1,225,226,228,237", "--unlabelled", "--year", YEAR, "--id-column", "CSBID",
               "--no-year-in-id", "--prefix", "csb", "--country", "US", "--source", "nass-csb",
               "--crop-code-column", f"CDL{YEAR}", "--metric-crs", CRS,
               "--keep", "CSBACRES=acres_reported", "CNTYFIPS=cnty_fips", "CNTY=county_name", "--out", frame_dir)
    run_py(CODE / "scripts" / "state_sample.py", "--frame", frame_dir / "fields.geojson", "--state", f, "--year", YEAR,
           "--blocks", BLOCKS, "--parts", PARTS, "--seed", SEED, "--out", OUTD / f"us_{f}_{YEAR}",
           "--target-csv", UPLOAD / TARGETS, "--state-name", name)'''),
        ('code', '''# 6. Small result zips (part field files + draw + state acres) and SHA256; full frames to Drive
from google.colab import files
for f, name in STATES:
    d = OUTD / f"us_{f}_{YEAR}"
    z_path = Path(f"/content/frames_{f}_{YEAR}.zip")
    with zipfile.ZipFile(z_path, "w", zipfile.ZIP_DEFLATED) as z:
        for k in range(1, PARTS + 1):
            z.write(d / f"s{k}" / "fields.geojson", f"s{k}/fields.geojson")
        for n in ("draw.json", "state_acres.json"):
            z.write(d / n, n)
        z.write(d / "frame" / "summary.json", "frame_summary.json")
        z.write(OUTD / f"csb1825_{f}.json", f"csb1825_{f}.json")
    s_path = Path(f"/content/frames_{f}_{YEAR}_SHA256SUMS.txt")
    s_path.write_text(f"{hashlib.sha256(z_path.read_bytes()).hexdigest()}  {z_path.name}\\n")
    print(z_path.name, z_path.stat().st_size // 1024, "KB")
    if Path("/content/drive/MyDrive").exists():
        dd = Path("/content/drive/MyDrive/fieldvision_frames_" + str(YEAR)); dd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d / "frame_fields.geojson", dd / f"frame_{f}_{YEAR}.geojson")
        shutil.copy2(OUTD / f"csb1825_{f}.parquet", dd)
    files.download(str(s_path)); files.download(str(z_path))'''),
    ]
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
          'cells': [{'cell_type': t, 'metadata': {}, 'source': s.splitlines(keepends=True),
                     **({'outputs': [], 'execution_count': None} if t == 'code' else {})} for t, s in cells]}
    nb_name = f'state_frames_{a.year}.ipynb'
    (a.dest / nb_name).write_text(json.dumps(nb, indent=1), encoding='utf-8')
    (a.dest / 'README.md').write_text(
        f'# State frames {a.year} ({", ".join(n for _, n in states)}): upload {code_zip}, {a.targets.name}, {nb_name} '
        f'to /content; run all cells top to bottom (1-6).\n\nBoundaries only (CSB 2018-2025 rev23), no imagery, nothing scored. '
        f'Returns frames_<fips>_{a.year}.zip + _SHA256SUMS.txt per state: s1..s{a.parts}/fields.geojson ({a.blocks} read blocks, '
        f'seed {a.seed}), draw.json, state_acres.json. Unzip to C:\\t\\fv\\data\\labels\\us_<fips>_{a.year}\\, then build one extraction '
        f'bundle per part with build_colab_run.py (template C:\\t\\fv\\colab\\_templates\\9cd0447b_{a.year}).\n', encoding='utf-8')
    print(a.dest, code_zip, code_sha[:12])


if __name__ == '__main__':
    main()
