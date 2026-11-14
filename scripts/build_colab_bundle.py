"""Build the upload folder for running one extraction on Google Colab.

Writes into --dest:
  code_<hash8>.zip        exact frozen extractor package (fieldvision/), fingerprint printed in the notebook
  fields_<run>.geojson    the label file, byte for byte (the run signature hashes these bytes)
  catalog_<run>.zip.NNN   the prefetched STAC cache, stripped to the assets the extractor reads, split <= 9 MB
  check_blocks.zip        two finished blocks (fields of their label block, scene list, reference rows, settings)
The notebook and README are written by the same script so paths and names match.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

PART = 8 * 1024 * 1024  # upload tool takes 10 MB per call; keep a margin
PINS = {"numpy": "2.5.4", "pandas": "3.1.1", "pyarrow": "26.0.1", "rasterio": "1.5.2", "shapely": "2.2.0",
        "pyproj": "3.8.0", "requests": "2.34.2", "urllib3": "2.8.0", "affine": "3.0.1"}
LAPTOP_LIBS = {"python": "3.12.8", "GDAL": "3.12.2", "PROJ": "9.8.1"}
BAND_KEYS_ES = ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22", "scl")
BAND_KEYS_PC = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL")


def fingerprint(package):
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted((package / "s2").glob("*.py")))).hexdigest()


def strip_item(item):
    """Drop assets and links the extractor never reads; properties and footprint stay as they are."""
    keys = BAND_KEYS_PC if item["collection"].startswith("planetary:") else BAND_KEYS_ES
    slim = {k: v for k, v in item.items() if k not in ("assets", "links")}
    slim["assets"] = {k: item["assets"][k] for k in keys if k in item["assets"]}
    return slim


def strip_catalog(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    data["items"] = [strip_item(i) for i in data["items"]]
    return json.dumps(data, separators=(",", ":"))


def write_split(dest, name, payload):
    parts = []
    for index in range(0, max(len(payload), 1), PART):
        part = dest / f"{name}.{index // PART:03d}"
        part.write_bytes(payload[index:index + PART])
        parts.append(part.name)
    return parts


CHECK_SCRIPT = r'''
"""Same-result check: re-extract finished laptop blocks and compare exactly."""
import json, shutil, sys, zipfile
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from fieldvision.s2.core import BANDS
from fieldvision.s2.extract import process_block
from fieldvision.s2.geometry import load_fields, make_blocks

work, check_zip, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
block_ids = sys.argv[4:]
root = work / "check"
shutil.rmtree(root, ignore_errors=True)
zipfile.ZipFile(check_zip).extractall(work)
EXACT = ["scene_id", "collection", "n_pixels", "n_clear", "clear_fraction", *BANDS]
failed = False
with ThreadPoolExecutor(max_workers=workers) as pool:
    for block_id in block_ids:
        d = root / block_id
        s = json.loads((d / "settings.json").read_text())
        p = s["parameters"]
        blocks = {b["id"]: b for b in make_blocks(load_fields(d / "fields.geojson", s["year"]), p["block_size"])}
        block = blocks[block_id]
        out = d / "out"
        (out / "blocks").mkdir(parents=True)
        (out / "catalog").mkdir()
        shutil.copy(d / "catalog.json", out / "catalog" / f"{block_id}.json")
        # Transient HTTP read failures are retried (the extractor resumes failed scenes);
        # only a block that reads completely is compared.
        for attempt in range(4):
            status = process_block(block, Namespace(**p, out=out, scene_batch=6), pool)
            errors = status.get("scene_errors", {})
            if status["status"] == "complete" or not all("RasterioIOError" in e or "CPLE_" in e for e in errors.values()):
                break
            print(f"{block_id}: {len(errors)} scene reads failed (network), retry {attempt + 1}", flush=True)
        old = pd.read_parquet(d / "reference.parquet").set_index(["field_id", "date"]).sort_index()
        new = pd.read_parquet(out / "blocks" / f"{block_id}.parquet").set_index(["field_id", "date"]).sort_index()
        problems = []
        if status["status"] != "complete":
            problems.append(f"block status {status['status']}: {status.get('scene_errors')}")
        if not old.index.equals(new.index):
            problems.append(f"rows differ: laptop {len(old)}, here {len(new)}, "
                            f"{len(old.index.symmetric_difference(new.index))} keys not shared")
            common = old.index.intersection(new.index)
            old, new = old.loc[common], new.loc[common]
        for column in EXACT:
            a, b = old[column].to_numpy(), new[column].to_numpy()
            same = (a == b) | (pd.isna(a) & pd.isna(b)) if a.dtype.kind == "f" else (a == b)
            if not np.all(same):
                first = np.flatnonzero(~same)[0]
                key = old.index[first]
                problems.append(f"{column}: {int((~same).sum())} rows differ, e.g. {key}: laptop {a[first]!r}, here {b[first]!r}")
        print(f"{block_id} ({s['source_run']}): {len(old)} rows, {status['scenes_succeeded']} scenes -> "
              + ("IDENTICAL" if not problems else "DIFFERENT"))
        for problem in problems:
            print("   ", problem)
        failed |= bool(problems)
print("CHECK", "FAILED" if failed else "PASSED")
sys.exit(1 if failed else 0)
'''


def write_notebook(path, bundle):
    run = bundle["run"]
    pins = " ".join(f"{k}=={v}" for k, v in bundle["pins"].items())

    def code(source):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": source.strip("\n").splitlines(keepends=True)}

    def md(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}

    cells = [
        md(f"""
# Sentinel-2 field time series: `{run}` on Colab

Runs the team's frozen extractor (fingerprint `{bundle['code_sha256'][:8]}`) with the same settings as the
laptop. Order: upload the files listed in `README.md` into `/content`, then run the cells top to bottom.
Step 4 re-extracts two blocks the laptop already finished and **stops the notebook if a single value differs**.
To resume after a restart, run all cells again: finished blocks are skipped.
"""),
        code(f"""
# 1. Settings (only WORKERS may be changed; it does not change results)
import os, sys, json, time, shutil, zipfile, hashlib, subprocess, threading
from pathlib import Path
UPLOAD = Path("/content")
WORK = Path("/content/fv")
bundle = json.loads((UPLOAD / "bundle.json").read_text())
RUN, YEAR = bundle["run"], bundle["year"]
OUT = WORK / RUN
CODE = WORK / "code"
DRIVE_BACKUP = Path("/content/drive/MyDrive/fieldvision_" + RUN)
CPUS = os.cpu_count() or 2
WORKERS = max(16, 8 * CPUS)   # I/O threads for (scene, band) reads
print(f"{{RUN}} {{YEAR}}: {{CPUS}} CPUs, {{WORKERS}} read threads; Python {{sys.version.split()[0]}}")
missing = [n for n in [bundle["code_zip"], bundle["fields"], "check_blocks.zip", *bundle["catalog_parts"]]
           if not (UPLOAD / n).exists()]
print("missing uploads:", missing or "none")
"""),
        code("""
# 2. Optional, recommended: mount Google Drive so checkpoints survive a recycled runtime
from google.colab import drive
drive.mount("/content/drive")
"""),
        code(f"""
# 3. Install the exact package versions used on the laptop (binary wheels only) and print them
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:", *"{pins}".split()], check=True)
versions = subprocess.run([sys.executable, "-c", (
    "import importlib.metadata as m, rasterio, pyproj, sys;"
    "print('python', sys.version.split()[0]);"
    "[print(p, m.version(p)) for p in {list(bundle['pins'])!r}];"
    "print('GDAL', rasterio.__gdal_version__); print('PROJ', pyproj.proj_version_str)")],
    capture_output=True, text=True, check=True).stdout
print(versions)
print("laptop:", bundle["laptop"], "(different GDAL/PROJ builds are allowed only if step 4 passes)")
"""),
        code("""
# 4. Unpack the frozen code, check its fingerprint, then the same-result check
shutil.rmtree(CODE, ignore_errors=True)
zipfile.ZipFile(UPLOAD / bundle["code_zip"]).extractall(CODE)
digest = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((CODE / "fieldvision" / "s2").glob("*.py")))).hexdigest()
print("code fingerprint", digest)
assert digest == bundle["code_sha256"], "code fingerprint differs from the laptop's frozen copy"
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "check.py").write_text(CHECK_SCRIPT)
env = dict(os.environ, PYTHONPATH=str(CODE))
passed_marker = WORK / "check_passed.txt"
stamp = digest + "\\n" + versions
if passed_marker.exists() and passed_marker.read_text() == stamp:
    print("same-result check already PASSED in this runtime with this code and these libraries; not repeated")
else:
    result = subprocess.run([sys.executable, "-W", "ignore", str(WORK / "check.py"), str(WORK), str(UPLOAD / "check_blocks.zip"),
                             str(min(WORKERS, 16)), *bundle["check_blocks"]], env=env, capture_output=True, text=True)
    print(result.stdout[-6000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise SystemExit("Same-result check FAILED: do not use this runtime for the extraction. Tell the coordinator.")
    passed_marker.write_text(stamp)
"""),
        code("""
# 5. Prepare the output folder: restore from Drive if this is a fresh runtime, add the catalog cache
OUT.mkdir(parents=True, exist_ok=True)
if DRIVE_BACKUP.exists() and not (OUT / "manifest.json").exists():
    shutil.copytree(DRIVE_BACKUP, OUT, dirs_exist_ok=True)
    print("restored checkpoints from Drive:", len(list((OUT / "blocks").glob("*.json"))), "block records")
parts = b"".join((UPLOAD / n).read_bytes() for n in bundle["catalog_parts"])
(WORK / "catalog.zip").write_bytes(parts)
added = 0
with zipfile.ZipFile(WORK / "catalog.zip") as archive:
    for name in archive.namelist():
        target = OUT / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            added += 1
print("catalog files added:", added, "| blocks already finished:",
      sum(json.loads(p.read_text()).get("status") in ("complete", "partial") for p in (OUT / "blocks").glob("*.json"))
      if (OUT / "blocks").exists() else 0)
"""),
        code("""
# 6. Run the extraction (same command and settings as the laptop lane). Rerun this cell to resume.
cmd = [sys.executable, "-W", "ignore", "-m", "fieldvision.s2.extract", "--fields", str(UPLOAD / bundle["fields"]),
       "--year", str(YEAR), "--start", bundle["start"], "--end", bundle["end"], "--out", str(OUT),
       "--workers", str(WORKERS), "--scene-batch", str(bundle["scene_batch"])]
print(" ".join(cmd))
total_blocks = len(list((OUT / "catalog").glob("*.json")))
stop = threading.Event()

def backup():
    while not stop.wait(300):
        if Path("/content/drive/MyDrive").exists():
            for item in ["manifest.json", "blocks"]:
                src = OUT / item
                if src.is_dir():
                    shutil.copytree(src, DRIVE_BACKUP / item, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("*.tmp"))
                elif src.exists():
                    DRIVE_BACKUP.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, DRIVE_BACKUP / item)

threading.Thread(target=backup, daemon=True).start()
started = time.time()
with open(OUT / "run.log", "a") as log:
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log.write(line)
        if "boto3" in line:
            continue
        if " fields, " in line or "WARNING" in line or "ERROR" in line or ": complete" in line or "partial:" in line or "complete:" in line:
            done = sum(json.loads(p.read_text()).get("status") in ("complete", "partial") for p in (OUT / "blocks").glob("*.json"))
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{total_blocks} blocks done | {line.strip()[24:]}", flush=True)
    code_ = proc.wait()
stop.set()
print("exit code", code_, "(0 complete, 2 partial: see scene errors in manifest)", f"{(time.time()-started)/60:.1f} min this session")
failures = {}
for p in (OUT / "blocks").glob("*.json"):
    for e in json.loads(p.read_text()).get("scene_errors", {}).values():
        kind = "network read (rerun this cell)" if ("RasterioIOError" in e or "CPLE_" in e) else e[:60]
        failures[kind] = failures.get(kind, 0) + 1
print("scene errors by kind:", failures or "none")
"""),
        code("""
# 7. Results: counts, then one zip with the same outputs as a laptop run (catalog left out)
import pandas as pd
manifest = json.loads((OUT / "manifest.json").read_text())
coverage = pd.read_csv(OUT / "fields_coverage.csv")
table = pd.read_parquet(OUT / "timeseries.parquet")
usable = table[(table.n_clear >= 1) & (table.clear_fraction >= 0.2)]
print("status", manifest["status"], "| rows", len(table), "| usable rows", len(usable), "| fields", len(coverage),
      "| field status", coverage.status.value_counts().to_dict())
print("usable dates per field min/median/max", coverage.usable_dates.min(), coverage.usable_dates.median(), coverage.usable_dates.max())
print("rows by source", table.collection.value_counts().to_dict(), "| scene failures", manifest["scene_failures"])
result_zip = Path(f"/content/{RUN}_colab_result.zip")
with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and "catalog" not in path.relative_to(OUT).parts and not path.name.endswith(".tmp"):
            archive.write(path, path.relative_to(OUT).as_posix())
    archive.writestr("colab_environment.txt", versions + f"\\ncode fingerprint {digest}\\nsame-result check PASSED\\n")
print(result_zip, f"{result_zip.stat().st_size/1e6:.1f} MB")
if Path("/content/drive/MyDrive").exists():
    shutil.copy2(result_zip, Path("/content/drive/MyDrive") / result_zip.name)
    print("copied to Drive:", result_zip.name)
from google.colab import files
files.download(str(result_zip))
"""),
    ]
    # The check script is embedded as a string constant in cell 1.
    cells[1]["source"] = (["CHECK_SCRIPT = " + repr(CHECK_SCRIPT) + "\n"] + cells[1]["source"])
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                              "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True, help="frozen copy containing fieldvision/")
    parser.add_argument("--run", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--check", nargs="+", required=True, help="run_dir:labels_file:year:block_id")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    package = args.code / "fieldvision"
    digest = fingerprint(package)
    code_name = f"code_{digest[:8]}.zip"
    with zipfile.ZipFile(args.dest / code_name, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*.py")):
            archive.write(path, path.relative_to(args.code).as_posix())
    fields_name = f"fields_{args.run}.geojson"
    shutil.copyfile(args.fields, args.dest / fields_name)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(args.catalog.glob("*.json")):
            archive.writestr(f"catalog/{path.name}", strip_catalog(path))
    catalog_parts = write_split(args.dest, f"catalog_{args.run}.zip", buffer.getvalue())

    checks = []
    with zipfile.ZipFile(args.dest / "check_blocks.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for spec in args.check:
            run_dir, labels, year, block_id = spec.split("|")
            run_dir, labels = Path(run_dir), Path(labels)
            label_block = None
            features = json.loads(labels.read_text(encoding="utf-8"))
            status = json.loads((run_dir / "blocks" / f"{block_id}.json").read_text())
            ids = set(status["field_ids"])
            label_block = next(f["properties"].get("block_id") for f in features["features"]
                               if f["properties"]["field_id"] in ids)
            subset = dict(features, features=[f for f in features["features"]
                                              if f["properties"].get("block_id") == label_block])
            prefix = f"check/{block_id}"
            archive.writestr(f"{prefix}/fields.geojson", json.dumps(subset))
            archive.writestr(f"{prefix}/catalog.json", strip_catalog(run_dir / "catalog" / f"{block_id}.json"))
            archive.write(run_dir / "blocks" / f"{block_id}.parquet", f"{prefix}/reference.parquet")
            params = json.loads((run_dir / "manifest.json").read_text())["parameters"]
            archive.writestr(f"{prefix}/settings.json", json.dumps({"year": int(year), "parameters": params,
                                                                     "source_run": run_dir.name}))
            checks.append(block_id)

    for stale in args.dest.glob(f"catalog_{args.run}.zip.*"):
        if stale.name not in catalog_parts:
            stale.unlink()
    manifest = {"run": args.run, "year": args.year, "start": "04-01", "end": "11-30", "scene_batch": 6,
                "code_zip": code_name, "code_sha256": digest, "fields": fields_name,
                "catalog_parts": catalog_parts, "check_blocks": checks, "pins": PINS, "laptop": LAPTOP_LIBS}
    (args.dest / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_notebook(args.dest / f"extract_{args.run}.ipynb", manifest)
    sizes = {p.name: p.stat().st_size for p in sorted(args.dest.iterdir()) if p.is_file()}
    manifest["file_sizes"] = sizes
    print(json.dumps(manifest, indent=2))
    too_big = [n for n, s in sizes.items() if s > PART]
    if too_big:
        raise SystemExit(f"files over 9 MB: {too_big}")


if __name__ == "__main__":
    main()
