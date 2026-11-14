"""Build a Colab upload folder that continues a laptop extraction run.

Unlike build_colab_bundle.py (fresh run), this one:
  * takes settings from the laptop run's manifest and passes only the options that
    differ from the extractor defaults, so the arguments (and run signature) match;
  * requires the code copy's fingerprint to equal the manifest's source_sha256;
  * ships the laptop's block checkpoints (complete, partial, running) so Colab
    resumes exactly where the laptop is: finished blocks are skipped, scenes that
    failed on the network are read again, unfinished blocks continue;
  * checks against finished blocks of the same run (or another run) before starting;
  * returns only the blocks Colab worked on (block parquet + checkpoint json, plus the
    manifest), zipped and split into parts with a SHA256SUMS file, for
    scripts/merge_colab_blocks.py on the laptop.
"""
import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_colab_bundle import LAPTOP_LIBS, PINS, fingerprint, strip_catalog, write_split  # noqa: E402

RESULT_PART_MB = 50


def extractor_defaults(code_dir):
    """Argument defaults of this exact code copy, read from its own parser."""
    sys.path.insert(0, str(code_dir))
    try:
        for name in [m for m in sys.modules if m == "fieldvision" or m.startswith("fieldvision.")]:
            del sys.modules[name]
        from fieldvision.s2 import extract
        captured = {}
        real_run = extract.run
        extract.run = lambda args: captured.update(vars(args)) or 0
        argv = sys.argv
        sys.argv = ["extract", "--fields", "f.geojson", "--year", "2000", "--out", "o"]
        try:
            extract.main()
        finally:
            sys.argv = argv
            extract.run = real_run
        return captured
    finally:
        sys.path.remove(str(code_dir))


def cli_from_manifest(params, defaults):
    cli = ["--year", str(params["year"])]
    for key, value in params.items():
        if key in ("year", "start_date", "end_date"):
            continue
        if key not in defaults:
            raise SystemExit(f"manifest parameter {key!r} unknown to this code copy")
        if value == defaults[key]:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            cli.append(flag)
        elif value is False or value is None:
            raise SystemExit(f"cannot express {key}={value!r} on the command line")
        else:
            cli += [flag, str(value)]
    return cli


CHECK_SCRIPT = r'''
"""Same-result check: re-extract finished laptop blocks and compare exactly."""
import json, shutil, sys, time, zipfile
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from fieldvision.s2.core import BANDS
from fieldvision.s2.extract import process_block
from fieldvision.s2.geometry import load_fields, make_blocks

work, check_zip, workers, main_fields = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
block_ids = sys.argv[5:]
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
        fields = main_fields if s.get("fields") == "MAIN" else d / "fields.geojson"
        blocks = {b["id"]: b for b in make_blocks(load_fields(fields, s["year"]), p["block_size"])}
        block = blocks[block_id]
        out = d / "out"
        (out / "blocks").mkdir(parents=True)
        (out / "catalog").mkdir()
        shutil.copy(d / "catalog.json", out / "catalog" / f"{block_id}.json")
        for attempt in range(4):
            status = process_block(block, Namespace(**p, out=out, scene_batch=6), pool)
            errors = status.get("scene_errors", {})
            if status["status"] == "complete" or not all("RasterioIOError" in e or "CPLE_" in e for e in errors.values()):
                break
            print(f"{block_id}: {len(errors)} scene reads failed (network), retry {attempt + 1}", flush=True)
            time.sleep(30 * (attempt + 1))
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
                problems.append(f"{column}: {int((~same).sum())} rows differ, e.g. {old.index[first]}: laptop {a[first]!r}, here {b[first]!r}")
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
# Sentinel-2 field time series: continue `{run}` on Colab

Continues the laptop run `{run}` with the same frozen extractor (fingerprint `{bundle['code_sha256'][:8]}`) and the
same arguments (`{' '.join(bundle['cli'])}`). The laptop's finished blocks are shipped and skipped here.
Order: upload every file listed in `README.md` into `/content`, then run the cells top to bottom.
Step 4 re-extracts finished laptop blocks and **stops the notebook if a single value differs**.
To resume after a restart, run all cells again.
"""),
        code("""
# 1. Settings (only WORKERS may be changed; it does not change results)
import os, sys, json, time, shutil, zipfile, hashlib, subprocess, threading
from pathlib import Path
UPLOAD = Path("/content")
WORK = Path("/content/fv")
bundle = json.loads((UPLOAD / "bundle.json").read_text())
RUN = bundle["run"]
OUT = WORK / RUN
CODE = WORK / "code"
DRIVE_BACKUP = Path("/content/drive/MyDrive/fieldvision_" + RUN)
CPUS = os.cpu_count() or 2
WORKERS = max(16, 8 * CPUS)   # I/O threads for (scene, band) reads
print(f"{RUN}: {CPUS} CPUs, {WORKERS} read threads; Python {sys.version.split()[0]}")
needed = [bundle["code_zip"], bundle["fields"], "check_blocks.zip", *bundle["catalog_parts"], *bundle["block_parts"]]
missing = [n for n in needed if not (UPLOAD / n).exists()]
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
assert digest == bundle["code_sha256"], "code fingerprint differs from the laptop run's code"
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "check.py").write_text(CHECK_SCRIPT)
env = dict(os.environ, PYTHONPATH=str(CODE))
passed_marker = WORK / "check_passed.txt"
stamp = digest + "\\n" + versions
if passed_marker.exists() and passed_marker.read_text() == stamp:
    print("same-result check already PASSED in this runtime with this code and these libraries; not repeated")
else:
    result = subprocess.run([sys.executable, "-W", "ignore", str(WORK / "check.py"), str(WORK), str(UPLOAD / "check_blocks.zip"),
                             str(min(WORKERS, 16)), str(UPLOAD / bundle["fields"]), *bundle["check_blocks"]],
                            env=env, capture_output=True, text=True)
    print(result.stdout[-6000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise SystemExit("Same-result check FAILED: do not use this runtime for the extraction. Tell the coordinator.")
    passed_marker.write_text(stamp)
"""),
        code("""
# 5. Prepare the output folder: restore from Drive, else unpack the laptop checkpoints; add the catalog cache
OUT.mkdir(parents=True, exist_ok=True)
if DRIVE_BACKUP.exists() and not (OUT / "blocks").exists():
    shutil.copytree(DRIVE_BACKUP, OUT, dirs_exist_ok=True)
    print("restored checkpoints from Drive")

def unpack(parts, name):
    if not parts:
        return 0
    (WORK / name).write_bytes(b"".join((UPLOAD / n).read_bytes() for n in parts))
    added = 0
    with zipfile.ZipFile(WORK / name) as archive:
        for member in archive.namelist():
            target = OUT / member
            if not target.exists():          # never overwrite work done here
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
                added += 1
    return added

print("catalog files added:", unpack(bundle["catalog_parts"], "catalog.zip"))
print("laptop block files added:", unpack(bundle["block_parts"], "blocks.zip"))
start_state = WORK / "start_state.json"
if not start_state.exists():
    state = {p.stem: json.loads(p.read_text()).get("status") for p in (OUT / "blocks").glob("*.json")}
    start_state.write_text(json.dumps(state))
state = json.loads(start_state.read_text())
print("blocks at start:", {s: list(state.values()).count(s) for s in set(state.values())}, "of", bundle["block_count"])
"""),
        code("""
# 6. Run the extraction (same arguments as the laptop run). Rerun this cell to resume.
cmd = [sys.executable, "-W", "ignore", "-m", "fieldvision.s2.extract", "--fields", str(UPLOAD / bundle["fields"]),
       *bundle["cli"], "--out", str(OUT), "--workers", str(WORKERS), "--scene-batch", "6"]
print(" ".join(cmd))
stop = threading.Event()

def backup():
    while not stop.wait(300):
        if Path("/content/drive/MyDrive").exists():
            shutil.copytree(OUT / "blocks", DRIVE_BACKUP / "blocks", dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp"))
            if (OUT / "manifest.json").exists():
                shutil.copy2(OUT / "manifest.json", DRIVE_BACKUP / "manifest.json")

def finished():
    return sum(json.loads(p.read_text()).get("status") in ("complete", "partial") for p in (OUT / "blocks").glob("*.json"))

threading.Thread(target=backup, daemon=True).start()
started = time.time()
with open(OUT / "run.log", "a") as log:
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log.write(line)
        if "boto3" in line:
            continue
        if " fields, " in line or " ERROR " in line or "complete:" in line or "partial:" in line:
            print(f"[{time.strftime('%H:%M:%S')}] {finished()}/{bundle['block_count']} blocks done | {line.strip()[24:]}", flush=True)
    code_ = proc.wait()
stop.set()
print("exit code", code_, "(0 complete, 2 partial)", f"{(time.time()-started)/60:.1f} min this session")
failures = {}
for p in (OUT / "blocks").glob("*.json"):
    for e in json.loads(p.read_text()).get("scene_errors", {}).values():
        kind = "network read (rerun this cell)" if ("RasterioIOError" in e or "CPLE_" in e) else e[:60]
        failures[kind] = failures.get(kind, 0) + 1
print("scene errors by kind:", failures or "none")
"""),
        code("""
# 7. Results: only blocks worked on here (parquet + checkpoint json) and the manifest, zipped,
#    split into parts of at most 50 MB with SHA256SUMS.txt. Download every part and the checksum file.
manifest = json.loads((OUT / "manifest.json").read_text())
print("status", manifest["status"], "| blocks", {b["status"] for b in manifest["blocks"].values()},
      "| scene failures", manifest["scene_failures"], "| fields with usable dates", manifest.get("fields_with_usable_dates"))
changed = []
for p in sorted((OUT / "blocks").glob("*.json")):
    now = json.loads(p.read_text()).get("status")
    if state.get(p.stem) != "complete" and now in ("complete", "partial"):
        changed.append(p.stem)
print(len(changed), "blocks to send back; still unfinished:",
      [p.stem for p in (OUT / "blocks").glob("*.json") if json.loads(p.read_text()).get("status") == "running"])
result = Path(f"/content/{RUN}_colab_blocks.zip")
with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for block in changed:
        archive.write(OUT / "blocks" / f"{block}.parquet", f"blocks/{block}.parquet")
        archive.write(OUT / "blocks" / f"{block}.json", f"blocks/{block}.json")
    archive.write(OUT / "manifest.json", "colab_manifest.json")
    archive.writestr("colab_environment.txt", versions + f"\\ncode fingerprint {digest}\\nsame-result check PASSED\\n")
data = result.read_bytes()
size = bundle["result_part_mb"] * 1024 * 1024
parts = []
for i in range(0, len(data), size):
    part = Path(f"{result}.{i // size:03d}")
    part.write_bytes(data[i:i + size])
    parts.append(part)
sums = Path(f"/content/{RUN}_SHA256SUMS.txt")
sums.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\\n" for p in parts)
                + f"{hashlib.sha256(data).hexdigest()}  {result.name}\\n")
print([f"{p.name} {p.stat().st_size/1e6:.1f} MB" for p in parts], sums.name)
if Path("/content/drive/MyDrive").exists():
    for p in [*parts, sums]:
        shutil.copy2(p, Path("/content/drive/MyDrive") / p.name)
    print("copied to Drive")
from google.colab import files
for p in [sums, *parts]:
    files.download(str(p))
"""),
    ]
    cells[1]["source"] = ["CHECK_SCRIPT = " + repr(CHECK_SCRIPT) + "\n"] + cells[1]["source"]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                              "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="laptop run folder (manifest, blocks, catalog)")
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True, help="folder containing fieldvision/ (frozen copy)")
    parser.add_argument("--check-local", nargs="*", default=[], help="finished block ids of this run")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.name
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    digest = fingerprint(args.code / "fieldvision")
    if digest != manifest["source_sha256"]:
        raise SystemExit(f"code copy {digest[:8]} does not match the run's {manifest['source_sha256'][:8]}")
    params = manifest["parameters"]
    cli = cli_from_manifest(params, extractor_defaults(args.code))
    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True)

    code_name = f"code_{digest[:8]}.zip"
    with zipfile.ZipFile(args.dest / code_name, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((args.code / "fieldvision").rglob("*.py")):
            archive.write(path, path.relative_to(args.code).as_posix())
    fields_name = f"fields_{run}.geojson"
    shutil.copyfile(args.fields, args.dest / fields_name)

    # Snapshot of the laptop checkpoints. A block's parquet is written before its json,
    # so json-then-parquet read order gives a consistent pair even while the laptop runs.
    statuses = {}
    blocks_buffer = io.BytesIO()
    with zipfile.ZipFile(blocks_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for status_path in sorted((args.run_dir / "blocks").glob("*.json")):
            status_bytes = status_path.read_bytes()
            parquet = status_path.with_suffix(".parquet")
            if not parquet.exists():
                continue
            statuses[status_path.stem] = json.loads(status_bytes)["status"]
            archive.writestr(f"blocks/{status_path.name}", status_bytes)
            archive.writestr(f"blocks/{parquet.name}", parquet.read_bytes())
    block_parts = write_split(args.dest, f"blocks_{run}.zip", blocks_buffer.getvalue()) if statuses else []

    catalog_buffer = io.BytesIO()
    n_catalog = 0
    with zipfile.ZipFile(catalog_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((args.run_dir / "catalog").glob("*.json")):
            if statuses.get(path.stem) == "complete":
                continue                      # finished blocks are skipped before the catalog is read
            archive.writestr(f"catalog/{path.name}", strip_catalog(path))
            n_catalog += 1
    catalog_parts = write_split(args.dest, f"catalog_{run}.zip", catalog_buffer.getvalue()) if n_catalog else []

    with zipfile.ZipFile(args.dest / "check_blocks.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for block_id in args.check_local:
            if statuses.get(block_id) != "complete":
                raise SystemExit(f"check block {block_id} is not complete on the laptop")
            status = json.loads((args.run_dir / "blocks" / f"{block_id}.json").read_text())
            if status.get("scene_errors"):
                raise SystemExit(f"check block {block_id} has scene errors")
            prefix = f"check/{block_id}"
            archive.writestr(f"{prefix}/catalog.json", strip_catalog(args.run_dir / "catalog" / f"{block_id}.json"))
            archive.write(args.run_dir / "blocks" / f"{block_id}.parquet", f"{prefix}/reference.parquet")
            archive.writestr(f"{prefix}/settings.json", json.dumps({"year": params["year"], "parameters": params,
                                                                     "source_run": run, "fields": "MAIN"}))

    bundle = {"run": run, "code_zip": code_name, "code_sha256": digest, "fields": fields_name, "cli": cli,
              "block_count": manifest["block_count"], "catalog_parts": catalog_parts, "block_parts": block_parts,
              "laptop_blocks_at_build": {s: list(statuses.values()).count(s) for s in set(statuses.values())},
              "catalog_files": n_catalog, "check_blocks": args.check_local, "pins": PINS, "laptop": LAPTOP_LIBS,
              "result_part_mb": RESULT_PART_MB}
    (args.dest / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    write_notebook(args.dest / f"extract_{run}.ipynb", bundle)
    sizes = {p.name: p.stat().st_size for p in sorted(args.dest.iterdir())}
    print(json.dumps(dict(bundle, file_sizes=sizes), indent=2))
    too_big = [n for n, s in sizes.items() if s > 9 * 1000 * 1000]
    if too_big:
        raise SystemExit(f"files over 9 MB: {too_big}")


if __name__ == "__main__":
    main()
