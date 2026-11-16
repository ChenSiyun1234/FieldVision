"""Build the eight-run pixel-chip upload bundle from prepared subset fields."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import textwrap
import zipfile

from build_colab_bundle import PINS

RUNS = (("at_2022", 2022), ("at_2023", 2023), ("at_2024", 2024),
        ("fr_2022", 2022), ("nl_2025", 2025), ("barc_2021", 2021),
        ("barc_2022", 2022), ("barc_2023", 2023))
S2_SHA256 = "9cd0447bcc1272bafdfb336059b4d60e2e0b56bd45825a580b410c43db6a0977"
CHUNK = 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_code(destination, expected=S2_SHA256, s2_source=None):
    """Zip the package. `s2_source` is a folder holding the frozen `s2/*.py` (the copy the
    existing extractions ran with); its files replace the checkout's, byte for byte."""
    package = Path(__file__).resolve().parents[1] / "src" / "fieldvision"
    sources = [(p.relative_to(package.parent).as_posix(), p.read_bytes())
               for p in sorted(package.rglob("*.py"))]
    if s2_source is not None:
        sources = [(n, b) for n, b in sources if not n.startswith("fieldvision/s2/")]
        sources += [(f"fieldvision/s2/{p.name}", p.read_bytes()) for p in sorted(Path(s2_source).glob("*.py"))]
        sources.sort()
    digest = hashlib.sha256()
    for name, payload in sources:
        if name.startswith("fieldvision/s2/") and name.count("/") == 2:
            digest.update(payload)
    if digest.hexdigest() != expected:
        raise ValueError(f"Frozen fieldvision/s2 fingerprint: actual {digest.hexdigest()}, expected {expected}")
    temporary = destination / "code_stream.zip.tmp"
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, payload in sources:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload, compresslevel=9)
        code_hash = sha256_file(temporary)
        name = f"code_stream_{code_hash[:8]}.zip"
        temporary.replace(destination / name)
    finally:
        temporary.unlink(missing_ok=True)
    return name, code_hash


CELLS = [r'''
# 1. Settings. RUNS may be shortened to divide years/runs across runtimes.
import os, sys, json, time, shutil, zipfile, hashlib, subprocess, threading, re
from pathlib import Path
UPLOAD = Path("/content")
WORK = Path("/content/fv_stream_px")
CODE = WORK / "code"
bundle = json.loads((UPLOAD / "bundle.json").read_text())
RUNS = [run["name"] for run in bundle["runs"]]  # e.g. ["at_2022", "barc_2021"]
WORKERS = 32 if (os.cpu_count() or 1) >= 8 else 16
DRIVE_BACKUP = Path("/content/drive/MyDrive/fieldvision_stream_px")
needed = [bundle["code_zip"], *[r["fields"] for r in bundle["runs"]],
          *[r["reference"] for r in bundle["runs"] if r["reference"]]]
missing = [name for name in needed if not (UPLOAD / name).is_file()]
print("missing uploads:", missing or "none")
print(f"{os.cpu_count()} CPUs, {WORKERS} read threads; Python {sys.version.split()[0]}")
WORK.mkdir(parents=True, exist_ok=True)
''', r'''
# 2. Optional, recommended: mount Google Drive so checkpoints survive a recycled runtime
USE_DRIVE = True  # Set False to run without Drive; download results before the runtime ends.
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_BACKUP.mkdir(parents=True, exist_ok=True)
''', r'''
# 3. Install pinned binary wheels and report the actual runtime versions.
pins = [f"{name}=={version}" for name, version in bundle["pins"].items()]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:", *pins], check=True)
version_code = (
    "import importlib.metadata as m, rasterio, pyproj, sys;"
    "print('python', sys.version.split()[0]);"
    f"[print(p, m.version(p)) for p in {list(bundle['pins'])!r}];"
    "print('GDAL', rasterio.__gdal_version__); print('PROJ', pyproj.proj_version_str)"
)
versions = subprocess.run([sys.executable, "-c", version_code],
                          capture_output=True, text=True, check=True).stdout
print(versions)
print("Pin source:", bundle["pins_source"])
''', r'''
# 4. Verify the upload ZIP, unpack Python sources, and verify frozen Sentinel-2 code.
def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

code_zip = UPLOAD / bundle["code_zip"]
assert file_hash(code_zip) == bundle["code_sha256"], "Code ZIP checksum differs"
with zipfile.ZipFile(code_zip) as archive:
    for member in archive.infolist():
        parts = member.filename.split("/")
        assert parts[0] == "fieldvision" and member.filename.endswith(".py")
        assert not any(p in ("", ".", "..") or ":" in p or "\\" in p for p in parts)
    shutil.rmtree(CODE, ignore_errors=True)
    archive.extractall(CODE)
digest = hashlib.sha256()
for path in sorted((CODE / "fieldvision" / "s2").glob("*.py")):
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
s2_digest = digest.hexdigest()
assert s2_digest == bundle["s2_source_sha256"], "Frozen Sentinel-2 fingerprint differs"
print("code ZIP", bundle["code_sha256"], "| frozen s2", s2_digest)
env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONUNBUFFERED="1")
''', r'''
# 5. Extract selected runs sequentially. Rerun this cell to resume unfinished blocks.
selected = {run["name"]: run for run in bundle["runs"]}
assert len(RUNS) == len(set(RUNS)) and set(RUNS) <= set(selected), "Unknown or duplicate RUNS"
CHECK_PASSED = set()
exit_codes = {}

def read_json(path):
    return json.loads(path.read_text())

def copy_atomic(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)

def copy_finished(source, target, restoring=False):
    if not source.exists():
        return
    source_manifest = source / "manifest.json"
    target_manifest = target / "manifest.json"
    if source_manifest.exists() and target_manifest.exists():
        if read_json(source_manifest).get("signature") != read_json(target_manifest).get("signature"):
            raise RuntimeError(f"Checkpoint signature mismatch for {target.name}")
    for status_path in sorted((source / "blocks").glob("*.json")):
        status_bytes = status_path.read_bytes()
        if json.loads(status_bytes).get("status") not in ("complete", "partial"):
            continue
        target_status = target / "blocks" / status_path.name
        if restoring and target_status.exists():
            local = read_json(target_status)
            if local.get("status") in ("complete", "partial") and all(
                    target_status.with_suffix(suffix).exists() for suffix in (".parquet", ".npz")):
                continue  # Existing finished local work takes precedence over a Drive snapshot.
        files = [status_path.with_suffix(suffix) for suffix in (".parquet", ".npz")]
        if not all(path.is_file() for path in files):
            continue
        target_status.parent.mkdir(parents=True, exist_ok=True)
        staged = []
        try:
            for path in files:
                dest = target_status.with_suffix(path.suffix)
                temp = dest.with_name(dest.name + ".tmp")
                shutil.copyfile(path, temp)
                staged.append((temp, dest))
            if status_path.read_bytes() != status_bytes:
                continue
            for temp, dest in staged:
                temp.replace(dest)
            temp_status = target_status.with_name(target_status.name + ".tmp")
            temp_status.write_bytes(status_bytes)
            temp_status.replace(target_status)
            catalog = source / "catalog" / status_path.name
            if catalog.exists():
                copy_atomic(catalog, target / "catalog" / catalog.name)
        finally:
            for temp, _ in staged:
                temp.unlink(missing_ok=True)
    if source_manifest.exists() and (not restoring or not target_manifest.exists()):
        copy_atomic(source_manifest, target_manifest)

def progress(out):
    manifest = read_json(out / "manifest.json") if (out / "manifest.json").exists() else {}
    count = sum(read_json(path).get("status") in ("complete", "partial")
                for path in (out / "blocks").glob("*.json"))
    return f"{count}/{manifest.get('block_count', '?')}"

for name in RUNS:
    run = selected[name]
    out = WORK / name
    out.mkdir(parents=True, exist_ok=True)
    if USE_DRIVE:
        copy_finished(DRIVE_BACKUP / name, out, restoring=True)
    stop = threading.Event()
    backup_errors = []
    def backup_loop():
        while not stop.wait(300):
            try:
                copy_finished(out, DRIVE_BACKUP / name)
            except Exception as exc:
                backup_errors.append(str(exc))
                print(f"{name}: Drive backup failed: {exc}", flush=True)
    thread = threading.Thread(target=backup_loop, daemon=True) if USE_DRIVE else None
    if thread:
        thread.start()
    cmd = [sys.executable, "-W", "ignore", "-m", "fieldvision.stream.pixels",
           "--fields", str(UPLOAD / run["fields"]), "--year", str(run["year"]),
           "--out", str(out), "--chip", str(bundle["chip_size"]), "--workers", str(WORKERS)]
    print(f"[{time.strftime('%H:%M:%S')}] {name} {progress(out)} blocks done | starting", flush=True)
    proc = None
    try:
        with (out / "run.log").open("a") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log.write(line)
                if any(word in line for word in ("Block ", "complete:", "partial:", "ERROR", "WARNING")):
                    print(f"[{time.strftime('%H:%M:%S')}] {name} {progress(out)} blocks done | {line.strip()}", flush=True)
            exit_codes[name] = proc.wait()
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        stop.set()
        if thread:
            thread.join()
            try:
                copy_finished(out, DRIVE_BACKUP / name)
            except Exception as exc:
                print(f"{name}: final Drive backup failed: {exc}")
    manifest = read_json(out / "manifest.json") if (out / "manifest.json").exists() else {}
    errors = {}
    for checkpoint in (out / "blocks").glob("*.json"):
        for error in read_json(checkpoint).get("scene_errors", {}).values():
            network = any(token in error for token in ("RasterioIOError", "CPLE_", "Timeout", "Connection", "HTTP"))
            kind = "network read (rerun this cell)" if network else error.split(":", 1)[0]
            errors[kind] = errors.get(kind, 0) + 1
    print(f"[{time.strftime('%H:%M:%S')}] {name} {progress(out)} blocks done | exit {exit_codes[name]}")
    print(name, "scene errors:", errors or "none")
print("Exit codes (0 complete, 2 partial):", exit_codes)
assert all(code in (0, 2) for code in exit_codes.values()), "Extraction failed; inspect the log and rerun this cell"
''', r'''
# 6. Check European field/date medians against the supplied references; print BARC counts only.
for name in RUNS:
    run = selected[name]
    out = WORK / name
    manifest = read_json(out / "manifest.json")
    counts = {key: manifest.get(key) for key in ("status", "input_fields", "fields_with_chips",
              "n_obs", "median_chips_per_field", "scene_failures")}
    print(name, json.dumps(counts))
    if run["reference"]:
        result = subprocess.run([sys.executable, "-m", "fieldvision.stream.pixels", "check",
                                 "--run", str(out), "--reference", str(UPLOAD / run["reference"]),
                                 "--min-agree", ".99"], env=env, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode:
            print(result.stderr)
            raise RuntimeError(f"{name}: same-result check FAILED; stop before exporting")
        CHECK_PASSED.add(name)
        print(name, "same-result check PASSED")
''', r'''
# 7. Export per-run ZIPs, <=50 MiB parts and SHA256SUMS; copy to Drive and print sizes.
for name in RUNS:
    run = selected[name]
    assert not run["reference"] or name in CHECK_PASSED, f"Run step 6 successfully for {name}"
    out = WORK / name
    result = UPLOAD / f"stream_px_{name}_colab_chips.zip"
    environment = versions + f"\ncode ZIP {bundle['code_sha256']}\nfrozen s2 {s2_digest}\n"
    environment += "same-result check PASSED\n" if name in CHECK_PASSED else "same-result check not applicable: no reference\n"
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for filename in ("chips.npz", "timeseries.parquet", "manifest.json"):
            archive.write(out / filename, filename)
        archive.writestr("colab_environment.txt", environment)
    for old in UPLOAD.glob(result.name + ".*"):
        if re.fullmatch(re.escape(result.name) + r"\.\d{3,}", old.name):
            old.unlink()
    parts, lines = [], []
    whole = hashlib.sha256()
    with result.open("rb") as source:
        index = 0
        while True:
            first = source.read(1024 * 1024)
            if not first:
                break
            part = result.with_name(result.name + f".{index:03d}")
            digest = hashlib.sha256()
            remaining = bundle["result_part_mb"] * 1024 * 1024
            with part.open("wb") as target:
                chunk = first
                while chunk:
                    target.write(chunk)
                    digest.update(chunk)
                    whole.update(chunk)
                    remaining -= len(chunk)
                    if not remaining:
                        break
                    chunk = source.read(min(1024 * 1024, remaining))
            parts.append(part)
            lines.append(f"{digest.hexdigest()}  {part.name}\n")
            index += 1
    lines.append(f"{whole.hexdigest()}  {result.name}\n")
    sums = UPLOAD / f"stream_px_{name}_SHA256SUMS.txt"
    sums.write_text("".join(lines), encoding="utf-8")
    exports = [result, *parts, sums]
    if USE_DRIVE:
        current_parts = {part.name for part in parts}
        for old in DRIVE_BACKUP.glob(result.name + ".*"):
            if re.fullmatch(re.escape(result.name) + r"\.\d{3,}", old.name) and old.name not in current_parts:
                old.unlink()
        for path in exports:
            copy_atomic(path, DRIVE_BACKUP / path.name)
    for path in exports:
        print(f"{path.name} {path.stat().st_size:,} bytes" + (" | copied to Drive" if USE_DRIVE else ""))
print("Download every .zip.NNN part and its SHA256SUMS.txt; the full ZIP is optional.")
''']


def write_notebook(destination):
    cells = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": textwrap.dedent(source).strip().splitlines(keepends=True)} for source in CELLS]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def build(args):
    import pandas as pd

    pins = dict(PINS)
    pins_source = "build_colab_bundle.PINS fallback; no restricted template bundle was read"
    if args.pins_from:
        document = json.loads(args.pins_from.read_text(encoding="utf-8"))
        pins = document.get("pins", document)
        pins_source = "explicit --pins-from JSON override"
    if not isinstance(pins, dict) or not pins or not all(isinstance(k, str) and isinstance(v, str)
                                                      for k, v in pins.items()):
        raise ValueError("Pins must be a nonempty mapping of package names to version strings")
    destination = args.bundle
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Bundle destination must be empty; existing files are preserved")
    destination.mkdir(parents=True, exist_ok=True)
    expected = args.expected_s2_sha256
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Expected Sentinel-2 fingerprint must be 64 lowercase hexadecimal characters")
    code_name, code_hash = package_code(destination, expected, args.s2_source)
    runs = []
    for name, year in RUNS:
        source = args.subsets / name / "fields.geojson"
        fields = json.loads(source.read_text(encoding="utf-8"))
        features = fields["features"]
        ids = {str(feature["properties"]["field_id"]) for feature in features}
        if not features or len(ids) != len(features):
            raise ValueError(f"{name}: subset fields must be nonempty and field_id unique")
        fields_name = f"fields_{name}.geojson"
        shutil.copyfile(source, destination / fields_name)
        reference_name = None
        if not name.startswith("barc_"):
            reference = pd.read_parquet(args.timeseries_root / name / "timeseries.parquet")
            reference = reference[reference.field_id.astype(str).isin(ids)]
            if reference.empty:
                raise ValueError(f"{name}: no reference rows match the selected fields")
            reference_name = f"ref_{name}.parquet"
            reference.to_parquet(destination / reference_name, index=False)
        runs.append({"name": name, "year": year, "fields": fields_name,
                     "reference": reference_name, "n_fields": len(features)})
    bundle = {"runs": runs, "code_zip": code_name, "code_sha256": code_hash,
              "s2_source_sha256": expected, "pins": pins, "pins_source": pins_source,
              "chip_size": 12, "result_part_mb": 50}
    (destination / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    write_notebook(destination / "extract_stream_px.ipynb")
    uploads = ["bundle.json", code_name, *[run["fields"] for run in runs],
               *[run["reference"] for run in runs if run["reference"]]]
    european = sum(run["n_fields"] for run in runs if run["reference"])
    other = sum(run["n_fields"] for run in runs if not run["reference"])
    readme = f"""# stream_px: upload {len(uploads)} files to /content (bundle.json, {code_name}, 8 fields_*.geojson, 5 ref_*.parquet) and open extract_stream_px.ipynb; run all cells top to bottom (1-7).

This pixel-chip job covers {european} labelled European fields plus {other} Beltsville field-years with labels removed. It reads with the frozen Sentinel-2 code (fingerprint {expected[:8]}) and saves a 12 x 12 pixel chip per field and usable date next to the same full-field medians.

Priority: run this AFTER the Maryland 2025 sample. Select a CPU high-RAM runtime with at least 8 cores; no GPU is needed. The job is about 200 read blocks (roughly 2 hours on one runtime). It can be split across up to three runtimes by shortening RUNS in cell 1: at_2022, at_2023, at_2024, barc_* / fr_2022 / nl_2025 (upload the same files to each).

Each run's files (parts + stream_px_<run>_SHA256SUMS.txt) go to the results folder stream_px_<run> (watcher: -Run stream_px_<run>). The training job (stream_train) can also read them from Drive (MyDrive/fieldvision_stream_px) if cell 2 mounted it.

Open extract_stream_px.ipynb in Colab. Upload these files into /content:

""" + "\n".join(f"- {name}" for name in uploads) + f"""

Cell 2 optionally mounts Drive. Finished block checkpoints are backed up every 300 seconds under MyDrive/fieldvision_stream_px and restored after a runtime restart. Rerun cell 5 to resume; complete blocks are skipped. Network errors mean rerun this cell. Cell 6 prints comparison counts and requires at least 99% agreement for European references; BARC prints manifest counts only.

Dependencies: {pins_source}. Explicit --pins-from accepts a JSON package/version mapping or a bundle JSON containing pins. The notebook prints Python, package, GDAL, and PROJ versions.

Cell 7 returns one stream_px_<run>_colab_chips.zip per run containing chips.npz, timeseries.parquet, manifest.json, and colab_environment.txt. It also writes <=50 MiB .zip.000, .zip.001, ... parts and stream_px_<run>_SHA256SUMS.txt. Parts and the full ZIP have SHA256 checksums. Drive receives the ZIPs, parts, and checksum files. No automatic browser downloads are started.

For each run, download every part and its checksum file into a stream_px_<run> folder, then unpack into that run's configured results folder:

```sh
python scripts/stream_unpack.py --parts-dir DOWNLOADED_PARTS --out RESULTS_FOLDER
```

The four result files are extracted directly into RESULTS_FOLDER. Pass a parts folder containing exactly one run's checksum file. Verified result files replace existing files atomically. The unpacker verifies every part, the full ZIP, ZIP CRCs, and safe member names before installing a run.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"runs": runs, "code_zip": code_name, "code_sha256": code_hash,
                      "files": {p.name: p.stat().st_size for p in sorted(destination.iterdir())}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--timeseries-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pins-from", type=Path, help="Explicit JSON pins mapping or bundle JSON")
    parser.add_argument("--s2-source", type=Path,
                        help="Folder with the frozen s2/*.py to package instead of the checkout's")
    parser.add_argument("--expected-s2-sha256", default=S2_SHA256,
                        help="Expected source fingerprint; change only with explicit approval")
    args = parser.parse_args()
    try:
        build(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
