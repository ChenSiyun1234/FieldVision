"""Build the Colab job that trains the sequence and stream models on a GPU runtime.

    python scripts/build_stream_train_bundle.py --reference <run>/predictions.parquet
        --timeseries-root <dir with <run>/timeseries.parquet> --subsets <dir with <run>/fields.geojson>
        --bundle <job folder> [--runs at_2022 at_2023 at_2024 fr_2022 nl_2025]

The job folder gets the code zip, the fold tables (all fields, and the
fields that have pixel chips), the per-field tables of the listed runs, a
notebook and a README naming the files to upload. The chip zips of the
extraction job are uploaded next to them (or read from Google Drive).
"""
import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS = ["at_2022", "at_2023", "at_2024", "fr_2022", "nl_2025"]


def code_zip(package):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(package.rglob("*.py")):
            info = zipfile.ZipInfo("fieldvision/" + path.relative_to(package).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    return buffer.getvalue()


CELLS = [
    ("markdown", """# Stream models: training on a GPU runtime

Trains small sequence models on the per-field band medians (the control) and, on the fields that have
pixel chips, the models that read the pixels through the season. Every setting is fixed in the code;
nothing is tuned here. Upload the files listed in `README.md` into `/content`, pick a GPU runtime
(T4 is enough; high RAM if offered), then run the cells top to bottom. Every training step can be
rerun: finished models are skipped. Step 4 needs no chips and can run before the extraction job is done.
"""),
    ("code", """# 1. Settings
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
UPLOAD, WORK, OUT = Path("/content"), Path("/content/work"), Path("/content/out")
bundle = json.loads((UPLOAD / "bundle.json").read_text())
RUNS = bundle["runs"]                     # runs whose chips are used; all must be present for steps 5-8
SEEDS = 3
need = [bundle["code_zip"], "folds_full.parquet", "folds_subset.parquet", *[f"ts_{r}.parquet" for r in RUNS]]
print("missing uploads:", [n for n in need if not (UPLOAD / n).exists()] or "none")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip() or "NO GPU: change the runtime type")
print("cpus", os.cpu_count())
"""),
    ("code", """# 2. Optional: Google Drive (chip zips are read from MyDrive/fieldvision_stream_px/ when they are not
#    in /content, and results are copied to MyDrive/fieldvision_stream_train/)
from google.colab import drive
drive.mount("/content/drive")
"""),
    ("code", """# 3. Unpack the code, check it, print versions
CODE = WORK / "code"
shutil.rmtree(CODE, ignore_errors=True)
assert hashlib.sha256((UPLOAD / bundle["code_zip"]).read_bytes()).hexdigest() == bundle["code_sha256"], "code zip differs"
zipfile.ZipFile(UPLOAD / bundle["code_zip"]).extractall(CODE)
env = dict(os.environ, PYTHONPATH=str(CODE))
OUT.mkdir(parents=True, exist_ok=True)
versions = subprocess.run([sys.executable, "-c", "import torch, numpy, pandas, sklearn, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__)"],
                          capture_output=True, text=True).stdout.strip()
print(versions)
DRIVE = Path("/content/drive/MyDrive")

def train(name, folds, timeseries, models, times, schemes, chips=()):
    \"\"\"One training process; returns it so several can run side by side.\"\"\"
    cmd = [sys.executable, "-W", "ignore", "-m", "fieldvision.stream.train", "--folds", str(UPLOAD / folds),
           "--timeseries", *map(str, timeseries), "--models", *models, "--time", *times, "--schemes", *schemes,
           "--seeds", str(SEEDS), "--threads", "2", "--out", str(OUT / name)]
    if chips:
        cmd += ["--chips", *map(str, chips)]
    path = OUT / f"{name}.log"
    START[name] = len(path.read_text().splitlines()) if path.exists() else 0
    return name, subprocess.Popen(cmd, env=env, stdout=open(path, "a"), stderr=subprocess.STDOUT, text=True)

START = {}

def wait(procs):
    \"\"\"Print each process's new result lines until all have ended; stop the cell if one failed.\"\"\"
    seen = {name: START.get(name, 0) for name, _ in procs}
    while True:
        alive = [p.poll() is None for _, p in procs]
        for name, _ in procs:
            lines = (OUT / f"{name}.log").read_text().splitlines()
            for line in lines[seen[name]:]:
                if "ROC-AUC" in line or "fields" in line or "Error" in line or "wrote" in line:
                    print(f"[{time.strftime('%H:%M:%S')}] {name} | {line}", flush=True)
            seen[name] = len(lines)
        if not any(alive):
            break
        time.sleep(20)
    bad = [name for name, p in procs if p.returncode != 0]
    if bad:
        for name in bad:
            print((OUT / f"{name}.log").read_text()[-3000:])
        raise SystemExit(f"failed: {bad} (rerun the cell to resume)")
    if DRIVE.exists():
        shutil.copytree(OUT, DRIVE / "fieldvision_stream_train", dirs_exist_ok=True)
"""),
    ("code", """# 4. Control on ALL labelled European fields (band medians only, no chips): each country held out, then
#    the spatial folds. About 15 minutes on a T4.
ts_full = [UPLOAD / f"ts_{r}.parquet" for r in bundle["all_runs"]]
wait([train("control_full", "folds_full.parquet", ts_full, ["seq"], ["greenup", "calendar"], ["loco", "spatial"])])
"""),
    ("code", """# 5. Chips: find each run's parts (in /content, else on Drive), verify, join, unpack
PX = WORK / "px"
for run in RUNS:
    if (PX / run / "chips.npz").exists():
        continue
    tag = f"stream_px_{run}"
    src = next((d for d in (UPLOAD, DRIVE / "fieldvision_stream_px") if (d / f"{tag}_SHA256SUMS.txt").exists()), None)
    assert src is not None, f"{tag}_SHA256SUMS.txt not found in /content or on Drive"
    parts = []
    for line in (src / f"{tag}_SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split()
        if name[-4] == "." and name[-3:].isdigit():
            data = (src / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, f"{name}: checksum differs, upload it again"
            parts.append((name, data))
    blob = b"".join(d for _, d in sorted(parts))
    (PX / run).mkdir(parents=True, exist_ok=True)
    (WORK / f"{tag}.zip").write_bytes(blob)
    with zipfile.ZipFile(WORK / f"{tag}.zip") as z:
        assert z.testzip() is None
        z.extractall(PX / run)
    m = json.loads((PX / run / "manifest.json").read_text())
    print(run, m.get("status"), "fields with chips", m.get("fields_with_chips"), "chips", m.get("n_obs"),
          f"{(PX / run / 'chips.npz').stat().st_size / 1e6:.0f} MB")
chips = [PX / r / "chips.npz" for r in RUNS]
ts_sub = [PX / r / "timeseries.parquet" for r in RUNS]
"""),
    ("code", """# 6. Stage A (the result that matters): fields with chips, each country held out in turn.
#    Four processes side by side; the control (seq) is retrained on exactly these fields.
def stage(schemes, chip_times):
    return [train("subset_seq", "folds_subset.parquet", ts_sub, ["seq"], ["greenup", "calendar"], schemes, chips),
            train("subset_pixelset", "folds_subset.parquet", ts_sub, ["pixelset"], chip_times, schemes, chips),
            train("subset_chipseq", "folds_subset.parquet", ts_sub, ["chipseq"], chip_times, schemes, chips),
            train("subset_conv3d", "folds_subset.parquet", ts_sub, ["conv3d"], chip_times, schemes, chips)]
wait(stage(["loco"], ["greenup"]))
"""),
    ("code", """# 7. Stage B: the same models on the spatial folds (gives the in-country number for the loss column)
wait(stage(["loco", "spatial"], ["greenup"]))
"""),
    ("code", """# 8. Stage C (only if there is time): the chip models with calendar days instead of days since green-up
wait(stage(["loco", "spatial"], ["greenup", "calendar"]))
"""),
    ("code", """# 9. Results so far, as one zip (run after step 4, 6, 7 or 8; the name says which stages are in it).
#    Fetch the zip and the checksum file from the file panel.
done = [d.name for d in sorted(OUT.iterdir()) if d.is_dir() and (d / "predictions.parquet").exists()]
import pandas as pd
frames = {d: pd.read_parquet(OUT / d / "predictions.parquet") for d in done}
schemes = sorted({s.split(":")[0] for f in frames.values() for s in f["scheme"].unique()})
models = sorted({m for f in frames.values() for m in f["model"].unique()})
stage_tag = ("0" if "control_full" in done else "") + ("A" if any(d.startswith("subset") for d in done) else "") \\
    + ("B" if any(d.startswith("subset") and "spatial" in set(frames[d]["scheme"]) for d in done) else "") \\
    + ("C" if any(m.endswith("-cal") and not m.startswith("seq") for m in models) else "")
result = Path(f"/content/stream_train_{stage_tag}_colab_result.zip")
with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for d in done:
        z.write(OUT / d / "predictions.parquet", f"{d}/predictions.parquet")
        z.write(OUT / d / "settings.json", f"{d}/settings.json")
        z.write(OUT / f"{d}.log", f"{d}.log")
    z.writestr("colab_environment.txt", versions + "\\n")
sums = Path(f"/content/stream_train_{stage_tag}_SHA256SUMS.txt")
sums.write_text(f"{hashlib.sha256(result.read_bytes()).hexdigest()}  {result.name}\\n")
print(result.name, result.stat().st_size // 1024, "KB;", sums.name)
print("folders:", done, "| schemes:", schemes, "| models:", models)
if DRIVE.exists():
    (DRIVE / "fieldvision_stream_train").mkdir(exist_ok=True)
    shutil.copy2(result, DRIVE / "fieldvision_stream_train"); shutil.copy2(sums, DRIVE / "fieldvision_stream_train")
"""),
]


def notebook():
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            compile(source, "<cell>", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}, "accelerator": "GPU",
                                         "colab": {"gpuType": "T4"}},
            "nbformat": 4, "nbformat_minor": 5}


def main(argv=None):
    import pandas as pd

    from fieldvision.stream.folds import folds_from_reference
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--timeseries-root", type=Path, required=True)
    ap.add_argument("--subsets", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", default=RUNS)
    args = ap.parse_args(argv)
    args.bundle.mkdir(parents=True, exist_ok=True)
    data = code_zip(ROOT / "src" / "fieldvision")
    digest = hashlib.sha256(data).hexdigest()
    for old in args.bundle.glob("code_stream_*.zip"):
        old.unlink()
    name = f"code_stream_{digest[:8]}.zip"
    (args.bundle / name).write_bytes(data)

    preds = pd.read_parquet(args.reference)
    full = folds_from_reference(preds)
    full.to_parquet(args.bundle / "folds_full.parquet", index=False)
    ids = []
    for run in args.runs:
        with open(args.subsets / run / "fields.geojson", encoding="utf-8") as fh:
            ids += [f["properties"]["field_id"] for f in json.load(fh)["features"]]
    subset = folds_from_reference(preds, field_ids=ids)
    subset.to_parquet(args.bundle / "folds_subset.parquet", index=False)
    for run in RUNS:
        shutil.copy2(args.timeseries_root / run / "timeseries.parquet", args.bundle / f"ts_{run}.parquet")
    (args.bundle / "bundle.json").write_text(json.dumps(
        {"code_zip": name, "code_sha256": digest, "runs": args.runs, "all_runs": RUNS,
         "fields_full": int(len(full)), "fields_subset": int(len(subset)),
         "subset_per_country": subset["country"].value_counts().to_dict()}, indent=2), encoding="utf-8")
    (args.bundle / "train_stream.ipynb").write_text(json.dumps(notebook(), indent=1), encoding="utf-8")
    uploads = [name, "bundle.json", "folds_full.parquet", "folds_subset.parquet",
               *[f"ts_{r}.parquet" for r in RUNS], "train_stream.ipynb"]
    chip_files = ", ".join(f"`stream_px_{r}_colab_chips.zip.NNN` + `stream_px_{r}_SHA256SUMS.txt`" for r in args.runs)
    (args.bundle / "README.md").write_text(
        f"# stream_train: upload {', '.join(uploads)} to /content; GPU runtime (T4 is enough, high RAM if offered); "
        "run cells 1-4 now, cells 5-9 once the chip files are there.\n\n"
        "Trains the stream experiment's models. Priority: after the Maryland 2025 statewide sample. "
        "Cell 4 (the control on all 8,000 labelled European fields, medians only, about 15 minutes) needs no "
        "chips and can run as soon as a GPU runtime is free; run cell 9 after it and fetch "
        "`stream_train_0_colab_result.zip` + `stream_train_0_SHA256SUMS.txt`.\n\n"
        f"Cells 5-8 need the results of the `stream_px` job, uploaded to /content as they came back "
        f"({chip_files}), or left on Drive in `MyDrive/fieldvision_stream_px/` by that job's last cell (then mount "
        "Drive in cell 2 and nothing has to be uploaded). Cell 6 is stage A, the result that matters (about 30-60 "
        "minutes): run cell 9 after it and fetch `stream_train_0A_colab_result.zip` + its checksum file. Cell 7 "
        "(stage B, spatial folds) and cell 8 (stage C, optional) follow if the runtime is still free; cell 9 "
        "after each gives a new zip whose name carries the stages. Every cell can be rerun after a disconnect: "
        "finished models are skipped. Results go to the results folder `stream_train_<stages>` "
        "(watcher: `-Run stream_train_<stages> -SingleZip`).\n\n"
        f"Fields: {len(full)} in the control, {len(subset)} with chips "
        f"({subset['country'].value_counts().to_dict()}).\n", encoding="utf-8")
    for p in sorted(args.bundle.iterdir()):
        print(f"{p.stat().st_size:>12,}  {p.name}")


if __name__ == "__main__":
    main()
