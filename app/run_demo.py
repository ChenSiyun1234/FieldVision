"""Build the demo's data from the team's files and serve the app.

    python app/run_demo.py                      # Beltsville 2021-2023 from C:\\t\\fv\\data, then http://localhost:8000
    python app/run_demo.py --data-root D:\\fv    # inputs somewhere else
    python app/run_demo.py --no-serve           # only (re)build app/data
    python app/run_demo.py --serve-only         # serve whatever app/data already holds
    python app/run_demo.py --sample             # synthetic sample regions instead (needs no inputs)

The Beltsville files are not in git. This script reads them (read-only) and writes the
small JSON files the app loads into app/data/, which git ignores:

    <data-root>/labels/barc_<year>/fields.geojson                                 fields + organisers' truth
    <data-root>/timeseries/barc_<year>/timeseries.parquet                         Sentinel-2 per field
    <data-root>/app/predictions_<model>/barc-<year>/predictions.parquet           stored predictions
    <data-root>/registry/<model>/provenance.json -> <training run>/run.json        training medians (optional)

Needs Python with numpy, pandas, pyarrow, geopandas and shapely. Only the satellite
base map needs the network; without it the fields are drawn on a plain background.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
REPO = APP.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_ROOT = Path(r"C:\t\fv\data")
YEARS = (2021, 2022, 2023)
MODEL = "eu_logistic_v1"
TRAINED_ON = "Austrian 2024 parcel registry labels only (2,400 fields); it never saw a US field"


def evidence_reference(root: Path, model: str):
    """The training run's run.json named in the model registry, if it is there."""
    prov = root / "registry" / model / "provenance.json"
    if not prov.exists():
        return None
    run = Path(json.loads(prov.read_text(encoding="utf-8")).get("training_run", "")) / "run.json"
    return run if run.exists() else None


def build_beltsville(root: Path, model: str, out: Path):
    from fieldvision.app import export

    missing = []
    jobs = []
    for year in YEARS:
        paths = {
            "fields": root / "labels" / f"barc_{year}" / "fields.geojson",
            "timeseries": root / "timeseries" / f"barc_{year}" / "timeseries.parquet",
            "predictions": root / "app" / f"predictions_{model}" / f"barc-{year}" / "predictions.parquet",
        }
        missing += [str(p) for p in paths.values() if not p.exists()]
        jobs.append((year, paths))
    if missing:
        sys.exit("Missing input files:\n  " + "\n  ".join(missing) + "\nPass --data-root, or --sample for synthetic data.")

    ref = evidence_reference(root, model)
    if ref is None:
        print("note: no training run found in the registry; evidence rows use the app's display thresholds")
    out.mkdir(parents=True, exist_ok=True)
    index = out / "regions.json"
    if index.exists():
        index.unlink()   # list only the regions built now
    for year, paths in jobs:
        meta = export.export_region(
            paths["fields"], out / f"barc-{year}", paths["timeseries"], paths["predictions"],
            title=f"Beltsville {year}", group="Beltsville, Maryland", year=year,
            min_clear=0.6,  # the feature code's usable-observation threshold
            note=f"{model}, trained on Austria 2024 only; never fitted on these fields",
            label_name="Organisers' truth", model_name=model, trained_on=TRAINED_ON,
            evidence_reference=ref,
        )
        m = meta["metrics"]["all"] or {}
        print(f"Beltsville {year}: {meta['metrics']['n_fields']} fields, "
              f"{m.get('tp', 0) + m.get('tn', 0)} of {m.get('n', 0)} right -> {out / f'barc-{year}'}")


def serve(port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(APP))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Demo running at http://localhost:{port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the demo data and serve the app.")
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT), help=r"Team data folder (default C:\t\fv\data)")
    ap.add_argument("--model", default=MODEL, help=f"Stored predictions to show (default {MODEL})")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-serve", action="store_true", help="Only build app/data")
    ap.add_argument("--serve-only", action="store_true", help="Do not rebuild; serve app/data as it is")
    ap.add_argument("--sample", action="store_true", help="Build synthetic sample regions instead of Beltsville")
    args = ap.parse_args(argv)

    out = APP / "data"
    if not args.serve_only:
        if args.sample:
            from fieldvision.app import sample_data
            index = out / "regions.json"
            if index.exists():
                index.unlink()
            sample_data.build(app_data=out)
        else:
            build_beltsville(Path(args.data_root), args.model, out)
    if not args.no_serve:
        serve(args.port)


if __name__ == "__main__":
    main()
