"""Per-field predictions from a trained run.

    python -m fieldvision.model.predict --model <run dir> --timeseries <...> \
        --fields <...> --out predictions.parquet [--which gbm] [--keep-columns county_fips]

Writes one row per field (see artifacts.PREDICTION_COLUMNS) and a small
`<out>.meta.json` with the model, timing and counts. Labels in the field
file, if any, are carried along but never used.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from .artifacts import load_run, predictions_frame, write_json


def add_input_args(ap):
    ap.add_argument("--model", required=True, help="run directory written by train")
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--which", default=None, help="model name in the run (default: its primary)")
    ap.add_argument("--rename", nargs="*", default=[], metavar="OURS=THEIRS",
                    help="map file columns to the contract, e.g. field_id=FID label=harvest_type")
    ap.add_argument("--label-map", nargs="*", default=[], metavar="VALUE=silage|grain",
                    help="map label values, e.g. Silage=silage 'Grain corn'=grain")
    ap.add_argument("--keep-columns", nargs="*", default=[],
                    help="extra field columns to copy into the output (e.g. a county code)")
    ap.add_argument("--peak-window", nargs=2, type=float, default=None, metavar=("START", "END"),
                    help="day-of-year window for the corn peak, overriding the run's; set it from "
                         "the regional crop calendar (e.g. Texas), never tuned on labels")


def parse_pairs(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def load_inputs(args, cfg):
    """fields, time series, features and timing, using the run's feature config."""
    if getattr(args, "peak_window", None):
        from dataclasses import replace
        cfg = replace(cfg, peak_window=tuple(args.peak_window))
    fields = load_fields(args.fields, rename=parse_pairs(args.rename) or None,
                         label_map=parse_pairs(args.label_map) or None,
                         keep_columns=args.keep_columns)
    t0 = time.perf_counter()
    ts, notes = load_timeseries(args.timeseries, field_ids=fields["field_id"])
    t1 = time.perf_counter()
    X, info = build_features(ts, fields, cfg)
    info["load_seconds"] = t1 - t0
    info["timeseries_notes"] = notes
    return fields, ts, X, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict silage vs grain per field.")
    add_input_args(ap)
    ap.add_argument("--out", required=True, help="output Parquet file")
    args = ap.parse_args(argv)

    model, meta, cfg, _ = load_run(args.model, args.which)
    fields, ts, X, info = load_inputs(args, cfg)
    t0 = time.perf_counter()
    p = model.predict_proba(X)
    pred_s = time.perf_counter() - t0
    name = args.which or meta["primary"]
    out = predictions_frame(X, fields, p, model.threshold, name, keep_columns=args.keep_columns)
    out["run_id"] = meta["run_id"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    n = max(len(out), 1)
    summary = {
        "run_id": meta["run_id"], "model": name, "threshold": model.threshold,
        "n_fields": int(len(out)),
        "n_without_timeseries": int((out["low_evidence_reason"] == "no_timeseries").sum()),
        "n_low_evidence": int(out["low_evidence"].sum()),
        "n_predicted_silage": int((out["predicted_label"] == "silage").sum()),
        "features_seconds_per_1000": info["seconds"] / n * 1000,
        "predict_seconds_per_1000": pred_s / n * 1000,
        "load_seconds": info["load_seconds"],
        "timeseries_notes": info["timeseries_notes"],
        "synthetic_training_data": meta.get("synthetic", False),
    }
    write_json(str(args.out) + ".meta.json", summary)
    print(f"{len(out)} fields -> {args.out}; silage {summary['n_predicted_silage']}, "
          f"low evidence {summary['n_low_evidence']}; "
          f"{summary['features_seconds_per_1000'] + summary['predict_seconds_per_1000']:.2f} s "
          "per 1000 fields (features + prediction)")
    return out


if __name__ == "__main__":
    main()
