"""Final thinned LightGBM on all labelled fields, written as a training run directory.

    python -m fieldvision.foundation.augment_train --timeseries ... --fields ... --out <run dir>

The model is the main line's 135-feature LightGBM (relative variant, in-fold Platt),
trained on every labelled field plus 3 date-thinned copies of each (foundation.augment,
kind "thin": each observation kept with probability 0.7, then one 10-20 day window
dropped; the same copies as in the three-country experiment, seed 101). Everything else
is the base setting.

The output directory has the layout model.train writes (run.json with frozen model ids,
models/<name>.pkl, models/majority.pkl, metrics.json, features.parquet), so
`python -m fieldvision.model.evaluate --model <run dir> --which gbm_thin ...`,
`python -m fieldvision.model.predict --model <run dir> --which gbm_thin ...` and
`python -m fieldvision.model.registry export --run <run dir> --which gbm_thin ...` take
it unchanged. metrics.json holds the same model's out-of-fold results (5 spatial folds
and leave-one-country-out), copies always on the training side only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig
from ..model.artifacts import save_run, write_json
from ..model.explain import reference_values
from ..model.harness import field_collection, summarize_predictions
from ..model.models import make_factories
from ..model.registry import portable_command, portable_path
from ..model.splits import leave_one_group_out, spatial_folds
from .augment import augment_fields, augment_timeseries, parent_id
from .augment_eval import run_arm

NAME = "gbm_thin"
AUG_SEED = 101          # the seed of the three-country experiment (its --seed 0 + 101)
COPIES = 3
KEY = ["harv_days_since_peak", "harv_mid_doy", "pre_ndvi_frac", "pre_ndmi", "sen_ndvi_slope_rel",
       "post_ndti", "peak_ndvi", "q_n_usable_season", "q_max_gap_days", "q_harv_width_days"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if (out / "run.json").exists():
        raise SystemExit(f"{out} already holds a run; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = load_fields(args.fields)
        fields = fields[np.isfinite(fields["y"].to_numpy(float))].reset_index(drop=True)
        fields["field_id"] = fields["field_id"].astype(str)
        ts, _ = load_timeseries(args.timeseries, field_ids=fields["field_id"])
        X, _ = build_features(ts, fields, cfg)
        X = X.reset_index(drop=True).drop(columns=["field_id"], errors="ignore")
        tsa = augment_timeseries(ts, ("thin",), COPIES, seed=AUG_SEED)
        afields = augment_fields(fields, COPIES)
        Xa, _ = build_features(tsa, afields, cfg)
        Xa = Xa.reset_index(drop=True).drop(columns=["field_id"], errors="ignore")
    del tsa
    y = fields["y"].to_numpy(int)
    pos = {f: i for i, f in enumerate(fields["field_id"])}
    par = np.array([pos[parent_id(f)] for f in afields["field_id"]])
    X["block_id"] = fields["block_id"].to_numpy()
    Xa["block_id"] = fields["block_id"].to_numpy()[par]
    print(f"{len(fields)} fields, {len(Xa)} thinned copies", flush=True)

    facs = make_factories(X.columns, "relative", seed=args.seed)
    make = facs["gbm"]
    model = make().fit(pd.concat([X, Xa], ignore_index=True), np.r_[y, y[par]])
    majority = facs["majority"]().fit(X, y)

    meta = fields.copy()
    meta["year"] = meta["year"].map(lambda v: str(int(v)) if np.isfinite(v) else "unknown")
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    splits = [("spatial", f"fold{i}", tr, te)
              for i, (tr, te) in enumerate(spatial_folds(meta["block_id"], y, 5, args.seed))]
    for g, tr, te, _ in leave_one_group_out(meta["country"], meta["block_id"], exclude_test_blocks=False):
        splits.append((f"loco:{g}", g, tr, te))
    preds = run_arm(NAME, make, X, meta, y, Xa, y[par], par, splits)
    preds.to_parquet(out / "cv_predictions.parquet", index=False)
    summary = summarize_predictions(preds, n_boot=args.n_boot, seed=args.seed)

    coll = field_collection(ts)
    meta["collection"] = meta["field_id"].map(coll).fillna("unknown").astype(str)
    created = dt.datetime.now().isoformat(timespec="seconds")
    run_id = hashlib.sha256(f"{out.resolve()}|{created}".encode()).hexdigest()[:12]
    describe = (f"{model.describe()}; trained on {len(fields)} fields plus {COPIES} date-thinned copies "
                "of each (keep probability 0.7, one 10-20 day gap, seed 101)")
    meta_run = {
        "run_id": run_id, "created": created, "primary": NAME, "variant": "relative",
        "feature_config": cfg.to_dict(),
        "threshold": {NAME: model.threshold, "majority": majority.threshold},
        "models": {NAME: describe, "majority": majority.describe()},
        "feature_names": {NAME: model.feature_names, "majority": []},
        "training": {"n_fields": int(len(fields)), "n_silage": int(y.sum()),
                     "countries": sorted(meta["country"].astype(str).unique()),
                     "years": sorted(meta["year"].unique()),
                     "collections": sorted(meta["collection"].unique()),
                     "fields_files": sorted(portable_path(f) for f in args.fields)},
        "augmentation": {"kind": "thin", "copies": COPIES, "keep_probability": 0.7,
                         "gap_days": [10, 20], "seed": AUG_SEED},
        "holdout_patterns": [], "synthetic": False, "train_prevalence": float(y.mean()),
        "evidence_reference": reference_values(X, y),
        "command": portable_command(sys.argv),
    }
    save_run(out, {NAME: model, "majority": majority}, meta_run)
    write_json(out / "metrics.json", {"run_id": run_id, "primary": NAME, "variant": "relative",
                                      "summary": summary, "calibration": {}, "prevalence_table": [],
                                      "final_models": meta_run["models"]})
    feats = X[[c for c in KEY if c in X.columns]].copy()
    feats.insert(0, "field_id", fields["field_id"].to_numpy())
    feats["label"] = np.where(y == 1, "silage", "grain")
    feats["split"] = "development"
    feats.to_parquet(out / "features.parquet", index=False)
    for s in ["spatial"] + sorted(k for k in summary if k.startswith("loco:")):
        r = summary[s]["relative"][NAME]["all"]
        print(f"{s}: AUC {r['roc_auc']:.3f}, recall {r['recall']:.3f}, FPR {r['fpr']:.3f}", flush=True)
    print(f"run {run_id} written", flush=True)


if __name__ == "__main__":
    main()
