"""Do augmentation or regularisation make the feature-rich model lose less out of country?

    python -m fieldvision.foundation.augment_eval --timeseries ... --fields ... --out <dir>

Arms (all fixed before any result):
  gbm|base      the main line's 135-feature LightGBM (relative variant, in-fold Platt)
  gbm|reg       the same features, regularised: 7 leaves, at least 100 fields per leaf,
                30% of features per tree, L2 5 (the relative variant already has no
                absolute-calendar feature, so there is nothing further to remove)
  gbm|shift, gbm|thin, gbm|gain, gbm|all   base GBM trained on the training fields plus
                3 augmented copies of each (foundation.augment), one kind or all three
  gbm|all+reg   both
  logit|base, logit|shift, logit|thin, logit|gain, logit|all   the 6-feature logistic,
                as a control
Splits: the main line's 5 spatial folds over all fields (in-country reference) and
leave-one-country-out. Copies are only ever added to a training part; every model is
scored on the real test fields. Paired comparisons to logit|base (the 6-feature
logistic) and to gbm|base on the same fields, block-bootstrap 95% intervals.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig
from ..model.harness import META_COLS, compare_models
from ..model.models import CalibratedModel, GBMModel, make_factories
from ..model.splits import leave_one_group_out, spatial_folds
from .augment import augment_fields, augment_timeseries, parent_id
from .compare import transfer_table

REG_PARAMS = dict(num_leaves=7, min_child_samples=100, colsample_bytree=0.3, reg_lambda=5.0)
AUG_SETS = {"shift": ("shift",), "thin": ("thin",), "gain": ("gain",), "all": ("shift", "thin", "gain")}
ARMS = [("gbm|base", "gbm", None, False), ("gbm|reg", "gbm", None, True),
        ("gbm|shift", "gbm", "shift", False), ("gbm|thin", "gbm", "thin", False),
        ("gbm|gain", "gbm", "gain", False), ("gbm|all", "gbm", "all", False),
        ("gbm|all+reg", "gbm", "all", True),
        ("logit|base", "logit", None, False), ("logit|shift", "logit", "shift", False),
        ("logit|thin", "logit", "thin", False), ("logit|gain", "logit", "gain", False),
        ("logit|all", "logit", "all", False)]
REF = "logit|base"


def model_columns(columns, seed=0):
    """The columns any arm reads (keeps memory low on the augmented tables)."""
    main = make_factories(columns, "relative", seed=seed)
    return list(dict.fromkeys(main["gbm"]().feature_names + main["logistic"]().feature_names
                              + ["low_evidence"]))


def features_for(ts, fields, cache, keep=None):
    if cache.exists():
        return pd.read_parquet(cache)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, _ = build_features(ts, fields, FeatureConfig())
    X = X.reset_index(drop=True).drop(columns=["field_id"], errors="ignore")
    if keep is not None:
        X = X[[c for c in keep if c in X.columns]]
    X.insert(0, "field_id", fields["field_id"].astype(str).to_numpy())
    X.to_parquet(cache, index=False)
    return X


def factory(columns, kind, reg, seed):
    main = make_factories(columns, "relative", seed=seed)
    if kind == "logit":
        return main["logistic"]
    if not reg:
        return main["gbm"]
    cols = main["gbm"]().feature_names
    return lambda: CalibratedModel(partial(GBMModel, cols, seed=seed, **REG_PARAMS), seed=seed)


def run_arm(name, make, X, meta, y, Xa, ya, parent_pos, splits):
    """Fit on each training part (+ its fields' copies when Xa is given), score the test part."""
    rows = []
    for scheme, fold, tr, te in splits:
        Xtr, ytr = X.iloc[tr], y[tr]
        if Xa is not None:
            sel = np.isin(parent_pos, tr)
            Xtr = pd.concat([Xtr, Xa[sel]], ignore_index=True)
            ytr = np.r_[ytr, ya[sel]]
        model = make().fit(Xtr, ytr)
        p = model.predict_proba(X.iloc[te])
        part = meta.iloc[te][META_COLS].copy()
        part["scheme"], part["variant"], part["fold"], part["model"] = scheme, "relative", str(fold), name
        part["y"], part["p"], part["threshold"] = y[te], p, model.threshold
        part["pred"] = (p >= model.threshold).astype(int)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arm names")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = load_fields(args.fields)
        fields = fields[np.isfinite(fields["y"].to_numpy(float))].reset_index(drop=True)
        ts, _ = load_timeseries(args.timeseries, field_ids=fields["field_id"])
    fields["field_id"] = fields["field_id"].astype(str)
    X = features_for(ts, fields, out / "features_base.parquet")
    keep = model_columns(X.columns, args.seed)
    X = X[["field_id"] + keep].copy()
    meta = fields.copy()
    meta["year"] = meta["year"].map(lambda v: str(int(v)) if np.isfinite(v) else "unknown")
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    y = meta["y"].to_numpy(int)
    X["block_id"] = meta["block_id"].to_numpy()       # inner calibration folds group by block
    pos = {f: i for i, f in enumerate(fields["field_id"])}
    print(f"{len(fields)} fields; base features {time.perf_counter() - t0:.0f}s", flush=True)

    aug = {}
    needed = {a for n, _k, a, _r in ARMS if a and (not args.arms or n in args.arms)}
    afields = augment_fields(fields, args.copies)
    for name in sorted(needed):
        t1 = time.perf_counter()
        cache = out / f"features_aug_{name}.parquet"
        if cache.exists():
            Xa = pd.read_parquet(cache)
        else:
            tsa = augment_timeseries(ts, AUG_SETS[name], args.copies, seed=args.seed + 101)
            Xa = features_for(tsa, afields, cache, keep)
            del tsa
        par = np.array([pos[parent_id(f)] for f in Xa["field_id"]])
        Xa = Xa.copy()
        Xa["block_id"] = meta["block_id"].to_numpy()[par]
        aug[name] = (Xa, y[par], par)
        print(f"augmented '{name}': {len(Xa)} copies, {time.perf_counter() - t1:.0f}s", flush=True)

    splits = [("spatial", f"fold{i}", tr, te)
              for i, (tr, te) in enumerate(spatial_folds(meta["block_id"], y, 5, args.seed))]
    for g, tr, te, _ in leave_one_group_out(meta["country"], meta["block_id"], exclude_test_blocks=False):
        splits.append((f"loco:{g}", g, tr, te))

    preds = []
    for name, kind, a, reg in ARMS:
        if args.arms and name not in args.arms:
            continue
        t1 = time.perf_counter()
        Xa, ya, par = aug[a] if a else (None, None, None)
        preds.append(run_arm(name, factory(X.columns, kind, reg, args.seed), X, meta, y, Xa, ya, par, splits))
        print(f"{name}: {time.perf_counter() - t1:.0f}s", flush=True)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(out / "predictions.parquet", index=False)

    models = [n for n, *_ in ARMS if n in set(preds["model"])]
    tabs = []
    for s in sorted(x for x in preds["scheme"].unique() if x.startswith("loco:")):
        t = transfer_table(preds, s, models=models, ref=REF, n_boot=args.n_boot, seed=args.seed)
        for col, ref in (("diff_vs_gbm_base", "gbm|base"),):
            vals, cis = [], []
            for m in t["model"]:
                c = compare_models(preds, m, ref, s, "relative", n_boot=args.n_boot, seed=args.seed) \
                    if m != ref else None
                vals.append(c["roc_auc"]["diff"] if c else 0.0)
                cis.append(c["roc_auc"]["ci"] if c else [0.0, 0.0])
            t[col], t[col + "_ci"] = vals, cis
        tabs.append(t)
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(out / "augment_table.csv", index=False)
    mean = tab.groupby("model", sort=False)[["auc_out", "auc_in", "auc_loss", "auc_diff_vs_ref"]].mean()
    mean.to_csv(out / "augment_mean.csv")
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "reg_params": REG_PARAMS,
                                                  "minutes": (time.perf_counter() - t0) / 60},
                                                 indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(tab[["scheme", "model", "auc_out", "auc_in", "auc_loss", "auc_diff_vs_ref", "auc_diff_vs_ref_ci",
               "diff_vs_gbm_base", "diff_vs_gbm_base_ci"]].round(3).to_string(index=False))
    print(mean.round(3).to_string())


if __name__ == "__main__":
    main()
