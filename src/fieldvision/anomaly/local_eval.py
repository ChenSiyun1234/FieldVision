"""European check of the local-threshold recipe (anomaly.local) on held-out years / countries.

    python -m fieldvision.anomaly.local_eval --timeseries ... --fields ... --out <dir>

For each held-out group: the main line's 6-feature logistic is trained on the other
groups (labels), source out-of-fold scores come from spatial CV inside those groups only,
and the held-out group is split by block into a reference county (grain + 1.3% silage)
and a test county (grain + 20% silage), many times (anomaly.local.emulate_counties).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig
from ..model.harness import run_splits
from ..model.models import make_factories
from ..model.splits import spatial_folds
from .local import DEFAULT_ALPHA, emulate_counties


def feature_table(timeseries, fields_files, cache=None):
    if cache is not None and Path(cache).exists():
        d = pd.read_parquet(cache)
        meta = d[["field_id", "block_id", "country", "year", "y", "low_evidence"]].copy()
        return d.drop(columns=["field_id", "block_id", "country", "year", "y"]), meta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = load_fields(fields_files)
        ts, _ = load_timeseries(timeseries, field_ids=fields["field_id"])
        fields = fields[np.isfinite(fields["y"].to_numpy(float))].reset_index(drop=True)
        X, _ = build_features(ts, fields, FeatureConfig())
    X = X.reset_index(drop=True).drop(columns=["field_id"], errors="ignore")
    meta = fields[["field_id", "block_id", "country", "year", "y"]].copy()
    meta["year"] = meta["year"].map(lambda v: str(int(v)) if np.isfinite(v) else "unknown")
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    if cache is not None:
        pd.concat([meta.drop(columns="low_evidence"), X], axis=1).to_parquet(cache, index=False)
    return X, meta


def source_and_target_scores(X, meta, test_mask, seed=0):
    """(source out-of-fold scores, source y, target scores from a model fit on all source)."""
    make = make_factories(X.columns, "relative", seed=seed)["logistic"]
    y = meta["y"].to_numpy(int)
    src = np.flatnonzero(~test_mask)
    Xs, ms, ys = X.iloc[src].reset_index(drop=True), meta.iloc[src].reset_index(drop=True), y[src]
    folds = [(f"f{i}", tr, te) for i, (tr, te) in enumerate(spatial_folds(ms["block_id"], ys, 5, seed))]
    oof, _, _ = run_splits(Xs, ms, ys, {"lr": make}, folds, "src", "relative", seed=seed)
    oof = oof.set_index("field_id").loc[ms["field_id"]]
    model = make().fit(Xs, ys)
    tgt = meta[test_mask].copy()
    tgt["p"] = model.predict_proba(X[test_mask])
    return oof["p"].to_numpy(), ys, tgt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[DEFAULT_ALPHA, 0.05, 0.20])
    ap.add_argument("--n-rep", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, meta = feature_table(args.timeseries, args.fields, out / "features.parquet")
    groups = []
    for col in ("year", "country"):
        if meta[col].nunique() > 1:
            groups += [(f"{'loyo' if col == 'year' else 'loco'}:{g}", (meta[col] == g).to_numpy())
                       for g in sorted(meta[col].unique())]
    rows, reps = [], []
    for name, mask in groups:
        s_src, y_src, tgt = source_and_target_scores(X, meta, mask, args.seed)
        for a in args.alphas:
            r = emulate_counties(tgt, s_src, y_src, alpha=a, n_rep=args.n_rep, seed=args.seed)
            r["held_out"], r["alpha"] = name, a
            reps.append(r)
            q = lambda c: (float(r[c].mean()), float(r[c].quantile(0.05)), float(r[c].quantile(0.95)))
            rows.append({"held_out": name, "alpha": a, "n_ref": int(r["n_ref"].mean()),
                         "n_test": int(r["n_test"].mean()), "true_share": float(r["true_share"].mean()),
                         **{k: q(k) for k in ("fixed_fpr", "fixed_recall", "fixed_share", "local_fpr",
                                              "local_recall", "local_flag_share", "local_adjusted_share")},
                         "tpr_source": float(r["tpr_source"].mean())})
            print(name, a, {k: round(v[0], 3) if isinstance(v, tuple) else v for k, v in rows[-1].items()},
                  flush=True)
    pd.concat(reps, ignore_index=True).to_csv(out / "local_threshold_reps.csv", index=False)
    (out / "local_threshold_summary.json").write_text(json.dumps({"args": vars(args), "rows": rows},
                                                                 indent=1, default=str), encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
