"""Does a pretrained time-series model help? Presto embeddings vs physical features.

    python -m fieldvision.foundation.compare --timeseries <dirs> --fields <geojsons> --out <dir>

Three inputs on identical folds and identical fields:
  emb   frozen Presto embeddings (128 numbers per field)
  phys  the main line's physical features (variant "relative")
  both  the two concatenated
each with a logistic-regression head and a LightGBM head, plus the main
line's rule and the majority-class baseline. Folds come from the modelling
code (model.splits.spatial_folds, 5 folds grouped by block_id, seed 0)
computed on the same field set the main line would use; fields that one of
the inputs cannot handle are then removed from every model's train and
test parts, and counted in the report.

Everything that learns, including scalers and Platt calibration, is fit
inside the training part of each fold (model.models). The Presto encoder
itself is frozen and was never trained on our labels.
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
from ..model.harness import compare_models, run_splits, summarize_predictions
from ..model.models import (CalibratedModel, GBMModel, LogisticModel, MajorityModel,
                            make_factories)
from ..model.splits import leave_one_group_out, spatial_folds
from .adapter import build_presto_inputs
from .embed import DEFAULT_WEIGHTS, EMB_COLS, embed_batch, file_sha256

TABLE_METRICS = ["precision", "recall", "fpr", "f1", "roc_auc"]


def make_comparison_factories(columns, seed=0, emb_cols=EMB_COLS, tag="emb"):
    """name -> constructor for the three inputs x two heads, plus rule and majority."""
    main = make_factories(columns, "relative", seed=seed)
    phys_logit = main["logistic"]().feature_names
    phys_gbm = main["gbm"]().feature_names
    emb_cols = list(emb_cols)

    def cal(make):
        return lambda m=make: CalibratedModel(m, seed=seed)

    return {
        "majority": MajorityModel,
        "phys-rule": main["rule"],
        "phys-logistic": main["logistic"],
        "phys-gbm": main["gbm"],
        f"{tag}-logistic": cal(partial(LogisticModel, emb_cols)),
        f"{tag}-gbm": cal(partial(GBMModel, emb_cols, seed=seed)),
        "both-logistic": cal(partial(LogisticModel, phys_logit + emb_cols)),
        "both-gbm": cal(partial(GBMModel, phys_gbm + emb_cols, seed=seed)),
    }


def restrict(splits, keep):
    """Drop fields outside `keep` (bool mask) from both sides of every split."""
    keep_idx = np.flatnonzero(keep)
    return [(name, np.intersect1d(tr, keep_idx), np.intersect1d(te, keep_idx))
            for name, tr, te in splits]


def fold_table(summary, scheme, variant="relative", metrics=TABLE_METRICS):
    """Per model: pooled value, mean and SD across folds, min-max across folds."""
    rows = []
    for model, rec in summary.get(scheme, {}).get(variant, {}).items():
        row = {"model": model, "n": rec["all"]["n"]}
        folds = rec["per_fold"]
        for m in metrics:
            vals = np.array([f[m] for f in folds.values()], dtype=float)
            row[m] = rec["all"][m]
            row[m + "_fold_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row[m + "_fold_sd"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else np.nan
            row[m + "_folds"] = [float(v) for v in vals]
            if "ci95" in rec["all"]:
                row[m + "_ci"] = rec["all"]["ci95"][m]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_fold_diffs(preds, a, b, scheme="spatial", metric="roc_auc"):
    """Per-fold metric difference a - b on the same test fields."""
    from ..model.metrics import classification_metrics
    pa = preds[(preds.scheme == scheme) & (preds.model == a)]
    pb = preds[(preds.scheme == scheme) & (preds.model == b)]
    m = pa.merge(pb[["field_id", "fold", "p", "pred"]], on=["field_id", "fold"], suffixes=("_a", "_b"))
    out = {}
    for fold, g in m.groupby("fold", sort=True):
        ma = classification_metrics(g["y"], g["pred_a"], g["p_a"])[metric]
        mb = classification_metrics(g["y"], g["pred_b"], g["p_b"])[metric]
        out[str(fold)] = float(ma - mb)
    return out


def verdict(diffs, boot):
    """Plain wording for a difference: only 'better'/'worse' when it beats the fold spread
    and the block-bootstrap interval excludes zero."""
    d = np.array(list(diffs.values()), dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        return "not enough folds"
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    lo, hi = boot["ci"]
    if abs(mean) > sd and (lo > 0 or hi < 0):
        return "better" if mean > 0 else "worse"
    return "no clear difference"


def _fmt(v, nd=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def markdown_table(tab):
    lines = ["| model | n | " + " | ".join(TABLE_METRICS) + " |",
             "|---|---:|" + "---:|" * len(TABLE_METRICS)]
    for _, r in tab.iterrows():
        cells = [f"{_fmt(r[m])} ({_fmt(r[m + '_fold_mean'], 2)} ± {_fmt(r[m + '_fold_sd'], 2)})"
                 for m in TABLE_METRICS]
        lines.append(f"| {r['model']} | {r['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--start-month", type=int, default=4)
    ap.add_argument("--end-month", type=int, default=11)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-blocks-with-data", action="store_true",
                    help="keep only label blocks with at least one time-series row "
                         "(for partial extraction runs; a complete run needs no filter)")
    ap.add_argument("--extra-embeddings", default=None,
                    help="parquet of other per-field vectors (e.g. fine-tuned) to add as model 'ft'")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    cfg = FeatureConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = load_fields(args.fields)
        ts, notes = load_timeseries(args.timeseries, field_ids=fields["field_id"])
    fields = fields[np.isfinite(fields["y"].to_numpy(float))].reset_index(drop=True)
    say(f"{len(fields)} labelled fields; {len(ts)} time-series rows for "
        f"{ts['field_id'].nunique()} fields")
    if args.only_blocks_with_data:
        has = set(fields.loc[fields["field_id"].isin(set(ts["field_id"])), "block_id"])
        before = len(fields)
        fields = fields[fields["block_id"].isin(has)].reset_index(drop=True)
        say(f"partial run: kept {len(fields)} of {before} fields in {len(has)} blocks with data")

    t0 = time.perf_counter()
    X, finfo = build_features(ts, fields, cfg)
    t_feat = time.perf_counter() - t0
    t0 = time.perf_counter()
    batch = build_presto_inputs(ts, fields, args.start_month, args.end_month, cfg)
    t_adapt = time.perf_counter() - t0
    t0 = time.perf_counter()
    E = embed_batch(batch, args.weights)
    t_embed = time.perf_counter() - t0
    E.to_parquet(out / "presto_embeddings.parquet", index=False)
    say(f"features {t_feat:.1f}s, Presto inputs {t_adapt:.1f}s, embeddings {t_embed:.1f}s "
        f"for {len(fields)} fields")

    meta = fields.copy()
    meta["year"] = meta["year"].map(lambda v: str(int(v)) if np.isfinite(v) else "unknown")
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    y = meta["y"].to_numpy(int)
    Xall = pd.concat([X.reset_index(drop=True), E[EMB_COLS + ["n_months_observed"]]], axis=1)
    Xall["block_id"] = meta["block_id"].to_numpy()
    extra_cols = []
    if args.extra_embeddings:
        ext = pd.read_parquet(args.extra_embeddings)
        extra_cols = [c for c in ext.columns if c != "field_id"]
        Xall = Xall.merge(ext, on="field_id", how="left")

    has_emb = np.isfinite(Xall[EMB_COLS].to_numpy()).all(axis=1)
    has_ts = X["q_n_rows"].to_numpy() > 0
    keep = has_emb
    counts = {
        "fields": int(len(fields)),
        "no_timeseries_rows": int((~has_ts).sum()),
        "no_usable_month_for_presto": int((~has_emb).sum()),
        "physical_low_evidence": int(X["low_evidence"].sum()),
        "physical_no_harvest_detected": int(X["harv_mid_doy"].isna().sum()),
        "evaluated_all_models": int(keep.sum()),
        "months_observed_distribution": {int(k): int(v) for k, v in
                                         pd.Series(batch.n_months_observed).value_counts().sort_index().items()},
    }
    say(f"evaluated on {keep.sum()} fields; dropped {(~keep).sum()} without any usable month "
        f"(dropped from every model)")

    facs = make_comparison_factories(Xall.columns, args.seed)
    if extra_cols:
        ft = make_comparison_factories(Xall.columns, args.seed, extra_cols, tag="ft")
        facs.update({k: v for k, v in ft.items() if k.startswith("ft-")})

    folds = spatial_folds(meta["block_id"], y, args.n_splits, args.seed)
    splits = restrict([(f"fold{i}", tr, te) for i, (tr, te) in enumerate(folds)], keep)
    preds, info = [], []
    p, inf, _ = run_splits(Xall, meta, y, facs, splits, "spatial", "relative", seed=args.seed)
    preds.append(p)
    info += inf
    schemes = ["spatial"]
    for key, groups, blocks in (("loco", meta["country"], None), ("loyo", meta["year"], meta["block_id"])):
        for g, tr, te, n_drop in leave_one_group_out(groups, blocks, exclude_test_blocks=False):
            (name, tr2, te2), = restrict([(g, tr, te)], keep)
            p, inf, _ = run_splits(Xall, meta, y, facs, [(name, tr2, te2)], f"{key}:{g}", "relative",
                                   seed=args.seed)
            preds.append(p)
            info += inf
            schemes.append(f"{key}:{g}")
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(out / "predictions.parquet", index=False)

    summary = summarize_predictions(preds, n_boot=args.n_boot, seed=args.seed)
    tab = fold_table(summary, "spatial")
    tab.to_csv(out / "spatial_table.csv", index=False)

    comparisons = {}
    for head in ("logistic", "gbm"):
        pairs = [(f"both-{head}", f"phys-{head}"), (f"emb-{head}", f"phys-{head}")]
        if extra_cols:
            pairs.append((f"ft-{head}", f"phys-{head}"))
        for a, b in pairs:
            boot = compare_models(preds, a, b, "spatial", "relative", n_boot=args.n_boot,
                                  seed=args.seed)["roc_auc"]
            diffs = paired_fold_diffs(preds, a, b)
            comparisons[f"{a} vs {b}"] = {"auc_diff_pooled": boot["diff"], "auc_diff_ci95": boot["ci"],
                                          "auc_diff_per_fold": diffs, "verdict": verdict(diffs, boot)}
    other = {}
    for s in schemes[1:]:
        t = fold_table(summary, s)
        other[s] = {r["model"]: {m: r[m] for m in TABLE_METRICS} for _, r in t.iterrows()}

    result = {"args": vars(args), "weights_sha256": file_sha256(args.weights), "counts": counts,
              "timing_seconds": {"features": t_feat, "presto_inputs": t_adapt, "embeddings": t_embed,
                                 "n_fields": len(fields)},
              "folds": [{k: v for k, v in r.items() if k != "models"} for r in info if r["scheme"] == "spatial"],
              "spatial": tab.to_dict(orient="records"), "comparisons": comparisons, "other_schemes": other,
              "load_notes": notes, "log": log}
    (out / "summary.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    md = [f"# Presto vs physical features: {out.name}", "",
          f"Fields evaluated: {keep.sum()} ({counts})", "",
          "Spatial 5-fold CV grouped by block. Cell = pooled value (fold mean ± fold SD).", "",
          markdown_table(tab), "", "## Paired ROC-AUC differences", ""]
    for k, c in comparisons.items():
        md.append(f"- {k}: pooled {c['auc_diff_pooled']:+.3f}, block-bootstrap 95% CI "
                  f"[{c['auc_diff_ci95'][0]:+.3f}, {c['auc_diff_ci95'][1]:+.3f}], per fold "
                  + ", ".join(f"{v:+.3f}" for v in c["auc_diff_per_fold"].values())
                  + f" -> {c['verdict']}")
    for s, rows in other.items():
        md += ["", f"## {s}", "", "| model | " + " | ".join(TABLE_METRICS) + " |",
               "|---|" + "---:|" * len(TABLE_METRICS)]
        for mname, r in rows.items():
            md.append(f"| {mname} | " + " | ".join(_fmt(r[m]) for m in TABLE_METRICS) + " |")
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    say("\n".join(md))
    return result


if __name__ == "__main__":
    main()
