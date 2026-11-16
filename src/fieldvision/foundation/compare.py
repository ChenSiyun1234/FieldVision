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


def make_comparison_factories(columns, inputs, combos=None, seed=0):
    """name -> constructor, for every input x {logistic, LightGBM}.

    inputs  {tag: [columns]} vector inputs, e.g. {"emb": EMB_COLS, "raw": [...]}
    combos  {name: tag} inputs to concatenate with the physical features,
            e.g. {"both": "emb"} gives both-logistic and both-gbm

    The physical models and the rule are exactly the main line's
    (make_factories, variant "relative"); the new heads use the same
    classes, hyper-parameters and in-fold Platt calibration.
    """
    main = make_factories(columns, "relative", seed=seed)
    phys_logit = main["logistic"]().feature_names
    phys_gbm = main["gbm"]().feature_names

    def cal(make):
        return lambda m=make: CalibratedModel(m, seed=seed)

    out = {"majority": MajorityModel, "always-grain": main["always_grain"],
           "timing-rule": main["timing_rule"], "phys-rule": main["rule"],
           "phys-logistic": main["logistic"], "phys-gbm": main["gbm"]}
    for tag, cols in inputs.items():
        out[f"{tag}-logistic"] = cal(partial(LogisticModel, list(cols)))
        out[f"{tag}-gbm"] = cal(partial(GBMModel, list(cols), seed=seed))
    for name, tag in (combos or {}).items():
        cols = list(inputs[tag])
        out[f"{name}-logistic"] = cal(partial(LogisticModel, phys_logit + cols))
        out[f"{name}-gbm"] = cal(partial(GBMModel, phys_gbm + cols, seed=seed))
    return out


def raw_composite_table(batch):
    """The monthly composites Presto sees, flattened (S2 bands + NDVI per month), NaN when masked.

    This is the no-pretraining control: the same input, without the encoder.
    """
    from .presto_bands import PRESTO_BANDS
    s2 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
    cols, data = [], []
    for t in range(batch.x.shape[1]):
        month = (batch.month + t) % 12 + 1
        for b in s2:
            k = PRESTO_BANDS.index(b)
            cols.append(f"raw_m{month:02d}_{b}")
            data.append(np.where(batch.mask[:, t, k] == 0, batch.x[:, t, k], np.nan))
    df = pd.DataFrame(np.column_stack(data), columns=cols)
    df.insert(0, "field_id", batch.field_id)
    return df


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


TRANSFER_REF = "phys-logistic"     # the 6-feature logistic regression of the main line
TRANSFER_MODELS = ["phys-logistic", "timing-rule", "phys-rule", "phys-gbm", "emb-logistic", "emb-gbm",
                   "raw-logistic", "raw-gbm", "both-logistic", "both-gbm", "physraw-logistic",
                   "physraw-gbm", "always-grain"]


def transfer_table(preds, scheme, models=TRANSFER_MODELS, ref=TRANSFER_REF, n_boot=1000, seed=0):
    """One row per model for a held-out group (e.g. 'loyo:2022').

    auc_out   trained without that year/country
    auc_in    the same fields as scored in spatial CV (their year was in training)
    auc_loss  auc_in - auc_out: what the model loses when the group is new
    auc_diff_vs_ref  paired difference to `ref` on the held-out fields, block-bootstrap CI
    """
    from ..model.metrics import classification_metrics
    out = preds[preds["scheme"] == scheme]
    test_ids = set(out["field_id"])
    sp = preds[(preds["scheme"] == "spatial") & preds["field_id"].isin(test_ids)]
    rows = []
    for m in models:
        g = out[out["model"] == m]
        if g.empty:
            continue
        mo = classification_metrics(g["y"], g["pred"], g["p"])
        gi = sp[sp["model"] == m]
        auc_in = classification_metrics(gi["y"], gi["pred"], gi["p"])["roc_auc"] if len(gi) else np.nan
        row = {"scheme": scheme, "model": m, "n": mo["n"], "accuracy": mo["accuracy"],
               "balanced_accuracy": mo["balanced_accuracy"], "recall": mo["recall"],
               "fpr": mo["fpr"], "precision": mo["precision"], "auc_out": mo["roc_auc"],
               "auc_in": auc_in, "auc_loss": auc_in - mo["roc_auc"],
               "auc_diff_vs_ref": 0.0, "auc_diff_vs_ref_ci": [0.0, 0.0]}
        if m != ref:
            c = compare_models(preds, m, ref, scheme, "relative", n_boot=n_boot, seed=seed)
            if c is not None:
                row["auc_diff_vs_ref"] = c["roc_auc"]["diff"]
                row["auc_diff_vs_ref_ci"] = c["roc_auc"]["ci"]
        rows.append(row)
    return pd.DataFrame(rows)


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
    # controls: same architecture with random weights, and the composites without any encoder
    import torch
    from .presto_encoder import Encoder
    torch.manual_seed(args.seed)
    R = embed_batch(batch, encoder=Encoder())
    rand_cols = [c.replace("presto_", "rand_") for c in EMB_COLS]
    R = R.rename(columns=dict(zip(EMB_COLS, rand_cols)))
    raw = raw_composite_table(batch)
    raw_cols = [c for c in raw.columns if c != "field_id"]

    meta = fields.copy()
    meta["year"] = meta["year"].map(lambda v: str(int(v)) if np.isfinite(v) else "unknown")
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    y = meta["y"].to_numpy(int)
    Xall = pd.concat([X.reset_index(drop=True), E[EMB_COLS + ["n_months_observed"]],
                      R[rand_cols], raw[raw_cols]], axis=1)
    Xall["block_id"] = meta["block_id"].to_numpy()
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

    inputs = {"emb": EMB_COLS, "raw": raw_cols, "rand": rand_cols}
    combos = {"both": "emb", "physraw": "raw"}
    facs = make_comparison_factories(Xall.columns, inputs, combos, args.seed)

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
    # one country-year to another country-year, e.g. AT 2022 -> FR 2022 (same imagery source)
    cy = (meta["country"].astype(str) + meta["year"].astype(str)).to_numpy()
    country = meta["country"].astype(str).to_numpy()
    for a in sorted(set(cy)):
        for b in sorted(set(cy)):
            ca, cb = country[cy == a][0], country[cy == b][0]
            if ca == cb:
                continue
            (name, tr2, te2), = restrict([(f"{a}->{b}", np.flatnonzero(cy == a),
                                           np.flatnonzero(cy == b))], keep)
            p, inf, _ = run_splits(Xall, meta, y, facs, [(name, tr2, te2)], f"pair:{a}->{b}",
                                   "relative", seed=args.seed)
            preds.append(p)
            info += inf
            schemes.append(f"pair:{a}->{b}")
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(out / "predictions.parquet", index=False)

    summary = summarize_predictions(preds, n_boot=args.n_boot, seed=args.seed)
    tab = fold_table(summary, "spatial")
    tab.to_csv(out / "spatial_table.csv", index=False)

    comparisons = {}
    for head in ("logistic", "gbm"):
        pairs = [(f"both-{head}", f"phys-{head}"),        # the question asked
                 (f"emb-{head}", f"phys-{head}"),
                 (f"emb-{head}", f"raw-{head}"),          # does pretraining beat the raw composites?
                 (f"emb-{head}", f"rand-{head}"),         # ... or the same network untrained?
                 (f"both-{head}", f"physraw-{head}")]     # extra info vs pretrained summary of it
        for a, b in pairs:
            boot = compare_models(preds, a, b, "spatial", "relative", n_boot=args.n_boot,
                                  seed=args.seed)["roc_auc"]
            diffs = paired_fold_diffs(preds, a, b)
            comparisons[f"{a} vs {b}"] = {"auc_diff_pooled": boot["diff"], "auc_diff_ci95": boot["ci"],
                                          "auc_diff_per_fold": diffs, "verdict": verdict(diffs, boot)}
    other = {}
    transfer = []
    for s in schemes[1:]:
        t = fold_table(summary, s)
        other[s] = {r["model"]: {m: r[m] for m in TABLE_METRICS} for _, r in t.iterrows()}
        transfer.append(transfer_table(preds, s, n_boot=args.n_boot, seed=args.seed))
    transfer = pd.concat(transfer, ignore_index=True) if transfer else pd.DataFrame()
    if len(transfer):
        transfer.to_csv(out / "transfer_table.csv", index=False)

    result = {"args": vars(args), "weights_sha256": file_sha256(args.weights), "counts": counts,
              "timing_seconds": {"features": t_feat, "presto_inputs": t_adapt, "embeddings": t_embed,
                                 "n_fields": len(fields)},
              "folds": [{k: v for k, v in r.items() if k != "models"} for r in info if r["scheme"] == "spatial"],
              "spatial": tab.to_dict(orient="records"), "comparisons": comparisons, "other_schemes": other,
              "transfer": transfer.to_dict(orient="records"),
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
    if len(transfer):
        md += ["", "## Transfer (held-out year / country)", "",
               "AUC out = trained on the other groups; AUC in = the same fields scored by the spatial "
               "CV models (trained on all groups, other blocks); diff vs phys-logistic is paired on the "
               "held-out fields with a block-bootstrap 95% CI.", "",
               "| held out | model | n | accuracy | recall | fpr | AUC out | AUC in | loss | "
               "AUC diff vs phys-logistic [95% CI] |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for _, r in transfer.iterrows():
            ci = r["auc_diff_vs_ref_ci"]
            diff = ("ref" if r["model"] == TRANSFER_REF else
                    f"{r['auc_diff_vs_ref']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
            md.append(f"| {r['scheme']} | {r['model']} | {r['n']} | {_fmt(r['accuracy'])} | "
                      f"{_fmt(r['recall'])} | {_fmt(r['fpr'])} | {_fmt(r['auc_out'])} | "
                      f"{_fmt(r['auc_in'])} | {_fmt(r['auc_loss'])} | {diff} |")
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    say("\n".join(md))
    return result


if __name__ == "__main__":
    main()
