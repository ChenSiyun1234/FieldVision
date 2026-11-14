"""Train the harvest-type models and evaluate them honestly.

    python -m fieldvision.model.train --timeseries <dir-or-parquet ...> \
        --fields <geojson ...> --out <run dir> [--holdout-blocks 'AT-E46*' ...]

What it does, in order:
 1. features for every field (labelled or not)
 2. spatial cross-validation grouped by block_id, for each feature variant
 3. leave-one-country-out and leave-one-year-out
 4. ablations of feature groups we want to test as hypotheses (residue, regreening)
 5. permutation importance on held-out folds
 6. a check of whether features shift with the imagery collection within a class
 7. final models fit on all labelled, non-holdout fields
 8. the region holdout, evaluated once with the final models
 9. response time, figures, metrics.json and REPORT.md

Positive class = silage. The majority-class baseline is evaluated in every
split next to the real models.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig, feature_group, signal_features
from . import plots
from .artifacts import (fingerprint, ledger_check_and_record, ledger_path, refuse_evaluation_fields,
                        save_run,
                        write_json)
from .explain import reference_values
from .harness import (collection_shift, compare_models, coverage_table, field_collection,
                      label_disagreement, prevalence_table,
                      importance_groups,
                      response_time, run_splits, summarize_importance, summarize_predictions)
from .metrics import (block_bootstrap_ci, calibration_summary, classification_metrics,
                      precision_at_prevalence)
from .models import MODEL_ORDER, make_factories
from .report import SYNTHETIC_BANNER, confusion_line, fmt, kv_table, metrics_table
from .splits import holdout_mask, leave_one_group_out, spatial_folds

ABLATIONS = {"residue": "no_residue", "regreening": "no_regreening"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train and evaluate silage/grain models.")
    ap.add_argument("--timeseries", nargs="+", required=True, help="Parquet files or directories")
    ap.add_argument("--fields", nargs="+", required=True, help="GeoJSON/GeoPackage files or dirs")
    ap.add_argument("--out", required=True, help="run directory (created)")
    ap.add_argument("--primary", default="gbm", choices=["timing_rule", "rule", "logistic", "gbm"],
                    help="model used by predict/evaluate by default (fixed before looking at results)")
    ap.add_argument("--variant", default="relative", choices=["relative", "calendar", "all"],
                    help="feature variant for the final models")
    ap.add_argument("--compare-variants", nargs="*", default=["calendar"],
                    help="other variants to run through spatial CV for comparison")
    ap.add_argument("--holdout-blocks", nargs="*", default=[],
                    help="block_id patterns (fnmatch) held out and evaluated once at the end")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loyo-exclude-test-blocks", action="store_true",
                    help="in leave-one-year-out, also drop training fields in the test year's blocks")
    ap.add_argument("--no-importance", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--min-clear-fraction", type=float, default=None)
    ap.add_argument("--max-harvest-interval", type=float, default=None,
                    help="low-evidence limit on the harvest interval width (days)")
    ap.add_argument("--peak-window", nargs=2, type=float, default=None, metavar=("START", "END"),
                    help="day-of-year window for the corn peak (default 160 280)")
    ap.add_argument("--prevalences", nargs="*", type=float, default=[0.07, 0.85],
                    help="silage shares to re-weight held-out metrics to (US average ~0.07; "
                         "a dairy-heavy county ~0.85)")
    ap.add_argument("--allow-repeat", action="store_true",
                    help="allow evaluating the same holdout twice with the same run id")
    return ap.parse_args(argv)


def _cfg(args):
    over = {}
    if args.min_clear_fraction is not None:
        over["min_clear_fraction"] = args.min_clear_fraction
    if args.max_harvest_interval is not None:
        over["low_evidence_max_interval_days"] = args.max_harvest_interval
    if args.peak_window is not None:
        over["peak_window"] = tuple(args.peak_window)
    return FeatureConfig(**over)


def _year_str(v):
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return "unknown"


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    cfg = _cfg(args)
    # ------------------------------------------------------------------ data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fields = load_fields(args.fields, keep_columns=["split"])
        refuse_evaluation_fields(fields)
        t0 = time.perf_counter()
        ts, ts_notes = load_timeseries(args.timeseries, field_ids=fields["field_id"])
        load_s = time.perf_counter() - t0
    load_warnings = sorted({str(w.message) for w in caught})
    say(f"{len(fields)} fields, {len(ts)} time-series rows "
        f"({ts['field_id'].nunique() if len(ts) else 0} fields with rows), loaded in {load_s:.1f}s")

    X, finfo = build_features(ts, fields, cfg)
    say(f"features: {X.shape[1]} columns, {finfo['seconds_per_1000_fields']:.2f} s per 1000 fields")
    meta = fields.reset_index(drop=True).copy()
    meta["year"] = meta["year"].map(_year_str)
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    coll = field_collection(ts)
    meta["collection"] = meta["field_id"].map(coll).fillna("none").to_numpy()
    y_all = meta["y"].to_numpy(dtype=float)
    labelled = np.isfinite(y_all)
    synthetic = bool((meta["source"].astype(str) == "synthetic").all())

    hold = holdout_mask(meta["block_id"], args.holdout_blocks) & labelled
    dev = labelled & ~hold
    if args.holdout_blocks and not hold.any():
        raise SystemExit(f"no labelled field matches the holdout patterns {args.holdout_blocks}")
    say(f"labelled {int(labelled.sum())} (silage {int(np.nansum(y_all[labelled]))}), "
        f"development {int(dev.sum())}, holdout {int(hold.sum())}")
    if dev.sum() < 20:
        raise SystemExit("fewer than 20 labelled development fields; nothing to train on")

    Xd = X[dev].reset_index(drop=True)
    md = meta[dev].reset_index(drop=True)
    Xd["block_id"] = md["block_id"].to_numpy()     # only for grouping the inner calibration CV
    yd = y_all[dev].astype(int)

    variants = [args.variant] + [v for v in args.compare_variants if v != args.variant]
    factories = {v: make_factories(X.columns, v, seed=args.seed) for v in variants}
    prim_fac = factories[args.variant]
    # ------------------------------------------------------------ spatial CV
    folds = spatial_folds(md["block_id"], yd, args.n_splits, args.seed)
    splits = [(f"fold{i}", tr, te) for i, (tr, te) in enumerate(folds)]
    say(f"spatial CV: {len(splits)} folds over {md['block_id'].nunique()} blocks")
    preds, fold_info, imp_records = [], [], []
    imp_groups = None
    if not args.no_importance:
        probe = {n: f() for n, f in prim_fac.items()}
        imp_groups = {"gbm": importance_groups(probe["gbm"].feature_names, individual=True),
                      "logistic": importance_groups(probe["logistic"].feature_names, individual=True)}
    for v in variants:
        p, info, imp = run_splits(Xd, md, yd, factories[v], splits, "spatial", v,
                                  importance=imp_groups if v == args.variant else None,
                                  seed=args.seed)
        preds.append(p)
        fold_info += info
        imp_records += imp

    # ---------------------------------------------------- feature ablations
    ablation_names = []
    if not args.no_ablation:
        gbm_cols = prim_fac["gbm"]().feature_names
        logit_cols = prim_fac["logistic"]().feature_names
        for group, tag in ABLATIONS.items():
            fac = None
            renamed = {}
            for name, cols in (("gbm", gbm_cols), ("logistic", logit_cols)):
                drop = [c for c in cols if feature_group(c) == group]
                if not drop:
                    continue
                fac = make_factories(X.columns, args.variant, exclude=drop, seed=args.seed)
                renamed[f"{name}-{tag}"] = fac[name]
                ablation_names.append((name, group, tag))
            if not renamed:
                continue
            p, info, _ = run_splits(Xd, md, yd, renamed, splits, "spatial", args.variant)
            preds.append(p)
            fold_info += info

    # ------------------------------------- leave one country / year out
    loco = leave_one_group_out(md["country"])
    for g, tr, te, _ in loco:
        p, info, _ = run_splits(Xd, md, yd, prim_fac, [(g, tr, te)], f"loco:{g}", args.variant)
        preds.append(p)
        fold_info += info
    loyo = leave_one_group_out(md["year"], md["block_id"], args.loyo_exclude_test_blocks)
    for g, tr, te, n_drop in loyo:
        p, info, _ = run_splits(Xd, md, yd, prim_fac, [(g, tr, te)], f"loyo:{g}", args.variant)
        for rec in info:
            rec["n_train_dropped_same_blocks"] = n_drop
        preds.append(p)
        fold_info += info
    say(f"leave-one-country-out: {[str(g) for g, *_ in loco] or 'only one country'}; "
        f"leave-one-year-out: {[str(g) for g, *_ in loyo] or 'only one year'}")

    preds = pd.concat(preds, ignore_index=True)
    summary = summarize_predictions(preds, n_boot=args.n_boot, seed=args.seed)

    comparisons = {}
    v = args.variant
    for a, b in (("gbm", "timing_rule"), ("logistic", "timing_rule"), ("rule", "timing_rule"),
                 ("gbm", "logistic"), ("gbm", "rule"), ("logistic", "rule"),
                 ("timing_rule", "majority"), ("gbm", "majority")):
        comparisons[f"{a} - {b} ({v})"] = compare_models(preds, a, b, "spatial", v,
                                                         n_boot=args.n_boot, seed=args.seed)
    # the organisers' test: does the model beat the timing rule on held-out geography / years?
    for scheme in [k for k in preds["scheme"].unique() if k.startswith(("loco:", "loyo:"))]:
        comparisons[f"gbm - timing_rule ({scheme})"] = compare_models(
            preds, "gbm", "timing_rule", scheme, v, n_boot=args.n_boot, seed=args.seed)
    for other in variants[1:]:
        for m in ("gbm", "logistic", "rule"):
            comparisons[f"{m}: {v} - {other}"] = compare_models(
                preds, m, m, "spatial", v, other, n_boot=args.n_boot, seed=args.seed)
    for m, group, tag in ablation_names:
        res = compare_models(preds, m, f"{m}-{tag}", "spatial", v,
                             n_boot=args.n_boot, seed=args.seed)
        if res is not None:
            comparisons[f"{m}: with {group} - without"] = res

    importance = summarize_importance(imp_records)
    sp_prim = preds[(preds.scheme == "spatial") & (preds.variant == v)
                    & (preds.model == args.primary)]
    coverage = coverage_table(sp_prim, Xd, cfg)
    sp_all = preds[(preds.scheme == "spatial") & (preds.variant == v)
                   & preds.model.isin(MODEL_ORDER)]
    calib = {m: calibration_summary(g["y"], g["p"]) for m, g in sp_all.groupby("model", sort=False)
             if m != "majority"}
    prev_tab = prevalence_table(sp_all, args.prevalences, float(yd.mean()), args.n_boot, args.seed)
    sp_prim_cy = sp_prim.assign(country_year=sp_prim["country"] + " " + sp_prim["year"])
    disagree = label_disagreement(sp_prim_cy, "country_year")

    # ------------------------------------------------- collection shift
    shift_cols = [c for c in signal_features(X.columns, "all")] + \
        ["q_n_usable_season", "q_max_gap_days", "q_mean_clear_fraction"]
    shift = collection_shift(Xd, md, yd, shift_cols, md["collection"])
    coll_year = pd.crosstab(md["collection"], [md["country"], md["year"]])
    # cleanest version: only blocks seen under more than one collection
    n_coll = md.groupby("block_id")["collection"].nunique()
    multi = md["block_id"].isin(n_coll[n_coll > 1].index).to_numpy()
    shift_matched = None
    if multi.sum() >= 40:
        shift_matched = collection_shift(Xd[multi], md[multi], yd[multi], shift_cols,
                                         md.loc[multi, "collection"])

    # ------------------------------------------------------- final models
    final = {}
    for name in MODEL_ORDER:
        final[name] = prim_fac[name]().fit(Xd, yd)
    run_id = hashlib.sha256(f"{out.resolve()}|{dt.datetime.now().isoformat()}".encode()).hexdigest()[:12]

    # ------------------------------------------------------ response time
    n_rt = min(len(fields), 2000)
    rt_ids = fields["field_id"].iloc[:n_rt]
    rt_fields = fields[fields["field_id"].isin(rt_ids)]
    rt_ts = ts[ts["field_id"].isin(rt_ids)]
    rtime = {}
    for name in ("timing_rule", "rule", "logistic", "gbm"):
        rtime[name] = response_time(lambda: build_features(rt_ts, rt_fields, cfg)[0],
                                    lambda F, m=final[name]: m.predict_proba(F), n_rt)
    rtime["note"] = ("features + prediction on already-loaded time series, single process; "
                     f"loading Parquet took {load_s:.1f}s for {len(fields)} fields")

    # ------------------------------------------------------------ holdout
    holdout = None
    if hold.any():
        Xh, mh, yh = X[hold].reset_index(drop=True), meta[hold].reset_index(drop=True), \
            y_all[hold].astype(int)
        fp = fingerprint(mh["field_id"])
        earlier = ledger_check_and_record(ledger_path(out), "holdout", fp, run_id, out,
                                          args.allow_repeat)
        hp = []
        for name, model in final.items():
            p = model.predict_proba(Xh)
            part = mh[["field_id", "block_id", "country", "year", "low_evidence"]].copy()
            part["scheme"], part["variant"], part["fold"], part["model"] = "holdout", v, "holdout", name
            part["y"], part["p"], part["threshold"] = yh, p, model.threshold
            part["pred"] = (p >= model.threshold).astype(int)
            hp.append(part)
        hp = pd.concat(hp, ignore_index=True)
        holdout = {"patterns": args.holdout_blocks, "fingerprint": fp,
                   "n_fields": int(hold.sum()), "n_blocks": int(mh["block_id"].nunique()),
                   "earlier_uses_of_this_holdout": len(earlier),
                   "summary": summarize_predictions(hp, n_boot=args.n_boot, seed=args.seed)
                   .get("holdout", {}).get(v, {})}
        preds = pd.concat([preds, hp], ignore_index=True)
        say(f"holdout evaluated once: {int(hold.sum())} fields; earlier uses of this holdout: "
            f"{len(earlier)}")

    # ------------------------------------------------------------ outputs
    feats_out = X.copy()
    for c in ("label", "y", "country", "year", "block_id", "source", "area_ha", "collection"):
        feats_out[c] = meta[c].to_numpy()
    feats_out["split"] = np.where(hold, "holdout", np.where(dev, "development", "unlabelled"))
    feats_out.to_parquet(out / "features.parquet", index=False)
    preds.to_parquet(out / "cv_predictions.parquet", index=False)
    importance.to_csv(out / "permutation_importance.csv", index=False)
    if not shift.empty:
        shift.to_csv(out / "collection_shift.csv", index=False)
    if shift_matched is not None and not shift_matched.empty:
        shift_matched.to_csv(out / "collection_shift_matched.csv", index=False)

    plots.reliability(calib, out / "figures" / "calibration.png",
                      title="calibration on held-out spatial folds" +
                      (" (SYNTHETIC)" if synthetic else ""))
    figs = _figures(out / "figures", ts, X, meta, Xd, yd, preds, importance, cfg, v, args.seed,
                    synthetic)

    meta_run = {
        "run_id": run_id, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "primary": args.primary, "variant": v, "feature_config": cfg.to_dict(),
        "threshold": {n: m.threshold for n, m in final.items()},
        "models": {n: m.describe() for n, m in final.items()},
        "feature_names": {n: m.feature_names for n, m in final.items()},
        "training": {"n_fields": int(dev.sum()), "n_silage": int(yd.sum()),
                     "countries": sorted(md["country"].unique()),
                     "years": sorted(md["year"].unique()),
                     "collections": sorted(md["collection"].unique()),
                     "fields_files": sorted(meta["fields_file"].unique())},
        "holdout_patterns": args.holdout_blocks, "synthetic": synthetic,
        "train_prevalence": float(yd.mean()),
        "evidence_reference": reference_values(Xd, yd),
        "command": " ".join(sys.argv),
    }
    save_run(out, final, meta_run)

    metrics = {
        "synthetic_data_software_check_only": synthetic,
        "run_id": run_id, "primary": args.primary, "variant": v,
        "data": _data_summary(meta, labelled, hold, finfo, ts_notes, load_warnings),
        "summary": summary, "comparisons": comparisons, "folds": fold_info,
        "holdout": holdout, "response_time": rtime,
        "low_evidence_reasons": X.loc[labelled, "low_evidence_reason"].replace("", "ok")
        .value_counts().to_dict(),
        "collection_by_country_year": {str(k): v_ for k, v_ in coll_year.to_dict().items()},
        "final_models": meta_run["models"],
        "coverage_vs_interval_limit": coverage.to_dict("records"),
        "calibration": calib, "train_prevalence": float(yd.mean()),
        "prevalence_table": prev_tab.to_dict("records"),
        "label_disagreement": disagree.to_dict("records"),
        "collection_shift_matched": None if shift_matched is None else {
            "n_blocks": int(md.loc[multi, "block_id"].nunique()), "n_fields": int(multi.sum()),
            "n_large": int(shift_matched.loc[shift_matched["abs_smd"] >= 0.5, "feature"].nunique())
            if not shift_matched.empty else 0,
            "top": shift_matched.head(12).to_dict("records") if not shift_matched.empty else []},
        "seconds_total": time.perf_counter() - t_start,
    }
    write_json(out / "metrics.json", metrics)
    _write_report(out, metrics, importance, shift, coll_year, figs, args, synthetic)
    say(f"done in {time.perf_counter() - t_start:.0f}s -> {out}")
    return metrics


def _data_summary(meta, labelled, hold, finfo, ts_notes, load_warnings):
    lab = meta[labelled]
    groups = (lab.groupby(["country", "year"])
              .agg(n=("y", "size"), silage=("y", "sum"), blocks=("block_id", "nunique"),
                   low_evidence=("low_evidence", "mean"))
              .reset_index())
    return {"n_fields": int(len(meta)), "n_labelled": int(labelled.sum()),
            "n_holdout": int(hold.sum()), "by_country_year": groups.to_dict("records"),
            "features": finfo, "timeseries_notes": ts_notes, "load_warnings": load_warnings}


def _figures(fdir, ts, X, meta, Xd, yd, preds, importance, cfg, variant, seed, synthetic):
    tag = " (SYNTHETIC - software check only)" if synthetic else ""
    figs = {}
    path = fdir / "class_distributions.png"
    plots.class_distributions(Xd, yd, path, title="development fields by declared class" + tag)
    figs["class_distributions"] = path.name
    sp = preds[(preds.scheme == "spatial") & (preds.variant == variant)
               & preds.model.isin(MODEL_ORDER)]
    if not sp.empty:
        path = fdir / "roc_pr_spatial_cv.png"
        plots.roc_pr_curves(sp, path, title="spatial CV, held-out folds" + tag)
        figs["roc_pr"] = path.name
        path = fdir / "score_histograms.png"
        plots.score_histograms(sp, path, title="spatial CV scores by declared class" + tag)
        figs["scores"] = path.name
    if importance is not None and not importance.empty:
        path = fdir / "permutation_importance.png"
        plots.importance_bars(importance, path, title="permutation importance, held-out folds" + tag)
        figs["importance"] = path.name
    # example curves: three of each class, plus the gbm's worst mistakes
    rng = np.random.default_rng(seed)
    md = meta[np.isfinite(meta["y"].to_numpy(float))]
    ids = []
    for val in (1.0, 0.0):
        pool = md.loc[md["y"] == val, "field_id"].to_numpy()
        ids += list(rng.choice(pool, size=min(3, len(pool)), replace=False))
    if ids:
        path = fdir / "example_fields.png"
        gb = sp[sp.model == "gbm"] if not sp.empty else None
        plots.example_curves(ts, X, meta, ids, path, cfg, preds=gb,
                             title="random examples (top: silage, bottom: grain)" + tag)
        figs["examples"] = path.name
    if not sp.empty:
        gb = sp[sp.model == "gbm"].copy()
        gb["err"] = np.abs(gb["y"] - gb["p"])
        worst = gb.sort_values("err", ascending=False).head(6)["field_id"].tolist()
        if worst:
            path = fdir / "worst_errors.png"
            plots.example_curves(ts, X, meta, worst, path, cfg, preds=gb,
                                 title="largest gbm errors in spatial CV" + tag)
            figs["worst_errors"] = path.name
    return figs


def _write_report(out, metrics, importance, shift, coll_year, figs, args, synthetic):
    s = metrics["summary"]
    v = metrics["variant"]
    L = []
    L.append(f"# Training run {metrics['run_id']}\n")
    if synthetic:
        L.append(SYNTHETIC_BANNER)
    d = metrics["data"]
    L.append(f"Created {dt.datetime.now():%Y-%m-%d %H:%M}. Positive class = silage. Primary model "
             f"`{metrics['primary']}`, feature variant `{v}`. Every table has the majority-class "
             "baseline (`majority`: predicts the training majority for every field). Intervals are "
             "95% block-bootstrap percentiles (whole 10 km blocks resampled).\n")
    L.append("## Validation design\n")
    L.append("- No random field splits. Spatial cross-validation keeps whole 10 km blocks "
             "together (neighbours share weather, soil and satellite passes).\n"
             "- Leave-one-country-out and leave-one-year-out train on everything else and test on "
             "the held-out country or year; these are the main evidence that the model transfers "
             "to a place or season it has not seen, which is what applying it to the US is.\n"
             "- Every learned quantity (rule thresholds, imputation, scaling, trees, probability "
             "calibration) is fit inside the training part of each split. No feature selection "
             "step; hyper-parameters fixed in advance.\n"
             "- Baselines in every table: `majority` and `timing_rule` (silage if the sustained "
             "NDVI decline comes before a day-of-year threshold), the transparent rule the "
             "organisers suggest. The other models have to beat `timing_rule` on held-out "
             "geography; see the paired comparisons.\n"
             "- A region holdout (if given) and the external US set are scored once, with a ledger.\n"
             "- Precision and recall are reported for both classes; positive class = silage.\n")
    L.append("## Data\n")
    L.append(f"{d['n_fields']} fields, {d['n_labelled']} labelled, {d['n_holdout']} in the holdout.\n")
    L.append(kv_table([(r["country"], r["year"], r["n"], int(r["silage"]),
                        fmt(r["silage"] / r["n"]), r["blocks"], fmt(r["low_evidence"]))
                       for r in d["by_country_year"]],
                      ["country", "year", "labelled", "silage", "silage share", "blocks",
                       "low-evidence share"]))
    L.append("\nThe class balance comes from how the label fetcher sampled parcels, not from the "
             "landscape. Precision depends on it; recall and FPR do not.\n")
    for n in d["timeseries_notes"] + d["load_warnings"]:
        L.append(f"- note: {n}")
    L.append("")

    sp = s.get("spatial", {}).get(v, {})
    L.append("## Spatial cross-validation (grouped by block_id)\n")
    L.append("All development fields:\n")
    L.append(metrics_table(sp, "all", MODEL_ORDER))
    L.append("\nConfident subset only (low-evidence fields removed; coverage = share kept):\n")
    L.append(metrics_table(sp, "confident", MODEL_ORDER,
                           extra_cols=[("coverage", lambda m: m.get("coverage"))]))
    prim = sp.get(metrics["primary"])
    if prim:
        m = prim["all"]
        L.append(f"\n`{metrics['primary']}` confusion (all fields): {confusion_line(m)}.")
        rows = []
        for prev in (0.1, 0.25, 0.5):
            rows.append((f"{prev:.0%}", fmt(precision_at_prevalence(m["recall"], m["fpr"], prev))))
        L.append("Precision the same recall and FPR would give at other silage shares: " +
                 ", ".join(f"{a}: {b}" for a, b in rows) + ".\n")
        pf = prim["per_fold"]
        L.append("Per fold (" + metrics["primary"] + "): " + "; ".join(
            f"{k} P {fmt(x['precision'], 2)} R {fmt(x['recall'], 2)} FPR {fmt(x['fpr'], 2)} "
            f"AUC {fmt(x['roc_auc'], 2)} (n={x['n']})" for k, x in pf.items()) + "\n")
        for col in ("per_country", "per_year"):
            if col in prim:
                L.append(f"Pooled CV by {col[4:]} ({metrics['primary']}): " + "; ".join(
                    f"{k} P {fmt(x['precision'], 2)} R {fmt(x['recall'], 2)} "
                    f"FPR {fmt(x['fpr'], 2)} (n={x['n']})" for k, x in prim[col].items()) + "\n")

    L.append("### At realistic silage shares\n")
    L.append(f"The training labels are balanced by design ({metrics['train_prevalence']:.0%} silage); "
             "about 6-8 % of US corn is silage, so 'always grain' is right 93 % of the time and "
             "precision at 50/50 overstates what a user would see. The held-out spatial-CV "
             "predictions are re-weighted so silage makes up the stated share. Recall and FPR "
             "do not change; precision, F1, accuracy and PR-AUC do. 'prior-shifted' moves the "
             "threshold by Bayes' rule to that share (fewer silage calls, higher precision, "
             "lower recall).\n")
    pt = metrics["prevalence_table"]
    head = ["silage share", "model", "operating point", "threshold", "precision", "recall", "FPR",
            "F1", "accuracy", "PR-AUC"]
    rows = []
    for r in pt:
        def cell(k, r=r):
            v_ = fmt(r[k])
            ci = r.get(k + "_ci")
            if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(
                    c is not None and np.isfinite(c) for c in ci):
                v_ += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
            return v_
        rows.append((f"{r['prevalence']:.0%}", r["model"], r["operating_point"], fmt(r["threshold"]),
                     cell("precision"), cell("recall"), cell("fpr"), cell("f1"), cell("accuracy"),
                     cell("pr_auc")))
    L.append(kv_table(rows, head))
    L.append("")
    L.append("### Calibration on held-out folds\n")
    L.append("Probabilities are Platt-calibrated inside each training fold (inner CV grouped by "
             "block), so they mean 'share of silage among similar fields at the training class "
             "balance'. Use `predict --prevalence` to move them to another share.\n")
    L.append(kv_table([(n, fmt(c["brier"]), fmt(c["ece"]),
                        ", ".join(f"{b['mean_p']:.2f}->{b['share_silage']:.2f} (n={b['n']})"
                                  for b in c["reliability"]))
                       for n, c in metrics["calibration"].items()],
                      ["model", "Brier", "ECE", "mean predicted -> observed silage share per bin"]))
    L.append("\nFigure: `figures/calibration.png`.\n")
    L.append("### Label check by country and year\n")
    L.append(f"Share of held-out fields where `{metrics['primary']}` strongly contradicts the "
             "declared label (p >= 0.9 for a declared grain field, p <= 0.1 for declared "
             "silage). Austrian labels are the end-of-year registry revision; Dutch labels are "
             "spring declarations, so late switches between silage and grain show up there as "
             "fields whose curve looks like the other class. A higher strong-disagreement share "
             "in NL than in AT, with similar overall error elsewhere, is evidence of label noise "
             "rather than of model error. It is not proof: some of these are real model mistakes.\n")
    L.append(kv_table([(r["country_year"], r["n"], fmt(r["error_rate"]),
                        fmt(r["grain_looks_silage"]), fmt(r["silage_looks_grain"]),
                        fmt(r["share_of_errors_that_are_strong"]))
                       for r in metrics["label_disagreement"]],
                      ["country, year", "n", "error rate", "declared grain, p>=0.9",
                       "declared silage, p<=0.1", "share of errors that are strong"]))
    L.append("")

    for other in [k for k in s.get("spatial", {}) if k != v]:
        L.append(f"### Feature variant `{other}` (spatial CV, all fields)\n")
        L.append("`relative` uses days since the field's own peak and fractions of its own "
                 "amplitude; `calendar` uses day of year and calendar half-months. The calendar "
                 "variant is expected to transfer worse to a different climate.\n")
        L.append(metrics_table(s["spatial"][other], "all", MODEL_ORDER))
        L.append("")

    for prefix, title, note in (
            ("loco:", "Leave one country out",
             "Train on the other countries, test on the one named. Closest thing we have to "
             "applying the model somewhere it has never seen, which is what the US is."),
            ("loyo:", "Leave one year out",
             "Train on the other years, test on the one named. " +
             ("Training fields in the test year's blocks were dropped."
              if args.loyo_exclude_test_blocks else
              "The same places may appear in training in other years, so this measures a new "
              "season, not a new region."))):
        keys = [k for k in s if k.startswith(prefix)]
        if not keys:
            L.append(f"## {title}\n\nNot run: the labelled data has only one "
                     f"{'country' if prefix == 'loco:' else 'year'}.\n")
            continue
        L.append(f"## {title}\n\n{note}\n")
        for k in keys:
            L.append(f"**Test = {k.split(':', 1)[1]}**\n")
            L.append(metrics_table(s[k][v], "all", MODEL_ORDER))
            L.append("")

    L.append("## Paired comparisons (spatial CV, same fields, block bootstrap)\n")
    rows = []
    for name, res in metrics["comparisons"].items():
        if not res:
            continue
        for key in ("roc_auc", "pr_auc", "recall", "fpr"):
            r = res[key]
            rows.append((name, key, fmt(r["diff"]), f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]"))
    L.append(kv_table(rows, ["comparison", "metric", "difference", "95% CI"]))
    L.append("\nAn interval that contains 0 means the data cannot tell the two apart.\n")

    if importance is not None and not importance.empty:
        L.append("## Permutation importance (held-out folds, drop in ROC-AUC)\n")
        for model in importance["model"].unique():
            g = importance[importance["model"] == model]
            grp = g[g.feature.str.startswith("group:")].head(10)
            ind = g[~g.feature.str.startswith("group:")].head(10)
            L.append(f"**{model}**, feature groups (shuffled jointly): " + ", ".join(
                f"{r.feature[6:]} {r.auc_drop_mean:.3f}" for r in grp.itertuples()))
            L.append(f"\n**{model}**, single features: " + ", ".join(
                f"{r.feature} {r.auc_drop_mean:.3f}" for r in ind.itertuples()) + "\n")
        L.append("Correlated features share credit, so a single feature can look unimportant "
                 "while its group matters.\n")

    L.append("## Does a feature shift with the imagery collection?\n")
    if shift is None or shift.empty:
        L.append("Not testable: the development data has a single collection per class "
                 "(or fewer than 20 fields in the other one).\n")
    else:
        big = shift[shift["abs_smd"] >= 0.5]
        L.append(f"Within the same declared class, {big['feature'].nunique()} features differ "
                 "by at least half a standard deviation between collections. Collection is "
                 "confounded with year and country (table below), so this flags a possible "
                 "processing effect; it does not prove one. Full table: `collection_shift.csv`.\n")
        L.append(kv_table([(r["class"], f"{r['collection_a']} ({r['years_a']})",
                            f"{r['collection_b']} ({r['years_b']})", r["feature"],
                            fmt(r["mean_a"]), fmt(r["mean_b"]), fmt(r["smd"], 2),
                            fmt(r["ks_stat"], 2)) for r in shift.head(12).to_dict("records")],
                          ["class", "collection a (years)", "collection b (years)", "feature",
                           "mean a", "mean b", "SMD", "KS"]))
    ms = metrics.get("collection_shift_matched")
    if ms:
        L.append("\n### Same blocks, different collection\n")
        L.append(f"Only the {ms['n_blocks']} blocks that have fields under more than one collection "
                 f"({ms['n_fields']} fields; e.g. AT 2022 vs AT 2023 on the same places). Place and "
                 "label source are fixed; year and imagery source still change together, so a "
                 "shift here is source or season, not geography. Full table: "
                 "`collection_shift_matched.csv`.\n")
        rows = ms["top"]
        if rows:
            L.append(kv_table([(r["class"], f"{r['collection_a']} ({r['years_a']})",
                                f"{r['collection_b']} ({r['years_b']})", r["feature"],
                                fmt(r["mean_a"]), fmt(r["mean_b"]), fmt(r["smd"], 2),
                                fmt(r["ks_stat"], 2)) for r in rows],
                              ["class", "collection a (years)", "collection b (years)", "feature",
                               "mean a", "mean b", "SMD", "KS"]))
            L.append(f"\n{ms['n_large']} features with |SMD| >= 0.5 within a class.")
        else:
            L.append("Too few fields per class and collection to compare.")
    if coll_year is not None and not coll_year.empty:
        L.append("\nFields per collection, country and year:\n")
        L.append(coll_year.to_markdown() if hasattr(coll_year, "to_markdown") and _has_tabulate()
                 else "```\n" + coll_year.to_string() + "\n```")
    L.append("")

    L.append("## Low-evidence fields\n")
    L.append("Flagged from data quality only (never from labels or scores): " + ", ".join(
        f"{k}: {n}" for k, n in metrics["low_evidence_reasons"].items()) + "\n")
    lim = args.max_harvest_interval or FeatureConfig().low_evidence_max_interval_days
    L.append(f"How the `{metrics['primary']}` spatial-CV numbers change with the harvest-interval "
             f"limit used for abstention (other criteria unchanged; this run uses {lim:g} days). "
             "Choose the limit here, on development data, never on a holdout or external set:\n")
    L.append(kv_table([(str(r["max_interval_days"]), fmt(r["coverage"]), r["n"], fmt(r["precision"]),
                        fmt(r["recall"]), fmt(r["fpr"]), fmt(r["roc_auc"]))
                       for r in metrics["coverage_vs_interval_limit"]],
                      ["max interval (days)", "coverage", "n", "precision", "recall", "FPR",
                       "ROC-AUC"]))
    L.append("")

    h = metrics.get("holdout")
    if h:
        L.append("## Region holdout (evaluated once, final models)\n")
        L.append(f"Blocks matching {h['patterns']}: {h['n_fields']} fields in {h['n_blocks']} "
                 f"blocks. Earlier evaluations of this same holdout (any run): "
                 f"{h['earlier_uses_of_this_holdout']}.\n")
        L.append(metrics_table(h["summary"], "all", MODEL_ORDER))
        L.append("\nConfident subset:\n")
        L.append(metrics_table(h["summary"], "confident", MODEL_ORDER,
                               extra_cols=[("coverage", lambda m: m.get("coverage"))]))
        L.append("")

    L.append("## Response time\n")
    rt = metrics["response_time"]
    L.append(kv_table([(n, fmt(r["features_seconds_per_1000"], 2), fmt(r["predict_seconds_per_1000"], 3),
                        fmt(r["total_seconds_per_1000"], 2)) for n, r in rt.items() if n != "note"],
                      ["model", "features s/1000 fields", "predict s/1000 fields", "total s/1000"]))
    L.append(f"\n{rt['note']}.\n")

    L.append("## Final models (fit on all development fields)\n")
    for n, desc in metrics["final_models"].items():
        L.append(f"- `{n}`: {desc}")
    L.append("\n## Figures\n")
    for k, f in figs.items():
        L.append(f"- `figures/{f}`")
    L.append("")
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
