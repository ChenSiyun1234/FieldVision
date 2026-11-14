"""Zero-shot evaluation of a trained run on an unseen labelled field set.

    python -m fieldvision.model.evaluate --model <run dir> --timeseries <...> \
        --fields <...> --out <dir> [--rename field_id=FID label=harvest] [--label-map ...]

Nothing is fit on the evaluation set: not the model, not the threshold, not
the feature settings, not the low-evidence rule. Everything comes from the
run directory. The set is recorded in a ledger and a second evaluation with
the same run is refused unless --allow-repeat is given.

Reports the same metrics as training, a per-year breakdown, the confident
subset with its coverage, the majority-class baseline of the training data,
and the misclassified fields with their key features.

For the clearly separate threshold-recalibrated variant see
fieldvision.model.evaluate_adapted.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import (fingerprint, ledger_check_and_record, ledger_path, load_run,
                        predictions_frame, write_json)
from .metrics import block_bootstrap_ci, classification_metrics
from .predict import add_input_args, load_inputs
from .report import SYNTHETIC_BANNER, TABLE_METRICS, confusion_line, fmt, fmt_ci, kv_table

KEY_FEATURES = ["harv_days_since_peak", "harv_mid_doy", "pre_ndvi_frac", "pre_ndmi",
                "sen_ndvi_slope_rel", "post_ndti", "peak_ndvi", "q_n_usable_season",
                "q_max_gap_days", "q_harv_width_days"]
MIN_BLOCKS_FOR_BLOCK_BOOTSTRAP = 8


def score_block(y, pred, p, groups, n_boot, seed):
    m = classification_metrics(y, pred, p)
    if len(y) and n_boot and len(np.unique(y)) == 2:
        m["ci95"] = block_bootstrap_ci(y, pred, p, groups, n_boot=n_boot, seed=seed)
    return m


def evaluate_frame(df, n_boot, seed, groups_col):
    """Metrics for all, confident, per year, for a frame with y, p, pred, low_evidence, year."""
    y, p, pred = df["y"].to_numpy(int), df["p"].to_numpy(float), df["pred"].to_numpy(int)
    g = df[groups_col].astype(str).to_numpy()
    conf = ~df["low_evidence"].to_numpy(bool)
    out = {"all": score_block(y, pred, p, g, n_boot, seed),
           "confident": score_block(y[conf], pred[conf], p[conf], g[conf], n_boot, seed)}
    out["confident"]["coverage"] = float(conf.mean()) if len(conf) else float("nan")
    out["per_year"] = {}
    for yr, sub in df.groupby("year"):
        out["per_year"][str(yr)] = score_block(sub["y"].to_numpy(int), sub["pred"].to_numpy(int),
                                               sub["p"].to_numpy(float),
                                               sub[groups_col].astype(str).to_numpy(), n_boot, seed)
    return out


def choose_groups(fields):
    """Resampling unit for the bootstrap: blocks if there are enough, else single fields."""
    n_blocks = fields["block_id"].nunique()
    if n_blocks >= MIN_BLOCKS_FOR_BLOCK_BOOTSTRAP:
        return "block_id", (f"block bootstrap over {n_blocks} blocks")
    return "field_id", (f"only {n_blocks} block(s), so the bootstrap resamples single fields. "
                        "Fields on one farm share weather and management, so these intervals "
                        "are too narrow")


def prepare(args):
    """Run, labelled evaluation fields, their features and predictions of every model in the run."""
    model, meta, cfg, majority = load_run(args.model, args.which)
    fields, ts, X, info = load_inputs(args, cfg)
    lab = np.isfinite(fields["y"].to_numpy(float))
    if not lab.any():
        raise SystemExit("no field in the evaluation set has a silage/grain label; "
                         "check --rename and --label-map")
    run_dir = Path(args.model)
    models = {}
    for path in sorted((run_dir / "models").glob("*.pkl")):
        import pickle
        with open(path, "rb") as fh:
            models[path.stem] = pickle.load(fh)
    name = args.which or meta["primary"]
    rows = []
    for mname, m in models.items():
        p = m.predict_proba(X)
        part = pd.DataFrame({"field_id": X["field_id"].to_numpy(), "model": mname, "p": p,
                             "threshold": m.threshold, "pred": (p >= m.threshold).astype(int)})
        rows.append(part)
    preds = pd.concat(rows, ignore_index=True)
    fmeta = fields[["field_id", "y", "label", "year", "block_id"]].copy()
    fmeta["year"] = fmeta["year"].map(lambda v: str(int(v)) if pd.notna(v) else "unknown")
    preds = preds.merge(fmeta, on="field_id").merge(
        X[["field_id", "low_evidence", "low_evidence_reason"]], on="field_id")
    preds = preds[np.isfinite(preds["y"].to_numpy(float))].copy()
    preds["y"] = preds["y"].astype(int)
    return dict(model=model, name=name, meta=meta, cfg=cfg, fields=fields, ts=ts, X=X,
                info=info, preds=preds, n_unlabelled=int((~lab).sum()), models=models)


def domain_shift_table(run_dir, X, fields, cols=KEY_FEATURES):
    """Per class medians of key features: training fields vs evaluation fields (descriptive)."""
    path = Path(run_dir) / "features.parquet"
    if not path.exists():
        return None
    tr = pd.read_parquet(path)
    tr = tr[tr["split"] == "development"]
    ev = X.merge(fields[["field_id", "label"]], on="field_id")
    rows = []
    for c in cols:
        if c not in tr.columns or c not in ev.columns:
            continue
        r = {"feature": c}
        for cls in ("silage", "grain"):
            r[f"train_{cls}"] = float(tr.loc[tr["label"] == cls, c].median())
            r[f"eval_{cls}"] = float(ev.loc[ev["label"] == cls, c].median())
            r[f"eval_{cls}_nan"] = float(ev.loc[ev["label"] == cls, c].isna().mean())
        rows.append(r)
    return pd.DataFrame(rows)


def misclassified(preds_model, X, fields):
    wrong = preds_model[preds_model["pred"] != preds_model["y"]]
    cols = ["field_id"] + [c for c in KEY_FEATURES if c in X.columns] + ["harv_lo_doy", "harv_hi_doy"]
    out = wrong[["field_id", "year", "label", "p", "low_evidence", "low_evidence_reason"]].merge(
        X[cols], on="field_id", how="left")
    out["error"] = np.where(out["label"] == "silage", "silage called grain (FN)",
                            "grain called silage (FP)")
    return out.sort_values("p", key=lambda s: -np.abs(s - 0.5)).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Zero-shot evaluation on an unseen labelled set.")
    add_input_args(ap)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-repeat", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    st = prepare(args)
    preds, meta, X, fields = st["preds"], st["meta"], st["X"], st["fields"]
    fp = fingerprint(preds["field_id"].unique())
    ledger = (ledger_path(out), "external_zero_shot", fp, meta["run_id"], out, args.allow_repeat)
    earlier = ledger_check_and_record(*ledger, record=False)
    groups_col, groups_note = choose_groups(fields[fields["field_id"].isin(preds["field_id"])])
    results = {}
    for mname, g in preds.groupby("model"):
        results[mname] = evaluate_frame(g, args.n_boot, args.seed, groups_col)

    prim = preds[preds["model"] == st["name"]]
    miss = misclassified(prim, X, fields)
    miss.to_csv(out / "misclassified.csv", index=False)
    pf = predictions_frame(X, fields, st["model"].predict_proba(X), st["model"].threshold,
                           st["name"], keep_columns=args.keep_columns, model=st["model"],
                           reference=meta.get("evidence_reference"))
    pf.to_parquet(out / "predictions.parquet", index=False)
    shift = domain_shift_table(args.model, X, fields)
    if shift is not None:
        shift.to_csv(out / "domain_shift.csv", index=False)

    n = max(len(X), 1)
    metrics = {
        "mode": "zero_shot", "fit_on_evaluation_set": "nothing",
        "run_id": meta["run_id"], "model": st["name"], "threshold": st["model"].threshold,
        "training_synthetic": meta.get("synthetic", False),
        "evaluation_set_fingerprint": fp, "earlier_evaluations_of_this_set": earlier,
        "n_fields": int(len(fields)), "n_labelled": int(prim.shape[0]),
        "n_unlabelled_ignored": st["n_unlabelled"],
        "bootstrap_unit": groups_col, "bootstrap_note": groups_note,
        "results": results,
        "response_time": {"features_seconds_per_1000": st["info"]["seconds"] / n * 1000,
                          "load_seconds": st["info"]["load_seconds"]},
        "timeseries_notes": st["info"]["timeseries_notes"],
        "low_evidence_reasons": X["low_evidence_reason"].replace("", "ok").value_counts().to_dict(),
    }
    write_json(out / "metrics.json", metrics)
    write_report(out, metrics, results, st, miss, shift, title="External evaluation: ZERO-SHOT",
                 intro=("Nothing was fit on these fields: model, threshold, feature settings and "
                        "the low-evidence rule all come from the training run. Evaluated once "
                        "(ledger entry written)."))
    ledger_check_and_record(*ledger, record=True)
    print(f"zero-shot on {prim.shape[0]} labelled fields -> {out}")
    return metrics


def metric_rows(results, order, subset="all"):
    head = ["model", "n (silage)"] + [h for _, h in TABLE_METRICS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in order:
        if name not in results:
            continue
        m = results[name][subset] if subset else results[name]
        cells = [name, f"{m['n']} ({m['n_silage']})"] + [fmt_ci(m, k) for k, _ in TABLE_METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out, metrics, results, st, miss, shift, title, intro, extra="", main_rows=None):
    meta = st["meta"]
    main_rows = main_rows or ["majority", st["name"]]
    others = [m for m in ("rule", "logistic", "gbm") if m not in main_rows]
    L = [f"# {title}\n"]
    if meta.get("synthetic"):
        L.append(SYNTHETIC_BANNER)
    L.append(f"{dt.datetime.now():%Y-%m-%d %H:%M}. Run `{meta['run_id']}`, primary model "
             f"`{metrics['model']}` (chosen at training time), training threshold "
             f"{fmt(st['model'].threshold)}. Positive class = silage.\n")
    L.append(intro + "\n")
    if metrics.get("earlier_evaluations_of_this_set"):
        L.append(f"**Warning:** this field set was evaluated before "
                 f"{len(metrics['earlier_evaluations_of_this_set'])} time(s) (any run). "
                 "See the ledger.\n")
    L.append(f"{metrics['n_labelled']} labelled fields ({metrics['n_unlabelled_ignored']} "
             f"unlabelled ignored). Intervals: {metrics['bootstrap_note']}.\n")
    if extra:
        L.append(extra + "\n")
    L.append("## Primary model and baseline\n")
    L.append(metric_rows(results, main_rows))
    m = results[st["name"]]["all"]
    L.append(f"\nConfusion: {confusion_line(m)}. `majority` predicts the training majority "
             "class for every field.\n")
    L.append("Confident subset (low-evidence fields removed):\n")
    L.append(metric_rows(results, main_rows, "confident"))
    L.append(f"\nCoverage {fmt(results[st['name']]['confident']['coverage'])}.\n")
    L.append("## Per year\n")
    head = ["year", "n (silage)"] + [h for _, h in TABLE_METRICS]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for yr, mm in results[st["name"]]["per_year"].items():
        rows.append("| " + " | ".join([yr, f"{mm['n']} ({mm['n_silage']})"] +
                                      [fmt_ci(mm, k) for k, _ in TABLE_METRICS]) + " |")
    L.append("\n".join(rows) + "\n")
    L.append("## Other models in the run (secondary; not used to choose the model)\n")
    L.append(metric_rows(results, others) + "\n")
    L.append("## Misclassified fields (primary model)\n")
    if miss.empty:
        L.append("None.\n")
    else:
        L.append(f"{len(miss)} fields, full list in `misclassified.csv`. Most confident mistakes:\n")
        cols = ["field_id", "year", "error", "p", "low_evidence", "harv_days_since_peak",
                "pre_ndmi", "pre_ndvi_frac", "post_ndti", "q_n_usable_season"]
        L.append(kv_table([[r[c] for c in cols] for r in miss.head(15).to_dict("records")], cols))
    if shift is not None and not shift.empty:
        L.append("\n## Key features: training vs this set (medians per declared class)\n")
        L.append("Descriptive only; nothing is adjusted from it.\n")
        cols = list(shift.columns)
        L.append(kv_table([[r[c] for c in cols] for r in shift.to_dict("records")], cols))
    L.append("\n## Low-evidence reasons\n")
    L.append(", ".join(f"{k}: {v}" for k, v in metrics["low_evidence_reasons"].items()))
    L.append(f"\n\nFeatures: {fmt(metrics['response_time']['features_seconds_per_1000'], 2)} s per "
             "1000 fields.\n")
    (Path(out) / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
