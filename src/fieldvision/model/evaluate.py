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
from .metrics import (block_bootstrap_ci, calibration_summary, classification_metrics,
                      exact_intervals)
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
    m["exact95"] = exact_intervals(m["confusion"])
    if len(y) and len(np.unique(y)) == 2:
        m["brier"] = float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))
    return m


def physical_unit(field_ids):
    """'barc-12-2022' -> 'barc-12': the same polygon in different years is one unit."""
    s = pd.Series(np.asarray(field_ids, dtype=object)).astype(str)
    return s.str.replace(r"-(19|20)\d\d$", "", regex=True).to_numpy()


def evaluate_frame(df, n_boot, seed, groups_col):
    """Metrics for all, confident, per year, for a frame with y, p, pred, low_evidence, year."""
    y, p, pred = df["y"].to_numpy(int), df["p"].to_numpy(float), df["pred"].to_numpy(int)
    g = df[groups_col].astype(str).to_numpy()
    conf = ~df["low_evidence"].to_numpy(bool)
    out = {"all": score_block(y, pred, p, g, n_boot, seed),
           "confident": score_block(y[conf], pred[conf], p[conf], g[conf], n_boot, seed)}
    out["confident"]["coverage"] = float(conf.mean()) if len(conf) else float("nan")
    for col in ("year", "collection"):
        if col not in df.columns:
            continue
        out[f"per_{col}"] = {}
        for key, sub in df.groupby(col):
            out[f"per_{col}"][str(key)] = score_block(
                sub["y"].to_numpy(int), sub["pred"].to_numpy(int), sub["p"].to_numpy(float),
                sub[groups_col].astype(str).to_numpy(), n_boot, seed)
    return out


def choose_groups(fields):
    """Resampling unit for the bootstrap: blocks if there are enough, else single fields."""
    n_blocks = fields["block_id"].nunique()
    if n_blocks >= MIN_BLOCKS_FOR_BLOCK_BOOTSTRAP:
        return "block_id", (f"block bootstrap over {n_blocks} blocks")
    n_units = len(np.unique(physical_unit(fields["field_id"])))
    return "unit", (f"only {n_blocks} 10 km block(s). Count metrics carry exact binomial "
                    "(Clopper-Pearson) 95% intervals; ROC-AUC and PR-AUC a bootstrap over the "
                    f"{n_units} physical fields (all years of a polygon resampled together). Both "
                    "treat fields as independent, which fields on one farm are not, so the true "
                    "uncertainty is larger")


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
    preds["unit"] = physical_unit(preds["field_id"])
    from .harness import field_collection
    coll = field_collection(ts)
    preds["collection"] = preds["field_id"].map(coll).fillna("none")
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


def check_frozen(run_dir, name, meta):
    """Identifier of the model being evaluated, checked against the one recorded at training."""
    from .artifacts import frozen_model_id
    now = frozen_model_id(run_dir, name)
    recorded = (meta.get("frozen_model_ids") or {}).get(name)
    if recorded and recorded != now:
        raise SystemExit(f"model {name} in {run_dir} changed after training (recorded {recorded}, "
                         f"now {now}); refusing to evaluate a model that is not the frozen one")
    return {"id": now, "recorded_at_training": recorded, "model": name,
            "run_created": meta.get("created"),
            "note": "sha256 of the model file and its training configuration (first 16 hex)"}


def access_log_rows(path):
    """The table rows of the team's access log for the evaluation set (markdown lines)."""
    if not path or not Path(path).exists():
        return None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln.startswith("|")]


def append_access_log(path, frozen_id, kind, out):
    """Add this evaluation as a row of the access log (after the report is written)."""
    if not path:
        return
    import sys
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    module = "fieldvision.model.evaluate" if kind == "zero-shot" else "fieldvision.model.evaluate_adapted"
    command = f"python -m {module} " + " ".join(sys.argv[1:])
    row = (f"| {dt.datetime.now():%H:%M} | modelling worker | {kind} evaluation, run once with "
           f"frozen model `{frozen_id}`: `{command}`; output `{out}` | This is the evaluation |")
    lines = text.splitlines()
    last = max((i for i, ln in enumerate(lines) if ln.startswith("|")), default=None)
    if last is None:
        lines.append(row)
    else:
        lines.insert(last + 1, row)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def counts_table(fields):
    """Evaluated field(-year)s per year and class, plus unlabelled ones that are not scored."""
    f = fields.copy()
    f["year"] = f["year"].map(lambda v: str(int(v)) if pd.notna(v) else "unknown")
    f["cls"] = f["label"].fillna("unlabelled (not scored)")
    t = pd.crosstab(f["year"], f["cls"])
    for c in ("silage", "grain", "unlabelled (not scored)"):
        if c not in t.columns:
            t[c] = 0
    t["scored"] = t["silage"] + t["grain"]
    t.loc["total"] = t.sum()
    return t.reset_index()[["year", "silage", "grain", "scored", "unlabelled (not scored)"]]


def abstentions(pred_frame, fields):
    """Labelled fields the method flags as low evidence, with its call and why."""
    lab = pred_frame[pred_frame["label"].notna()] if "label" in pred_frame else pred_frame
    cols = [c for c in ("field_id", "year", "label", "p_silage", "predicted_label",
                        "low_evidence_reason", "n_usable_dates", "max_gap_days",
                        "harvest_interval_days", "reason") if c in lab.columns]
    return lab.loc[lab["low_evidence"], cols].reset_index(drop=True)


def transfer_evidence(run_dir):
    """Leave-one-country/year-out numbers from the training run, for the external report."""
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        return None
    import json
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    v = m.get("variant", "relative")
    out = []
    for scheme, by_var in m.get("summary", {}).items():
        if not scheme.startswith(("loco:", "loyo:", "spatial")):
            continue
        for model in ("majority", "timing_rule", "gbm"):
            r = by_var.get(v, {}).get(model, {}).get("all")
            if r:
                out.append({"scheme": scheme, "model": model, "n": r["n"],
                            "balanced_accuracy": r["balanced_accuracy"], "roc_auc": r["roc_auc"],
                            "recall": r["recall"], "fpr": r["fpr"],
                            "grain_recall": r.get("grain_recall")})
    return out


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
    ap.add_argument("--access-log", default=None,
                    help="team log of looks at the evaluation set; quoted in the report and "
                         "this run is appended to it")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    st = prepare(args)
    preds, meta, X, fields = st["preds"], st["meta"], st["X"], st["fields"]
    fp = fingerprint(preds["field_id"].unique())
    frozen = check_frozen(args.model, st["name"], meta)
    ledger = (ledger_path(out), "external_zero_shot", fp, frozen["id"], out, args.allow_repeat)
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
    abstain = abstentions(pf, fields)
    abstain.to_csv(out / "abstentions.csv", index=False)

    n = max(len(X), 1)
    metrics = {
        "mode": "zero_shot", "fit_on_evaluation_set": "nothing",
        "run_id": meta["run_id"], "model": st["name"], "threshold": st["model"].threshold,
        "frozen_model": frozen,
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
        "counts_by_year": counts_table(fields).to_dict("records"),
        "calibration": calibration_summary(prim["y"], prim["p"], n_bins=5),
        "transfer_evidence_from_training": transfer_evidence(args.model),
        "access_log": access_log_rows(args.access_log),
    }
    write_json(out / "metrics.json", metrics)
    write_report(out, metrics, results, st, miss, shift, title="External evaluation: ZERO-SHOT",
                 intro=("Nothing was fit on these fields: model, threshold, feature settings and "
                        "the low-evidence rule all come from the training run. Evaluated once "
                        "(ledger entry written)."))
    ledger_check_and_record(*ledger, record=True)
    append_access_log(args.access_log, frozen["id"], "zero-shot", out)
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
    main_rows = main_rows or ["majority", "always_grain", "timing_rule", st["name"]]
    others = [m for m in ("timing_rule", "rule", "logistic", "gbm") if m not in main_rows]
    L = [f"# {title}\n"]
    if meta.get("synthetic"):
        L.append(SYNTHETIC_BANNER)
    L.append(f"{dt.datetime.now():%Y-%m-%d %H:%M}. Run `{meta['run_id']}`, primary model "
             f"`{metrics['model']}` (chosen at training time), training threshold "
             f"{fmt(st['model'].threshold)}. Positive class = silage.\n")
    L.append(intro + "\n")
    fz = metrics.get("frozen_model")
    if fz:
        L.append(f"Frozen model identifier: `{fz['id']}` (model `{fz['model']}`, trained "
                 f"{fz.get('run_created')}; {fz['note']}; matches the value written at training: "
                 f"{'yes' if fz.get('recorded_at_training') == fz['id'] else 'not recorded'}). "
                 "The ledger refuses a second zero-shot evaluation of this set with the same "
                 "frozen model.\n")
    if metrics.get("earlier_evaluations_of_this_set"):
        L.append(f"**Warning:** this field set was evaluated before "
                 f"{len(metrics['earlier_evaluations_of_this_set'])} time(s) (any run). "
                 "See the ledger.\n")
    L.append(f"{metrics['n_labelled']} labelled fields ({metrics['n_unlabelled_ignored']} "
             f"unlabelled ignored). Intervals: {metrics['bootstrap_note']}.\n")
    if extra:
        L.append(extra + "\n")
    L.append("## Claim and validation design\n")
    L.append("Claim: the timing and state of the harvest drop seen by Sentinel-2 separates silage "
             "from grain corn, and a model learned on European parcel registries (Austria 2022-24, "
             "Netherlands 2025, France 2022) transfers to US fields without retraining.\n")
    L.append("Design: all model choices were made on the European data with splits grouped by "
             "10 km block (never random fields), plus leave-one-country-out and leave-one-year-out. "
             "This set was not used for any fitting, tuning or feature choice and is scored once. "
             "With this few fields it confirms or contradicts the European transfer results "
             "below; it cannot replace them.\n")
    log = metrics.get("access_log")
    L.append("### What was seen of this set before the freeze\n")
    if log:
        L.append("Quoted from the team's access log (kept by the coordinating session; the row "
                 "for this evaluation is added after this report is written):\n")
        L.append("\n".join(log) + "\n")
    else:
        L.append("No access log was given to this command (`--access-log`), so this report cannot "
                 "state who looked at the set before the freeze.\n")
    L.append("### Fields evaluated, by year and class\n")
    ct = metrics.get("counts_by_year") or []
    if ct:
        cols = list(ct[0].keys())
        L.append(kv_table([[r[c] for c in cols] for r in ct], cols))
        L.append("\nBlank labels are 'not labelled', not negatives, and are not scored. One "
                 "polygon can appear in several years.\n")
    L.append("## Results: baselines and primary model\n")
    L.append("`majority` predicts the training majority class for every field; `timing_rule` "
             "is the transparent baseline (silage if the sustained NDVI decline comes before a "
             "day-of-year threshold fitted on the European training data only). Precision and "
             "recall are given for both classes.\n")
    L.append(metric_rows(results, main_rows))
    m = results[st["name"]]["all"]
    c = m["confusion"]
    L.append(f"\nConfusion matrix, `{st['name']}` (rows = declared, columns = predicted):\n")
    L.append(kv_table([("silage", c["tp"], c["fn"]), ("grain", c["fp"], c["tn"])],
                      ["declared \\ predicted", "silage", "grain"]))
    if "timing_rule" in results:
        c2 = results["timing_rule"]["all"]["confusion"]
        L.append("\nConfusion matrix, `timing_rule`:\n")
        L.append(kv_table([("silage", c2["tp"], c2["fn"]), ("grain", c2["fp"], c2["tn"])],
                          ["declared \\ predicted", "silage", "grain"]))
    L.append("\n### Abstention\n")
    L.append("Fields flagged low evidence still get a call, but the method says it should not be "
             "trusted. Confident subset (low-evidence fields removed):\n")
    L.append(metric_rows(results, main_rows, "confident"))
    L.append(f"\nCoverage {fmt(results[st['name']]['confident']['coverage'])}.\n")
    ab_path = Path(out) / "abstentions.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        if len(ab):
            cols = [c_ for c_ in ("field_id", "label", "p_silage", "predicted_label",
                                  "low_evidence_reason", "n_usable_dates", "max_gap_days")
                    if c_ in ab.columns]
            L.append(f"{len(ab)} labelled fields abstained on (all in `abstentions.csv`):\n")
            L.append(kv_table([[r[c_] for c_ in cols] for r in ab.head(30).to_dict("records")], cols))
            L.append("")
    L.append("## Per year\n")
    head = ["year", "n (silage)"] + [h for _, h in TABLE_METRICS]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for yr, mm in results[st["name"]]["per_year"].items():
        rows.append("| " + " | ".join([yr, f"{mm['n']} ({mm['n_silage']})"] +
                                      [fmt_ci(mm, k) for k, _ in TABLE_METRICS]) + " |")
    L.append("\n".join(rows) + "\n")
    L.append("Per-year counts are tiny; read these as a check for one bad year, not as estimates.\n")
    pc = results[st["name"]].get("per_collection")
    if pc:
        L.append("### By imagery source\n")
        L.append("Each field-year is assigned the collection most of its usable dates came from "
                 "('+mixed' if more than one). For Beltsville, 2022 comes from a different "
                 "processing source (Planetary Computer) than 2021 and 2023; the training report's "
                 "same-block check (Austria 2022 vs 2023) shows how much that source moves the "
                 "features.\n")
        rows = ["| " + " | ".join(["collection", "n (silage)"] + [h for _, h in TABLE_METRICS]) + " |",
                "|" + "---|" * (2 + len(TABLE_METRICS))]
        for key, mm in pc.items():
            rows.append("| " + " | ".join([key, f"{mm['n']} ({mm['n_silage']})"] +
                                          [fmt_ci(mm, k) for k, _ in TABLE_METRICS]) + " |")
        L.append("\n".join(rows) + "\n")
    cal = metrics.get("calibration")
    if cal:
        L.append("## Calibration on this set\n")
        L.append(f"Brier {fmt(cal['brier'])}, ECE {fmt(cal['ece'])}. Probabilities are calibrated "
                 "for the 50/50 training balance, so with fewer silage fields here they will read "
                 "high on average. Per bin (mean predicted -> observed silage share): " +
                 ", ".join(f"{b['mean_p']:.2f}->{b['share_silage']:.2f} (n={b['n']})"
                           for b in cal["reliability"]) + ".\n")
    te = metrics.get("transfer_evidence_from_training")
    if te:
        L.append("## Transfer evidence from the European data (main evidence)\n")
        L.append("From the training run: spatial CV, and each country or year held out entirely. "
                 "This is where the transfer claim is tested with hundreds of fields per split.\n")
        L.append(kv_table([(r["scheme"], r["model"], r["n"], fmt(r["balanced_accuracy"]),
                            fmt(r["roc_auc"]), fmt(r["recall"]), fmt(r["grain_recall"]), fmt(r["fpr"]))
                           for r in te],
                          ["split", "model", "n", "bal. acc.", "ROC-AUC", "silage recall",
                           "grain recall", "FPR"]))
        L.append("")
    L.append("## Limits\n")
    L.append("- Geography: one research farm in Maryland. It says nothing direct about the Corn "
             "Belt, the South (earlier harvest calendars) or irrigated double-cropped regions.\n"
             "- Time: 2021-2023, with imagery from more than one processing source; see the "
             "per-year table and the training report's collection-shift check.\n"
             "- Labels: European labels are farmer declarations; the Dutch ones are made in spring "
             "and can differ from the actual use.\n"
             "- The harvest date is bracketed by the last clear image before and the first after; "
             "cloud gaps widen it, and wide brackets are what drive abstention.\n")
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
