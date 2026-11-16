"""Stage 1 of the anomaly view, on labelled European fields: can "unlike normal grain" find silage?

    python -m fieldvision.anomaly.evaluate --timeseries <run>/timeseries.parquet ... \
        --fields <labels>/fields.geojson ... --out <dir>

Inputs (each scored by mahalanobis, iforest and knn, see anomaly.scorers):
  phys6  the six physical features of the small logistic model
  rel    the peak-aligned curve: index medians in 15-day bins around each field's own peak
  raw    monthly composites Apr-Nov of the S2 bands and NDVI (no encoder)

Splits are the main line's: 5 spatial folds grouped by block (model.splits.spatial_folds,
seed 0), then leave-one-year-out and, with two countries, leave-one-country-out. The
one-class models see only the grain fields of the training part (plus, for the
contamination runs, silage mixed in at 1%, 5% or 20% of the normal set); the supervised
6-feature logistic and the LightGBM of the main line are fit on the same training parts
with labels, for reference. ROC-AUC intervals are block-bootstrap (model.harness).
"""
from __future__ import annotations

import argparse
import json
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig
from ..model.harness import compare_models, run_splits, summarize_predictions
from ..model.models import LOGISTIC_FEATURES, make_factories
from ..model.splits import leave_one_group_out, spatial_folds
from .scorers import SCORERS, OneClassModel

CONTAMINATION = (0.0, 0.01, 0.05, 0.20)
REF = "phys-logistic"

# what a flagged grain field looks like: a field can carry several of these
FLAGS = {
    "no harvest found": lambda X, g: X["harv_mid_doy"].isna(),
    "sparse or cloudy series": lambda X, g: X["low_evidence"].astype(bool) | (X["q_max_gap_days"] > 40),
    "second green-up": lambda X, g: (X["q_n_green_cycles"] >= 2) | (X["post_regreen_flag"] == 1)
                                    | (X["q_spring_above_summer"] == 1),
    "weak crop (peak NDVI < 0.6)": lambda X, g: X["peak_ndvi"] < 0.6,
    "early harvest (grain 10th pct)": lambda X, g: X["harv_days_since_peak"]
                                                   < g["harv_days_since_peak"].quantile(0.10),
    "late harvest (grain 90th pct)": lambda X, g: X["harv_days_since_peak"]
                                                  > g["harv_days_since_peak"].quantile(0.90),
}
DESCRIBE_COLS = ["peak_ndvi", "amplitude", "peak_doy", "harv_mid_doy", "harv_days_since_peak",
                 "pre_ndmi", "post_ndti", "q_n_usable_season", "q_max_gap_days", "q_n_green_cycles",
                 "post_regreen_flag", "low_evidence"]


def variant_name(c, seed):
    return "c00" if c <= 0 else f"c{int(round(c * 100)):02d}s{seed}"


def oneclass_factories(inputs, contamination=0.0, seed=0, quantile=0.98, scorers=SCORERS):
    return {f"{sc}-{tag}": partial(OneClassModel, list(cols), sc, contamination, quantile, seed=seed)
            for tag, cols in inputs.items() for sc in scorers}


def auc_by_contamination(summary, scheme, models):
    """model -> {contamination: mean pooled AUC over the seeds}."""
    out = {}
    for variant, recs in summary.get(scheme, {}).items():
        c = 0.0 if variant == "c00" else int(variant[1:3]) / 100
        for m in models:
            if m in recs:
                out.setdefault(m, {}).setdefault(c, []).append(recs[m]["all"]["roc_auc"])
    return {m: {c: float(np.mean(v)) for c, v in d.items()} for m, d in out.items()}


def stage1_table(preds, summary, scheme, models, n_boot=1000, seed=0):
    """One row per model: clean AUC with CI and fold spread, recall/FPR at the fixed
    threshold, AUC with 1/5/20% silage in the normal set, paired diff to the logistic."""
    from ..foundation.compare import paired_fold_diffs, verdict
    base = summary[scheme]["c00"]
    by_c = auc_by_contamination(summary, scheme, models)
    clean = preds[preds["variant"] == "c00"]
    rows = []
    for m in models:
        if m not in base:
            continue
        rec = base[m]
        folds = np.array([f["roc_auc"] for f in rec["per_fold"].values()], dtype=float)
        row = {"scheme": scheme, "model": m, "n": rec["all"]["n"], "auc": rec["all"]["roc_auc"],
               "auc_ci": rec["all"].get("ci95", {}).get("roc_auc", [np.nan, np.nan]),
               "auc_fold_mean": float(np.nanmean(folds)),
               "auc_fold_sd": float(np.nanstd(folds, ddof=1)) if len(folds) > 1 else np.nan,
               "recall": rec["all"]["recall"], "fpr": rec["all"]["fpr"]}
        for c in CONTAMINATION[1:]:
            row[f"auc_c{int(c * 100):02d}"] = by_c.get(m, {}).get(c, np.nan)
        if m != REF:
            cmp = compare_models(clean, m, REF, scheme, "c00", n_boot=n_boot, seed=seed)
            row["diff_vs_logistic"] = cmp["roc_auc"]["diff"]
            row["diff_ci"] = cmp["roc_auc"]["ci"]
            row["verdict"] = (verdict(paired_fold_diffs(clean, m, REF, scheme), cmp["roc_auc"])
                              if scheme == "spatial" else
                              ("better" if cmp["roc_auc"]["ci"][0] > 0 else
                               "worse" if cmp["roc_auc"]["ci"][1] < 0 else "no clear difference"))
        rows.append(row)
    return pd.DataFrame(rows)


def flagged_grain(preds, X, meta, model, top=50):
    """The `top` most anomalous grain fields (held-out scores, scaled by each fold's own
    threshold so folds are comparable) and how often each FLAGS pattern occurs among
    them versus among all grain fields."""
    g = preds[(preds["scheme"] == "spatial") & (preds["variant"] == "c00") & (preds["model"] == model)
              & (preds["y"] == 0)].copy()
    g["rel_score"] = g["p"] / g["threshold"]
    g = g.sort_values("rel_score", ascending=False)
    feats = X.assign(field_id=meta["field_id"].to_numpy()).set_index("field_id")
    grain = feats.loc[g["field_id"]]
    topf = feats.loc[g["field_id"].head(top)]
    share = {name: {"top": float(f(topf, grain).mean()), "all_grain": float(f(grain, grain).mean())}
             for name, f in FLAGS.items()}
    detail = g.head(top)[["field_id", "country", "year", "rel_score"]].merge(
        feats[DESCRIBE_COLS].reset_index(), on="field_id")
    medians = grain[DESCRIBE_COLS].median()
    return share, detail, medians


def tier_table(preds, anomaly_model, classifier=REF, variant="c00"):
    """Per scheme and confidence tier (anomaly.api.confidence_tier): share of fields, the
    classifier's accuracy, recall and FPR inside the tier, and the share that is silage."""
    from .api import confidence_tier
    rows = []
    for scheme, g in preds[preds["variant"] == variant].groupby("scheme", sort=False):
        c = g[g["model"] == classifier].set_index("field_id")
        a = g[g["model"] == anomaly_model].set_index("field_id")
        m = c[["y", "pred", "low_evidence"]].join(a[["pred"]].rename(columns={"pred": "flag"}), how="inner")
        m["tier"] = confidence_tier(m["pred"], m["flag"], m["low_evidence"])
        for tier in ("agree", "disagree", "abstain", "all"):
            t = m if tier == "all" else m[m["tier"] == tier]
            if t.empty:
                continue
            yy, pp = t["y"].to_numpy(), t["pred"].to_numpy()
            rows.append({"scheme": scheme, "tier": tier, "n": len(t), "share": len(t) / len(m),
                         "silage_share": float(yy.mean()), "classifier_accuracy": float((yy == pp).mean()),
                         "classifier_recall": float(pp[yy == 1].mean()) if (yy == 1).any() else np.nan,
                         "classifier_fpr": float(pp[yy == 0].mean()) if (yy == 0).any() else np.nan})
    return pd.DataFrame(rows)


def fmt(v, nd=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def markdown(tab):
    lines = ["| held out | model | n | AUC [95% CI] | fold mean ± SD | recall | FPR | AUC 1% | AUC 5% | AUC 20% "
             "| diff vs logistic [95% CI] | verdict |",
             "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|"]
    for _, r in tab.iterrows():
        ci = r["auc_ci"]
        d = ("ref" if r["model"] == REF else f"{r['diff_vs_logistic']:+.3f} "
             f"[{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]")
        sup = r["model"].startswith("phys-")
        lines.append(f"| {r['scheme']} | {r['model']} | {r['n']} | {fmt(r['auc'])} [{fmt(ci[0])}, {fmt(ci[1])}] | "
                     f"{fmt(r['auc_fold_mean'], 2)} ± {fmt(r['auc_fold_sd'], 2)} | {fmt(r['recall'])} | "
                     f"{fmt(r['fpr'])} | " + " | ".join("" if sup else fmt(r[k]) for k in
                                                         ("auc_c01", "auc_c05", "auc_c20"))
                     + f" | {d} | {r.get('verdict', '') if not sup or r['model'] != REF else ''} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3, help="repeats of each contaminated run")
    ap.add_argument("--quantile", type=float, default=0.98)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from ..foundation.adapter import build_presto_inputs
    from ..foundation.compare import raw_composite_table
    cfg = FeatureConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = load_fields(args.fields)
        ts, _ = load_timeseries(args.timeseries, field_ids=fields["field_id"])
        fields = fields[np.isfinite(fields["y"].to_numpy(float))].reset_index(drop=True)
        X, _ = build_features(ts, fields, cfg)
        raw = raw_composite_table(build_presto_inputs(ts, fields, cfg=cfg))
    raw_cols = [c for c in raw.columns if c != "field_id"]
    X = pd.concat([X.reset_index(drop=True), raw[raw_cols]], axis=1)
    meta = fields.copy()
    meta["year"] = meta["year"].map(lambda v: str(int(v)) if np.isfinite(v) else "unknown")
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    y = meta["y"].to_numpy(int)
    inputs = {"phys6": [c for c in LOGISTIC_FEATURES["relative"] if c in X.columns],
              "rel": [c for c in X.columns if c.startswith("rel_")],
              "raw": raw_cols}
    print(f"{len(fields)} fields ({int(y.sum())} silage); inputs "
          + ", ".join(f"{k} {len(v)}" for k, v in inputs.items()), flush=True)

    splits = {"spatial": [(f"fold{i}", tr, te) for i, (tr, te) in
                          enumerate(spatial_folds(meta["block_id"], y, 5, args.seed))]}
    for key, groups in (("loyo", meta["year"]), ("loco", meta["country"])):
        if groups.nunique() > 1:
            for g, tr, te, _ in leave_one_group_out(groups, meta["block_id"], exclude_test_blocks=False):
                splits[f"{key}:{g}"] = [(g, tr, te)]

    main_facs = make_factories(X.columns, "relative", seed=args.seed)
    sup = {"phys-logistic": main_facs["logistic"], "phys-gbm": main_facs["gbm"]}
    preds = []
    for scheme, sp in splits.items():
        p, _, _ = run_splits(X, meta, y, sup, sp, scheme, "c00", seed=args.seed)
        preds.append(p)
        for c in CONTAMINATION:
            for s in ([args.seed] if c <= 0 else range(args.seed, args.seed + args.seeds)):
                p, _, _ = run_splits(X, meta, y, oneclass_factories(inputs, c, s, args.quantile), sp,
                                     scheme, variant_name(c, s), seed=s)
                preds.append(p)
        print(f"done {scheme}", flush=True)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(out / "predictions.parquet", index=False)
    summary = summarize_predictions(preds, n_boot=args.n_boot, seed=args.seed)

    models = list(sup) + list(oneclass_factories(inputs))
    tabs = pd.concat([stage1_table(preds, summary, s, models, args.n_boot, args.seed) for s in splits],
                     ignore_index=True)
    tabs.to_csv(out / "stage1_table.csv", index=False)

    sp_tab = tabs[(tabs["scheme"] == "spatial") & ~tabs["model"].str.startswith("phys-")]
    best = sp_tab.sort_values("auc", ascending=False)["model"].iloc[0]
    share, detail, medians = flagged_grain(preds, X, meta, best)
    detail.to_csv(out / f"flagged_grain_{best}.csv", index=False)

    md = [f"# Anomaly view, stage 1: {out.name}", "",
          f"{len(fields)} fields, {int(y.sum())} silage. Normal set = grain fields of the training part; "
          f"flag threshold = the normal set's {args.quantile:.0%} quantile (cross-fitted). "
          f"AUC 1/5/20% = mean over {args.seeds} draws with that share of silage in the normal set.", "",
          markdown(tabs), "", f"## Most anomalous grain fields ({best}, spatial folds, top {len(detail)})", "",
          "| pattern | share in top | share among all grain |", "|---|---:|---:|"]
    md += [f"| {k} | {v['top']:.2f} | {v['all_grain']:.2f} |" for k, v in share.items()]
    md += ["", "Grain medians: " + ", ".join(f"{k} {fmt(v, 2)}" for k, v in medians.items())]
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(
        {"args": vars(args), "inputs": inputs, "best_anomaly_model": best, "flag_share": share,
         "table": tabs.to_dict(orient="records")}, indent=1, default=str), encoding="utf-8")
    print("\n".join(md), flush=True)
    return tabs


if __name__ == "__main__":
    main()
