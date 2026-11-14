"""How early can we know? Evaluation with the time series cut off at a date.

    python -m fieldvision.model.asof --timeseries <...> --fields <...> --out <dir> \
        [--cutoffs 08-01 08-15 09-01 ...]

For each cutoff (month-day, applied in each field's own season year) every
field keeps only the acquisitions up to that day. Features are rebuilt from
what was visible then, and the models are trained and tested with the same
block-grouped spatial folds at every cutoff, so an early-season model is
judged on early-season data only.

Reported per cutoff: metrics on all fields, the share of fields that can be
called confidently (not low evidence) and the metrics on those, the share
with a harvest drop already seen, and precision re-weighted to a realistic
silage share.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig
from .artifacts import write_json
from .harness import run_splits
from .metrics import block_bootstrap_ci, classification_metrics, prevalence_weights
from .models import make_factories
from .report import SYNTHETIC_BANNER, fmt, kv_table
from .splits import spatial_folds

DEFAULT_CUTOFFS = ["08-01", "08-15", "09-01", "09-15", "10-01", "10-15", "11-01", "11-15",
                   "12-01", "12-31"]


def truncate(ts, fields, cutoff):
    """Rows of `ts` on or before month-day `cutoff` of each field's season year."""
    m, d = (int(x) for x in cutoff.split("-"))
    year = fields.set_index("field_id")["year"]
    y = pd.to_numeric(ts["field_id"].map(year), errors="coerce")
    y = y.fillna(ts["date"].dt.year).astype(int)
    limit = pd.to_datetime(dict(year=y, month=m, day=d))
    return ts[ts["date"] <= limit]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Accuracy and confident coverage vs cutoff date.")
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoffs", nargs="*", default=DEFAULT_CUTOFFS, help="MM-DD")
    ap.add_argument("--variant", default="relative")
    ap.add_argument("--models", nargs="*", default=["majority", "timing_rule", "rule", "logistic", "gbm"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prevalence", type=float, default=0.07)
    ap.add_argument("--exclude-blocks", nargs="*", default=[],
                    help="block patterns to leave out entirely (e.g. the training holdout)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    fields = load_fields(args.fields)
    fields = fields[np.isfinite(fields["y"].to_numpy(float))].reset_index(drop=True)
    if args.exclude_blocks:
        from .splits import holdout_mask
        fields = fields[~holdout_mask(fields["block_id"], args.exclude_blocks)].reset_index(drop=True)
    ts, _ = load_timeseries(args.timeseries, field_ids=fields["field_id"])
    y = fields["y"].to_numpy(int)
    meta = fields.copy()
    meta["year"] = meta["year"].map(lambda v: str(int(v)) if pd.notna(v) else "unknown")
    folds = spatial_folds(meta["block_id"], y, args.n_splits, args.seed)
    splits = [(f"fold{i}", tr, te) for i, (tr, te) in enumerate(folds)]
    synthetic = bool((fields["source"].astype(str) == "synthetic").all())
    cfg = FeatureConfig()

    rows, all_preds = [], []
    for cut in args.cutoffs:
        sub = truncate(ts, fields, cut)
        X, _ = build_features(sub, fields, cfg)
        X["block_id"] = meta["block_id"].to_numpy()
        meta["low_evidence"] = X["low_evidence"].to_numpy()
        fac = {k: f for k, f in make_factories(X.columns, args.variant, seed=args.seed).items()
               if k in args.models}
        preds, _, _ = run_splits(X, meta, y, fac, splits, f"asof:{cut}", args.variant)
        preds["cutoff"] = cut
        all_preds.append(preds)
        seen = np.isfinite(X["harv_mid_doy"].to_numpy(float))
        for name, g in preds.groupby("model", sort=False):
            yy, pp, pr = g["y"].to_numpy(int), g["p"].to_numpy(float), g["pred"].to_numpy(int)
            m = classification_metrics(yy, pr, pp)
            ci = block_bootstrap_ci(yy, pr, pp, g["block_id"], n_boot=args.n_boot, seed=args.seed)
            conf = ~g["low_evidence"].to_numpy(bool)
            mc = classification_metrics(yy[conf], pr[conf], pp[conf]) if conf.any() else {}
            mw = classification_metrics(yy, pr, pp, weights=prevalence_weights(yy, args.prevalence))
            rows.append({
                "cutoff": cut, "model": name, "n": m["n"],
                "harvest_seen_share": float(seen.mean()),
                "harvest_seen_silage": float(seen[y == 1].mean()),
                "harvest_seen_grain": float(seen[y == 0].mean()),
                "confident_share": float(conf.mean()),
                "balanced_accuracy": m["balanced_accuracy"],
                "balanced_accuracy_ci": ci["balanced_accuracy"],
                "roc_auc": m["roc_auc"], "roc_auc_ci": ci["roc_auc"],
                "precision": m["precision"], "recall": m["recall"], "fpr": m["fpr"],
                "confident_balanced_accuracy": mc.get("balanced_accuracy", np.nan),
                "confident_precision": mc.get("precision", np.nan),
                "confident_recall": mc.get("recall", np.nan),
                "confident_fpr": mc.get("fpr", np.nan),
                f"precision_at_{args.prevalence:g}": mw["precision"],
            })
        print(f"cutoff {cut}: harvest seen {seen.mean():.0%}, confident "
              f"{rows[-1]['confident_share']:.0%}", flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "asof.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_parquet(out / "asof_predictions.parquet", index=False)
    write_json(out / "asof.json", {"synthetic_data_software_check_only": synthetic,
                                   "cutoffs": args.cutoffs, "rows": tab.to_dict("records"),
                                   "n_fields": int(len(fields)), "n_blocks": int(meta["block_id"].nunique()),
                                   "seconds": time.perf_counter() - t0})
    _plot(tab, out / "asof.png", synthetic, args.prevalence)
    _report(tab, out, args, synthetic, len(fields), meta["block_id"].nunique())
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}")
    return tab


def _plot(tab, path, synthetic, prevalence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    cuts = list(dict.fromkeys(tab["cutoff"]))
    x = np.arange(len(cuts))
    for name, g in tab.groupby("model", sort=False):
        g = g.set_index("cutoff").reindex(cuts)
        style = "--" if name == "majority" else "-o"
        a1.plot(x, g["balanced_accuracy"], style, ms=4, label=f"{name}")
        if name != "majority":
            lo = [c[0] if isinstance(c, (list, tuple)) else np.nan for c in g["balanced_accuracy_ci"]]
            hi = [c[1] if isinstance(c, (list, tuple)) else np.nan for c in g["balanced_accuracy_ci"]]
            a1.fill_between(x, lo, hi, alpha=0.15)
    g0 = tab[tab["model"] == tab["model"].iloc[0]].set_index("cutoff").reindex(cuts)
    a2.plot(x, g0["harvest_seen_silage"], "-o", ms=4, color="#2a9d8f", label="harvest seen: silage")
    a2.plot(x, g0["harvest_seen_grain"], "-o", ms=4, color="#e9a23b", label="harvest seen: grain")
    a2.plot(x, g0["confident_share"], "-s", ms=4, color="k", label="confident call possible")
    for a in (a1, a2):
        a.set_xticks(x)
        a.set_xticklabels(cuts, rotation=45, fontsize=8)
        a.set_ylim(0, 1.02)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    a1.set_ylabel("balanced accuracy (held-out blocks)")
    a1.set_title("accuracy using only data up to the date", fontsize=9)
    a2.set_ylabel("share of fields")
    a2.set_title("what can be called by then", fontsize=9)
    fig.suptitle("How early can we know?" + (" (SYNTHETIC - software check only)" if synthetic else ""),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _report(tab, out, args, synthetic, n_fields, n_blocks):
    L = [f"# How early can we know? (as-of-date evaluation)\n"]
    if synthetic:
        L.append(SYNTHETIC_BANNER)
    L.append(f"{dt.datetime.now():%Y-%m-%d %H:%M}. {n_fields} labelled fields in {n_blocks} blocks, "
             f"{args.n_splits}-fold spatial CV grouped by block, the same folds at every cutoff. "
             "At each cutoff the time series are cut at that day of each field's season year, "
             "features are rebuilt and the models retrained, so nothing after the cutoff is used. "
             f"Balanced accuracy with a 95% block-bootstrap interval; precision also re-weighted to "
             f"{args.prevalence:.0%} silage.\n")
    L.append("'confident' = not low evidence (enough dates, a harvest drop seen and confirmed, "
             "interval narrow enough). Before a field is cut, the honest answer is 'not yet'.\n")
    for name in tab["model"].unique():
        g = tab[tab["model"] == name]
        L.append(f"## {name}\n")
        L.append(kv_table([(r["cutoff"], fmt(r["harvest_seen_silage"], 2), fmt(r["harvest_seen_grain"], 2),
                            fmt(r["confident_share"], 2),
                            f"{fmt(r['balanced_accuracy'])} [{fmt(r['balanced_accuracy_ci'][0], 2)}, "
                            f"{fmt(r['balanced_accuracy_ci'][1], 2)}]",
                            fmt(r["roc_auc"]), fmt(r["recall"]), fmt(r["fpr"]),
                            fmt(r[f"precision_at_{args.prevalence:g}"]),
                            fmt(r["confident_balanced_accuracy"]), fmt(r["confident_recall"]),
                            fmt(r["confident_fpr"]))
                           for r in g.to_dict("records")],
                          ["cutoff", "cut seen (silage)", "cut seen (grain)", "confident share",
                           "bal. acc. [CI]", "ROC-AUC", "recall", "FPR",
                           f"precision @{args.prevalence:.0%}", "confident bal. acc.",
                           "confident recall", "confident FPR"]))
        L.append("")
    L.append("Figure: `asof.png`.\n")
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
