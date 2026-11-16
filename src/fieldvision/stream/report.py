"""Hold-out table and figure for the stream models, next to the reference models.

    python -m fieldvision.stream.report --stream <train output dir ...>
        --reference <reference run>/predictions.parquet --out <dir>
        [--retrain-reference --timeseries <parquet ...> --fields <geojson ...>]

For every held-out country: ROC-AUC out of country, the same fields'
ROC-AUC when their country was in training (spatial folds), the loss
between the two, and the paired difference to the 6-feature logistic
with a block-bootstrap 95 % interval, all on the fields the stream
models were scored on. Reference rows come from the reference run
(trained on all its fields); with `--retrain-reference` the logistic and
the boosted model are also retrained on exactly the stream models'
training fields, which is the like-for-like row when the stream models
only had a subset.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..model.harness import compare_models, run_splits
from ..model.metrics import classification_metrics

REF = "phys-logistic"
REFERENCE_MODELS = ["phys-logistic", "phys-gbm", "raw-gbm", "emb-logistic"]
LABELS = {"phys-logistic": "6-feature logistic", "phys-gbm": "boosted trees, 135 features",
          "raw-gbm": "boosted trees, monthly band values", "emb-logistic": "pretrained encoder (Presto)",
          "phys-logistic@same": "6-feature logistic, same training fields",
          "phys-gbm@same": "boosted trees, same training fields",
          "seq-rel": "sequence of band medians, days since green-up",
          "seq-cal": "sequence of band medians, calendar days",
          "pixelset-rel": "pixel sets, days since green-up", "pixelset-cal": "pixel sets, calendar days",
          "chipseq-rel": "chips + 2D conv, days since green-up", "chipseq-cal": "chips + 2D conv, calendar days",
          "conv3d-rel": "chips + 3D conv, grid from green-up", "conv3d-cal": "chips + 3D conv, calendar grid"}
COUNTRY = {"AT": "Austria", "FR": "France", "NL": "the Netherlands"}


def load_stream(paths):
    frames = []
    for p in paths:
        p = Path(p)
        frames.append(pd.read_parquet(p / "predictions.parquet" if p.is_dir() else p))
    preds = pd.concat(frames, ignore_index=True)
    preds["field_id"] = preds["field_id"].astype(str)
    preds["year"] = preds["year"].astype(str)
    return preds.drop_duplicates(["scheme", "model", "field_id"], keep="last")


def align_reference(stream, reference, models=REFERENCE_MODELS):
    """Reference predictions for the fields and schemes the stream models were scored on."""
    parts = []
    for scheme, g in stream.groupby("scheme"):
        ids = set(g["field_id"])
        r = reference[(reference["scheme"] == scheme) & reference["model"].isin(models)
                      & reference["field_id"].astype(str).isin(ids)]
        parts.append(r)
    ref = pd.concat(parts, ignore_index=True) if parts else reference.iloc[:0]
    ref = ref.copy()
    ref["field_id"] = ref["field_id"].astype(str)
    ref["year"] = ref["year"].astype(str)
    return ref


def retrain_reference(stream, timeseries, fields, seed=0):
    """6-feature logistic and boosted model trained on the stream models' own training fields."""
    from ..features.build import build_features
    from ..features.loading import load_fields, load_timeseries
    from ..features.season import FeatureConfig
    from ..model.models import make_factories
    one = stream[stream["model"] == stream["model"].iloc[0]]
    sp = one[one["scheme"] == "spatial"].drop_duplicates("field_id")
    ids = sorted(set(one["field_id"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = load_fields(fields)
        f = f[f["field_id"].isin(ids)].reset_index(drop=True)
        ts, _ = load_timeseries(timeseries, field_ids=f["field_id"])
        X, _ = build_features(ts, f, FeatureConfig())
    meta = f.copy()
    meta["year"] = meta["year"].map(lambda v: str(int(v)))
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    y = meta["y"].to_numpy(int)
    main = make_factories(X.columns, "relative", seed=seed)
    facs = {"phys-logistic@same": main["logistic"], "phys-gbm@same": main["gbm"]}
    X = X.reset_index(drop=True)
    X["block_id"] = meta["block_id"].to_numpy()
    out = []
    fold_of = sp.set_index("field_id")["fold"].reindex(meta["field_id"]).to_numpy()
    if len(sp):
        splits = [(k, np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k))
                  for k in sorted(pd.unique(fold_of[pd.notna(fold_of)]))]
        out.append(run_splits(X, meta, y, facs, splits, "spatial", "relative", seed=seed)[0])
    country = meta["country"].astype(str).to_numpy()
    for c in sorted(set(country)):
        if f"loco:{c}" in set(stream["scheme"]):
            split = [(c, np.flatnonzero(country != c), np.flatnonzero(country == c))]
            out.append(run_splits(X, meta, y, facs, split, f"loco:{c}", "relative", seed=seed)[0])
    return pd.concat(out, ignore_index=True)


def seed_spread(g):
    cols = [c for c in g.columns if c.startswith("p_seed") and g[c].notna().all()]
    if len(cols) < 2:
        return float("nan"), len(cols)
    aucs = [classification_metrics(g["y"], (g[c] >= 0.5).astype(int), g[c])["roc_auc"] for c in cols]
    return float(np.std(aucs, ddof=1)), len(cols)


def transfer_table(preds, models, ref=REF, n_boot=1000, seed=0):
    """One row per (held-out country, model); same definitions as the reference comparison."""
    rows = []
    for scheme in sorted(s for s in preds["scheme"].unique() if s.startswith("loco:")):
        out = preds[preds["scheme"] == scheme]
        ids = set(out["field_id"])
        sp = preds[(preds["scheme"] == "spatial") & preds["field_id"].isin(ids)]
        for m in models:
            g = out[out["model"] == m]
            if g.empty:
                continue
            mo = classification_metrics(g["y"], g["pred"], g["p"])
            gi = sp[sp["model"] == m]
            auc_in = classification_metrics(gi["y"], gi["pred"], gi["p"])["roc_auc"] if len(gi) else np.nan
            sd, n_seeds = seed_spread(g)
            row = {"scheme": scheme, "country": scheme.split(":")[1], "model": m, "n": mo["n"],
                   "auc_out": mo["roc_auc"], "auc_in": auc_in, "auc_loss": auc_in - mo["roc_auc"],
                   "recall": mo["recall"], "fpr": mo["fpr"], "seed_sd": sd, "n_seeds": n_seeds,
                   "diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
            if m != ref:
                c = compare_models(preds, m, ref, scheme, "relative", n_boot=n_boot, seed=seed)
                if c is not None:
                    row["diff"], (row["ci_lo"], row["ci_hi"]) = c["roc_auc"]["diff"], c["roc_auc"]["ci"]
            rows.append(row)
    return pd.DataFrame(rows)


def markdown(table, models):
    countries = [c for c in ("AT", "FR", "NL") if c in set(table["country"])]
    head = "| model | " + " | ".join(f"test {COUNTRY.get(c, c)} | loss" for c in countries) + " | mean |"
    lines = [head, "|---|" + "---|---:|" * len(countries) + "---:|"]
    for m in models:
        t = table[table["model"] == m].set_index("country")
        if t.empty:
            continue
        cells = []
        for c in countries:
            if c not in t.index:
                cells += ["n/a", "n/a"]
                continue
            r = t.loc[c]
            cell = (f"{r['auc_out']:.3f} (ref)" if m == REF else
                    f"{r['auc_out']:.3f}, {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
            cells += [cell, "n/a" if not np.isfinite(r["auc_loss"]) else f"{r['auc_loss']:.3f}"]
        lines.append(f"| {LABELS.get(m, m)} (`{m}`) | " + " | ".join(cells) + f" | {t['auc_out'].mean():.3f} |")
    return "\n".join(lines)


def figure(table, models, path, title):
    """Held-out ROC-AUC per country for every model, with the mean as a bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in models if m in set(table["model"])]
    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    colours = {"AT": "#2a78d6", "FR": "#eb6834", "NL": "#1baf7a"}   # fixed order, one hue per country
    marks = {"AT": "o", "FR": "s", "NL": "^"}                        # shape repeats what colour says
    fig, ax = plt.subplots(figsize=(9.6, 0.46 * len(models) + 1.7), dpi=170, facecolor=surface)
    ax.set_facecolor(surface)
    means = table.groupby("model")["auc_out"].mean()
    y = np.arange(len(models))[::-1]
    ax.barh(y, [means[m] - 0.5 for m in models], left=0.5, height=0.34, color="#d9d8d3", zorder=1)
    for c in ("AT", "FR", "NL"):
        t = table[table["country"] == c].set_index("model")
        if t.empty:
            continue
        ax.scatter([t.loc[m, "auc_out"] for m in models], y, marker=marks[c], s=64, color=colours[c],
                   edgecolor=surface, linewidth=1.4, zorder=3, label=f"held out: {COUNTRY[c]}")
    if REF in means.index:
        ax.axvline(means[REF], color=muted, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    for yi, m in zip(y, models):
        ax.text(0.503, yi + 0.33, f"mean {means[m]:.3f}", va="center", ha="left", fontsize=7.5, color=muted)
    ax.set_yticks(y, [LABELS.get(m, m) for m in models], fontsize=9, color=ink)
    ax.set_ylim(-0.6, len(models) - 0.25)
    ax.set_xlim(0.5, max(0.9, float(table["auc_out"].max()) + 0.03))
    ax.tick_params(axis="both", length=0, labelcolor=ink)
    ax.tick_params(axis="x", labelsize=8, labelcolor=muted)
    ax.set_xlabel("ROC-AUC on the country that was held out\n"
                  "grey bar: mean of the three countries; dashed line: the 6-feature logistic's mean",
                  fontsize=8.5, color=muted)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", va="top", fontsize=11.5, color=ink)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
              borderaxespad=0.1, handletextpad=0.2, columnspacing=1.6, labelcolor=ink)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e9e8e4", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream", nargs="+", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--retrain-reference", action="store_true")
    ap.add_argument("--timeseries", nargs="*", default=[])
    ap.add_argument("--fields", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--title", default="Each country held out in turn")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stream = load_stream(args.stream)
    ref = align_reference(stream, pd.read_parquet(args.reference))
    parts = [ref, stream]
    if args.retrain_reference:
        parts.insert(1, retrain_reference(stream, args.timeseries, args.fields))
    preds = pd.concat(parts, ignore_index=True)
    order = ([m for m in REFERENCE_MODELS if m in set(preds["model"])]
             + [m for m in ("phys-logistic@same", "phys-gbm@same") if m in set(preds["model"])]
             + sorted(set(stream["model"]), key=lambda m: (list(LABELS).index(m) if m in LABELS else 99, m)))
    table = transfer_table(preds, order, n_boot=args.n_boot)
    table.to_csv(out / "stream_table.csv", index=False)
    md = markdown(table, order)
    n = {c: int(table[(table["country"] == c)]["n"].iloc[0]) for c in table["country"].unique()}
    (out / "REPORT.md").write_text(f"# Stream models, each country held out\n\nFields scored: {n}\n\n{md}\n",
                                   encoding="utf-8")
    figure(table, order, out / "stream_holdout.png", args.title)
    (out / "summary.json").write_text(json.dumps(
        {"fields_scored": n, "models": order,
         "mean_auc_out": table.groupby("model")["auc_out"].mean().to_dict()}, indent=1), encoding="utf-8")
    print(md)
    return table


if __name__ == "__main__":
    main()
