"""Cross-validation, summaries, permutation importance and shift checks.

Everything that learns (thresholds, imputation, scaling, trees) happens in
model.fit() on the training part of a split. The test part is only ever
passed to predict_proba().
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..features.season import feature_group
from .metrics import ALL_METRICS, block_bootstrap_ci, classification_metrics, paired_bootstrap_diff

META_COLS = ["field_id", "block_id", "country", "year", "low_evidence"]


def run_splits(X, meta, y, factories, splits, scheme, variant, importance=None, seed=0):
    """Fit every model on each training part and predict the matching test part.

    X          feature table (rows aligned with meta and y)
    meta       frame with META_COLS
    y          1 silage / 0 grain, finite for every row
    factories  name -> zero-argument constructor
    splits     list of (fold_name, train_idx, test_idx)
    importance None, or dict model name -> list of (label, [columns]) to permute
               on each test fold

    Returns (predictions, fold_info, importance_records).
    """
    y = np.asarray(y).astype(int)
    rows, info, imp = [], [], []
    rng = np.random.default_rng(seed)
    for fold, tr, te in splits:
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        fold_rec = {"scheme": scheme, "variant": variant, "fold": str(fold),
                    "n_train": int(len(tr)), "n_test": int(len(te)),
                    "n_train_silage": int(y[tr].sum()), "n_test_silage": int(y[te].sum()),
                    "n_test_blocks": int(meta["block_id"].iloc[te].nunique()), "models": {}}
        for name, make in factories.items():
            t0 = time.perf_counter()
            model = make().fit(Xtr, y[tr])
            fit_s = time.perf_counter() - t0
            p = model.predict_proba(Xte)
            part = meta.iloc[te][META_COLS].copy()
            part["scheme"] = scheme
            part["variant"] = variant
            part["fold"] = str(fold)
            part["model"] = name
            part["y"] = y[te]
            part["p"] = p
            part["threshold"] = model.threshold
            part["pred"] = (p >= model.threshold).astype(int)
            rows.append(part)
            fold_rec["models"][name] = {"describe": model.describe(), "fit_seconds": fit_s}
            if importance and name in importance and len(np.unique(y[te])) == 2:
                imp.extend(permutation_drop(model, Xte, y[te], importance[name], rng,
                                            extra={"scheme": scheme, "variant": variant,
                                                   "fold": str(fold), "model": name}))
        info.append(fold_rec)
    preds = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return preds, info, imp


def permutation_drop(model, X, y, groups, rng, n_repeats=3, extra=None):
    """Fall in ROC-AUC when a column (or a group of columns jointly) is shuffled.

    Done on held-out data so it measures what the model uses to generalise,
    not what it memorised.
    """
    from .metrics import weighted_auc
    base_auc, _ = weighted_auc(y, model.predict_proba(X))
    out = []
    for label, cols in groups:
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            auc, _ = weighted_auc(y, model.predict_proba(Xp))
            drops.append(base_auc - auc)
        out.append({**(extra or {}), "feature": label, "n_columns": len(cols),
                    "base_auc": base_auc, "auc_drop": float(np.mean(drops))})
    return out


def importance_groups(feature_names, individual=True):
    """(label, columns) pairs: one per physical group and, optionally, one per column."""
    by_group = {}
    for c in feature_names:
        by_group.setdefault("group:" + feature_group(c), []).append(c)
    pairs = sorted(by_group.items())
    if individual:
        pairs += [(c, [c]) for c in feature_names]
    return pairs


def _metric_block(y, pred, p, blocks, n_boot, seed):
    m = classification_metrics(y, pred, p)
    if len(y) and n_boot:
        m["ci95"] = block_bootstrap_ci(y, pred, p, blocks, n_boot=n_boot, seed=seed)
    return m


def summarize_predictions(preds, n_boot=1000, seed=0, by=("country", "year")):
    """Metrics per (scheme, variant, model): all fields, the confident subset,
    per fold and per group in `by`.

    The confident subset drops fields flagged low evidence; its coverage is
    the share of fields kept.
    """
    out = {}
    if preds.empty:
        return out
    for (scheme, variant, model), g in preds.groupby(["scheme", "variant", "model"], sort=False):
        y = g["y"].to_numpy()
        pred = g["pred"].to_numpy()
        p = g["p"].to_numpy()
        blocks = g["block_id"].to_numpy()
        rec = {"all": _metric_block(y, pred, p, blocks, n_boot, seed)}
        conf = ~g["low_evidence"].to_numpy(dtype=bool)
        rec["confident"] = _metric_block(y[conf], pred[conf], p[conf], blocks[conf], n_boot, seed)
        rec["confident"]["coverage"] = float(conf.mean()) if len(conf) else float("nan")
        rec["per_fold"] = {str(f): classification_metrics(gg["y"], gg["pred"], gg["p"])
                           for f, gg in g.groupby("fold", sort=False)}
        for col in by:
            if col in g.columns and g[col].nunique() > 1:
                rec[f"per_{col}"] = {str(k): classification_metrics(gg["y"], gg["pred"], gg["p"])
                                     for k, gg in g.groupby(col, sort=True)}
        out.setdefault(scheme, {}).setdefault(variant, {})[model] = rec
    return out


def compare_models(preds, a, b, scheme, variant_a, variant_b=None, n_boot=1000, seed=0):
    """Paired block-bootstrap difference of model a minus model b on the same fields."""
    variant_b = variant_b or variant_a
    pa = preds[(preds.scheme == scheme) & (preds.variant == variant_a) & (preds.model == a)]
    pb = preds[(preds.scheme == scheme) & (preds.variant == variant_b) & (preds.model == b)]
    m = pa.merge(pb[["field_id", "p", "pred"]], on="field_id", suffixes=("_a", "_b"))
    if m.empty:
        return None
    return paired_bootstrap_diff(m["y"], m["pred_a"], m["p_a"], m["pred_b"], m["p_b"],
                                 m["block_id"], n_boot=n_boot, seed=seed)


def summarize_importance(records):
    if not records:
        return pd.DataFrame(columns=["model", "feature", "auc_drop_mean", "auc_drop_std", "n_folds"])
    df = pd.DataFrame(records)
    agg = (df.groupby(["model", "feature"])["auc_drop"]
           .agg(auc_drop_mean="mean", auc_drop_std="std", n_folds="count").reset_index())
    return agg.sort_values(["model", "auc_drop_mean"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------
# does a feature shift with the imagery collection, within one crop class?
# --------------------------------------------------------------------------

def field_collection(ts):
    """Most common `collection` among a field's rows with usable bands, per field."""
    if "collection" not in ts.columns or ts.empty:
        return pd.Series(dtype=object)
    ok = ts["B04"].notna()
    sub = ts.loc[ok, ["field_id", "collection"]].copy()
    sub["collection"] = sub["collection"].astype(str)
    counts = sub.groupby(["field_id", "collection"]).size().reset_index(name="n")
    counts = counts.sort_values(["field_id", "n"], ascending=[True, False])
    top = counts.drop_duplicates("field_id").set_index("field_id")
    n_coll = counts.groupby("field_id").size()
    out = top["collection"].where(n_coll.reindex(top.index) == 1, top["collection"] + "+mixed")
    return out


def collection_shift(X, meta, y, columns, collection, min_n=20):
    """Standardised mean difference and KS statistic between collections, per class.

    Compares, for silage and grain separately, each pair of collections
    with at least `min_n` fields. A large difference within the same crop
    class means the feature moves with the processing, not the harvest.
    Year and country are usually confounded with collection; the table
    reports both so nobody reads too much into it.
    """
    from scipy.stats import ks_2samp
    rows = []
    coll = pd.Series(np.asarray(collection, dtype=object), index=X.index).fillna("unknown")
    y = pd.Series(np.asarray(y), index=X.index)
    for cls_val, cls in ((1, "silage"), (0, "grain")):
        in_cls = y == cls_val
        counts = coll[in_cls].value_counts()
        names = [c for c, n in counts.items() if n >= min_n]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = in_cls & (coll == names[i])
                b = in_cls & (coll == names[j])
                years_a = sorted(map(str, meta.loc[a, "year"].dropna().unique()))
                years_b = sorted(map(str, meta.loc[b, "year"].dropna().unique()))
                for c in columns:
                    va = X.loc[a, c].to_numpy(dtype=float)
                    vb = X.loc[b, c].to_numpy(dtype=float)
                    va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                    if len(va) < min_n or len(vb) < min_n:
                        continue
                    sd = np.sqrt(0.5 * (va.var(ddof=1) + vb.var(ddof=1)))
                    smd = (va.mean() - vb.mean()) / sd if sd > 0 else np.nan
                    ks = ks_2samp(va, vb)
                    rows.append({"class": cls, "collection_a": names[i], "collection_b": names[j],
                                 "years_a": ",".join(years_a), "years_b": ",".join(years_b),
                                 "feature": c, "n_a": len(va), "n_b": len(vb),
                                 "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                                 "smd": float(smd), "ks_stat": float(ks.statistic),
                                 "ks_p": float(ks.pvalue)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["abs_smd"] = df["smd"].abs()
        df = df.sort_values("abs_smd", ascending=False).reset_index(drop=True)
    return df


def response_time(build_fn, predict_fn, n_fields, repeats=1):
    """Seconds for features and for prediction, scaled to 1,000 fields."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        feats = build_fn()
    t1 = time.perf_counter()
    for _ in range(repeats):
        predict_fn(feats)
    t2 = time.perf_counter()
    scale = 1000.0 / max(n_fields, 1) / repeats
    return {"n_fields": int(n_fields),
            "features_seconds_per_1000": (t1 - t0) * scale,
            "predict_seconds_per_1000": (t2 - t1) * scale,
            "total_seconds_per_1000": (t2 - t0) * scale}


def headline(summary, scheme, variant, model, subset="all"):
    """Flat dict of the main numbers with their intervals, for tables."""
    try:
        m = summary[scheme][variant][model][subset]
    except KeyError:
        return None
    row = {"n": m["n"], "n_silage": m["n_silage"]}
    for k in ["precision", "recall", "fpr", "f1", "balanced_accuracy", "roc_auc", "pr_auc"]:
        row[k] = m[k]
        if "ci95" in m:
            row[k + "_ci"] = m["ci95"][k]
    if "coverage" in m:
        row["coverage"] = m["coverage"]
    return row


__all__ = ["run_splits", "summarize_predictions", "compare_models", "permutation_drop",
           "importance_groups", "summarize_importance", "field_collection", "collection_shift",
           "response_time", "headline", "ALL_METRICS"]


def coverage_table(preds_model, X, cfg, limits=(10, 15, 20, 30, 45, None)):
    """How precision/recall/FPR change as the harvest-interval limit for abstention moves.

    preds_model: held-out predictions of one model (field_id, y, pred, p).
    The other low-evidence criteria stay as configured. Read it to choose
    the limit on development data, never on a holdout or external set.
    """
    from dataclasses import replace
    from ..features.season import evidence_flags
    from .metrics import classification_metrics
    f = X.set_index("field_id").loc[preds_model["field_id"]]
    rows = []
    for lim in limits:
        c = replace(cfg, low_evidence_max_interval_days=float("inf") if lim is None else float(lim))
        low, _ = evidence_flags(f["q_n_usable_season"], f["harv_mid_doy"],
                                f["q_harv_width_days"], f["q_harv_confirmed"], c)
        keep = ~low
        m = classification_metrics(preds_model["y"].to_numpy()[keep],
                                   preds_model["pred"].to_numpy()[keep],
                                   preds_model["p"].to_numpy()[keep])
        rows.append({"max_interval_days": "none" if lim is None else lim,
                     "coverage": float(keep.mean()), "n": m["n"], "precision": m["precision"],
                     "recall": m["recall"], "fpr": m["fpr"], "roc_auc": m["roc_auc"]})
    m = classification_metrics(preds_model["y"], preds_model["pred"], preds_model["p"])
    rows.append({"max_interval_days": "no abstention", "coverage": 1.0, "n": m["n"],
                 "precision": m["precision"], "recall": m["recall"], "fpr": m["fpr"],
                 "roc_auc": m["roc_auc"]})
    return pd.DataFrame(rows)
