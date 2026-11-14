"""Binary classification metrics with silage as the positive class.

Undefined ratios (for example precision when nothing was predicted
positive) are NaN, not 0, so a table never shows a number that was not
measured.
"""
from __future__ import annotations

import numpy as np

COUNT_METRICS = ["precision", "recall", "fpr", "f1", "accuracy", "balanced_accuracy"]
SCORE_METRICS = ["roc_auc", "pr_auc"]
ALL_METRICS = COUNT_METRICS + SCORE_METRICS


def confusion_counts(y_true, y_pred):
    """tp, fp, fn, tn with 1 = silage."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def _div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def metrics_from_counts(tp, fp, fn, tn):
    """Threshold metrics from confusion counts. Works on scalars or arrays."""
    tp, fp, fn, tn = (np.asarray(v, dtype=float) for v in (tp, fp, fn, tn))
    recall = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    return {
        "precision": _div(tp, tp + fp),
        "recall": recall,
        "fpr": _div(fp, fp + tn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "accuracy": _div(tp + tn, tp + fp + fn + tn),
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def precision_at_prevalence(recall, fpr, prevalence):
    """Precision the same recall and FPR would give if silage made up `prevalence` of fields."""
    num = recall * prevalence
    return float(_div(num, num + fpr * (1 - prevalence)))


def _score_groups(score):
    """Descending order of scores and the start index of each group of tied scores."""
    order = np.argsort(-score, kind="stable")
    s = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    return order, starts


def weighted_auc(y_true, score, weights=None):
    """ROC-AUC and PR-AUC (average precision) with row weights.

    weights: None, shape (n,) or shape (B, n) for B weightings at once.
    Returns two arrays of length B (or two floats). NaN when a class has
    zero weight.
    """
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    single = weights is None or np.ndim(weights) == 1
    W = np.ones((1, len(y))) if weights is None else np.atleast_2d(np.asarray(weights, dtype=float))
    order, starts = _score_groups(score)
    Wo = W[:, order]
    yo = y[order]
    P = np.add.reduceat(Wo * yo, starts, axis=1)
    N = np.add.reduceat(Wo * (1 - yo), starts, axis=1)
    p_tot = P.sum(axis=1)
    n_tot = N.sum(axis=1)
    n_lower = n_tot[:, None] - np.cumsum(N, axis=1)
    roc = _div((P * (n_lower + 0.5 * N)).sum(axis=1), p_tot * n_tot)
    tp_cum = np.cumsum(P, axis=1)
    fp_cum = np.cumsum(N, axis=1)
    prec = _div(tp_cum, tp_cum + fp_cum)
    ap = _div(np.nansum(P * prec, axis=1), p_tot)
    ap = np.where(n_tot + p_tot > 0, ap, np.nan)
    if single:
        return float(roc[0]), float(ap[0])
    return roc, ap


def classification_metrics(y_true, y_pred, y_score=None):
    """All metrics for one set of predictions, as plain floats and ints."""
    y_true = np.asarray(y_true).astype(int)
    counts = confusion_counts(y_true, y_pred)
    out = {k: float(v) for k, v in metrics_from_counts(**counts).items()}
    if y_score is not None and len(y_true):
        out["roc_auc"], out["pr_auc"] = weighted_auc(y_true, y_score)
    else:
        out["roc_auc"], out["pr_auc"] = float("nan"), float("nan")
    out["confusion"] = counts
    out["n"] = int(len(y_true))
    out["n_silage"] = int((y_true == 1).sum())
    out["n_grain"] = int((y_true == 0).sum())
    out["prevalence"] = float(np.mean(y_true)) if len(y_true) else float("nan")
    return out


def _block_weights(blocks, n_boot, seed):
    """Resample whole blocks with replacement: (n_boot, n_blocks) multiplicities."""
    uniq, inverse = np.unique(np.asarray(blocks).astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    k = len(uniq)
    mult = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot)
    return mult, inverse


def _boot_metrics(y, pred, score, mult, inverse, chunk=200):
    k = mult.shape[1]
    counts = np.zeros((k, 4))
    for j, m in enumerate(((y == 1) & (pred == 1), (y == 0) & (pred == 1),
                           (y == 1) & (pred == 0), (y == 0) & (pred == 0))):
        counts[:, j] = np.bincount(inverse, weights=m.astype(float), minlength=k)
    boot = mult @ counts
    out = metrics_from_counts(boot[:, 0], boot[:, 1], boot[:, 2], boot[:, 3])
    if score is not None:
        roc, ap = [], []
        for a in range(0, mult.shape[0], chunk):
            r, p = weighted_auc(y, score, mult[a:a + chunk][:, inverse])
            roc.append(r)
            ap.append(p)
        out["roc_auc"] = np.concatenate(roc)
        out["pr_auc"] = np.concatenate(ap)
    return out


def _interval(samples, alpha):
    samples = np.asarray(samples, dtype=float)
    if not np.isfinite(samples).any():
        return [float("nan"), float("nan")]
    lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def block_bootstrap_ci(y_true, y_pred, y_score, blocks, n_boot=1000, seed=0, alpha=0.05):
    """Percentile confidence intervals from resampling blocks, not fields.

    Fields in one 10 km block share weather and satellite passes, so they
    are not independent draws; resampling blocks keeps that dependence.
    Returns {metric: [lo, hi]} plus the number of blocks.
    """
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    score = None if y_score is None else np.asarray(y_score, dtype=float)
    mult, inverse = _block_weights(blocks, n_boot, seed)
    boot = _boot_metrics(y, pred, score, mult, inverse)
    ci = {name: _interval(vals, alpha) for name, vals in boot.items()}
    ci["n_blocks"] = int(mult.shape[1])
    ci["n_boot"] = int(n_boot)
    return ci


def paired_bootstrap_diff(y_true, pred_a, score_a, pred_b, score_b, blocks,
                          n_boot=1000, seed=0, alpha=0.05):
    """Difference A - B of each metric with a block-bootstrap interval.

    Both sets of predictions are evaluated on the same resampled blocks.
    """
    y = np.asarray(y_true).astype(int)
    mult, inverse = _block_weights(blocks, n_boot, seed)
    a = _boot_metrics(y, np.asarray(pred_a).astype(int), np.asarray(score_a, dtype=float), mult, inverse)
    b = _boot_metrics(y, np.asarray(pred_b).astype(int), np.asarray(score_b, dtype=float), mult, inverse)
    point_a = classification_metrics(y, pred_a, score_a)
    point_b = classification_metrics(y, pred_b, score_b)
    out = {}
    for name in ALL_METRICS:
        out[name] = {"diff": float(point_a[name] - point_b[name]),
                     "ci": _interval(a[name] - b[name], alpha)}
    return out
