"""Binary classification metrics with silage as the positive class.

Undefined ratios (for example precision when nothing was predicted
positive) are NaN, not 0, so a table never shows a number that was not
measured.
"""
from __future__ import annotations

import numpy as np

COUNT_METRICS = ["precision", "recall", "fpr", "f1", "accuracy", "balanced_accuracy",
                 "grain_precision", "grain_recall"]
SCORE_METRICS = ["roc_auc", "pr_auc"]
ALL_METRICS = COUNT_METRICS + SCORE_METRICS


def confusion_counts(y_true, y_pred, weights=None):
    """tp, fp, fn, tn with 1 = silage (weighted sums if weights are given)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if weights is None:
        return {
            "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
            "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
            "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        }
    w = np.asarray(weights, dtype=float)
    return {
        "tp": float(w[(y_true == 1) & (y_pred == 1)].sum()),
        "fp": float(w[(y_true == 0) & (y_pred == 1)].sum()),
        "fn": float(w[(y_true == 1) & (y_pred == 0)].sum()),
        "tn": float(w[(y_true == 0) & (y_pred == 0)].sum()),
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
        # the same with grain as the positive class
        "grain_precision": _div(tn, tn + fn),
        "grain_recall": specificity,
    }


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval for k successes out of n (NaN when n == 0)."""
    from scipy.stats import beta
    if n <= 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if k <= 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def exact_intervals(confusion, alpha=0.05):
    """Clopper-Pearson intervals for the count metrics of one confusion matrix.

    Treats fields as independent, which fields on one farm are not; with a
    few dozen fields it is still the most honest single number to show.
    """
    tp, fp, fn, tn = (int(round(confusion[k])) for k in ("tp", "fp", "fn", "tn"))
    return {
        "precision": clopper_pearson(tp, tp + fp, alpha),
        "recall": clopper_pearson(tp, tp + fn, alpha),
        "fpr": clopper_pearson(fp, fp + tn, alpha),
        "grain_precision": clopper_pearson(tn, tn + fn, alpha),
        "grain_recall": clopper_pearson(tn, tn + fp, alpha),
        "accuracy": clopper_pearson(tp + tn, tp + fp + fn + tn, alpha),
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


def classification_metrics(y_true, y_pred, y_score=None, weights=None):
    """All metrics for one set of predictions, as plain floats and ints.

    weights: optional row weights, e.g. from prevalence_weights().
    """
    y_true = np.asarray(y_true).astype(int)
    counts = confusion_counts(y_true, y_pred, weights)
    out = {k: float(v) for k, v in metrics_from_counts(**counts).items()}
    if y_score is not None and len(y_true):
        out["roc_auc"], out["pr_auc"] = weighted_auc(y_true, y_score, weights)
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


def _boot_metrics(y, pred, score, mult, inverse, chunk=200, weights=None):
    k = mult.shape[1]
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    counts = np.zeros((k, 4))
    for j, m in enumerate(((y == 1) & (pred == 1), (y == 0) & (pred == 1),
                           (y == 1) & (pred == 0), (y == 0) & (pred == 0))):
        counts[:, j] = np.bincount(inverse, weights=m.astype(float) * w, minlength=k)
    boot = mult @ counts
    out = metrics_from_counts(boot[:, 0], boot[:, 1], boot[:, 2], boot[:, 3])
    if score is not None:
        roc, ap = [], []
        for a in range(0, mult.shape[0], chunk):
            r, p = weighted_auc(y, score, mult[a:a + chunk][:, inverse] * w[None, :])
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


def block_bootstrap_ci(y_true, y_pred, y_score, blocks, n_boot=1000, seed=0, alpha=0.05,
                       weights=None):
    """Percentile confidence intervals from resampling blocks, not fields.

    Fields in one 10 km block share weather and satellite passes, so they
    are not independent draws; resampling blocks keeps that dependence.
    Returns {metric: [lo, hi]} plus the number of blocks.
    """
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    score = None if y_score is None else np.asarray(y_score, dtype=float)
    mult, inverse = _block_weights(blocks, n_boot, seed)
    boot = _boot_metrics(y, pred, score, mult, inverse, weights=weights)
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


# --------------------------------------------------------------------------
# realistic class shares and calibration
# --------------------------------------------------------------------------

def prevalence_weights(y_true, prevalence):
    """Row weights that make silage `prevalence` of the total weight.

    Our training sets are balanced by design; in the US about 6-8 % of corn
    is silage. Recall and FPR do not change under these weights; precision,
    F1, accuracy and PR-AUC do.
    """
    y = np.asarray(y_true).astype(int)
    p_obs = y.mean() if len(y) else np.nan
    if not (0 < p_obs < 1):
        return np.ones(len(y))
    return np.where(y == 1, prevalence / p_obs, (1 - prevalence) / (1 - p_obs))


def prior_shift(p, prevalence, train_prevalence):
    """Move a probability calibrated at `train_prevalence` to another class share (Bayes)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    a = p * prevalence / train_prevalence
    b = (1 - p) * (1 - prevalence) / (1 - train_prevalence)
    return a / (a + b)


def shifted_threshold(prevalence, train_prevalence, threshold=0.5):
    """Score threshold equivalent to `threshold` on prior-shifted probabilities."""
    r = (prevalence / train_prevalence) / ((1 - prevalence) / (1 - train_prevalence))
    t = threshold / (1 - threshold)
    return float(t / (t + r))


def calibration_summary(y_true, p, n_bins=10):
    """Brier score, expected calibration error and a reliability table (equal-width bins)."""
    y = np.asarray(y_true).astype(float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_p": float(p[m].mean()), "share_silage": float(y[m].mean())})
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
            "ece": float(ece), "reliability": rows}
