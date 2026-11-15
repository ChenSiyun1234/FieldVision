"""Local threshold: keep a classifier's direction, learn the cut-off from local grain.

A distance-to-normal score does not separate silage from grain (docs/anomaly-view.md):
silage is a shifted part of the grain distribution, not outside it, and a two-sided
distance flags late, double-cropped or cloudy grain just as readily. What does carry
over from the "learn only normal" idea is using a local, mostly-grain reference set to
set the threshold, while the direction (which way is silage) comes from the labelled
source data.

Recipe, fixed before looking at any target county:
  1. score the reference fields (mostly grain, no labels read) with the source classifier;
     threshold t = the (1 - alpha) quantile of those scores, alpha = 0.10.
     So about alpha of local grain is flagged, whatever the classifier's calibration.
  2. on the labelled source data, with out-of-fold scores, find the score s_t that flags
     the same share alpha of source grain, and the share of source silage above it:
     tpr. This assumes the separation (not the location) of the two classes carries over.
  3. per field: silage call = score >= t.
     county share: adjusted count (flag_rate - fpr) / (tpr - fpr), clipped to [0, 1],
     with fpr = the reference's own flag rate (about alpha).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ALPHA = 0.10


def matched_tpr(source_scores, source_y, alpha=DEFAULT_ALPHA):
    """Share of source silage above the score that flags `alpha` of source grain."""
    s = np.asarray(source_scores, dtype=float)
    y = np.asarray(source_y).astype(int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError("source needs both classes")
    t_s = float(np.quantile(s[y == 0], 1 - alpha))
    return t_s, float((s[y == 1] >= t_s).mean())


def adjusted_share(flag_rate, fpr, tpr):
    """Adjusted classify-and-count; nan when tpr <= fpr (no separation to correct with)."""
    if not np.isfinite(tpr - fpr) or tpr - fpr <= 1e-9:
        return float("nan")
    return float(np.clip((flag_rate - fpr) / (tpr - fpr), 0.0, 1.0))


class LocalThreshold:
    """Fit on classifier scores; nothing here reads target labels.

    Pass reference and source scores from the same kind of field the target share will use
    (fields that are not low evidence): source scores of low-evidence fields understate the
    matched recall and overstate the adjusted share. NaN scores are ignored.
    """

    def __init__(self, alpha=DEFAULT_ALPHA):
        self.alpha = alpha

    def fit(self, reference_scores, source_scores, source_y):
        r = np.asarray(reference_scores, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 50:
            raise ValueError(f"reference has {len(r)} scored fields; need at least 50")
        self.threshold = float(np.quantile(r, 1 - self.alpha))
        self.reference_flag_rate = float((r >= self.threshold).mean())
        self.source_threshold, self.tpr = matched_tpr(source_scores, source_y, self.alpha)
        self.n_reference = int(len(r))
        return self

    def calls(self, scores):
        s = np.asarray(scores, dtype=float)
        out = (s >= self.threshold).astype(float)
        out[~np.isfinite(s)] = np.nan
        return out

    def county_share(self, scores):
        """(raw flagged share, adjusted share) over the fields with a score."""
        c = self.calls(scores)
        c = c[np.isfinite(c)]
        flag = float(c.mean()) if len(c) else float("nan")
        return flag, adjusted_share(flag, self.reference_flag_rate, self.tpr)

    def describe(self):
        return (f"local threshold {self.threshold:.3f} = reference's {1 - self.alpha:.0%} quantile "
                f"({self.n_reference} fields); source tpr at matched fpr {self.alpha:.0%}: {self.tpr:.3f}")


def _wmean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() and np.sum(w[ok]) > 0 else float("nan")


def local_share(ref_p, src_p, src_y, tgt_p, alpha=DEFAULT_ALPHA, ref_w=None, tgt_w=None, adjust=True):
    """One county-wide local threshold; shares weighted by `*_w` (e.g. area) when given."""
    ref_p = np.asarray(ref_p, dtype=float)
    t = float(np.nanquantile(ref_p, 1 - alpha))
    ref_flag = _wmean(ref_p >= t, ref_w)
    flag = _wmean(np.asarray(tgt_p, dtype=float) >= t, tgt_w)
    if not adjust:
        return flag
    _, tpr = matched_tpr(src_p, src_y, alpha)
    return adjusted_share(flag, ref_flag, tpr)


def stratified_share(ref_p, ref_s, src_p, src_y, src_s, tgt_p, tgt_s, alpha=DEFAULT_ALPHA,
                     ref_w=None, tgt_w=None, min_ref=30):
    """Local threshold per stratum of evidence quality (threshold, reference flagged share and
    matched recall all within the stratum), combined by the target's weight per stratum.
    A stratum with fewer than `min_ref` reference fields uses the county-wide threshold."""
    ref_p, ref_s = np.asarray(ref_p, float), np.asarray(ref_s)
    src_p, src_y, src_s = np.asarray(src_p, float), np.asarray(src_y).astype(int), np.asarray(src_s)
    tgt_p, tgt_s = np.asarray(tgt_p, float), np.asarray(tgt_s)
    rw = np.ones(len(ref_p)) if ref_w is None else np.asarray(ref_w, float)
    tw = np.ones(len(tgt_p)) if tgt_w is None else np.asarray(tgt_w, float)
    est, wsum = 0.0, 0.0
    for k in np.unique(tgt_s):
        tk = tgt_s == k
        rk, sk = ref_s == k, src_s == k
        fin = np.isfinite(src_p)
        if rk.sum() < min_ref or (fin & sk & (src_y == 0)).sum() < min_ref or (fin & sk & (src_y == 1)).sum() < min_ref:
            rk, sk = np.ones(len(ref_p), bool), np.ones(len(src_p), bool)
        e = local_share(ref_p[rk], src_p[sk], src_y[sk], tgt_p[tk], alpha, rw[rk], tw[tk])
        if not np.isfinite(e):
            return float("nan")
        est += e * tw[tk].sum()
        wsum += tw[tk].sum()
    return est / wsum if wsum > 0 else float("nan")


def emulate_counties(target, source_scores, source_y, alpha=DEFAULT_ALPHA, ref_silage=0.013,
                     test_silage=0.20, n_rep=20, seed=0):
    """European check of the recipe on one held-out group (a year or a country).

    target  frame with field_id, block_id, y, p (classifier score from a model trained
            without this group) for every field of the held-out group
    Each repeat splits the group's blocks in two halves: a 'reference county' (its grain
    plus silage drawn to `ref_silage`, labels then forgotten) and a 'test county' (its
    grain plus silage drawn to `test_silage`). Reports, on the test county, the FPR and
    recall of the classifier at its own 0.5 cut-off versus the local threshold, and the
    true versus estimated silage share.
    """
    rng = np.random.default_rng(seed)
    blocks = target["block_id"].unique()
    rows = []
    for rep in range(n_rep):
        ref_b = set(rng.permutation(blocks)[: len(blocks) // 2])
        in_ref = target["block_id"].isin(ref_b).to_numpy()

        def county(mask, share):
            g = target[mask]
            grain, sil = g[g["y"] == 0], g[g["y"] == 1]
            n = int(round(share / (1 - share) * len(grain)))
            return pd.concat([grain, sil.sample(min(n, len(sil)), random_state=int(rng.integers(1 << 31)))])

        ref, test = county(in_ref, ref_silage), county(~in_ref, test_silage)
        lt = LocalThreshold(alpha).fit(ref["p"], source_scores, source_y)
        y, p = test["y"].to_numpy(), test["p"].to_numpy()
        call = lt.calls(p)
        fixed = (p >= 0.5).astype(float)
        flag, est = lt.county_share(p)
        rows.append({"rep": rep, "n_ref": len(ref), "n_test": len(test), "true_share": float(y.mean()),
                     "fixed_fpr": fixed[y == 0].mean(), "fixed_recall": fixed[y == 1].mean(),
                     "fixed_share": fixed.mean(),
                     "local_fpr": call[y == 0].mean(), "local_recall": call[y == 1].mean(),
                     "local_flag_share": flag, "local_adjusted_share": est,
                     "threshold": lt.threshold, "tpr_source": lt.tpr})
    return pd.DataFrame(rows)
