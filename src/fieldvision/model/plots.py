"""Diagnostic figures. Small PNGs, written next to the run, never into git."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..features.indices import BANDS, ndmi, ndvi  # noqa: E402
from ..features.season import usable_mask  # noqa: E402

COLORS = {"silage": "#2a9d8f", "grain": "#e9a23b"}
DIST_FEATURES = [
    ("harv_mid_doy", "harvest mid-point (day of year)"),
    ("harv_days_since_peak", "harvest, days after own NDVI peak"),
    ("pre_ndmi", "NDMI at last standing obs."),
    ("pre_ndvi_frac", "NDVI at last standing obs. / own amplitude"),
    ("sen_ndvi_slope_rel", "NDVI slope before harvest / amplitude"),
    ("post_ndti", "NDTI after harvest (residue)"),
]


def class_distributions(X, y, path, features=DIST_FEATURES, title=""):
    y = np.asarray(y)
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows))
    for ax, (c, lab) in zip(np.ravel(axes), features):
        if c not in X.columns:
            ax.set_visible(False)
            continue
        v = X[c].to_numpy(dtype=float)
        fin = np.isfinite(v)
        if not fin.any():
            ax.text(0.5, 0.5, "all NaN", ha="center", transform=ax.transAxes)
            ax.set_title(lab, fontsize=9)
            continue
        lo, hi = np.nanpercentile(v[fin], [1, 99])
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 30)
        for val, name in ((1, "silage"), (0, "grain")):
            sel = fin & (y == val)
            ax.hist(np.clip(v[sel], lo, hi), bins=bins, alpha=0.55, color=COLORS[name],
                    density=True, label=f"{name} (n={int(sel.sum())})")
        miss = {name: float(np.mean(~fin[y == val])) if (y == val).any() else np.nan
                for val, name in ((1, "silage"), (0, "grain"))}
        ax.set_title(f"{lab}\nNaN: silage {miss['silage']:.0%}, grain {miss['grain']:.0%}",
                     fontsize=9)
        ax.legend(fontsize=7)
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def example_curves(ts, X, fields, field_ids, path, cfg, preds=None, title=""):
    """NDVI and NDMI over the season with the detected harvest interval shaded."""
    field_ids = list(field_ids)
    if not field_ids:
        return
    cols = 3
    rows = int(np.ceil(len(field_ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.9 * rows), squeeze=False)
    fx = X.set_index("field_id")
    fmeta = fields.set_index("field_id")
    pmap = {} if preds is None else dict(zip(preds["field_id"], preds["p"]))
    for ax, fid in zip(np.ravel(axes), field_ids):
        sub = ts[ts["field_id"] == fid].sort_values("date")
        year = fmeta.at[fid, "year"] if fid in fmeta.index else np.nan
        if not np.isfinite(pd.to_numeric(year)):
            year = int(sub["date"].dt.year.mode().iloc[0]) if len(sub) else 2000
        jan1 = pd.Timestamp(int(year), 1, 1)
        doy = ((sub["date"] - jan1).dt.days + 1).to_numpy(dtype=float)
        bands = sub[BANDS].to_numpy(dtype=float)
        ok = usable_mask(sub["clear_fraction"].to_numpy(), bands, cfg,
                         sub["n_clear"].to_numpy() if "n_clear" in sub else None)
        b = {k: bands[:, j] for j, k in enumerate(BANDS)}
        nv, nm = ndvi(b["B08"], b["B04"]), ndmi(b["B8A"], b["B11"])
        ax.plot(doy[ok], nv[ok], "o-", ms=3, color="#2a6f3a", label="NDVI")
        ax.plot(doy[ok], nm[ok], "s-", ms=3, color="#3a6ea5", label="NDMI")
        rejected = ~ok & np.isfinite(nv)
        ax.plot(doy[rejected], nv[rejected], "x", ms=4, color="0.6", label="NDVI not usable")
        if fid in fx.index:
            lo, hi = fx.at[fid, "harv_lo_doy"], fx.at[fid, "harv_hi_doy"]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.axvspan(lo, hi, color="#d62828", alpha=0.18, label="harvest interval")
        lab = fmeta.at[fid, "label"] if fid in fmeta.index else None
        txt = f"{fid} | label {lab}"
        if fid in pmap:
            txt += f" | p={pmap[fid]:.2f}"
        if fid in fx.index and bool(fx.at[fid, "low_evidence"]):
            txt += " | low ev."
        ax.set_title(txt, fontsize=8)
        ax.set_xlim(120, 360)
        ax.set_ylim(-0.4, 1.0)
        ax.axhline(0, color="0.8", lw=0.5)
    for ax in np.ravel(axes)[len(field_ids):]:
        ax.set_visible(False)
    np.ravel(axes)[0].legend(fontsize=6, loc="lower left")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _roc(y, p):
    order = np.argsort(-p, kind="stable")
    y = y[order]
    tp = np.r_[0, np.cumsum(y)]
    fp = np.r_[0, np.cumsum(1 - y)]
    P, N = max(tp[-1], 1), max(fp[-1], 1)
    return fp / N, tp / P, np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)


def roc_pr_curves(preds, path, title=""):
    """ROC and precision-recall curves for each model, pooled over test folds."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, g in preds.groupby("model", sort=False):
        y = g["y"].to_numpy(int)
        p = g["p"].to_numpy(float)
        fpr, tpr, prec = _roc(y, p)
        style = "--" if name == "majority" else "-"
        a1.plot(fpr, tpr, style, label=name)
        a2.plot(tpr[1:], prec[1:], style, label=name)
    a1.plot([0, 1], [0, 1], color="0.8", lw=0.8)
    a1.set_xlabel("false positive rate")
    a1.set_ylabel("recall (silage)")
    a2.set_xlabel("recall (silage)")
    a2.set_ylabel("precision (silage)")
    a1.legend(fontsize=8)
    a1.set_title("ROC", fontsize=9)
    a2.set_title("precision-recall", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def score_histograms(preds, path, title=""):
    models = [m for m in preds["model"].unique() if m != "majority"]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3), squeeze=False)
    for ax, name in zip(axes[0], models):
        g = preds[preds["model"] == name]
        bins = np.linspace(0, 1, 26)
        for val, lab in ((1, "silage"), (0, "grain")):
            ax.hist(g.loc[g["y"] == val, "p"], bins=bins, alpha=0.55, color=COLORS[lab],
                    label=lab)
        thr = float(g["threshold"].iloc[0])
        ax.axvline(thr, color="k", lw=0.8)
        ax.set_title(f"{name}: score on held-out fields", fontsize=9)
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def importance_bars(imp, path, top=20, title=""):
    if imp is None or imp.empty:
        return
    models = list(imp["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 0.28 * top + 1.5),
                             squeeze=False)
    for ax, name in zip(axes[0], models):
        g = imp[imp["model"] == name].sort_values("auc_drop_mean", ascending=False).head(top)
        g = g.iloc[::-1]
        colors = ["#264653" if f.startswith("group:") else "#8ab17d" for f in g["feature"]]
        ax.barh(g["feature"], g["auc_drop_mean"], xerr=g["auc_drop_std"].fillna(0), color=colors)
        ax.set_title(f"{name}: drop in held-out ROC-AUC when shuffled", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.axvline(0, color="0.5", lw=0.6)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
