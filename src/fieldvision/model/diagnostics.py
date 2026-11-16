"""Diagnostics on the training data: double cropping, self-normalised vs raw
features, feature redundancy and where in time the signal sits.

Nothing here is fit on held-out data. The heatmap cells are single-feature
ROC-AUCs (no fitting at all; 0.5 = no separation, below 0.5 = the feature is
lower for silage) with block-bootstrap intervals; the raw-vs-normalised
comparison uses the same block-grouped splits as the main evaluation.
"""
from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..features.build import build_features  # noqa: E402
from ..features.indices import BANDS, compute_indices  # noqa: E402
from ..features.season import SELF_NORMALISED_INFO, usable_mask  # noqa: E402
from .metrics import weighted_auc  # noqa: E402

SEP_VARS = ["ndvi", "ndmi", "nbr", "ndti", "ndre", "psri", "B11", "B12"]
HALF_MONTHS = ["jul1", "jul2", "aug1", "aug2", "sep1", "sep2", "oct1", "oct2", "nov1", "nov2"]
REL_BINS = [(-28, -21), (-21, -14), (-14, -7), (-7, 0), (0, 7), (7, 14), (14, 21), (21, 28)]
REL_LABELS = ["-4w", "-3w", "-2w", "-1w", "+1w", "+2w", "+3w", "+4w"]

RAW_SET = ["pre_ndvi", "pre_ndmi", "pre_ndre", "drop_ndvi", "drop_rate_ndvi",
           "harv_days_since_peak", "post_ndti", "post_bright", "sen_ndmi_slope"]
NORM_SET = ["pre_ndvi_over_peak", "pre_ndmi_over_peak", "pre_ndre_over_peak", "drop_over_peak",
            "drop_rate_over_peak", "harv_drydown_days", "post_ndti_minus_soil",
            "post_bright_minus_soil", "sen_dndmi_minus_dndvi"]


# --------------------------------------------------------------------------
# double cropping
# --------------------------------------------------------------------------

def double_crop_table(ts, fields, X, cfg, preds=None, shifts=((145, 295), (175, 265))):
    """Fields whose spring greenness beats the summer peak, and how stable their harvest is.

    Stability: the harvest interval is recomputed with the peak window moved;
    'stable' = the same last-standing and first-harvested dates every time.
    """
    f = fields.set_index("field_id")
    X = X.set_index("field_id")
    cy = f["country"].astype(str) + " " + f["year"].map(lambda v: str(int(v)) if pd.notna(v) else "?")
    spring = X["q_spring_above_summer"] == 1
    ids = list(X.index[spring])
    stable = pd.Series(np.nan, index=X.index)
    if ids:
        sub_f = fields[fields["field_id"].isin(ids)]
        sub_t = ts[ts["field_id"].isin(ids)]
        same = pd.Series(True, index=ids)
        for w in shifts:
            Xa, _ = build_features(sub_t, sub_f, replace(cfg, peak_window=tuple(w)))
            Xa = Xa.set_index("field_id").reindex(ids)
            for c in ("harv_lo_doy", "harv_hi_doy"):
                a, b = X.loc[ids, c].to_numpy(float), Xa[c].to_numpy(float)
                same &= (a == b) | (np.isnan(a) & np.isnan(b))
        stable.loc[ids] = same.astype(float).to_numpy()
    err = None
    if preds is not None and not preds.empty:
        err = (preds.set_index("field_id")["pred"] != preds.set_index("field_id")["y"]).astype(float)
    rows = []
    for key in sorted(cy.unique()):
        m = (cy == key).reindex(X.index).fillna(False).to_numpy()
        s = spring.to_numpy() & m
        o = ~spring.to_numpy() & m
        r = {"country_year": key, "n": int(m.sum()), "spring_above_summer": int(s.sum()),
             "share": float(s.sum() / max(m.sum(), 1)),
             "harvest_found_spring": float(X.loc[s, "harv_mid_doy"].notna().mean()) if s.any() else np.nan,
             "harvest_found_others": float(X.loc[o, "harv_mid_doy"].notna().mean()) if o.any() else np.nan,
             "median_harvest_doy_spring": float(X.loc[s, "harv_mid_doy"].median()) if s.any() else np.nan,
             "median_harvest_doy_others": float(X.loc[o, "harv_mid_doy"].median()) if o.any() else np.nan,
             "stable_under_window_shift": float(stable[s].mean()) if s.any() else np.nan}
        if err is not None:
            e = err.reindex(X.index)
            r["error_rate_spring"] = float(e[s].mean()) if s.any() else np.nan
            r["error_rate_others"] = float(e[o].mean()) if o.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# single-feature separability (no fitting), with block bootstrap
# --------------------------------------------------------------------------

def _auc_with_ci(y, v, blocks, n_boot=200, seed=0):
    m = np.isfinite(v)
    y, v, blocks = y[m], v[m], np.asarray(blocks)[m]
    n_s, n_g = int((y == 1).sum()), int((y == 0).sum())
    if n_s < 10 or n_g < 10:
        return np.nan, np.nan, np.nan, n_s, n_g
    auc, _ = weighted_auc(y, v)
    uniq, inv = np.unique(blocks.astype(str), return_inverse=True)
    if len(uniq) >= 3 and n_boot:
        rng = np.random.default_rng(seed)
        mult = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot)
        r, _ = weighted_auc(y, v, mult[:, inv])
        lo, hi = np.nanpercentile(r, [2.5, 97.5])
    else:
        lo = hi = np.nan
    return float(auc), float(lo), float(hi), n_s, n_g


def drop_relative_medians(ts, X, cfg):
    """Per field and variable, medians in weekly bins before the last standing and after the
    first harvested observation."""
    hv = X.set_index("field_id")[["harv_lo_doy", "harv_hi_doy"]].dropna()
    t = ts[ts["field_id"].isin(hv.index)].copy()
    if t.empty:
        return pd.DataFrame()
    bands = t[BANDS].to_numpy(float)
    ok = usable_mask(t["clear_fraction"].to_numpy(), bands, cfg,
                     t["n_clear"].to_numpy() if "n_clear" in t else None)
    t = t[ok]
    bands = bands[ok]
    idx = compute_indices({b: bands[:, j] for j, b in enumerate(BANDS)})
    year = t["date"].dt.year.to_numpy()
    # season year = the field's harvest year; take it from the dates closest to the harvest
    jan1 = pd.to_datetime(pd.Series(year).astype(str) + "-01-01").to_numpy()
    doy = (t["date"].to_numpy() - jan1) / np.timedelta64(1, "D") + 1.0
    lo = t["field_id"].map(hv["harv_lo_doy"]).to_numpy()
    hi = t["field_id"].map(hv["harv_hi_doy"]).to_numpy()
    before = doy - lo
    after = doy - hi
    lab = np.full(len(t), None, dtype=object)
    for (a, b), name in zip(REL_BINS, REL_LABELS):
        if b <= 0:
            m = (before > a) & (before <= b)
        else:
            m = (after >= a) & (after < b)
        lab[m] = name
    data = pd.DataFrame({"field_id": t["field_id"].to_numpy(), "bin": lab})
    for v in SEP_VARS:
        data[v] = idx[v] if v in idx else bands[:, BANDS.index(v)]
    data = data[data["bin"].notna()]
    return data.groupby(["field_id", "bin"])[SEP_VARS].median()


def separability(ts, X, meta, y, cfg, n_boot=200, seed=0):
    """Long table: scope (pooled or a country), axis (calendar / drop-relative), variable, bin,
    ROC-AUC of the raw value for silage, its block-bootstrap interval and the class counts."""
    meta = meta.set_index("field_id")
    ymap = pd.Series(np.asarray(y, float), index=meta.index)
    rel = drop_relative_medians(ts, X, cfg)
    Xi = X.set_index("field_id")
    scopes = [("pooled", meta.index)] + [(c, meta.index[meta["country"] == c])
                                         for c in sorted(meta["country"].unique())]
    if len(scopes) == 2:
        scopes = scopes[:1]
    rows = []
    for scope, ids in scopes:
        yy = ymap.reindex(ids).to_numpy()
        bl = meta.loc[ids, "block_id"].to_numpy()
        for v in SEP_VARS:
            for hm in HALF_MONTHS:
                col = f"cal_{v.lower()}_{hm}"
                if col not in Xi.columns:
                    continue
                vals = Xi.reindex(ids)[col].to_numpy(float)
                a, lo, hi, ns, ng = _auc_with_ci(yy, vals, bl, n_boot, seed)
                rows.append(dict(scope=scope, axis="calendar", var=v, bin=hm, auc=a, lo=lo, hi=hi,
                                 n_silage=ns, n_grain=ng))
            for lab in REL_LABELS:
                if rel.empty:
                    continue
                try:
                    s = rel.xs(lab, level="bin")[v]
                except KeyError:
                    continue
                vals = s.reindex(ids).to_numpy(float)
                a, lo, hi, ns, ng = _auc_with_ci(yy, vals, bl, n_boot, seed)
                rows.append(dict(scope=scope, axis="drop-relative", var=v, bin=lab, auc=a, lo=lo,
                                 hi=hi, n_silage=ns, n_grain=ng))
    return pd.DataFrame(rows)


def plot_separability(tab, path, title=""):
    scopes = list(dict.fromkeys(tab["scope"]))
    fig, axes = plt.subplots(len(scopes), 2, figsize=(14, 3.6 * len(scopes) + 0.8), squeeze=False,
                             gridspec_kw={"hspace": 0.75, "wspace": 0.18})
    for i, scope in enumerate(scopes):
        for j, (axis, bins) in enumerate((("calendar", HALF_MONTHS), ("drop-relative", REL_LABELS))):
            ax = axes[i, j]
            t = tab[(tab.scope == scope) & (tab.axis == axis)]
            M = t.pivot(index="var", columns="bin", values="auc").reindex(index=SEP_VARS, columns=bins)
            N = t.pivot(index="var", columns="bin", values="n_silage").reindex(index=SEP_VARS, columns=bins)
            Ng = t.pivot(index="var", columns="bin", values="n_grain").reindex(index=SEP_VARS, columns=bins)
            im = ax.imshow(M.to_numpy(float), cmap="RdBu_r", vmin=0.15, vmax=0.85, aspect="auto")
            for r in range(M.shape[0]):
                for c in range(M.shape[1]):
                    v = M.iat[r, c]
                    if np.isfinite(v):
                        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                                color="white" if abs(v - 0.5) > 0.25 else "black")
            nmin = (N.fillna(0) + Ng.fillna(0)).min(axis=0)
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels([f"{b}\nn>={int(n)}" for b, n in zip(bins, nmin)], fontsize=7)
            ax.set_yticks(range(len(SEP_VARS)))
            ax.set_yticklabels(SEP_VARS, fontsize=8)
            ax.set_title(f"{scope}: {axis}" + (" (weeks around the drop)" if j else
                                                " (half-months Jul-Nov)"), fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="ROC-AUC for silage (0.5 = none)")
    fig.suptitle(title or "Where the signal sits: single-index separability", fontsize=10)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def separability_summary(tab):
    """Plain-words comparison of calendar vs drop-relative separability, per scope."""
    out = []
    for scope, g in tab.groupby("scope", sort=False):
        g = g.dropna(subset=["auc"])
        if g.empty:
            continue
        g = g.assign(dist=(g["auc"] - 0.5).abs())
        best = {ax: g[g.axis == ax].sort_values("dist", ascending=False).head(1)
                for ax in ("calendar", "drop-relative")}
        parts = []
        for ax, b in best.items():
            if not b.empty:
                r = b.iloc[0]
                parts.append(f"best {ax} cell {r['var']} {r['bin']} AUC {r['auc']:.2f} "
                             f"[{r['lo']:.2f}, {r['hi']:.2f}], n={int(r['n_silage'])}+{int(r['n_grain'])}")
        med = {ax: float(g[g.axis == ax]["dist"].median()) for ax in ("calendar", "drop-relative")
               if (g.axis == ax).any()}
        out.append({"scope": scope, "text": "; ".join(parts),
                    "median_distance_calendar": med.get("calendar"),
                    "median_distance_drop_relative": med.get("drop-relative")})
    return out


# --------------------------------------------------------------------------
# redundancy
# --------------------------------------------------------------------------

def correlation_heatmap(X, features, kept, path, max_nan=0.5):
    """Clustered Spearman correlation; features the models keep are marked with '*'."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    cols = [c for c in dict.fromkeys(features) if c in X.columns and X[c].isna().mean() <= max_nan
            and X[c].nunique() > 2]
    if len(cols) < 3:
        return None
    rho = X[cols].corr(method="spearman").fillna(0.0)
    d = 1 - rho.abs().to_numpy()
    np.fill_diagonal(d, 0)
    order = leaves_list(linkage(squareform(np.clip(d, 0, None), checks=False), "average"))
    rho = rho.iloc[order, order]
    n = len(cols)
    fig, ax = plt.subplots(figsize=(0.24 * n + 3, 0.24 * n + 2))
    im = ax.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [("* " if c in kept else "") + c for c in rho.index]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman rho")
    ax.set_title("Feature redundancy (clustered). * = used by the logistic model or among the "
                 "GBM's most important", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return rho


def redundant_pairs(rho, threshold=0.9):
    if rho is None:
        return []
    out = []
    c = list(rho.columns)
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            if abs(rho.iat[i, j]) >= threshold:
                out.append((c[i], c[j], float(rho.iat[i, j])))
    return sorted(out, key=lambda r: -abs(r[2]))


def univariate_by_country(X, meta, y, pairs=None):
    """Single-feature ROC-AUC of each self-normalised feature and its raw counterpart, per country."""
    pairs = pairs or [(k, v[0]) for k, v in SELF_NORMALISED_INFO.items()]
    rows = []
    y = np.asarray(y)
    for scope in ["pooled"] + sorted(meta["country"].unique()):
        m = np.ones(len(meta), bool) if scope == "pooled" else (meta["country"] == scope).to_numpy()
        for norm, raw in pairs:
            r = {"scope": scope, "normalised": norm, "raw": raw}
            for k, c in (("auc_normalised", norm), ("auc_raw", raw)):
                v = X[c].to_numpy(float)[m] if c in X.columns else np.full(m.sum(), np.nan)
                ok = np.isfinite(v)
                yy = y[m][ok]
                r[k] = weighted_auc(yy, v[ok])[0] if ok.sum() > 20 and 0 < yy.mean() < 1 else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    df["distance_gain"] = (df["auc_normalised"] - 0.5).abs() - (df["auc_raw"] - 0.5).abs()
    return df
