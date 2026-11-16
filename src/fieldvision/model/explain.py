"""Readable evidence behind each prediction.

Two kinds of explanation, kept apart on purpose:

- physical evidence: the handful of numbers a skeptic can check on the curve
  (the drop, how green and wet the field was just before it, what the ground
  looked like after). Each is compared with the typical value of each class
  in the training data, so the reader sees which way it points. This does
  not depend on the model.
- model factors: the features that moved this model's score most for this
  field (LightGBM's own contributions, or coef * value for the logistic
  model). This says what the model actually used.

A field can have the evidence pointing one way and the model the other;
showing both is the point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# feature, short name, what it means
EVIDENCE = [
    ("harv_days_since_peak", "days peak->cut", "days from the field's own NDVI peak to the drop"),
    ("sen_ndvi_slope_rel", "dry-down slope", "NDVI slope in the 30 days before the drop, per unit of own amplitude"),
    ("pre_ndvi_frac", "greenness before", "NDVI at the last standing observation as a share of own peak amplitude"),
    ("pre_ndmi", "wetness before", "NDMI (canopy water) at the last standing observation"),
    ("post_ndti", "residue after", "NDTI after the drop; higher = more dry stalks on the ground"),
]


def reference_values(X, y):
    """Median of each evidence feature per class on the training fields."""
    y = np.asarray(y)
    out = {}
    for c, *_ in EVIDENCE:
        if c not in X.columns:
            continue
        v = X[c].to_numpy(dtype=float)
        out[c] = {"silage": float(np.nanmedian(v[y == 1])) if (y == 1).any() else None,
                  "grain": float(np.nanmedian(v[y == 0])) if (y == 0).any() else None}
    return out


def evidence_sides(X, reference):
    """+1 where a value is closer to the silage median, -1 closer to grain, 0 not observed."""
    sides = pd.DataFrame(index=X.index)
    for c, *_ in EVIDENCE:
        ref = reference.get(c)
        if not ref or c not in X.columns or ref["silage"] is None or ref["grain"] is None:
            continue
        v = X[c].to_numpy(dtype=float)
        ds, dg = np.abs(v - ref["silage"]), np.abs(v - ref["grain"])
        sides[c] = np.where(~np.isfinite(v), 0, np.where(ds < dg, 1, -1))
    return sides


def top_factors(model, X, k=3):
    """'feature +0.84; feature -0.31; ...' (log-odds units, + pushes towards silage)."""
    try:
        contrib = model.explain(X)
    except Exception:          # explanation must never break prediction
        contrib = None
    if contrib is None or contrib.empty:
        return np.full(len(X), "", dtype=object)
    a = contrib.to_numpy(dtype=float)
    a = np.where(np.isfinite(a), a, 0.0)
    names = np.array(contrib.columns)
    order = np.argsort(-np.abs(a), axis=1)[:, :k]
    out = []
    for i, idx in enumerate(order):
        out.append("; ".join(f"{names[j]} {a[i, j]:+.2f}" for j in idx if abs(a[i, j]) > 1e-6))
    return np.array(out, dtype=object)


def _fmt_date(d):
    return "?" if pd.isna(d) else pd.Timestamp(d).strftime("%d %b")


def reason_strings(pred, X, reference):
    """One sentence per field built from the evidence and the low-evidence flag."""
    sides = evidence_sides(X, reference)
    texts = []
    for i in range(len(pred)):
        r = pred.iloc[i]
        x = X.iloc[i]
        lab = r["predicted_label"]
        parts = [f"{lab} (p={r['p_silage']:.2f})."]
        if np.isfinite(x.get("harv_mid_doy", np.nan)):
            parts.append(
                f"NDVI fell from {x['pre_ndvi']:.2f} to {x['post_ndvi']:.2f} between "
                f"{_fmt_date(r['harvest_start'])} and {_fmt_date(r['harvest_end'])}, "
                f"{x['harv_days_since_peak']:.0f} days after its peak.")
            def side(c):
                s = sides[c].iloc[i] if c in sides.columns else 0
                return {1: "silage-like", -1: "grain-like", 0: "not observed"}[int(s)]
            if np.isfinite(x.get("pre_ndvi_frac", np.nan)):
                parts.append(f"Just before: {100 * x['pre_ndvi_frac']:.0f}% of its own peak "
                             f"greenness ({side('pre_ndvi_frac')}), NDMI {x['pre_ndmi']:.2f} "
                             f"({side('pre_ndmi')}).")
            if np.isfinite(x.get("post_ndti", np.nan)):
                parts.append(f"After: NDTI {x['post_ndti']:.2f} ({side('post_ndti')}).")
        else:
            parts.append("No clear harvest drop was seen.")
        n_obs = int((sides.iloc[i] != 0).sum()) if sides.shape[1] else 0
        n_sil = int((sides.iloc[i] == 1).sum()) if sides.shape[1] else 0
        if n_obs:
            parts.append(f"{n_sil} of {n_obs} signals look like silage.")
        if r["low_evidence"]:
            parts.append(f"LOW EVIDENCE: {str(r['low_evidence_reason']).replace('+', ', ')}"
                         f" ({int(r['n_usable_dates'])} usable dates, largest gap "
                         f"{r['max_gap_days']:.0f} days).")
        texts.append(" ".join(parts))
    return np.array(texts, dtype=object), sides


def explanation_columns(pred, X, model, reference):
    """Columns added to the predictions table."""
    X = X.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    texts, sides = reason_strings(pred, X, reference)
    out = pd.DataFrame({
        "ndvi_before_drop": X["pre_ndvi"].to_numpy(dtype=float),
        "ndvi_after_drop": X["post_ndvi"].to_numpy(dtype=float),
        "days_peak_to_harvest": X["harv_days_since_peak"].to_numpy(dtype=float),
        "greenness_before_frac": X["pre_ndvi_frac"].to_numpy(dtype=float),
        "drydown_slope_rel": X["sen_ndvi_slope_rel"].to_numpy(dtype=float),
        "residue_ndti_after": X["post_ndti"].to_numpy(dtype=float),
        "regreen_frac_after": X["post_regreen_frac"].to_numpy(dtype=float),
        "n_signals_observed": (sides != 0).sum(axis=1).to_numpy() if sides.shape[1] else 0,
        "n_signals_silage_like": (sides == 1).sum(axis=1).to_numpy() if sides.shape[1] else 0,
        "reason": texts,
        "top_model_factors": top_factors(model, X),
    })
    return out
