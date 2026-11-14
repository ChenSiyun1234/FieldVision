"""From per-field predictions to a predicted silage share of corn area per county.

Works on plain data frames so it does not depend on the file formats of the
US field set or the USDA county table, which we have not seen yet. The
caller says which column holds the county key.

Shares are by area, not by field count, because USDA reports acres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def county_silage_share(pred, county_col, area_col="area_ha", p_col="p_silage",
                        label_col="predicted_label", low_col="low_evidence"):
    """One row per county.

    silage_share_confident  area share predicted silage among confident fields
                            (hard labels); the headline number
    silage_share_all        same with low-evidence fields included
    silage_share_expected   sum(p_silage * area) / area over all fields; uses the
                            score rather than the hard label (only meaningful if
                            the score is roughly calibrated)
    low_evidence_area_share share of corn area whose prediction rests on thin data
    """
    df = pred[[county_col, area_col, p_col, label_col, low_col]].copy()
    df = df[df[county_col].notna()]
    df[area_col] = pd.to_numeric(df[area_col], errors="coerce")
    n_no_area = int(df[area_col].isna().sum())
    df = df[df[area_col] > 0]
    df["is_silage"] = (df[label_col] == "silage").astype(float)
    df["low"] = df[low_col].astype(bool)
    df["a_sil"] = df[area_col] * df["is_silage"]
    df["a_p"] = df[area_col] * pd.to_numeric(df[p_col], errors="coerce")
    df["a_low"] = df[area_col] * df["low"]
    df["a_conf"] = df[area_col] * ~df["low"]
    df["a_sil_conf"] = df["a_sil"] * ~df["low"]
    g = df.groupby(county_col)
    out = pd.DataFrame({
        "n_fields": g.size(),
        "n_low_evidence": g["low"].sum().astype(int),
        "area_ha": g[area_col].sum(),
        "area_confident_ha": g["a_conf"].sum(),
    })
    with np.errstate(invalid="ignore", divide="ignore"):
        out["silage_share_confident"] = g["a_sil_conf"].sum() / out["area_confident_ha"]
        out["silage_share_all"] = g["a_sil"].sum() / out["area_ha"]
        out["silage_share_expected"] = g["a_p"].sum() / out["area_ha"]
        out["low_evidence_area_share"] = g["a_low"].sum() / out["area_ha"]
    out.loc[out["area_confident_ha"] <= 0, "silage_share_confident"] = np.nan
    out = out.reset_index()
    out.attrs["n_fields_without_area"] = n_no_area
    return out


def reference_share(table, county_col, silage_col, grain_col):
    """Observed silage share of harvested corn area from a county table.

    silage / (silage + grain), NaN where the denominator is zero or a value
    is withheld (USDA uses '(D)' for undisclosed numbers).
    """
    t = table[[county_col]].copy()
    sil = pd.to_numeric(table[silage_col].astype(str).str.replace(",", ""), errors="coerce")
    grn = pd.to_numeric(table[grain_col].astype(str).str.replace(",", ""), errors="coerce")
    tot = sil + grn
    t["reference_silage_share"] = np.where(tot > 0, sil / tot, np.nan)
    t["reference_corn_area"] = tot
    return t


def compare_county_shares(predicted, reference, county_col,
                          pred_col="silage_share_confident", ref_col="reference_silage_share"):
    """Join predicted and observed shares and summarise the agreement."""
    m = predicted.merge(reference, on=county_col, how="inner")
    ok = m[pred_col].notna() & m[ref_col].notna()
    a, b = m.loc[ok, pred_col].to_numpy(float), m.loc[ok, ref_col].to_numpy(float)
    summary = {"n_counties": int(ok.sum())}
    if ok.sum() >= 1:
        summary["mean_abs_error"] = float(np.mean(np.abs(a - b)))
        summary["mean_error"] = float(np.mean(a - b))
    if ok.sum() >= 3:
        from scipy.stats import pearsonr, spearmanr
        summary["pearson_r"] = float(pearsonr(a, b).statistic)
        summary["spearman_r"] = float(spearmanr(a, b).statistic)
    m["error"] = m[pred_col] - m[ref_col]
    return m, summary
