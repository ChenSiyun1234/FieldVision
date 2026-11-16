"""TempDiff-US: the TempDiff-SMID silage rule (Agronomy 2025, 15(6), 1438) set up for US corn fields.

The paper (Hohhot, China) separates silage from grain maize with one moisture test: silage is cut
earlier, so on a date when silage is already harvested and grain still stands, a silage field reads
as bare and dry (low LSWI) and a grain field does not. Like the paper, the series are 15-day
composites smoothed with a Whittaker filter.

What is ours, fixed before scoring Beltsville (no Beltsville label was used to set them):
- the date: halfway between the middles of Maryland's most-active harvest windows for corn silage
  (Sep 9 - Oct 1) and corn grain (Sep 22 - Oct 22), USDA NASS Usual Planting and Harvesting Dates
  (Agricultural Handbook 628, 2010), which gives September 28;
- the threshold: LSWI below 0.05, the bare-soil moisture level FieldVision's own season code uses;
- LSWI is FieldVision's NDMI, (B8A - B11) / (B8A + B11), the paper's NIR-SWIR formula.
The paper's own thresholds are not used: they were tuned to Hohhot and printed only as images.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RULE = {
    "name": "tempdiff_us_v1",
    "date": "09-28",
    "date_source": "USDA NASS Agricultural Handbook 628 (2010), Maryland: corn silage most active Sep 9-Oct 1, "
                   "corn grain most active Sep 22-Oct 22; midpoint of the two window middles",
    "lswi_threshold": 0.05,
    "threshold_source": "FieldVision features.season bare-soil NDMI level (0.05)",
    "bin_days": 15,
    "season": ("04-01", "11-30"),
    "whittaker_lambda": 10.0,
    "max_gap_days": 20,           # nearest usable observation farther than this from the date: low evidence
}


def whittaker(y, w, lam=10.0, d=2):
    """Weighted Whittaker smoother: minimise sum w (y - z)^2 + lam * sum (diff^d z)^2."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    n = len(y)
    if n <= d:
        return y.copy()
    D = np.diff(np.eye(n), n=d, axis=0)
    A = np.diag(w) + lam * D.T @ D
    return np.linalg.solve(A, w * np.nan_to_num(y))


def composite(doy, values, start_doy, end_doy, bin_days=15):
    """Median per bin of bin_days; returns bin centres, medians (NaN when empty), weights (1 or 0)."""
    edges = np.arange(start_doy, end_doy + bin_days, bin_days)
    centres = (edges[:-1] + edges[1:] - 1) / 2
    med = np.full(len(centres), np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (doy >= a) & (doy < b) & np.isfinite(values)
        if sel.any():
            med[i] = np.median(values[sel])
    return centres, med, np.isfinite(med).astype(float)


def _doy(year, mmdd):
    return pd.Timestamp(f"{year}-{mmdd}").dayofyear


def score_field(dates, lswi, ndvi, year, rule=RULE):
    """One field-season: smoothed LSWI and NDVI on the rule's date, the call and its evidence."""
    doy = pd.to_datetime(pd.Series(dates)).dt.dayofyear.to_numpy()
    start, end, target = _doy(year, rule["season"][0]), _doy(year, rule["season"][1]), _doy(year, rule["date"])
    out = {"date": f"{year}-{rule['date']}", "n_usable": int(np.isfinite(lswi).sum())}
    if out["n_usable"] < 3:
        return {**out, "call": None, "low_evidence": True, "reason": "fewer than 3 usable observations"}
    smooth = {}
    for name, v in (("lswi", np.asarray(lswi, float)), ("ndvi", np.asarray(ndvi, float))):
        c, m, w = composite(doy, v, start, end, rule["bin_days"])
        z = whittaker(m, w, rule["whittaker_lambda"])
        smooth[name] = float(np.interp(target, c, z))
    gap = int(np.min(np.abs(doy[np.isfinite(lswi)] - target)))
    silage = smooth["lswi"] < rule["lswi_threshold"]
    return {**out, "lswi_on_date": round(smooth["lswi"], 3), "ndvi_on_date": round(smooth["ndvi"], 3),
            "nearest_obs_days": gap, "low_evidence": gap > rule["max_gap_days"],
            "call": "silage" if silage else "grain", "score": round(rule["lswi_threshold"] - smooth["lswi"], 3),
            "reason": (f"moisture (LSWI) {smooth['lswi']:.2f} on {year}-{rule['date']} is "
                       f"{'below' if silage else 'above'} {rule['lswi_threshold']}: "
                       f"{'already harvested and bare, like silage' if silage else 'crop still standing, like grain'}")}


# --- Beltsville ------------------------------------------------------------------

def _fieldvision_indices(ts):
    """LSWI (= NDMI) and NDVI on FieldVision's usable observations only, as its own features do."""
    from fieldvision.features.indices import BANDS, ndmi, ndvi
    from fieldvision.features.season import usable_mask
    ok = usable_mask(ts["clear_fraction"].to_numpy(), ts[BANDS].to_numpy(),
                     n_clear=ts["n_clear"].to_numpy() if "n_clear" in ts else None)
    ts = ts.assign(lswi=np.where(ok, ndmi(ts["B8A"], ts["B11"]), np.nan),
                   ndvi=np.where(ok, ndvi(ts["B08"], ts["B04"]), np.nan))
    return ts


def _label(crop_name):
    s = str(crop_name).lower()
    return "silage" if "silage" in s else "grain" if "grain" in s else None


def score_beltsville(county_dir, years=(2021, 2022, 2023), rule=RULE):
    import geopandas as gpd
    rows = []
    for year in years:
        folder = Path(county_dir) / str(year)
        fields = gpd.read_file(folder / "fields.geojson")
        ts = _fieldvision_indices(pd.read_parquet(folder / "timeseries.parquet"))
        for _, f in fields.iterrows():
            s = ts[ts["field_id"] == f["field_id"]].sort_values("date")
            r = score_field(s["date"].to_numpy(), s["lswi"].to_numpy(), s["ndvi"].to_numpy(), year, rule)
            rows.append({"field_id": f["field_id"], "year": year, "label": _label(f.get("crop_name")), **r})
    return pd.DataFrame(rows)


def metrics(df):
    d = df[df["label"].notna()]
    called = d["call"].fillna("grain")                    # no call counts as grain, the majority class
    y, p = d["label"] == "silage", called == "silage"
    tp, fp, fn, tn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((~y & ~p).sum())
    out = {"n": int(len(d)), "silage": int(y.sum()), "correct": tp + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "silage_precision": round(tp / (tp + fp), 3) if tp + fp else None,
           "silage_recall": round(tp / (tp + fn), 3) if tp + fn else None,
           "fpr": round(fp / (fp + tn), 3) if fp + tn else None,
           "low_evidence": int(d["low_evidence"].sum()), "always_grain_correct": int((~y).sum())}
    s = d["score"].astype(float)
    if s.notna().all() and 0 < y.sum() < len(y):
        pos, neg = s[y].to_numpy(), s[~y].to_numpy()
        out["roc_auc"] = round(float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                                     / (len(pos) * len(neg))), 3)
    return out


def sensitivity(county_dir, dates=("09-14", "09-21", "09-28", "10-05"), thresholds=(0.0, 0.05, 0.10)):
    """Post hoc: how much the result depends on the two fixed settings. Not a model choice."""
    rows = []
    for dt in dates:
        for th in thresholds:
            m = metrics(score_beltsville(county_dir, rule={**RULE, "date": dt, "lswi_threshold": th}))
            rows.append({"date": dt, "lswi_threshold": th, "correct": m["correct"], "n": m["n"],
                         "silage_recall": m["silage_recall"], "fpr": m["fpr"]})
    return pd.DataFrame(rows)


def run(county_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = score_beltsville(county_dir)
    m = metrics(df)
    by_year = {int(y): metrics(g) for y, g in df.groupby("year")}
    sens = sensitivity(county_dir)
    df.to_csv(out / "beltsville_predictions.csv", index=False)
    sens.to_csv(out / "sensitivity.csv", index=False)
    (out / "beltsville_result.json").write_text(json.dumps({"rule": RULE, "all": m, "by_year": by_year,
                                                            "comparison_from_model_cards": {
                                                                "eu_logistic_v1": "42 of 44 (zero-shot)",
                                                                "eu_timing_rule_v1": "38 of 44 (zero-shot)"}},
                                                           indent=1))
    return df, m, by_year, sens
