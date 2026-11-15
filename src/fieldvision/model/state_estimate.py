"""State silage / grain acres from a random sample of whole read blocks.

    python -m fieldvision.model.state_estimate --sample <folder with fields.geojson and
        timeseries.parquet> --reference-fips 24003 24037 24035 24005 --census-shares <json>
        --class1-acres <state corn acres in CDL class 1> [--double-crop-acres N]
        --headline counted_clear --out <dir>

Every estimate is a share of corn AREA, computed over the scored (class-1) corn fields of the
sampled blocks:

  counted_clear       share of the clear-tier area that the classifier calls silage (p >= its
                      threshold)
  reference_calibrated  counted_clear corrected with the false-positive rate measured in the
                      reference blocks, whose silage share is taken from the 2022 Census shares of
                      their counties (area-weighted); recall from European held-out fields
  local_threshold     threshold = 90th percentile of the reference blocks' clear-tier scores;
                      (flagged - reference flagged) / (matched recall - reference flagged)
  local_flagged       the flagged share without adjustment
  local_per_tier      the local threshold inside the clear and the wide tier, combined by area
  wide_called         share of the wide-tier area called silage (context)
  raw_sum             area-weighted mean probability over non-abstained fields (context)

Uncertainty: the sampled blocks are resampled with replacement (reference blocks included, so
the reference's own sampling error is carried) and every estimate recomputed. The shift guard is
the counted clear share inside the reference blocks: a season in which the classifier's
operating point has moved shows up there first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..anomaly.local import adjusted_share
from .artifacts import write_json

ESTIMATES = ("window_zero", "counted_clear", "reference_calibrated", "local_threshold",
             "local_flagged", "local_per_tier", "window_upper", "blind_share", "lower_bound",
             "seen_share", "lower_seen", "upper_seen", "wide_called", "raw_sum")
# harvest-window regions, fixed from geography before each state's sample was opened
# (Maryland logged 18:55; Pennsylvania and Wisconsin logged 22:5x). Counties not listed fall in
# the state's last, default region.
STATE_REGIONS = {
    "24": ({"eastern_shore": ("015", "029", "035", "011", "041", "019", "045", "039", "047"),
            "western": ("023", "001", "043", "021")}, "central_southern"),
    "42": ({"southeast_southcentral": ("001", "011", "017", "029", "041", "043", "045", "055", "067",
                                       "071", "075", "077", "087", "091", "095", "099", "101", "107",
                                       "133"),
            "western": ("003", "005", "007", "009", "013", "019", "021", "023", "031", "033", "039",
                        "047", "049", "051", "053", "059", "063", "065", "073", "083", "085", "111",
                        "121", "123", "125", "129")}, "central_northern"),
    "55": ({"southern": ("025", "105", "045", "065", "043", "049", "127", "059", "101", "079", "133",
                         "055", "027", "021", "111", "103", "023", "123", "089", "131", "047", "077",
                         "039", "117"),
            "eastern": ("071", "015", "139", "087", "009", "061", "029", "115", "135", "137", "083",
                        "075", "078")}, "central_western_northern"),
}
MD_REGIONS = {k: tuple("24" + c for c in v) for k, v in STATE_REGIONS["24"][0].items()}


def region_of(fips):
    fips = str(fips).zfill(5)
    regions, default = STATE_REGIONS.get(fips[:2], ({}, "all"))
    for name, members in regions.items():
        if fips[2:] in members:
            return name
    return default


HA_TO_ACRE = 2.47105


def _share(a, x):
    s = float(np.sum(a))
    return float(np.sum(a * x) / s) if s > 0 else float("nan")


def estimates(df, ref_mask, ref_share, tpr_clear, local_tpr, alpha=0.10, min_ref_fields=30):
    """All estimates for one set of fields. df columns: acres, p, call, tier ('clear', 'wide',
    'abstain'). ref_mask: boolean array (reference blocks). ref_share: the reference's assumed
    silage share. local_tpr: {'clear': r, 'wide': r} matched recall per tier."""
    a, p, call, tier = (df["acres"].to_numpy(float), df["p"].to_numpy(float),
                        df["call"].to_numpy(bool), df["tier"].to_numpy())
    clear, wide, ok = tier == "clear", tier == "wide", tier != "abstain"
    out = {e: float("nan") for e in ESTIMATES}
    out["counted_clear"] = _share(a[clear], call[clear])
    out["wide_called"] = _share(a[wide], call[wide])
    out["raw_sum"] = _share(a[ok], p[ok])
    out["lower_bound"] = float(np.sum(a[clear] * call[clear]) / np.sum(a)) if np.sum(a) > 0 else float("nan")
    if "drop_doy" in df:                 # harvest-window weighting within regions
        from .windows import window_share
        reg = df["region"].to_numpy() if "region" in df else np.full(len(df), "all")
        drop = df["drop_doy"].to_numpy(float)
        z = u = blind = 0.0
        for r in np.unique(reg):
            m = reg == r
            ww, _ = window_share(drop[m], clear[m], call[m], a[m])
            if np.isfinite(ww["silage_acres_zero"]):
                z += ww["silage_acres_zero"]
                u += ww["silage_acres_upper"]
                blind += ww["blind_share"] * float(np.sum(a[m]))
        tot = float(np.sum(a))
        out["window_zero"], out["window_upper"], out["blind_share"] = z / tot, u / tot, blind / tot
    if "mid_doy" in df:                  # seen / unseen corn (amendment logged 23:17), within regions
        from .windows import seen_bounds
        reg = df["region"].to_numpy() if "region" in df else np.full(len(df), "all")
        mid = df["mid_doy"].to_numpy(float)
        lo = up = seen = 0.0
        for r in np.unique(reg):
            m = reg == r
            sb, _ = seen_bounds(mid[m], clear[m], call[m], a[m], min_clear_area=0.5)
            w = float(np.sum(a[m]))
            if np.isfinite(sb["lower"]):
                lo += sb["lower"] * w
                up += sb["upper"] * w
                seen += sb["seen_share"] * w
        tot = float(np.sum(a))
        out["seen_share"], out["lower_seen"], out["upper_seen"] = seen / tot, lo / tot, up / tot
    rc = clear & ref_mask
    out["reference_counted_clear"] = _share(a[rc], call[rc])
    out["clear_share_of_area"] = _share(a, clear)
    if rc.sum() < min_ref_fields:
        return out
    fpr_ref = float(np.clip((out["reference_counted_clear"] - tpr_clear * ref_share)
                            / (1 - ref_share), 0, 0.95))
    out["reference_calibrated"] = adjusted_share(out["counted_clear"], fpr_ref, tpr_clear)
    flags, num, den = {}, 0.0, 0.0
    for k, m in (("clear", clear), ("wide", wide)):
        rk = m & ref_mask
        if rk.sum() < min_ref_fields or not m.any():
            num = float("nan")
            continue
        t = float(np.quantile(p[rk], 1 - alpha))
        ref_flag = _share(a[rk], p[rk] >= t)
        flag = _share(a[m], p[m] >= t)
        flags[k] = (flag, ref_flag)
        share_k = adjusted_share(flag, ref_flag, local_tpr[k])
        num += share_k * float(np.sum(a[m]))
        den += float(np.sum(a[m]))
    if "clear" in flags:
        flag, ref_flag = flags["clear"]
        out["local_flagged"] = flag
        out["local_threshold"] = adjusted_share(flag, ref_flag, local_tpr["clear"])
    out["local_per_tier"] = num / den if den > 0 else float("nan")
    return out


def block_bootstrap(df, ref_mask, ref_share, tpr_clear, local_tpr, n_boot=2000, seed=0):
    """Point estimates and 5-95 % ranges, resampling whole blocks with replacement."""
    point = estimates(df, ref_mask, ref_share, tpr_clear, local_tpr)
    blocks = df["block_id"].to_numpy()
    uniq = np.unique(blocks)
    idx_by_block = {b: np.flatnonzero(blocks == b) for b in uniq}
    rng = np.random.default_rng(seed)
    keys = list(point)
    sims = np.full((n_boot, len(keys)), np.nan)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_block[b] for b in pick])
        e = estimates(df.iloc[rows], ref_mask[rows], ref_share, tpr_clear, local_tpr)
        sims[i] = [e[k] for k in keys]
    rng_out = {}
    for j, k in enumerate(keys):
        col = sims[:, j][np.isfinite(sims[:, j])]
        rng_out[k] = ([float(np.percentile(col, 5)), float(np.percentile(col, 95))]
                      if len(col) else [float("nan"), float("nan")])
    return point, rng_out, len(uniq)


def double_crop_silage_acres(dc, class_acres, min_clear=30):
    """Silage acres in double-crop corn: each class's counted clear share (pooled share when the
    class has fewer than `min_clear` clear fields) times that class's acres from boundaries."""
    clear = (dc["tier"] == "clear").to_numpy()
    a, call = dc["acres"].to_numpy(float), dc["call"].to_numpy(bool)
    pooled = _share(a[clear], call[clear])
    total, per = 0.0, {}
    for cls, acres in class_acres.items():
        m = clear & (dc["crop_code"].astype(str).to_numpy() == str(cls))
        share = _share(a[m], call[m]) if m.sum() >= min_clear else pooled
        share = 0.0 if not np.isfinite(share) else share
        per[str(cls)] = share
        total += share * float(acres)
    return total, per


def combined_bootstrap(table, dc, class1_acres, class_acres, n_boot=2000, seed=0):
    """State share over single-crop plus double-crop corn (headline: counted clear in both), with
    5-95 % ranges from resampling whole blocks across both strata."""
    def one(t, d):
        c = (t["tier"] == "clear").to_numpy()
        single = _share(t["acres"].to_numpy(float)[c], t["call"].to_numpy(bool)[c])
        dca, per = double_crop_silage_acres(d, class_acres)
        tot = class1_acres + sum(float(v) for v in class_acres.values())
        return (single * class1_acres + dca) / tot, single * class1_acres + dca, per
    share, acres, per = one(table, dc)
    blocks = np.union1d(table["block_id"].unique(), dc["block_id"].unique())
    it = {b: np.flatnonzero(table["block_id"].to_numpy() == b) for b in blocks}
    id_ = {b: np.flatnonzero(dc["block_id"].to_numpy() == b) for b in blocks}
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_boot):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        t = table.iloc[np.concatenate([it[b] for b in pick])]
        d = dc.iloc[np.concatenate([id_[b] for b in pick])]
        sims.append(one(t, d)[0])
    sims = np.array(sims)
    return {"share_all_corn": share, "silage_acres": acres, "double_crop_share_by_class": per,
            "range_5_95": [float(np.nanpercentile(sims, 5)), float(np.nanpercentile(sims, 95))]}


def prior_updates(table, prior_landuse=None, prior_official=None, n_boot=2000, seed=0):
    """Version A of the field-level prior update (fieldvision.model.windows), with the seen
    fields found within regions; point and 5-95 % range over whole-block resamples."""
    from .windows import prior_updated_share, seen_field_mask
    priors = {}
    if prior_landuse:
        pt = pd.read_csv(prior_landuse, dtype={"fips": str})
        pt["fips"] = pt["fips"].str.zfill(5)
        col = "prior_share" if "prior_share" in pt else "prior_silage_share"
        pm = pt.set_index("fips")[col].astype(float)
        pr = table["county_fips"].astype(str).str.zfill(5).map(pm).to_numpy(float)
        priors["landuse"] = np.where(np.isfinite(pr), pr, np.nanmedian(pr))
    if prior_official is not None:
        priors["official"] = np.full(len(table), float(prior_official))
    if not priors:
        return None
    reg = table["region"].to_numpy() if "region" in table else np.full(len(table), "all")
    clear = (table["tier"] == "clear").to_numpy()
    a, mid, p = table["acres"].to_numpy(float), table["mid_doy"].to_numpy(float), table["p"].to_numpy(float)

    def seen_of(idx):
        s = np.zeros(len(idx), bool)
        r = reg[idx]
        for g in np.unique(r):
            m = r == g
            s[m] = seen_field_mask(mid[idx][m], clear[idx][m], a[idx][m])
        return s
    allidx = np.arange(len(table))
    out = {}
    seen = seen_of(allidx)
    blocks = table["block_id"].to_numpy()
    uniq = np.unique(blocks)
    by = {b: np.flatnonzero(blocks == b) for b in uniq}
    rng = np.random.default_rng(seed)
    picks = [np.concatenate([by[b] for b in rng.choice(uniq, len(uniq), replace=True)]) for _ in range(n_boot)]
    for name, pr in priors.items():
        point = prior_updated_share(p, pr, seen, a)
        sims = [prior_updated_share(p[i], pr[i], seen_of(i), a[i]) for i in picks]
        out[name] = {"prior_mean": float(np.sum(a * pr) / np.sum(a)), "share": point,
                     "range_5_95": [float(np.nanpercentile(sims, 5)), float(np.nanpercentile(sims, 95))]}
    out["seen_fields_area_share"] = float(np.sum(a[seen]) / np.sum(a))
    return out


def to_acres(share, lo, hi, class1_acres, double_crop_acres=0.0):
    """Silage and grain acres of class-1 corn, and the range when the double-crop corn outside
    the scored set is counted as none or all silage."""
    sil = share * class1_acres
    out = {"silage_acres": sil, "grain_acres": class1_acres - sil,
           "silage_acres_low": lo * class1_acres, "silage_acres_high": hi * class1_acres,
           "class1_corn_acres": class1_acres, "double_crop_corn_acres": double_crop_acres}
    if double_crop_acres:
        tot = class1_acres + double_crop_acres
        out["share_all_corn_if_double_crop_grain"] = sil / tot
        out["share_all_corn_if_double_crop_silage"] = (sil + double_crop_acres) / tot
    return out


def double_crop_table(sample, registry, model_name):
    """Per-field table of the double-crop stratum (timeseries_dblcrop.parquet), with crop_code."""
    import geopandas as gpd
    from ..features.loading import load_fields, load_timeseries
    from ..toolbox import evidence_tier
    from .classify import results as classify_rows
    from .registry import load as load_model
    d = Path(sample)
    raw = gpd.read_file(d / "fields.geojson", ignore_geometry=True).set_index("field_id")
    ids = raw.index[raw["stratum"].astype(str) != "corn"]
    fields = load_fields(d / "fields.geojson")
    fields = fields[fields["field_id"].isin(ids)].drop(columns=["label", "y"], errors="ignore")
    fields = fields.reset_index(drop=True)
    ts, _ = load_timeseries(d / "timeseries_dblcrop.parquet", field_ids=fields["field_id"])
    model, mj, cfg = load_model(registry, model_name)
    rows, _ = classify_rows(fields, ts, model, model_name, cfg, mj["threshold"])
    r = pd.DataFrame(rows)
    fid = r["source_field_id"]
    return pd.DataFrame({"field_id": fid, "block_id": fid.map(raw["block_id"]).to_numpy(),
                         "crop_code": fid.map(raw["crop_code"]).astype(str).to_numpy(),
                         "acres": fid.map(raw["area_ha"]).to_numpy(float) * HA_TO_ACRE,
                         "call": (r["label"] == "silage").to_numpy(),
                         "tier": [evidence_tier(x) for x in rows]})


def field_table(sample, registry, model_name, double_crop_codes=(225, 226, 228, 237, 241), landsat=False):
    """Per-field table for a sample folder: block, county, acres, score, call, evidence tier."""
    from ..features.loading import load_fields, load_timeseries
    from ..toolbox import evidence_tier
    from .classify import results as classify_rows
    from .registry import load as load_model
    d = Path(sample)
    fields = load_fields(d / "fields.geojson")
    fields = fields.drop(columns=[c for c in ("label", "y") if c in fields], errors="ignore")
    import geopandas as gpd
    raw = gpd.read_file(d / "fields.geojson", ignore_geometry=True).set_index("field_id")
    if "stratum" in raw:            # the sample's own flag: corn / dblcrop_sg_corn
        dc = (raw["stratum"].reindex(fields["field_id"]).astype(str) != "corn").to_numpy()
    elif "crop_code" in raw:
        dc = raw["crop_code"].reindex(fields["field_id"]).isin(double_crop_codes).to_numpy()
    else:
        dc = np.zeros(len(fields), bool)
    scored = fields[~np.asarray(dc)].reset_index(drop=True)
    ts, _ = load_timeseries(d / "timeseries.parquet", field_ids=scored["field_id"])
    if landsat and (d / "timeseries_landsat.parquet").exists():
        # union with Landsat, per-band line fitted on this sample's own same-field pairs (no labels)
        from ..features.landsat import fit_adjustment, merge, pair_check
        ls, _ = load_timeseries(d / "timeseries_landsat.parquet", field_ids=scored["field_id"])
        _, prs = pair_check(ts, ls, max_days=2)
        ts = merge(ts, ls, fit_adjustment(prs) if len(prs) >= 30 else None)
    model, mj, cfg = load_model(registry, model_name)
    rows, _ = classify_rows(scored, ts, model, model_name, cfg, mj["threshold"])
    r = pd.DataFrame(rows)
    meta = scored.set_index("field_id")
    if "county_fips" not in meta:      # the loader keeps the contract columns only
        meta["county_fips"] = (raw["county_fips"].reindex(meta.index).to_numpy()
                               if "county_fips" in raw else "")
    ids = r["source_field_id"]
    table = pd.DataFrame({
        "field_id": ids, "block_id": ids.map(meta["block_id"]).to_numpy(),
        "county_fips": ids.map(meta["county_fips"].astype(str)).to_numpy(),
        "acres": ids.map(meta["area_ha"]).to_numpy(float) * HA_TO_ACRE,
        "p": r["p_silage"].to_numpy(float), "call": (r["label"] == "silage").to_numpy(),
        "tier": [evidence_tier(x) for x in rows],
        "drop_doy": np.asarray(pd.to_datetime([x["evidence"].get("drop_date") for x in rows],
                                              errors="coerce").dayofyear, dtype=float),
        "standing_doy": np.asarray(pd.to_datetime([x["evidence"].get("last_standing_date") for x in rows],
                                                  errors="coerce").dayofyear, dtype=float)})
    table["mid_doy"] = (table["drop_doy"] + table["standing_doy"]) / 2
    table["region"] = [region_of(f) for f in table["county_fips"]]
    dca = float(fields.loc[np.asarray(dc), "area_ha"].sum() * HA_TO_ACRE)
    return table, dca


def main(argv=None):
    ap = argparse.ArgumentParser(description="State silage/grain acres from a block sample.")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--registry", default="registry")
    ap.add_argument("--model", default="eu2_logistic_v1")
    ap.add_argument("--reference-fips", nargs="+", required=True)
    ap.add_argument("--census-shares", required=True,
                    help="JSON {fips: 2022 Census silage share} for the reference counties")
    ap.add_argument("--class1-acres", type=float, required=True)
    ap.add_argument("--double-crop-acres", type=float, default=None,
                    help="state double-crop corn acres; default: none given (sample's own shown)")
    ap.add_argument("--headline", default="counted_clear", choices=ESTIMATES[:6])
    ap.add_argument("--guard", type=float, default=0.10,
                    help="reference counted clear share above which the headline is flagged")
    ap.add_argument("--guard-over-reference", type=float, default=None,
                    help="flag when the reference's counted clear share exceeds its Census share by "
                         "more than this (00:18 rule for Pennsylvania and Wisconsin)")
    ap.add_argument("--secondary-model", default=None,
                    help="also report this model's counted clear share (shown beside)")
    ap.add_argument("--min-clear-fields", type=int, default=300)
    ap.add_argument("--min-clear-area", type=float, default=0.15)
    ap.add_argument("--double-crop-class-acres", default=None,
                    help="JSON {crop_code: acres} (state_acres.json by_class); scores the double-crop "
                         "stratum and adds its silage acres (rule logged 19:21)")
    ap.add_argument("--min-seen", type=float, default=0.50,
                    help="below this seen share of corn area the headline is the range (23:17)")
    ap.add_argument("--landsat", action="store_true",
                    help="union with timeseries_landsat.parquet in the sample folder")
    ap.add_argument("--prior-landuse", default=None,
                    help="county prior table (fips, prior_share) for the prior-updated figure")
    ap.add_argument("--prior-official", type=float, default=None,
                    help="last official state share before the test year, for the second prior-updated figure")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table, sample_dc_acres = field_table(a.sample, a.registry, a.model, landsat=a.landsat)
    table.to_parquet(out / "fields.parquet", index=False)
    rates = json.loads((Path(a.registry) / a.model / "rates.json").read_text(encoding="utf-8"))
    tpr_clear = rates["sets"][rates["primary_clear"]]["tpr"]
    local_tpr = {k: rates["local"][k]["matched_tpr"] for k in ("clear", "wide")}
    shares = {str(k): float(v) for k, v in json.loads(Path(a.census_shares).read_text()).items()}
    ref_mask = table["county_fips"].isin([str(f) for f in a.reference_fips]).to_numpy()
    rc = ref_mask & (table["tier"] == "clear").to_numpy()
    w = table[rc].groupby("county_fips")["acres"].sum()
    ref_share = float(sum(shares[f] * w[f] for f in w.index) / w.sum()) if len(w) else float("nan")
    point, ranges, n_blocks = block_bootstrap(table, ref_mask, ref_share, tpr_clear, local_tpr,
                                              n_boot=a.n_boot)
    n_clear = int((table["tier"] == "clear").sum())
    enough = n_clear >= a.min_clear_fields and point["clear_share_of_area"] >= a.min_clear_area
    guard_limit = (ref_share + a.guard_over_reference if a.guard_over_reference is not None
                   else a.guard)
    blind = point.get("blind_share", 0.0)
    headline = a.headline
    seen = point.get("seen_share", 1.0)
    as_range = bool(np.isfinite(seen) and seen < a.min_seen)   # logged 23:17: headline is the range
    head = point[headline] if enough and not as_range else float("nan")
    lo, hi = ranges[headline] if enough and not as_range else (float("nan"), float("nan"))
    report = {
        "headline_estimate": headline, "blind_share": blind, "seen_share": seen,
        "headline_is_range": as_range,
        "headline_range_seen": [point.get("lower_seen"), point.get("upper_seen")], "headline_share": head, "headline_range_5_95": [lo, hi],
        "clear_tier_fields": n_clear, "headline_reported": bool(enough),
        "minimum": {"clear_fields": a.min_clear_fields, "clear_share_of_area": a.min_clear_area},
        "shift_guard": {"reference_counted_clear": point["reference_counted_clear"],
                        "limit": guard_limit,
                        "flagged": bool(point["reference_counted_clear"] > guard_limit)},
        "estimates": point, "ranges_5_95": ranges, "n_blocks": n_blocks,
        "n_fields_scored": int(len(table)), "reference_fips": a.reference_fips,
        "reference_blocks": int(table.loc[ref_mask, "block_id"].nunique()),
        "reference_share_from_2022_census": ref_share, "tpr_clear": tpr_clear, "local_tpr": local_tpr,
        "sample_double_crop_acres": sample_dc_acres,
        "acres": (to_acres(head, lo, hi, a.class1_acres,
                           a.double_crop_acres if a.double_crop_acres is not None else 0.0)
                  if enough else {"note": "clear tier below the minimum: no state figure"}),
        "model": a.model,
    }
    if a.secondary_model:
        t2, _ = field_table(a.sample, a.registry, a.secondary_model)
        r2 = ref_mask if len(t2) == len(table) else t2["county_fips"].isin(
            [str(f) for f in a.reference_fips]).to_numpy()
        p2, g2, _ = block_bootstrap(t2, r2, ref_share, tpr_clear, local_tpr, n_boot=a.n_boot)
        report["secondary_model"] = {"model": a.secondary_model, "counted_clear": p2["counted_clear"],
                                     "range_5_95": g2["counted_clear"],
                                     "reference_counted_clear": p2["reference_counted_clear"]}
    if a.double_crop_class_acres:
        ca = json.loads(Path(a.double_crop_class_acres).read_text(encoding="utf-8"))
        if "by_class" in ca:           # state_acres.json layout: {"225 Dbl Crop ...": {"acres": x}}
            ca = {k.split()[0]: v["acres"] for k, v in ca["by_class"].items() if not k.startswith("1 ")}
        dc = double_crop_table(a.sample, a.registry, a.model)
        report["with_double_crops"] = combined_bootstrap(table, dc, a.class1_acres, ca, n_boot=a.n_boot)
    report["prior_updated"] = prior_updates(table, a.prior_landuse, a.prior_official, a.n_boot)
    write_json(out / "state_estimate.json", report)
    print(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    main()
