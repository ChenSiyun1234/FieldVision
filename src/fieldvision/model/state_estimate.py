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

ESTIMATES = ("counted_clear", "reference_calibrated", "local_threshold", "local_flagged",
             "local_per_tier", "wide_called", "raw_sum")
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


def field_table(sample, registry, model_name, double_crop_codes=(225, 226, 228, 237, 241)):
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
        "tier": [evidence_tier(x) for x in rows]})
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
    ap.add_argument("--headline", required=True, choices=ESTIMATES[:5])
    ap.add_argument("--guard", type=float, default=0.10,
                    help="reference counted clear share above which the headline is flagged")
    ap.add_argument("--secondary-model", default=None,
                    help="also report this model's counted clear share (shown beside)")
    ap.add_argument("--min-clear-fields", type=int, default=300)
    ap.add_argument("--min-clear-area", type=float, default=0.15)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table, sample_dc_acres = field_table(a.sample, a.registry, a.model)
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
    head = point[a.headline] if enough else float("nan")
    lo, hi = ranges[a.headline] if enough else (float("nan"), float("nan"))
    report = {
        "headline_estimate": a.headline, "headline_share": head, "headline_range_5_95": [lo, hi],
        "clear_tier_fields": n_clear, "headline_reported": bool(enough),
        "minimum": {"clear_fields": a.min_clear_fields, "clear_share_of_area": a.min_clear_area},
        "shift_guard": {"reference_counted_clear": point["reference_counted_clear"],
                        "limit": a.guard,
                        "flagged": bool(point["reference_counted_clear"] > a.guard)},
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
    write_json(out / "state_estimate.json", report)
    print(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    main()
