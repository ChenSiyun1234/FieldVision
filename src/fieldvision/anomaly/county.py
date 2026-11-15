"""Stage 2: fit on European labels + a local mostly-grain reference county, score a target county.

    python -m fieldvision.anomaly.county --source-features <local_eval out>/features.parquet \
        --reference <county dir> --target <county dir> --out <dir>

County dirs hold timeseries.parquet and fields.geojson (labels, if any, are never read).
The recipe is written down in docs/anomaly-view.md before any target county is scored:
  classifier  the main line's 6-feature logistic, trained on every labelled source field
  direction   source out-of-fold scores (5 spatial folds) give the matched recall
  threshold   the reference county's 90th percentile of classifier scores (alpha = 0.10)
  share       adjusted count, by field count and by area, over fields that are not low evidence
Also reported: the classifier at its own 0.5 cut-off, and the distance view
(robust Mahalanobis on monthly spectra, flag above the reference's cross-fitted
98th percentile) as a diagnostic.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig
from .api import fit_reference, score_target
from .local import DEFAULT_ALPHA, LocalThreshold, adjusted_share
from .local_eval import source_and_target_scores


def county_table(folder):
    folder = Path(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = load_fields([folder / "fields.geojson"])
        ts, _ = load_timeseries([folder / "timeseries.parquet"], field_ids=fields["field_id"])
        X, _ = build_features(ts, fields, FeatureConfig())
        from ..foundation.adapter import build_presto_inputs
        from ..foundation.compare import raw_composite_table
        raw = raw_composite_table(build_presto_inputs(ts, fields, cfg=FeatureConfig()))
    X = X.reset_index(drop=True).drop(columns=["field_id"], errors="ignore")
    X = pd.concat([X, raw.drop(columns="field_id")], axis=1)
    X.insert(0, "field_id", fields["field_id"].to_numpy())
    X["area_ha"] = pd.to_numeric(fields.get("area_ha"), errors="coerce").to_numpy()
    return X


def shares(calls, area, ok):
    c, a = calls[ok], area[ok]
    by_n = float(np.mean(c)) if len(c) else float("nan")
    by_area = float(np.sum(c * a) / np.sum(a)) if np.nansum(a) > 0 else float("nan")
    return by_n, by_area


def failure_checks(ref_p, ref_a, tgt_p, tgt_a, src_p, src_y, est, alphas=(0.05, 0.10, 0.20, 0.30)):
    """Signs that the reference county's grain does not stand in for the target's grain.

    alpha_sweep  adjusted area share at several alphas; if local grain behaves like the
                 reference's they agree within noise, if the target's grain sits higher
                 (or lower) the estimate drifts systematically with alpha
    lower_tail   target area below the reference's median score, observed vs expected from
                 the estimate (reference share below its median x (1 - est) + est x the share
                 of source silage below the source grain median); a ratio far from 1 means
                 the bulk of the target's grain is shifted relative to the reference
    clipped      the estimate hit 0 or 1
    """
    from .local import local_share
    f = np.isfinite(src_p)
    sp, sy = src_p[f], np.asarray(src_y)[f]
    sweep = {f"{a:.2f}": local_share(ref_p, sp, sy, tgt_p, a, ref_a, tgt_a) for a in alphas}
    med = float(np.median(ref_p))
    ref_below = float(np.sum(ref_a * (ref_p < med)) / np.sum(ref_a))
    sil_below = float((sp[sy == 1] < np.median(sp[sy == 0])).mean())
    obs = float(np.sum(tgt_a * (tgt_p < med)) / np.sum(tgt_a))
    exp = ref_below * (1 - est) + est * sil_below if np.isfinite(est) else float("nan")
    vals = np.array(list(sweep.values()), dtype=float)
    return {"alpha_sweep": sweep, "alpha_sweep_range": float(np.nanmax(vals) - np.nanmin(vals)),
            "lower_tail_observed": obs, "lower_tail_expected": exp,
            "lower_tail_ratio": obs / exp if exp else float("nan"),
            "clipped": bool(np.isfinite(est) and (est <= 0 or est >= 1))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-features", required=True, help="features.parquet from anomaly.local_eval")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    src = pd.read_parquet(args.source_features)
    meta_cols = ["field_id", "block_id", "country", "year", "y"]
    Xs, ms = src.drop(columns=meta_cols), src[meta_cols].copy()
    ms["low_evidence"] = Xs["low_evidence"].to_numpy()
    ref, tgt = county_table(args.reference), county_table(args.target)

    # classifier on all source fields; source out-of-fold scores for the matched recall
    allX = pd.concat([Xs, ref.drop(columns=["field_id", "area_ha"]), tgt.drop(columns=["field_id", "area_ha"])],
                     ignore_index=True)[Xs.columns]
    allm = pd.concat([ms, pd.DataFrame({"field_id": ref["field_id"], "block_id": "ref", "y": 0}),
                      pd.DataFrame({"field_id": tgt["field_id"], "block_id": "tgt", "y": 0})], ignore_index=True)
    is_src = np.r_[np.ones(len(Xs), bool), np.zeros(len(ref) + len(tgt), bool)]
    s_src, y_src, scored = source_and_target_scores(allX, allm, ~is_src, args.seed)
    p_ref = scored["p"].to_numpy()[: len(ref)]
    p_tgt = scored["p"].to_numpy()[len(ref):]

    ok_ref = ~ref["low_evidence"].astype(bool).to_numpy() & np.isfinite(p_ref)
    ok_tgt = ~tgt["low_evidence"].astype(bool).to_numpy() & np.isfinite(p_tgt)
    lt = LocalThreshold(args.alpha).fit(p_ref[ok_ref], s_src, y_src)
    a_ref, a_tgt = ref["area_ha"].to_numpy(float), tgt["area_ha"].to_numpy(float)
    ref_calls, tgt_calls = lt.calls(p_ref), lt.calls(p_tgt)
    fixed_ref, fixed_tgt = (p_ref >= 0.5).astype(float), (p_tgt >= 0.5).astype(float)
    fpr_n, fpr_a = shares(ref_calls, a_ref, ok_ref)
    flag_n, flag_a = shares(tgt_calls, a_tgt, ok_tgt)

    # distance view, as a diagnostic
    from ..foundation.compare import raw_composite_table  # noqa: F401  (columns come from county_table)
    raw_cols = [c for c in ref.columns if c.startswith("raw_m")]
    dist = fit_reference(ref, features=raw_cols, scorer="mahalanobis", quantile=0.98)
    d_ref = score_target(dist, ref)["anomaly_flag"].to_numpy(float)
    d_tgt = score_target(dist, tgt)["anomaly_flag"].to_numpy(float)

    res = {
        "args": vars(args), "recipe": lt.describe(), "distance_view": dist.describe(),
        "n_source": int(len(Xs)), "source_tpr_at_alpha": lt.tpr,
        "reference": {"n": int(len(ref)), "n_scored": int(ok_ref.sum()),
                      "fixed_share_n_area": shares(fixed_ref, a_ref, ok_ref),
                      "local_flag_share_n_area": (fpr_n, fpr_a),
                      "distance_flag_share_n_area": shares(d_ref, a_ref, ok_ref)},
        "target": {"n": int(len(tgt)), "n_scored": int(ok_tgt.sum()),
                   "fixed_share_n_area": shares(fixed_tgt, a_tgt, ok_tgt),
                   "local_flag_share_n_area": (flag_n, flag_a),
                   "local_adjusted_share_n": adjusted_share(flag_n, fpr_n, lt.tpr),
                   "local_adjusted_share_area": adjusted_share(flag_a, fpr_a, lt.tpr),
                   "distance_flag_share_n_area": shares(d_tgt, a_tgt, ok_tgt)},
    }
    res["target"]["diagnostics"] = failure_checks(p_ref[ok_ref], a_ref[ok_ref], p_tgt[ok_tgt], a_tgt[ok_tgt],
                                                  s_src, y_src, res["target"]["local_adjusted_share_area"])
    from .local import stratified_share
    from .local_eval import stratum
    s_s = stratum(Xs)
    res["target"]["stratified_adjusted_share_area"] = stratified_share(
        p_ref[ok_ref], stratum(ref)[ok_ref], s_src, y_src, s_s, p_tgt[ok_tgt], stratum(tgt)[ok_tgt],
        args.alpha, a_ref[ok_ref], a_tgt[ok_tgt])
    res["target"]["stratum_area_share"] = {k: float(a_tgt[ok_tgt][stratum(tgt)[ok_tgt] == k].sum()
                                                    / a_tgt[ok_tgt].sum()) for k in ("tight", "loose")}
    res["reference"]["stratum_area_share"] = {k: float(a_ref[ok_ref][stratum(ref)[ok_ref] == k].sum()
                                                       / a_ref[ok_ref].sum()) for k in ("tight", "loose")}
    o = ok_tgt
    res["target"]["agreement"] = {
        "local_and_fixed_both_flag": int(((tgt_calls == 1) & (fixed_tgt == 1))[o].sum()),
        "local_only": int(((tgt_calls == 1) & (fixed_tgt == 0))[o].sum()),
        "fixed_only": int(((tgt_calls == 0) & (fixed_tgt == 1))[o].sum()),
        "distance_and_fixed_both_flag": int(((d_tgt == 1) & (fixed_tgt == 1))[o].sum()),
        "distance_and_local_both_flag": int(((d_tgt == 1) & (tgt_calls == 1))[o].sum()),
    }
    pd.DataFrame({"field_id": tgt["field_id"], "area_ha": a_tgt, "low_evidence": tgt["low_evidence"],
                  "p": p_tgt, "fixed_call": fixed_tgt, "local_call": tgt_calls, "distance_flag": d_tgt}
                 ).to_parquet(out / "target_fields.parquet", index=False)
    (out / "county_summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
