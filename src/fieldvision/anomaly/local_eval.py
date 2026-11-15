"""European check of the local-threshold recipe (anomaly.local) on held-out years / countries.

    python -m fieldvision.anomaly.local_eval --timeseries ... --fields ... --out <dir>

For each held-out group: the main line's 6-feature logistic is trained on the other
groups (labels), source out-of-fold scores come from spatial CV inside those groups only,
and the held-out group is split by block into a reference county (grain + 1.3% silage)
and a test county (grain + 20% silage), many times (anomaly.local.emulate_counties).
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
from ..model.harness import run_splits
from ..model.models import make_factories
from ..model.splits import spatial_folds
from .local import DEFAULT_ALPHA, emulate_counties


META = ["field_id", "block_id", "country", "year", "y", "area_ha"]
SHARES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
ESTIMATORS = ("counted", "adj_eu", "local", "local_pooled", "local_flagged", "local_strat")


def stratum(X):
    """Evidence-quality stratum, fixed in advance: harvest bracket (days between the last
    clear image before the drop and the first after it) at most 10 days, or longer."""
    w = pd.to_numeric(X["q_harv_width_days"], errors="coerce").to_numpy(float)
    return np.where(np.isfinite(w) & (w <= 10), "tight", "loose")


def feature_table(timeseries, fields_files, cache=None):
    if cache is not None and Path(cache).exists():
        d = pd.read_parquet(cache)
        if "area_ha" not in d.columns:          # caches written before area was kept
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                area = load_fields(fields_files).set_index("field_id")["area_ha"]
            d.insert(5, "area_ha", d["field_id"].map(area).to_numpy(float))
            d.to_parquet(cache, index=False)
        meta = d[META + ["low_evidence"]].copy()
        return d.drop(columns=META), meta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = load_fields(fields_files)
        ts, _ = load_timeseries(timeseries, field_ids=fields["field_id"])
        fields = fields[np.isfinite(fields["y"].to_numpy(float))].reset_index(drop=True)
        X, _ = build_features(ts, fields, FeatureConfig())
    X = X.reset_index(drop=True).drop(columns=["field_id"], errors="ignore")
    meta = fields[META].copy()
    meta["year"] = meta["year"].map(lambda v: str(int(v)) if np.isfinite(v) else "unknown")
    meta["low_evidence"] = X["low_evidence"].to_numpy()
    if cache is not None:
        pd.concat([meta.drop(columns="low_evidence"), X], axis=1).to_parquet(cache, index=False)
    return X, meta


def source_and_target_scores(X, meta, test_mask, seed=0, drop_low_evidence=True):
    """(source out-of-fold scores, source y, target scores from a model fit on all source).

    drop_low_evidence  set the out-of-fold score of low-evidence source fields to NaN, so the
                       matched recall is measured on the same kind of field the target
                       estimate uses (targets are scored on fields that are not low evidence;
                       low-evidence fields separate worse, and keeping them in the source
                       understates the recall and overstates the adjusted share)
    """
    make = make_factories(X.columns, "relative", seed=seed)["logistic"]
    y = meta["y"].to_numpy(int)
    src = np.flatnonzero(~test_mask)
    Xs, ms, ys = X.iloc[src].reset_index(drop=True), meta.iloc[src].reset_index(drop=True), y[src]
    folds = [(f"f{i}", tr, te) for i, (tr, te) in enumerate(spatial_folds(ms["block_id"], ys, 5, seed))]
    oof, _, _ = run_splits(Xs, ms, ys, {"lr": make}, folds, "src", "relative", seed=seed)
    oof = oof.set_index("field_id").loc[ms["field_id"]]
    model = make().fit(Xs, ys)
    tgt = meta[test_mask].copy()
    tgt["p"] = model.predict_proba(X[test_mask])
    s = oof["p"].to_numpy(float).copy()
    if drop_low_evidence:
        s[ms["low_evidence"].astype(bool).to_numpy()] = np.nan
    return s, ys, tgt


def _draw(pool, share, n, rng):
    """n fields with silage share `share` by count (at least one of each class)."""
    sil = pool.index[pool["y"] == 1].to_numpy()
    gra = pool.index[pool["y"] == 0].to_numpy()
    k = int(np.clip(round(share * n), 1, n - 1))
    return np.r_[rng.choice(sil, min(k, len(sil)), replace=False),
                 rng.choice(gra, min(n - k, len(gra)), replace=False)]


def county_estimates(c, ref, src, alpha=DEFAULT_ALPHA):
    """All estimators of the area share for one target county `c` (frame y, p, area, stratum,
    abstain), with reference county `ref` and source out-of-fold (p, y, stratum)."""
    from .local import local_share, stratified_share
    sp, sy, ss, sg = src
    ok, rok = ~c["abstain"].to_numpy(), ~ref["abstain"].to_numpy()
    p, a, s = c["p"].to_numpy()[ok], c["area"].to_numpy()[ok], c["stratum"].to_numpy()[ok]
    rp, ra, rs = ref["p"].to_numpy()[rok], ref["area"].to_numpy()[rok], ref["stratum"].to_numpy()[rok]
    counted = float(np.sum(a * (p >= 0.5)) / np.sum(a))
    f = np.isfinite(sp)
    tpr05, fpr05 = float((sp[f & (sy == 1)] >= 0.5).mean()), float((sp[f & (sy == 0)] >= 0.5).mean())
    from .local import adjusted_share
    return {"truth": float(np.sum(c["area"] * c["y"]) / np.sum(c["area"])),
            "counted": counted, "adj_eu": adjusted_share(counted, fpr05, tpr05),
            "local": local_share(rp, sp, sy, p, alpha, ra, a, src_g=sg),
            "local_pooled": local_share(rp, sp, sy, p, alpha, ra, a),
            "local_flagged": local_share(rp, sp, sy, p, alpha, ra, a, adjust=False),
            "local_strat": stratified_share(rp, rs, sp, sy, ss, p, s, alpha, ra, a, src_g=sg)}


def pseudo_counties(pool, src, construction="random", n=250, n_ref=250, ref_share=0.01, reps=200,
                    seed=0, alpha=DEFAULT_ALPHA):
    """Pseudo-counties drawn from one held-out group (as in the modelling track's
    county_validation: a ~1% silage reference county and target counties at SHARES).
    construction 'random': fields drawn at random; 'blocks': reference and target come from
    different random halves of the group's blocks (neighbouring, not identical, counties)."""
    rng = np.random.default_rng(seed)
    blocks = pool["block_id"].unique()
    rows = []
    for rep in range(reps):
        if construction == "blocks":
            half = set(rng.permutation(blocks)[: len(blocks) // 2])
            in_ref = pool["block_id"].isin(half).to_numpy()
            ref_pool, tgt_pool = pool[in_ref], pool[~in_ref]
        else:
            ref_pool = tgt_pool = pool
        ref = pool.loc[_draw(ref_pool, ref_share, n_ref, rng)]
        if construction != "blocks":
            tgt_pool = pool.drop(index=ref.index)
        for share in SHARES:
            c = pool.loc[_draw(tgt_pool, share, n, rng)]
            rows.append({"rep": rep, "target_share": share, **county_estimates(c, ref, src, alpha)})
    return pd.DataFrame(rows)


def summarise(sim):
    out = []
    for share, g in list(sim.groupby("target_share")) + [("all", sim)]:
        r = {"target_share": share, "truth_mean": float(g["truth"].mean())}
        for e in ESTIMATORS:
            err = g[e] - g["truth"]
            r[f"{e}_mae"], r[f"{e}_bias"] = float(err.abs().mean()), float(err.mean())
        out.append(r)
    return pd.DataFrame(out)


def pseudo_main(argv=None):
    """python -m fieldvision.anomaly.local_eval pseudo --cache <features.parquet> --out <dir>"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--fields", nargs="*", default=None, help="field files, to add area to an old cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, meta = feature_table(None, args.fields, args.cache)
    meta = meta.reset_index(drop=True)
    meta["stratum"] = stratum(X)
    groups = [(f"loyo:{g}", (meta["year"] == g).to_numpy()) for g in sorted(meta["year"].unique())]
    groups += [(f"loco:{g}", (meta["country"] == g).to_numpy()) for g in sorted(meta["country"].unique())
               if meta["country"].nunique() > 1]
    allsum = []
    for name, mask in groups:
        s_src, y_src, tgt = source_and_target_scores(X, meta, mask, args.seed)
        src_meta = meta.loc[~mask]
        src = (s_src, y_src, src_meta["stratum"].to_numpy(),
               (src_meta["country"].astype(str) + src_meta["year"].astype(str)).to_numpy())
        pool = pd.DataFrame({"y": tgt["y"].to_numpy(int), "p": tgt["p"].to_numpy(),
                             "area": tgt["area_ha"].to_numpy(float), "stratum": tgt["stratum"].to_numpy(),
                             "abstain": tgt["low_evidence"].astype(bool).to_numpy(),
                             "block_id": tgt["block_id"].to_numpy()}, index=tgt["field_id"].to_numpy())
        pool = pool[np.isfinite(pool["area"])]
        for construction in ("random", "blocks"):
            sim = pseudo_counties(pool, src, construction, reps=args.reps, seed=args.seed)
            sim.to_csv(out / f"sim_{name.replace(':', '_')}_{construction}.csv", index=False)
            tab = summarise(sim)
            tab.insert(0, "construction", construction)
            tab.insert(0, "held_out", name)
            allsum.append(tab)
            r = tab[tab["target_share"] == "all"].iloc[0]
            print(name, construction, " ".join(f"{e} {r[f'{e}_mae']:.3f}" for e in ESTIMATORS), flush=True)
    pd.concat(allsum, ignore_index=True).to_csv(out / "pseudo_county_summary.csv", index=False)


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "pseudo":
        return pseudo_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[DEFAULT_ALPHA, 0.05, 0.20])
    ap.add_argument("--n-rep", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, meta = feature_table(args.timeseries, args.fields, out / "features.parquet")
    groups = []
    for col in ("year", "country"):
        if meta[col].nunique() > 1:
            groups += [(f"{'loyo' if col == 'year' else 'loco'}:{g}", (meta[col] == g).to_numpy())
                       for g in sorted(meta[col].unique())]
    rows, reps = [], []
    for name, mask in groups:
        s_src, y_src, tgt = source_and_target_scores(X, meta, mask, args.seed, drop_low_evidence=False)
        for a in args.alphas:
            r = emulate_counties(tgt, s_src, y_src, alpha=a, n_rep=args.n_rep, seed=args.seed)
            r["held_out"], r["alpha"] = name, a
            reps.append(r)
            q = lambda c: (float(r[c].mean()), float(r[c].quantile(0.05)), float(r[c].quantile(0.95)))
            rows.append({"held_out": name, "alpha": a, "n_ref": int(r["n_ref"].mean()),
                         "n_test": int(r["n_test"].mean()), "true_share": float(r["true_share"].mean()),
                         **{k: q(k) for k in ("fixed_fpr", "fixed_recall", "fixed_share", "local_fpr",
                                              "local_recall", "local_flag_share", "local_adjusted_share")},
                         "tpr_source": float(r["tpr_source"].mean())})
            print(name, a, {k: round(v[0], 3) if isinstance(v, tuple) else v for k, v in rows[-1].items()},
                  flush=True)
    pd.concat(reps, ignore_index=True).to_csv(out / "local_threshold_reps.csv", index=False)
    (out / "local_threshold_summary.json").write_text(json.dumps({"args": vars(args), "rows": rows},
                                                                 indent=1, default=str), encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
