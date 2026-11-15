"""Do county-total estimators recover a known silage share? Checked on European pseudo-counties.

    python -m fieldvision.model.county_validation --run <run dir with composite out-of-fold
        predictions> --rates-run <run dir the error rates come from> --out <dir>

A pseudo-county is a random set of held-out fields (out-of-fold predictions, so no field was
seen by the model that scored it) drawn to a chosen silage share. A separate pseudo-county with
about 1 % silage plays the reference county. Estimators of the area share:

  raw        sum of area x p over the non-abstained area (assumes the training balance)
  counted    share of the non-abstained area called silage (the module's headline so far)
  adj_eu     counted share corrected with TPR and FPR measured in Europe:
             (counted - FPR) / (TPR - FPR)
  ref        counted share corrected with the FPR measured in the reference county:
             FPR_ref = (counted_ref - TPR * s_ref) / (1 - s_ref), s_ref the reference county's
             known share (in the US: its Census share), TPR from Europe
  em         prior shift by expectation maximisation from the county's own probabilities

Two settings: the fields come from the same year the rates were measured in (spatial CV), or
from a year the model never saw (leave-one-year-out), with the rates taken from the training
year, which is how it would be used in a new US season.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..model.artifacts import write_json
from ..model.registry import COMPOSITE_MEMBERS, rates_from_run

SHARES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def composite_pool(run_dir, scheme, year=None, spread_limit=0.5):
    """Out-of-fold composite probabilities, abstain flag and area for one scheme of a run."""
    cv = pd.read_parquet(Path(run_dir) / "cv_predictions.parquet")
    variant = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))["variant"]
    g = cv[(cv["variant"] == variant) & (cv["scheme"] == scheme)]
    P = g[g["model"].isin(COMPOSITE_MEMBERS)].pivot(index="field_id", columns="model", values="p")
    meta = g.drop_duplicates("field_id").set_index("field_id").loc[P.index]
    feats = pd.read_parquet(Path(run_dir) / "features.parquet", columns=["field_id", "area_ha"])
    area = feats.set_index("field_id")["area_ha"].reindex(P.index).to_numpy(float)
    votes = (P.to_numpy() >= 0.5).sum(axis=1)
    spread = (P.max(axis=1) - P.min(axis=1)).to_numpy()
    pool = pd.DataFrame({"y": meta["y"].to_numpy(int), "p": P.mean(axis=1).to_numpy(),
                         "abstain": meta["low_evidence"].to_numpy(bool)
                         | ((votes > 0) & (votes < P.shape[1]) & (spread >= spread_limit)),
                         "area": area, "year": meta["year"].astype(str).to_numpy()}, index=P.index)
    if year is not None:
        pool = pool[pool["year"] == str(year)]
    return pool[np.isfinite(pool["area"])]


def draw_county(pool, share, n, rng, exclude=()):
    """n fields whose silage share by count is `share` (at least one of each class)."""
    avail = pool.drop(index=list(exclude), errors="ignore")
    sil = avail.index[avail["y"] == 1].to_numpy()
    gra = avail.index[avail["y"] == 0].to_numpy()
    k = int(np.clip(round(share * n), 1, n - 1))
    ids = np.r_[rng.choice(sil, size=min(k, len(sil)), replace=False),
                rng.choice(gra, size=min(n - k, len(gra)), replace=False)]
    return avail.loc[ids]


def em_share(p, a, train_prior=0.5, tol=1e-6, max_iter=500):
    pi = 0.5
    p = np.clip(p, 1e-6, 1 - 1e-6)
    for _ in range(max_iter):
        num = p * pi / train_prior
        q = num / (num + (1 - p) * (1 - pi) / (1 - train_prior))
        new = float(np.sum(a * q) / np.sum(a))
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def local_threshold_share(c, ref, source, alpha=0.10, tpr_offset=0.0):
    """The local-threshold county layer (fieldvision.anomaly.local), area-weighted here.

    Threshold = the reference county's (1 - alpha) quantile of scores (labels not read);
    TPR = share of source silage above the score that flags alpha of source grain
    (out-of-fold source scores); share = (flagged - reference flagged) / (TPR - reference
    flagged), clipped to [0, 1]. Non-abstained fields only.
    """
    rok = ~ref["abstain"].to_numpy()
    rp, ra = ref["p"].to_numpy()[rok], ref["area"].to_numpy()[rok]
    t = float(np.quantile(rp, 1 - alpha))
    ref_flag = float(np.sum(ra * (rp >= t)) / np.sum(ra))
    sp, sy = source
    t_s = float(np.quantile(sp[sy == 0], 1 - alpha))
    tpr = float((sp[sy == 1] >= t_s).mean()) + (tpr_offset or 0.0)
    ok = ~c["abstain"].to_numpy()
    p, a = c["p"].to_numpy()[ok], c["area"].to_numpy()[ok]
    flag = float(np.sum(a * (p >= t)) / np.sum(a))
    if tpr_offset is None:          # the plain locally flagged share, no adjustment
        return flag
    den = tpr - ref_flag
    return float(np.clip((flag - ref_flag) / den, 0, 1)) if den > 1e-9 else np.nan


def estimates(c, tpr, fpr, ref_counted=None, ref_share=None, ref=None, source=None, tpr_offset=0.0):
    ok = ~c["abstain"].to_numpy()
    a, p = c["area"].to_numpy()[ok], c["p"].to_numpy()[ok]
    called = p >= 0.5
    counted = float(np.sum(a * called) / np.sum(a))

    def adj(fp):
        den = tpr - fp
        return float(np.clip((counted - fp) / den, 0, 1)) if den > 0.05 else np.nan
    out = {"truth": float(np.sum(c["area"] * c["y"]) / np.sum(c["area"])),
           "truth_not_abstained": float(np.sum(a * c["y"].to_numpy()[ok]) / np.sum(a)),
           "raw": float(np.sum(a * p) / np.sum(a)), "counted": counted, "adj_eu": adj(fpr),
           "em": em_share(p, a)}
    if ref_counted is not None:
        fpr_ref = float(np.clip((ref_counted - tpr * ref_share) / (1 - ref_share), 0, 0.95))
        out["ref"] = adj(fpr_ref)
        out["fpr_ref"] = fpr_ref
    if ref is not None and source is not None:
        out["local"] = local_threshold_share(c, ref, source, tpr_offset=tpr_offset)
        out["local_flagged"] = local_threshold_share(c, ref, source, tpr_offset=None)
    return out


def counted_share(c):
    ok = ~c["abstain"].to_numpy()
    a = c["area"].to_numpy()[ok]
    return float(np.sum(a * (c["p"].to_numpy()[ok] >= 0.5)) / np.sum(a))


def simulate(pool, tpr, fpr, n_fields=250, n_ref=250, ref_share=0.01, reps=200, seed=0,
             tpr_offsets=(0.0,), source=None):
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        ref = draw_county(pool, ref_share, n_ref, rng)
        ref_counted = counted_share(ref)
        true_ref = float(np.sum(ref["area"] * ref["y"]) / np.sum(ref["area"]))
        for share in SHARES:
            c = draw_county(pool, share, n_fields, rng, exclude=ref.index)
            for off in tpr_offsets:
                e = estimates(c, tpr + off, fpr, ref_counted, true_ref, ref=ref, source=source,
                              tpr_offset=off)
                rows.append({"rep": rep, "target_share": share, "tpr_offset": off, **e})
    return pd.DataFrame(rows)


ESTIMATORS = ("raw", "counted", "adj_eu", "ref", "local", "local_flagged", "em")


def summarise(sim, setting):
    out = []
    for (share, off), g in sim.groupby(["target_share", "tpr_offset"]):
        r = {"setting": setting, "target_share": share, "tpr_offset": off,
             "truth_mean": g["truth"].mean()}
        for e in ESTIMATORS:
            err = g[e] - g["truth"]
            r[f"{e}_bias"] = float(err.mean())
            r[f"{e}_mae"] = float(err.abs().mean())
        out.append(r)
    tab = pd.DataFrame(out)
    overall = {"setting": setting, "target_share": "all", "tpr_offset": tab["tpr_offset"].iloc[0]}
    for e in ESTIMATORS:
        overall[f"{e}_mae"] = float((sim[e] - sim["truth"]).abs().mean())
        overall[f"{e}_bias"] = float((sim[e] - sim["truth"]).mean())
    return tab, overall


def main(argv=None):
    ap = argparse.ArgumentParser(description="County estimators on European pseudo-counties.")
    ap.add_argument("--run", required=True, help="run with Austria 2022 + 2024 out-of-fold predictions")
    ap.add_argument("--rates-run", required=True, help="run whose spatial CV gives the European rates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rates = rates_from_run(a.rates_run, "composite")["spatial"]
    tpr, fpr = rates["tpr"], rates["fpr"]
    # source scores for the local-threshold layer: out-of-fold composite scores of the rates run
    src = composite_pool(a.rates_run, "spatial")
    src = src[~src["abstain"]]
    source = (src["p"].to_numpy(), src["y"].to_numpy(int))
    settings = {
        "same year (2024 fields, spatial CV)": composite_pool(a.run, "spatial", year=2024),
        "held-out year (2022 fields, model trained on 2024)": composite_pool(a.run, "loyo:2022"),
    }
    tables, overall = [], []
    for name, pool in settings.items():
        sim = simulate(pool, tpr, fpr, reps=a.reps, seed=a.seed, tpr_offsets=(0.0, -0.1, 0.1),
                       source=source)
        sim.to_csv(out / f"sim_{len(tables)}.csv", index=False)
        for off in (0.0, -0.1, 0.1):
            t, o = summarise(sim[sim["tpr_offset"] == off], name)
            tables.append(t)
            overall.append(o)
    tab = pd.concat(tables, ignore_index=True)
    tab.to_csv(out / "pseudo_county_table.csv", index=False)
    ov = pd.DataFrame(overall)
    ov.to_csv(out / "pseudo_county_overall.csv", index=False)
    write_json(out / "pseudo_county_rates.json", {"tpr": tpr, "fpr": fpr, "source": "composite, spatial CV"})
    pd.set_option("display.width", 250)
    print(f"European rates (composite, spatial CV of the rates run): TPR {tpr:.3f}, FPR {fpr:.3f}")
    print(ov.round(4).to_string(index=False))
    show = tab[tab["tpr_offset"] == 0.0][["setting", "target_share", "truth_mean"]
                                          + [f"{e}_mae" for e in ESTIMATORS]]
    print(show.round(3).to_string(index=False))
    return tab, ov


if __name__ == "__main__":
    main()
