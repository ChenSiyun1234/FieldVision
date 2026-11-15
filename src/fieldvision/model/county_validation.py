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
                         "area": area, "year": meta["year"].astype(str).to_numpy(),
                         "country": meta["country"].astype(str).to_numpy()}, index=P.index)
    if year is not None:
        pool = pool[pool["year"] == str(year)]
    return pool[np.isfinite(pool["area"])]


def model_pool(run_dir, scheme, model="composite", year=None, max_bracket=None, spread_limit=0.5):
    """Out-of-fold pool for one model of a run ('composite' = member average); with
    max_bracket, fields whose harvest bracket is wider count as abstained (clear tier only)."""
    if model == "composite":
        pool = composite_pool(run_dir, scheme, spread_limit=spread_limit)
    else:
        cv = pd.read_parquet(Path(run_dir) / "cv_predictions.parquet")
        variant = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))["variant"]
        m = cv[(cv["variant"] == variant) & (cv["scheme"] == scheme)
               & (cv["model"] == model)].set_index("field_id")
        feats = pd.read_parquet(Path(run_dir) / "features.parquet", columns=["field_id", "area_ha"])
        pool = pd.DataFrame({"y": m["y"].to_numpy(int), "p": m["p"].to_numpy(float),
                             "abstain": m["low_evidence"].to_numpy(bool),
                             "area": feats.set_index("field_id")["area_ha"].reindex(m.index)
                             .to_numpy(float),
                             "year": m["year"].astype(str).to_numpy(),
                             "country": m["country"].astype(str).to_numpy()}, index=m.index)
    if max_bracket is not None:
        br = pd.read_parquet(Path(run_dir) / "features.parquet",
                             columns=["field_id", "q_harv_width_days"]).set_index("field_id")
        width = br["q_harv_width_days"].reindex(pool.index).to_numpy(float)
        pool["abstain"] = pool["abstain"].to_numpy(bool) | ~(width <= max_bracket)
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
    """The local-threshold county layer, area-weighted, using fieldvision.anomaly.local.

    Threshold = the reference county's (1 - alpha) quantile of scores over its non-abstained
    fields (labels not read); matched recall from non-abstained source fields, measured inside
    each source domain (country-year) and averaged (anomaly.local.matched_tpr); share =
    adjusted_share(target flagged, reference flagged, recall). Non-abstained fields only.
    `source` = (scores, labels, groups) or (scores, labels).
    """
    from ..anomaly.local import adjusted_share, matched_tpr
    rok = ~ref["abstain"].to_numpy()
    rp, ra = ref["p"].to_numpy()[rok], ref["area"].to_numpy()[rok]
    t = float(np.quantile(rp, 1 - alpha))
    ref_flag = float(np.sum(ra * (rp >= t)) / np.sum(ra))
    ok = ~c["abstain"].to_numpy()
    p, a = c["p"].to_numpy()[ok], c["area"].to_numpy()[ok]
    flag = float(np.sum(a * (p >= t)) / np.sum(a))
    if tpr_offset is None:          # the plain locally flagged share, no adjustment
        return flag
    sp, sy = source[0], source[1]
    sg = source[2] if len(source) > 2 else None
    _, tpr = matched_tpr(sp, sy, alpha, sg)
    return adjusted_share(flag, ref_flag, tpr + tpr_offset)


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


def heldout_source(run, scheme, model, max_bracket=None):
    """Rates and source scores for one setting: the run's spatial out-of-fold fields that are not
    in the held-out group (loco:X drops country X, loyo:Y drops year Y), non-abstained only.
    Returns (tpr, fpr, (scores, labels, country-year groups))."""
    src = model_pool(run, "spatial", model, max_bracket=max_bracket)
    src = src[~src["abstain"]]
    kind, _, grp = scheme.partition(":")
    if kind == "loco":
        src = src[src["country"] != grp]
    elif kind == "loyo":
        src = src[src["year"] != grp]
    y, call = src["y"].to_numpy(int), src["p"].to_numpy() >= 0.5
    tpr = float((call & (y == 1)).sum() / max((y == 1).sum(), 1))
    fpr = float((call & (y == 0)).sum() / max((y == 0).sum(), 1))
    groups = (src["country"] + "-" + src["year"]).to_numpy()
    return tpr, fpr, (src["p"].to_numpy(), y, groups)


def parse_setting(text):
    """NAME=RUN_DIR|SCHEME[|YEAR]"""
    name, rest = text.split("=", 1)
    parts = rest.split("|")
    return name, parts[0], parts[1], (parts[2] if len(parts) > 2 else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description="County estimators on European pseudo-counties.")
    ap.add_argument("--run", help="run with Austria 2022 + 2024 out-of-fold predictions "
                                  "(default settings; not needed with --setting)")
    ap.add_argument("--rates-run", required=True, help="run whose spatial CV gives the European rates "
                                                      "and the source scores of the local threshold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="composite", help="model name in the runs (composite = member mean)")
    ap.add_argument("--max-bracket", type=float, default=None,
                    help="clear tier only: fields with a wider harvest bracket count as abstained")
    ap.add_argument("--setting", action="append", default=[],
                    help="NAME=RUN_DIR|SCHEME[|YEAR], repeatable; replaces the default settings")
    ap.add_argument("--n-fields", type=int, default=250)
    ap.add_argument("--source-from-run", action="store_true",
                    help="per setting, take the European rates and the local threshold's source "
                         "from the setting's own run: spatial out-of-fold fields outside the "
                         "held-out country or year, recall measured per country-year")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rates = rates_from_run(a.rates_run, a.model, max_bracket=a.max_bracket)["spatial"]
    tpr, fpr = rates["tpr"], rates["fpr"]
    # source scores for the local-threshold layer: out-of-fold scores of the rates run
    src = model_pool(a.rates_run, "spatial", a.model, max_bracket=a.max_bracket)
    src = src[~src["abstain"]]
    source = (src["p"].to_numpy(), src["y"].to_numpy(int))
    sources = {}
    if a.setting:
        settings = {}
        for text in a.setting:
            name, run, scheme, year = parse_setting(text)
            settings[name] = model_pool(run, scheme, a.model, year=year, max_bracket=a.max_bracket)
            if a.source_from_run:
                sources[name] = heldout_source(run, scheme, a.model, a.max_bracket)
    else:
        settings = {
            "same year (2024 fields, spatial CV)": model_pool(a.run, "spatial", a.model, year=2024,
                                                              max_bracket=a.max_bracket),
            "held-out year (2022 fields, model trained on 2024)": model_pool(
                a.run, "loyo:2022", a.model, max_bracket=a.max_bracket),
        }
    tables, overall = [], []
    for k, (name, pool) in enumerate(settings.items()):
        s_tpr, s_fpr, s_source = sources.get(name, (tpr, fpr, source))
        sim = simulate(pool, s_tpr, s_fpr, n_fields=a.n_fields, n_ref=a.n_fields, reps=a.reps,
                       seed=a.seed, tpr_offsets=(0.0, -0.1, 0.1), source=s_source)
        sim.to_csv(out / f"sim_{k}.csv", index=False)
        for off in (0.0, -0.1, 0.1):
            t, o = summarise(sim[sim["tpr_offset"] == off], name)
            tables.append(t)
            overall.append(o)
    tab = pd.concat(tables, ignore_index=True)
    tab.to_csv(out / "pseudo_county_table.csv", index=False)
    ov = pd.DataFrame(overall)
    ov.to_csv(out / "pseudo_county_overall.csv", index=False)
    write_json(out / "pseudo_county_rates.json",
               {"tpr": tpr, "fpr": fpr, "model": a.model, "max_bracket": a.max_bracket,
                "source": "spatial CV of the rates run"})
    pd.set_option("display.width", 250)
    print(f"European rates ({a.model}, spatial CV of the rates run): TPR {tpr:.3f}, FPR {fpr:.3f}")
    print(ov.round(4).to_string(index=False))
    show = tab[tab["tpr_offset"] == 0.0][["setting", "target_share", "truth_mean"]
                                          + [f"{e}_mae" for e in ESTIMATORS]]
    print(show.round(3).to_string(index=False))
    return tab, ov


if __name__ == "__main__":
    main()
