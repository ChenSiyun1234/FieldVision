"""Run registry models on every corn field of one county-year and compare with the USDA Census.

No labels exist for these fields; the county's Census silage share is the only test. Each model is
run with FieldVision's own classify CLI into its own folder, so the team's results/ stay untouched.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..settings import COUNTY_DATA, FIELDVISION, FIELDVISION_DEPS, FIELDVISION_FINAL, QUICKSTATS_CSV as QUICKSTATS, RUNS, WORKSPACE


def classify(model, fips, year, out_root, state="MD"):
    folder = COUNTY_DATA / str(fips) / str(year)
    out = Path(out_root) / model
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(FIELDVISION_FINAL / "src"), str(FIELDVISION_DEPS)])}
    cmd = [sys.executable, "-m", "fieldvision.model.classify", "--model", model,
           "--registry", str(FIELDVISION_FINAL / "registry"),
           "--timeseries", str(folder / "timeseries.parquet"), "--fields", str(folder / "fields.geojson"),
           "--state", state, "--out", str(out)]
    r = subprocess.run(cmd, cwd=FIELDVISION_FINAL, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{model}: {r.stderr[-1500:]}")
    return out / f"{state}_{year}.jsonl"


def census_share(fips, year):
    df = pd.read_csv(QUICKSTATS, dtype=str)
    fips = str(fips).zfill(5)
    d = df[(df.measurement == "harvested") & (df.domain == "TOTAL") & (df.reference_period == "YEAR") &
           (df.aggregation_level == "COUNTY") & (df.source == "CENSUS") & (df.year == str(year)) &
           (df.state_ansi.str.zfill(2) == fips[:2]) & (df.county_ansi.fillna("").str.zfill(3) == fips[2:])]
    v = d.groupby("crop_use")["value"].first().str.replace(",", "").astype(float)
    return float(v["silage"] / (v["silage"] + v["grain"]))


def shares(results_jsonl, fields_geojson, prior):
    """Area-weighted silage shares among decided fields: called silage, mean p, mean p at the county prior."""
    sys.path.insert(0, str(WORKSPACE))
    from fieldvision_agents.fieldvision_bridge.prevalence import shift_probability
    import geopandas as gpd
    area = gpd.read_file(fields_geojson).set_index("field_id")["area_ha"]
    r = pd.read_json(results_jsonl, lines=True)
    key = "source_field_id" if r["source_field_id"].isin(area.index).any() else "field_id"
    r["area_ha"] = r[key].map(area)
    d = r[~r["abstain"].astype(bool) & r["area_ha"].notna()]
    a = d["area_ha"].to_numpy(float)
    p = d["p_silage"].to_numpy(float)
    return {"fields": int(len(r)), "decided": int(len(d)), "abstained": int(r["abstain"].sum()),
            "called_silage_pct": round(100 * float((a * (p >= 0.5)).sum() / a.sum()), 1),
            "mean_p_pct": round(100 * float((a * p).sum() / a.sum()), 1),
            "mean_p_at_prior_pct": round(100 * float((a * shift_probability(p, prior)).sum() / a.sum()), 1)}


def module_v2_headline(fips, year):
    """The final module's own county figure (team Toolbox, policy module_v2)."""
    sys.path.insert(0, str(WORKSPACE))
    from fieldvision_agents.fieldvision_bridge.toolbox import load_team_toolbox
    tb = load_team_toolbox(standard_model="eu2_logistic_v1", module_model="eu2_logistic_v1")
    tb.classify(fips, year, "module_v2")
    agg = tb.aggregate(fips, year)
    keys = ("silage_share_headline", "headline_method", "clear_tier_fields", "clear_tier_share_of_area",
            "clear_silage_share_counted", "clear_wide_share_called_silage", "window_weighted_share")
    return {k: agg.get(k) for k in keys}


def run(fips="24021", year=2022, models=("eu2_logistic_v1", "eu_logistic_v1", "eu_barc_logistic_v1"),
        existing={"composite_v1": "results/MD_2022_24021_standard.jsonl"}, out_root=None):
    sys.path.insert(0, str(WORKSPACE))
    from fieldvision_agents.fieldvision_bridge.prevalence import silage_share_prior
    out_root = Path(out_root or RUNS / "county_check" / f"{fips}_{year}")
    fields = COUNTY_DATA / str(fips) / str(year) / "fields.geojson"
    prior = silage_share_prior(QUICKSTATS, fips, year)
    truth = census_share(fips, year)
    rows = []
    for m in models:
        rows.append({"model": m, **shares(classify(m, fips, year, out_root), fields, prior["share"])})
    for m, rel in existing.items():
        rows.append({"model": m + " (team run)", **shares(FIELDVISION / rel, fields, prior["share"])})
    table = pd.DataFrame(rows)
    out = {"fips": fips, "year": year, "census_silage_pct": round(100 * truth, 1),
           "module_v2": module_v2_headline(fips, year),
           "prior_pct": round(100 * prior["share"], 1), "prior_source": prior["source"], "rows": rows,
           "note": "Model estimates, not NASS statistics. Census share is silage / (silage + grain) harvested acres."}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "comparison.json").write_text(json.dumps(out, indent=1))
    return out, table
