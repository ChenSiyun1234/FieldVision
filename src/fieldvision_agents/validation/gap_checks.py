"""Checks on the seven known gaps, from results already on disk. Nothing is re-trained or tuned.

    python -m fieldvision_agents.validation.gap_checks        # writes handoff/results/agents/gap_checks.{json,md}

1. ground truth    how wide the Beltsville score really is (44 field-years), and the sampling range on county shares
2. Corn Belt       the European-trained model against NASS 2025 in Wisconsin, Iowa and Nebraska
3. year gap        how much of Lancaster's 2023-vs-2022-Census gap a one-year change can explain
4. unjudged fields bounds on the county share if the fields we did not judge were all grain, all silage, or as the model leans
5. price           whether our silage error is small enough to move a price model
6. blind test      every comparison where the USDA answer was not available to the pipeline or the critic
7. scale           see scripts/pod_batch_counties.py

Model estimates, not NASS statistics.
"""

import json
import math

import numpy as np
import pandas as pd

from ..settings import COUNTY_DATA, REPO

AGENTS = REPO / "handoff" / "results" / "agents"
RESULTS = REPO / "handoff" / "results"
NATIONAL_ACRES_PER_POINT = 971_350          # 1 point of silage share of US harvested corn, 2025 (price_impact.json)


def _json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def wilson(k, n, z=1.96):
    """95% interval for a proportion k/n."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def _fields(fips, year):
    """One row per field: acres, call, the model's p(silage), evidence tier and abstain reason."""
    from ..app import dairy
    sm = dairy._silage_match()
    p = pd.read_parquet(sm.prediction_path(fips, year, dairy._ensure_predictions(fips, year)))
    ev = p["evidence"].map(lambda e: e if isinstance(e, dict) else json.loads(e or "{}"))
    p["tier"], p["reason"] = ev.map(lambda e: e.get("evidence_tier")), ev.map(lambda e: e.get("abstain_reason"))
    g = json.loads((COUNTY_DATA / fips / str(year) / "fields.geojson").read_text())["features"]
    acres = {str(f["properties"]["field_id"]): float(f["properties"].get("acres_reported")
                                                     or (f["properties"].get("area_ha") or 0) * 2.47105) for f in g}
    p["acres"] = p["field_id"].astype(str).map(acres).fillna(0.0)
    p["call"] = np.where(p["abstain"], "not judged", p["label"])
    return p


def _bootstrap_share(d, n=2000, seed=0):
    """Acre-weighted silage share of the clear tier, resampling fields. Sampling spread only, not model error."""
    d = d[(d["tier"] == "clear") & (d["call"] != "not judged")]
    if len(d) < 20:
        return None
    rng = np.random.default_rng(seed)
    a, s = d["acres"].to_numpy(), (d["call"] == "silage").to_numpy()
    idx = rng.integers(0, len(d), size=(n, len(d)))
    shares = (a[idx] * s[idx]).sum(1) / a[idx].sum(1)
    return {"share_pct": round(100 * (a * s).sum() / a.sum(), 1),
            "range_95_pct": [round(float(np.percentile(shares, 2.5)) * 100, 1),
                             round(float(np.percentile(shares, 97.5)) * 100, 1)], "clear_fields": int(len(d))}


# 1 ------------------------------------------------------------------------------------------------------------
def ground_truth():
    out = {"beltsville": {"right": 42, "of": 44, "right_range_95_pct": wilson(42, 44),
                          "always_grain": 32, "always_grain_range_95_pct": wilson(32, 44),
                          "silage_found": 11, "silage_total": 12, "silage_found_range_95_pct": wilson(11, 12),
                          "source": "handoff/results/REPORT.md, scored once"},
           "county_share_ranges": {}}
    for fips, year in (("24021", 2022), ("42071", 2023)):
        if (COUNTY_DATA / fips / str(year) / "timeseries.parquet").is_file():
            out["county_share_ranges"][f"{fips}-{year}"] = _bootstrap_share(_fields(fips, year))
    b = out["beltsville"]
    out["finding"] = (f"42 of 44 is {b['right_range_95_pct'][0]}–{b['right_range_95_pct'][1]}% right at 95%; always-grain is "
                      f"{b['always_grain_range_95_pct'][0]}–{b['always_grain_range_95_pct'][1]}%. The ranges do not overlap, "
                      "so the model beats the baseline, but 44 field-years from one farm cannot say how it does elsewhere. "
                      "No other US field labels are public; county shares are checked against the Census instead.")
    return out


# 2 ------------------------------------------------------------------------------------------------------------
def corn_belt():
    s = _json(AGENTS / "states_2025_comparison.json")
    rows = []
    for r in (s or {}).get("shares", []):
        if r["model"] != "eu2_logistic_v1":
            continue
        k = round(r["headline_pct"] / 100 * r["clear_fields"])
        ci = wilson(k, r["clear_fields"])
        rows.append({"state": r["state"], "ours_pct": r["headline_pct"], "usda_2025_pct": r["usda_2025_silage_pct"],
                     "error_pts": round(r["headline_pct"] - r["usda_2025_silage_pct"], 1), "clear_fields": r["clear_fields"],
                     "ours_range_95_pct": ci, "usda_inside_range": ci[0] <= r["usda_2025_silage_pct"] <= ci[1],
                     "corn_belt": r["state"] in ("IA", "NE", "WI")})
    belt = [r for r in rows if r["corn_belt"]]
    return {"states": rows,
            "corn_belt_mean_abs_error_pts": round(float(np.mean([abs(r["error_pts"]) for r in belt])), 1) if belt else None,
            "finding": _corn_belt_finding(rows)}


def _corn_belt_finding(rows):
    belt = [r for r in rows if r["corn_belt"]]
    if not belt:
        return None
    errs = ", ".join(f"{r['state']} {r['error_pts']:+}" for r in belt)
    inside = [r["state"] for r in belt if r["usda_inside_range"]]
    east = "".join(f" {r['state']} is {abs(r['error_pts'])} points {'low' if r['error_pts'] < 0 else 'high'}, so the big miss "
                   "is in the East, not the Corn Belt." for r in rows if not r["corn_belt"])
    return (f"In the Corn Belt the model lands within {max(abs(r['error_pts']) for r in belt)} points of NASS 2025 ({errs}); "
            f"USDA's figure is inside our sampling range in {', '.join(inside) or 'no state'}. Where silage is rare a few "
            "points is still a large relative miss, so low-silage states need a higher bar before a field is called "
            f"silage.{east} Each state rests on {min(r['clear_fields'] for r in belt)}–{max(r['clear_fields'] for r in belt)} "
            "clearly seen fields.")


# 3 ------------------------------------------------------------------------------------------------------------
def year_gap():
    la = _json(AGENTS / "lancaster_2023_county_check.json")
    if not la:
        return None
    from ..app import geo
    h = [x for x in la["census_history"] if x.get("silage_share_pct") is not None]
    yrs, pct = np.array([x["year"] for x in h]), np.array([x["silage_share_pct"] for x in h])
    slope, icept = np.polyfit(yrs, pct, 1)
    trend_2023 = round(float(slope * 2023 + icept), 1)
    a = geo._acres()
    pa = a[(a["level"] == "state") & (a["state"] == "42") & (a["source"] == "SURVEY")].set_index("year")["silage_share_pct"]
    pa_2022 = float(pa[2022]) if 2022 in pa else float(np.interp(2022, pa.index, pa.values))   # no 2022 survey row: interpolate
    pa_2023 = float(pa[2023])
    expected = round(la["census_silage_pct"] * pa_2023 / pa_2022, 1)       # Lancaster keeps its 2022 ratio to the state
    ours = round(100 * la["module_v2"]["silage_share_headline"], 1)
    rng = (_bootstrap_share(_fields("42071", 2023)) or {}).get("range_95_pct")
    gap = round(la["census_silage_pct"] and ours - la["census_silage_pct"], 1)
    return {"ours_2023_pct": ours, "ours_range_95_pct": rng, "census_2022_pct": la["census_silage_pct"],
            "census_trend_pts_per_year": round(float(slope), 2), "census_trend_2023_pct": trend_2023,
            "pa_survey_2022_pct": round(pa_2022, 1), "pa_survey_2022_interpolated": 2022 not in pa,
            "pa_survey_2023_pct": round(pa_2023, 1), "expected_2023_pct": expected,
            "gap_vs_census_pts": gap, "gap_explained_by_year_pts": round(expected - la["census_silage_pct"], 1),
            "gap_left_pts": round(ours - expected, 1),
            "finding": (f"Pennsylvania's survey share jumped to {pa_2023:.1f}% in 2023 from about {pa_2022:.1f}% in 2022, so "
                        f"if Lancaster kept its 2022 ratio to the state its 2023 share would be about {expected}%. That "
                        f"explains {expected - la['census_silage_pct']:.1f} of the {gap} points between us and the 2022 Census; "
                        f"{ours - expected:.1f} points are left, at the edge of our sampling range "
                        f"({rng[0]}–{rng[1]}%). The Census trend alone ({slope:+.1f} points a year) would point the other way, "
                        "so the year gap is real but does not close the difference.")}


# 4 ------------------------------------------------------------------------------------------------------------
def unjudged():
    out = {}
    for fips, year, name in (("24021", 2022, "Frederick 2022"), ("42071", 2023, "Lancaster 2023")):
        if not (COUNTY_DATA / fips / str(year) / "timeseries.parquet").is_file():
            continue
        d = _fields(fips, year)
        acres = d.groupby("call")["acres"].sum()
        S, G, A = (float(acres.get(k, 0)) for k in ("silage", "grain", "not judged"))
        basis = "field outlines (each field's own acres)"
        saved = _json(AGENTS / "lancaster_2023_county_check.json") if fips == "42071" else None
        if saved:   # the agent's aggregate weights fields by corn cover: use the acres the critic checked
            S, G, A = (float(saved["acres"][k]) for k in ("silage_acres", "grain_acres", "abstained_acres"))
            basis = "the agent's county aggregate (the acres the critic checked)"
        nj = d[d["call"] == "not judged"]
        judged = d[d["call"] != "not judged"]
        lean = float((nj["p_silage"] * nj["acres"]).sum() / nj["acres"].sum()) if len(nj) else None
        out[name] = {
            "acres_basis": basis, "fields": int(len(d)), "fields_not_judged": int(len(nj)),
            "not_judged_share_of_acres_pct": round(100 * A / (S + G + A), 1),
            "share_if_all_grain_pct": round(100 * S / (S + G + A), 1),
            "share_if_all_silage_pct": round(100 * (S + A) / (S + G + A), 1),
            "share_judged_only_pct": round(100 * S / (S + G), 1),
            "share_with_model_lean_pct": round(100 * (S + (lean or 0) * A) / (S + G + A), 1),
            "model_lean_on_not_judged_pct": round(100 * lean, 1) if lean is not None else None,
            "median_acres_judged": round(float(judged["acres"].median()), 1),
            "median_acres_not_judged": round(float(nj["acres"].median()), 1) if len(nj) else None,
            "reasons": nj["reason"].fillna("unknown").value_counts().to_dict()}
    la = out.get("Lancaster 2023")
    out["finding"] = (None if not la else
                      f"In Lancaster the share could be anywhere from {la['share_if_all_grain_pct']}% to "
                      f"{la['share_if_all_silage_pct']}% depending on the unjudged fields. Filling them with the model's own "
                      f"lean gives {la['share_with_model_lean_pct']}%. Most unjudged fields were cut inside a long gap between "
                      "clear images (wide_harvest_interval); radar (Sentinel-1) is the fix, and the 'radar_first' route is wired for it, "
                      "but these counties have no radar series extracted yet.")
    return out


# 5 ------------------------------------------------------------------------------------------------------------
def price():
    p = _json(AGENTS / "price_impact.json")
    if not p:
        return None
    errs = [r["error_pts"] for r in corn_belt()["states"]]
    signed = p["our_four_states_2025"]["silage_acres_ours_minus_usda"]
    pts = signed / NATIONAL_ACRES_PER_POINT
    effects = [v["price_change_pct"] for v in p["silage_plus_1pt_2025"].values()]
    loyo = {r["model"]: r["mae_pct"] for r in p["leave_one_year_out_2001_2024"]}
    return {"our_net_error_4_states_acres": signed, "as_national_points": round(pts, 2),
            "price_effect_of_that_error_pct": [round(pts * min(effects), 2), round(pts * max(effects), 2)],
            "state_errors_pts": [min(errs), max(errs)], "loyo_mae_pct": loyo,
            "finding": (f"Our four-state silage error nets out to {signed:,} acres, about {pts:.2f} points of US corn, which "
                        f"the price models turn into {pts * min(effects):.1f}–{pts * max(effects):.1f}% on price. State errors run "
                        f"from {min(errs)} to +{max(errs)} points, so the sign is not settled. On clean leave-one-year-out "
                        f"tests no model beats last year's price by much ({loyo.get('no_change')}% vs "
                        f"{min(v for k, v in loyo.items() if k != 'no_change')}% average error). So the price page stays a "
                        "labelled what-if; it is not a forecast and our silage map should not feed one yet.")}


# 6 ------------------------------------------------------------------------------------------------------------
def blind_test():
    rows = []
    fr = _json(AGENTS / "frederick_2022_county_check.json")
    if fr:
        rows.append({"place": "Frederick County, MD", "season": 2022, "ours_pct": round(100 * fr["module_v2"]["silage_share_headline"], 1),
                     "usda_pct": fr["census_silage_pct"], "usda_source": "2022 Census",
                     "hidden": "yes: the critic only reads Census years before 2022 (2012, 2017)"})
    md = _json(RESULTS / "state_estimate.json")
    if md:
        rows.append({"place": "Maryland", "season": 2025, "ours_pct": round(100 * md["headline_share"], 1), "usda_pct": 8.9,
                     "usda_source": "NASS 2025 survey", "hidden": "yes: the state run used 2022 Census reference blocks"})
    for r in corn_belt()["states"]:
        rows.append({"place": r["state"], "season": 2025, "ours_pct": r["ours_pct"], "usda_pct": r["usda_2025_pct"],
                     "usda_source": "NASS 2025 survey", "hidden": "yes: priors come from earlier years"})
    la = _json(AGENTS / "lancaster_2023_county_check.json")
    if la:
        rows.append({"place": "Lancaster County, PA", "season": 2023, "ours_pct": round(100 * la["module_v2"]["silage_share_headline"], 1),
                     "usda_pct": la["census_silage_pct"], "usda_source": "2022 Census (no 2023 county figure)",
                     "hidden": "no: nearest year, and the critic saw it"})
    for r in rows:
        r["error_pts"] = round(r["ours_pct"] - r["usda_pct"], 1)
    blind = [r for r in rows if r["hidden"].startswith("yes")]
    within5 = sum(abs(r["error_pts"]) <= 5 for r in blind)
    return {"rows": rows, "blind_n": len(blind), "blind_within_5_pts": within5,
            "blind_mean_abs_error_pts": round(float(np.mean([abs(r["error_pts"]) for r in blind])), 1),
            "finding": (f"{within5} of {len(blind)} blind comparisons land within 5 points of USDA. The misses: "
                        + ", ".join(f"{r['place']} {r['error_pts']:+}" for r in blind if abs(r["error_pts"]) > 5)
                        + ". Frederick's critic range (24.3–32.1%, from the 2012 and 2017 Census) would only have warned "
                        "on the true 19.6% as well, so the history check cannot see a real change coming.")}


def build():
    return {"ground_truth": ground_truth(), "corn_belt": corn_belt(), "year_gap": year_gap(), "unjudged": unjudged(),
            "price": price(), "blind_test": blind_test(),
            "scale": {"script": "scripts/pod_batch_counties.py",
                      "finding": "Satellite pulls are the bottleneck; the batch script lines up the top silage counties for the pod."},
            "note": "Model estimates, not NASS statistics. Built from results on disk; nothing re-trained or tuned."}


def markdown(g):
    L = ["# Known gaps: what the data says", "", g["note"], ""]
    for key, title in (("ground_truth", "1. Thin ground truth"), ("corn_belt", "2. Corn Belt"), ("year_gap", "3. Year gap"),
                       ("unjudged", "4. Unjudged fields"), ("price", "5. Price story"), ("blind_test", "6. Blind test"),
                       ("scale", "7. Scale")):
        sec = g.get(key) or {}
        L += [f"## {title}", "", sec.get("finding") or "No data on disk.", ""]
    bt = g["blind_test"]
    L += ["| Place | Season | Ours % | USDA % | Error | USDA source | Hidden from pipeline |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['place']} | {r['season']} | {r['ours_pct']} | {r['usda_pct']} | {r['error_pts']:+} | {r['usda_source']} | {r['hidden']} |"
          for r in bt["rows"]]
    return "\n".join(L) + "\n"


def main():
    g = build()
    AGENTS.mkdir(parents=True, exist_ok=True)
    (AGENTS / "gap_checks.json").write_text(json.dumps(g, indent=1, default=str))
    (AGENTS / "gap_checks.md").write_text(markdown(g))
    print(markdown(g))


if __name__ == "__main__":
    main()
