"""Data behind the US map: per state, a silage share for every county to shade it by; per county, everything
the pipeline knows about it.

USDA numbers come from the Quick Stats file the critic uses (Census for counties in 2012, 2017 and 2022,
the annual survey for states) and from NASS's 2025 state figures; ours from agent runs, county data on
disk and the hand-over results. Withheld Census cells, "(D)", stay missing, never zero.
"""

import json
from functools import lru_cache

import pandas as pd

from ..settings import COUNTY_DATA, QUICKSTATS_CSV, REPO
from . import service

RESULTS = REPO / "handoff" / "results"
STATE_NAMES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California", "08": "Colorado",
    "09": "Connecticut", "10": "Delaware", "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
    "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
    "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada", "33": "New Hampshire", "34": "New Jersey",
    "35": "New Mexico", "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma",
    "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}
ABBR = {"MD": "24", "PA": "42", "WI": "55", "IA": "19", "NE": "31"}


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return float("nan")


@lru_cache(maxsize=1)
def _acres():
    """Harvested acres, one row per place, year, source and use (grain or silage)."""
    q = pd.read_csv(QUICKSTATS_CSV, low_memory=False)
    q = q[(q["domain"] == "TOTAL") & (q["reference_period"] == "YEAR") & q["crop_use"].isin(["grain", "silage"])].copy()
    q["acres"] = q["value"].map(_num)
    q["state"] = q["state_ansi"].map(lambda x: f"{int(x):02d}" if x == x else None)
    q["fips"] = [f"{s}{int(c):03d}" if s and c == c else None for s, c in zip(q["state"], q["county_ansi"])]
    q["level"] = q["aggregation_level"].str.lower()
    # a Census county often has a second, withheld row: keep the reported number
    g = q.groupby(["level", "state", "fips", "county_name", "year", "source", "crop_use"], dropna=False)["acres"].max()
    wide = g.unstack("crop_use").reset_index()
    wide["silage_share_pct"] = 100 * wide["silage"] / (wide["silage"] + wide["grain"])
    return wide


def _records(df, cols):
    df = df[cols].astype(object).where(df[cols].notna(), None)
    return df.to_dict("records")


def _round(rows):
    for r in rows:
        for k in ("grain", "silage"):
            if r.get(k) is not None:
                r[k] = round(r[k])
        if r.get("silage_share_pct") is not None:
            r["silage_share_pct"] = round(r["silage_share_pct"], 1)
    return rows


@lru_cache(maxsize=1)
def _nass_2025():
    path = RESULTS / "nass_2025_state_targets.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path, comment="#")
    out = {}
    for _, r in df.iterrows():
        fips = next((f for f, n in STATE_NAMES.items() if n.upper() == r["state"]), None)
        if fips:
            g, s = r["grain_acres_harvested"], r["silage_acres_harvested"]
            out[fips] = {"grain": None if g != g else int(g), "silage": None if s != s else int(s),
                         "silage_share_pct": round(100 * s / (s + g), 1) if g == g and s == s and (s + g) else None,
                         "source": "NASS 2025 survey (the challenge's evaluation figures)"}
    return out


def _ours_by_state():
    """Our state-level estimates: the four-state 2025 run and Maryland 2025."""
    out = {}
    path = RESULTS / "agents" / "states_2025_comparison.json"
    if path.is_file():
        for r in json.loads(path.read_text())["shares"]:
            if r["model"] == "eu2_logistic_v1":
                out.setdefault(ABBR[r["state"]], []).append({
                    "what": "2025 state sample (300 fields, final model)", "ours_pct": r["headline_pct"],
                    "usda_pct": r["usda_2025_silage_pct"], "ours_acres": r["model_silage_acres"], "usda_acres": r["usda_2025_silage_acres"]})
    md = RESULTS / "state_estimate.json"
    if md.is_file():
        m = json.loads(md.read_text())
        out.setdefault("24", []).append({
            "what": f"2025 state sample ({m['n_fields_scored']:,} fields, clearly seen fields counted)",
            "ours_pct": round(100 * m["headline_share"], 2), "usda_pct": 8.9,
            "ours_acres": round(m["acres"]["silage_acres"]), "usda_acres": 40000})
    return out


def state_summary(state):
    state = str(state).zfill(2)
    a = _acres()
    counties = a[(a["level"] == "county") & (a["state"] == state) & (a["source"] == "CENSUS")]
    latest = counties.sort_values("year").dropna(subset=["silage_share_pct"]).groupby("fips").tail(1)
    on_disk = {c["fips"] for c in service.counties()}
    with_runs = {r["fips"] for r in service.runs(limit=500)}
    shade = {r["fips"]: {"share": round(r["silage_share_pct"], 1), "year": int(r["year"]),
                         "silage": None if r["silage"] != r["silage"] else round(r["silage"]),
                         "grain": None if r["grain"] != r["grain"] else round(r["grain"])} for _, r in latest.iterrows()}
    hist = a[(a["level"] == "state") & (a["state"] == state) & (a["source"] == "SURVEY")].sort_values("year")
    return {"state": state, "name": STATE_NAMES.get(state, state), "counties": shade,
            "fields_on_disk": sorted(f for f in on_disk if f.startswith(state)),
            "with_runs": sorted(f for f in with_runs if f.startswith(state)),
            "history": _round(_records(hist, ["year", "source", "grain", "silage", "silage_share_pct"])),
            "nass_2025": _nass_2025().get(state), "ours": _ours_by_state().get(state, [])}


def states():
    """Every state's 2025 silage share from NASS, for shading the national map."""
    on_disk = {c["fips"][:2] for c in service.counties()}
    return {s: {**v, "name": STATE_NAMES[s], "fields_on_disk": s in on_disk} for s, v in _nass_2025().items()}


def _round_coords(geom, nd=5):
    def r(c):
        return [round(c[0], nd), round(c[1], nd)] if isinstance(c[0], (int, float)) else [r(x) for x in c]
    return {"type": geom["type"], "coordinates": r(geom["coordinates"])}


@lru_cache(maxsize=8)
def field_calls(fips, year):
    """Every field of a county-season with the final module's call: silage, grain or not judged.
    Classifies once if the season has no saved predictions yet."""
    from . import dairy
    fips, year = str(fips), int(year)
    policy = dairy._ensure_predictions(fips, year)
    sm = dairy._silage_match()
    preds = pd.read_parquet(sm.prediction_path(fips, year, policy)).set_index("field_id")
    g = json.loads((COUNTY_DATA / fips / str(year) / "fields.geojson").read_text())
    feats, counts = [], {"silage": 0, "grain": 0, "not judged": 0}
    for f in g["features"]:
        fid = str(f["properties"]["field_id"])
        if fid not in preds.index:
            continue
        p = preds.loc[fid]
        ev = p["evidence"] if isinstance(p["evidence"], dict) else json.loads(p["evidence"] or "{}")
        call = "not judged" if bool(p["abstain"]) else str(p["label"])
        counts[call] = counts.get(call, 0) + 1
        props = f["properties"]
        feats.append({"type": "Feature", "geometry": _round_coords(f["geometry"]), "properties": {
            "field_id": fid, "call": call, "p_silage": round(float(p["p_silage"]), 3),
            "acres": round(float(props.get("acres_reported") or (props.get("area_ha") or 0) * 2.47105), 1),
            "drop_date": ev.get("drop_date"), "reason": ev.get("abstain_reason"),
            "gap_days": ev.get("days_to_drop") or ev.get("largest_gap_days")}})
    return {"type": "FeatureCollection", "features": feats, "counts": counts, "model": str(preds["model"].iloc[0]),
            "fips": fips, "year": year}


def dairy_points():
    from . import dairy
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
         "properties": {k: d.get(k) for k in ("dairy_id", "county", "state", "head")}} for d in dairy.dairies()]}


def county_detail(fips):
    fips = str(fips).zfill(5)
    a = _acres()
    rows = a[(a["level"] == "county") & (a["fips"] == fips)].sort_values(["year", "source"])
    name = rows["county_name"].dropna().iloc[0].title() if len(rows) else None
    seasons = [c for c in service.counties() if c["fips"] == fips]
    runs = [r for r in service.runs(limit=500) if r["fips"] == fips]
    extras = {}
    for key, path in (("rotation_2022", RESULTS / "rotation_check_md_2022.csv"), ("landscape_2022", RESULTS / "landscape_groups_md_2022.csv")):
        if path.is_file():
            d = pd.read_csv(path, dtype={"fips": str})
            d = d[d["fips"] == fips]
            if len(d):
                extras[key] = json.loads(d.to_json(orient="records"))
    check = RESULTS / "agents" / "frederick_2022_county_check.json"
    if check.is_file() and json.loads(check.read_text()).get("fips") == fips:
        extras["county_check_2022"] = json.loads(check.read_text())
    prior = None
    try:
        from ..fieldvision_bridge.prevalence import silage_share_prior
        p = silage_share_prior(QUICKSTATS_CSV, fips, 2026)
        prior = {"share_pct": round(100 * p["share"], 1), "source": p.get("source"), "level": p.get("level")}
    except Exception:
        pass
    return {"fips": fips, "name": f"{name} County, {STATE_NAMES.get(fips[:2], '')}" if name else service.COUNTY_NAMES.get(fips, fips),
            "state": fips[:2], "history": _round(_records(rows, ["year", "source", "grain", "silage", "silage_share_pct"])),
            "seasons": seasons, "runs": runs[:10], "extras": extras, "prior_2026": prior}
