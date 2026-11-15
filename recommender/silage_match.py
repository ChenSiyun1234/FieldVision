"""Silage Match: point dairies that are short on feed to nearby corn fields that could be chopped.

Inputs (all already in the workspace):
  - field polygons + centroids      data/counties/<fips>/<year>/fields.geojson
  - field predictions               results/<ST>_<year>_<fips>_<policy>.parquet  (p_silage, label, evidence.drop_date)
  - Sentinel-2 time series          data/counties/<fips>/<year>/timeseries.parquet (canopy greenness as of a date)
  - permitted dairies               recommender/data/dairies_md.csv  (fetch_dairies.py, MDE public layer)
  - QuickStats history              via rag-pipeline (silage prevalence prior)

Process:
  1. Prevalence fix: the model's p_silage assumes a 50/50 class balance; shift it to the county's
     historical silage share (years before the target year, so no peeking at the answer).
  2. Candidate fields as of a date: not abstained, not yet harvested, grain-bound (p_adj < 0.5),
     big enough for a crew trip, canopy still green (so whole-plant moisture is still in silage range).
  3. Each dairy places a request (tons). Default scenario: a share of its annual need, e.g. a dry year.
  4. Min-cost flow matches all requests at once, so two dairies are never sent to the same field.
     Field capacity = expected tons that are NOT already someone's silage = (1 - p_adj) * acres * t/ac.
     Cost = haul $ (straight line x detour factor) + a risk penalty x p_adj. Edges only inside the max haul distance.
  5. Code checks the result (critic), and optionally Nemotron writes the explanation.

Outputs: recommender/out/match_<fips>_<year>_<asof>.json (+ .geojson for the map).

Every number that is an assumption lives in Params, with its source in SOURCES.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REC_DIR = Path(__file__).resolve().parent            # .../FieldVision/recommender
REPO = REC_DIR.parent                                 # .../FieldVision
# .deps must come first on sys.path before pandas is imported (pyarrow lives there)
for _p in (REPO / "rag-pipeline", REPO / "src", REPO / ".deps"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA_ROOT = REPO / "data" / "counties"
RESULTS = REPO / "results"
DAIRIES_CSV = REC_DIR / "data" / "dairies_md.csv"
OUT_DIR = REC_DIR / "out"
STATE_ABBR = {"24": "MD", "42": "PA", "51": "VA", "10": "DE", "54": "WV"}
EARTH_MILES = 3958.8

SOURCES = {
    "haul_limit": "Silage economics fall apart over long hauls (NDSU Extension EC2264); ~20 mi practical limit (WLJ, extension specialist).",
    "haul_rate": "Long-haul surcharge of $0.50 to $0.75 per ton per loaded mile (Virginia Tech, Buying Standing Corn for Silage, 1997 dollars).",
    "standing_price": "Standing corn silage priced at about 7 to 8 x the corn price per bushel, per wet ton (Vita Plus).",
    "tons_per_head": "55 lb/cow/day as-fed (U. Kentucky forage planning sheet): about 10 tons per head per year.",
    "dairies": "Maryland Department of the Environment, AFO/CAFO permit map service. Permitted operations only.",
    "fields": "USDA NASS Crop Sequence Boundaries; FieldVision silage/grain predictions.",
    "prior": "USDA NASS QuickStats silage share of harvested corn acres, years before the target year.",
}


@dataclass
class Params:
    tons_per_acre: float = 20.0      # as-fed silage yield; our QuickStats file has acres only
    tons_per_head: float = 10.0      # as-fed per dairy animal per year (55 lb/day x 365 / 2000)
    shortfall_frac: float = 0.15     # default scenario: each dairy asks for this share of its annual need
    max_road_miles: float = 20.0
    detour: float = 1.3              # road miles per straight-line mile (assumption until we have a road graph)
    haul_low: float = 0.50           # $ per ton per loaded mile
    haul_high: float = 0.75
    corn_price: float = 4.00         # $ per bushel
    mult_low: float = 7.0            # standing silage $/wet ton = corn price x multiplier
    mult_high: float = 8.0
    min_acres: float = 5.0           # smaller fields are not worth moving a chopper crew
    min_ndvi: float = 0.50           # canopy still green as of the date
    min_clear: float = 0.6           # clear-sky fraction for an observation to count
    risk_usd_per_ton: float = 8.0    # matching penalty x p_adj: prefer fields that are surely not someone's silage
    default_head: float = 300.0      # herd size used only when the dairy file has none (assumption, flagged)
    prior: float | None = None       # silage prevalence; None = county history before the target year
    use_prior: bool = True


# ----------------------------------------------------------------------------- loading

def load_fields(fips: str, year: int, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    g = json.loads((data_root / fips / str(year) / "fields.geojson").read_text())
    rows = []
    for f in g["features"]:
        p = f["properties"]
        acres = p.get("acres_reported") or (p.get("area_ha") or 0) * 2.47105
        rows.append({"field_id": p["field_id"], "lat": float(p["centroid_lat"]),
                     "lon": float(p["centroid_lon"]), "acres": float(acres)})
    return pd.DataFrame(rows)


def prediction_path(fips: str, year: int, policy: str = "standard", results: Path = RESULTS) -> Path:
    st = STATE_ABBR.get(fips[:2])
    p = results / f"{st}_{year}_{fips}_{policy}.parquet"
    if p.exists():
        return p
    hits = sorted(results.glob(f"*_{year}_{fips}_{policy}.parquet"))
    if not hits:
        raise FileNotFoundError(f"No predictions for {fips} {year} policy={policy} in {results}. "
                                "Run toolbox.classify first.")
    return hits[0]


def load_predictions(fips: str, year: int, policy: str = "standard", results: Path = RESULTS) -> pd.DataFrame:
    d = pd.read_parquet(prediction_path(fips, year, policy, results))
    ev = d["evidence"].map(lambda s: json.loads(s) if isinstance(s, str) else (s or {}))
    d["drop_date"] = pd.to_datetime(ev.map(lambda e: e.get("drop_date")), errors="coerce")
    return d[["field_id", "p_silage", "label", "abstain", "drop_date", "model"]]


def ndvi_as_of(fips: str, year: int, as_of: pd.Timestamp, min_clear: float = 0.6,
               data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Last clear NDVI on or before as_of, per field (no data after as_of is used)."""
    t = pd.read_parquet(data_root / fips / str(year) / "timeseries.parquet",
                        columns=["field_id", "date", "clear_fraction", "B04", "B08"])
    t["date"] = pd.to_datetime(t["date"])
    t = t[(t["date"] <= as_of) & (t["clear_fraction"] >= min_clear)].copy()
    t["ndvi"] = (t["B08"] - t["B04"]) / (t["B08"] + t["B04"])
    last = t.sort_values("date").groupby("field_id").tail(1)
    return last.set_index("field_id")[["date", "ndvi"]].rename(columns={"date": "ndvi_date"})


def normalize_dairies(d: pd.DataFrame) -> pd.DataFrame:
    """Accept our MDE file or the EPA FRS export (Name, Description, kml_layer, longitude, latitude).
    Keeps an id, location, county/state and herd size only; facility names are dropped."""
    if {"latitude", "longitude"} <= set(d.columns) and "lat" not in d.columns:
        desc = d.get("Description", pd.Series([""] * len(d))).fillna("")
        reg = desc.str.extract(r"REGISTRY ID.*?(\d{9,})", expand=False)
        county = desc.str.extract(r"COUNTY NAME</td><td>([^<]+)", expand=False).str.replace(" COUNTY", "")
        state = desc.str.extract(r", ([A-Z]{2}) \d{5}", expand=False)
        d = pd.DataFrame({"dairy_id": ("FRS-" + reg.fillna(pd.Series(range(len(d))).astype(str))),
                          "county": county.str.title(), "state": state,
                          "lat": d["latitude"].astype(float), "lon": d["longitude"].astype(float),
                          "head": d["head"] if "head" in d.columns else np.nan,
                          "source": "EPA FRS dairy facilities (team file)"})
    return d


def load_dairies(path: Path = DAIRIES_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python3 recommender/fetch_dairies.py")
    d = normalize_dairies(pd.read_csv(path, dtype={"dairy_id": str}))
    need = {"dairy_id", "lat", "lon"}
    if not need <= set(d.columns):
        raise ValueError(f"dairy file needs columns {need}, has {list(d.columns)}")
    if "head" not in d.columns:
        d["head"] = np.nan
    return d


def history_prior(fips: str, year: int) -> tuple[float | None, str]:
    """Mean silage share from QuickStats for years before the target year: county if 2+ years, else state."""
    try:
        import critic_toolbox as ct
        import rag_pipeline as rp
    except Exception as e:  # rag-pipeline missing: run without the prior
        return None, f"no prior (rag-pipeline import failed: {e})"
    state, county = ct.fips_to_names(fips)
    if state is None:
        return None, "no prior (unknown state)"
    years = list(range(year - 15, year))
    h = rp.silage_share_history(state, county, years=years) if county else {"summary": {}}
    if h["summary"].get("n_years", 0) >= 2:
        return h["summary"]["mean"], f"{county} County mean of {h['summary']['n_years']} years before {year}"
    h = rp.silage_share_history(state, years=years)
    if h["summary"].get("n_years", 0) >= 1:
        return h["summary"]["mean"], f"{state} state mean of {h['summary']['n_years']} years before {year}"
    return None, "no prior (no QuickStats history)"


# ----------------------------------------------------------------------------- math

def adjust_prior(p, pi: float, base: float = 0.5):
    """Shift probabilities calibrated at class balance base to prevalence pi (Bayes odds update)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    odds = p / (1 - p) * (pi / (1 - pi)) / (base / (1 - base))
    return odds / (1 + odds)


def straight_miles(lat1, lon1, lat2, lon2):
    """Haversine miles; arrays broadcast (fields as column, dairies as row)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_MILES * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- core

def build_candidates(fields: pd.DataFrame, preds: pd.DataFrame, ndvi: pd.DataFrame,
                     as_of: pd.Timestamp, P: Params, pi: float | None) -> tuple[pd.DataFrame, dict]:
    f = fields.merge(preds, on="field_id", how="inner").join(ndvi, on="field_id")
    f["p_adj"] = adjust_prior(f["p_silage"], pi) if (P.use_prior and pi) else f["p_silage"].astype(float)
    f["tons"] = f["acres"] * P.tons_per_acre
    f["usable_tons"] = (1 - f["p_adj"]) * f["tons"]
    harvested = f["drop_date"].notna() & (f["drop_date"] <= as_of)
    silage_bound = f["p_adj"] >= 0.5
    small = f["acres"] < P.min_acres
    brown = f["ndvi"].isna() | (f["ndvi"] < P.min_ndvi)

    f["status"] = "candidate"
    # first reason wins, in this order, so the funnel adds up
    for mask, why in [(brown, "browned_down_or_no_view"), (small, "too_small"),
                      (silage_bound, "silage_bound"), (harvested, "harvested_by_date"),
                      (f["abstain"].astype(bool), "abstained")]:
        f.loc[mask, "status"] = why
    funnel = {"fields": int(len(f))}
    for why in ["abstained", "harvested_by_date", "silage_bound", "too_small", "browned_down_or_no_view",
                "candidate"]:
        funnel[why] = int((f["status"] == why).sum())
    funnel["candidate_usable_tons"] = int(f.loc[f["status"] == "candidate", "usable_tons"].sum())
    return f, funnel


def default_requests(dairies: pd.DataFrame, P: Params) -> pd.DataFrame:
    d = dairies.copy()
    d["head_assumed"] = d["head"].isna()
    d["head"] = d["head"].fillna(P.default_head)
    d["annual_need_tons"] = d["head"] * P.tons_per_head
    d["request_tons"] = (d["annual_need_tons"] * P.shortfall_frac).round()
    return d


def match(cands: pd.DataFrame, dairies: pd.DataFrame, P: Params) -> tuple[list[dict], pd.DataFrame]:
    """Min-cost max-flow: source -> field (usable tons) -> dairy (haul cost) -> sink (request)."""
    c = cands.reset_index(drop=True)
    d = dairies[dairies["request_tons"] > 0].reset_index(drop=True)
    if c.empty or d.empty:
        return [], np.zeros((len(c), len(d)))
    road = straight_miles(c["lat"].values[:, None], c["lon"].values[:, None],
                          d["lat"].values[None, :], d["lon"].values[None, :]) * P.detour
    haul_mid = (P.haul_low + P.haul_high) / 2
    G = nx.DiGraph()
    for i, row in c.iterrows():
        cap = int(row["usable_tons"])
        if cap <= 0:
            continue
        for j in np.where(road[i] <= P.max_road_miles)[0]:
            cost = road[i, j] * haul_mid + row["p_adj"] * P.risk_usd_per_ton   # $ per ton
            G.add_edge(("f", i), ("d", int(j)), capacity=cap, weight=int(round(cost * 100)))
        if G.has_node(("f", i)):
            G.add_edge("S", ("f", i), capacity=cap, weight=0)
    for j, row in d.iterrows():
        G.add_edge(("d", j), "T", capacity=int(row["request_tons"]), weight=0)
    if "S" not in G:
        return [], road
    flow = nx.max_flow_min_cost(G, "S", "T")
    out = []
    for i in range(len(c)):
        for node, t in flow.get(("f", i), {}).items():
            if t <= 0:
                continue
            j = node[1]
            miles = float(road[i, j])
            out.append({"dairy_id": d.at[j, "dairy_id"], "field_id": c.at[i, "field_id"], "tons": int(t),
                        "road_miles": round(miles, 1), "acres": round(float(c.at[i, "acres"]), 1),
                        "p_silage_adj": round(float(c.at[i, "p_adj"]), 3),
                        "ndvi_as_of": round(float(c.at[i, "ndvi"]), 3),
                        "ndvi_date": str(c.at[i, "ndvi_date"].date()),
                        "haul_usd_per_ton": [round(miles * P.haul_low, 2), round(miles * P.haul_high, 2)],
                        "lat": round(float(c.at[i, "lat"]), 5), "lon": round(float(c.at[i, "lon"]), 5)})
    return out, road


def critic(result: dict, P: Params) -> list[dict]:
    """Deterministic checks on the match. Code decides the status; text only explains."""
    if not result["dairies"]:
        return [{"check": "dairies_in_reach", "status": "warn",
                 "why": f"No dairy in {result.get('dairy_source', 'the dairy file')} is within "
                        f"{P.max_road_miles:g} road miles of this county's fields. Check that the file covers this state."}]
    checks = []
    req = sum(x["request_tons"] for x in result["dairies"])
    met = sum(x["matched_tons"] for x in result["dairies"])
    frac = met / req if req else 0.0
    checks.append({"check": "requests_covered", "status": "pass" if frac >= 0.8 else "warn",
                   "observed_pct": round(100 * frac, 1),
                   "why": f"{met:,} of {req:,} requested tons matched to fields within {P.max_road_miles:g} road miles."})
    a = result["assignments"]
    long = [x for x in a if x["road_miles"] > 15]
    checks.append({"check": "long_haul", "status": "warn" if long else "pass", "n": len(long),
                   "why": "Hauls over 15 road miles cost the most; check them first." if long
                   else "Every haul is 15 road miles or less."})
    tons = sum(x["tons"] for x in a) or 1
    risky = sum(x["tons"] for x in a if x["p_silage_adj"] >= 0.3)
    pct = 100 * risky / tons
    checks.append({"check": "uncertain_sources", "status": "warn" if pct > 25 else "pass",
                   "observed_pct": round(pct, 1),
                   "why": "Share of matched tons from fields that might already be someone's silage (p_adj >= 0.3)."})
    outside = [x["dairy_id"] for x in result["dairies"] if not x["in_county"]]
    checks.append({"check": "partial_view", "status": "info" if outside else "pass", "dairies": outside,
                   "why": "These dairies sit outside the classified county, so fields on their side of the "
                          "line are not counted." if outside else "All dairies are inside the classified county."})
    assumed = sum(1 for x in result["dairies"] if x.get("head_assumed"))
    checks.append({"check": "dairy_coverage", "status": "info",
                   "why": f"{len(result['dairies'])} dairies in reach from {result.get('dairy_source', 'the dairy file')}."
                          + (f" {assumed} have no herd size, so {P.default_head:g} head was assumed." if assumed else "")
                          + " Dairies missing from the file are not counted, so real demand may be higher."})
    if result.get("hindsight"):
        h = result["hindsight"]
        checks.append({"check": "window_held_hindsight", "status": "pass" if h["held_pct"] >= 80 else "warn",
                       "observed_pct": h["held_pct"],
                       "why": f"Backtest only: share of matched tons on fields that really stayed standing "
                              f"{h['days']} days after the date."})
    return checks


def recommend(fips: str, year: int, as_of: str | None = None, P: Params | None = None,
              policy: str = "standard", requests: pd.DataFrame | None = None,
              dairies: pd.DataFrame | None = None, data_root: Path = DATA_ROOT,
              results: Path = RESULTS, hindsight_days: int = 14, dairies_path: Path | None = None) -> dict:
    P = P or Params()
    as_of_ts = pd.Timestamp(as_of or f"{year}-09-01")
    fields = load_fields(fips, year, data_root)
    preds = load_predictions(fips, year, policy, results)
    ndvi = ndvi_as_of(fips, year, as_of_ts, P.min_clear, data_root)
    pi, pi_src = (P.prior, "given") if P.prior is not None else history_prior(fips, year)
    f, funnel = build_candidates(fields, preds, ndvi, as_of_ts, P, pi)

    # county silage share before and after the prior, a sanity check against Census
    dec = f[~f["abstain"].astype(bool)]
    share_raw = float((dec["p_silage"] * dec["acres"]).sum() / dec["acres"].sum()) if len(dec) else None
    share_adj = float((dec["p_adj"] * dec["acres"]).sum() / dec["acres"].sum()) if len(dec) else None

    d = normalize_dairies(dairies) if dairies is not None else load_dairies(dairies_path or DAIRIES_CSV)
    d = default_requests(d, P)
    if requests is not None:  # explicit requests override the scenario
        d = d.drop(columns=["request_tons"]).merge(requests[["dairy_id", "request_tons"]], on="dairy_id",
                                                   how="left").fillna({"request_tons": 0})
    # keep dairies that can reach at least one field in the county
    lat0, lon0 = f["lat"].values[:, None], f["lon"].values[:, None]
    near = (straight_miles(lat0, lon0, d["lat"].values[None, :], d["lon"].values[None, :]) * P.detour
            <= P.max_road_miles).any(axis=0)
    d = d[near].reset_index(drop=True)
    cands = f[f["status"] == "candidate"]
    assignments, _ = match(cands, d, P)

    lo, hi = P.corn_price * P.mult_low, P.corn_price * P.mult_high
    county_name = None
    try:
        import critic_toolbox as ct
        county_name = ct.fips_to_names(fips)[1]
    except Exception:
        pass
    dairies_out = []
    for _, row in d.iterrows():
        mine = [x for x in assignments if x["dairy_id"] == row["dairy_id"]]
        t = sum(x["tons"] for x in mine)
        haul = [sum(x["tons"] * x["haul_usd_per_ton"][k] for x in mine) / t if t else 0 for k in (0, 1)]
        dairies_out.append({
            "dairy_id": row["dairy_id"], "county": row.get("county"), "head": int(row["head"] or 0),
            "head_assumed": bool(row["head_assumed"]),
            "in_county": bool(county_name and str(row.get("county", "")).upper() == county_name.upper()),
            "annual_need_tons": int(row["annual_need_tons"]), "request_tons": int(row["request_tons"]),
            "matched_tons": int(t), "n_fields": len(mine),
            "est_acres_to_chop": round(t / P.tons_per_acre, 1),
            "avg_haul_usd_per_ton": [round(haul[0], 2), round(haul[1], 2)],
            "delivered_usd_per_ton": [round(lo + haul[0], 2), round(hi + haul[1], 2)],
            "lat": row["lat"], "lon": row["lon"]})

    result = {
        "fips": fips, "year": year, "as_of": str(as_of_ts.date()), "policy": policy,
        "model": str(preds["model"].iloc[0]) if len(preds) else None,
        "prior": {"pi": pi, "source": pi_src, "used": bool(P.use_prior and pi)},
        "county_silage_share_pct": {"raw": round(100 * share_raw, 1) if share_raw is not None else None,
                                    "after_prior": round(100 * share_adj, 1) if share_adj is not None else None},
        "funnel": funnel,
        "standing_price_usd_per_ton": [round(lo, 2), round(hi, 2)],
        "dairy_source": str(d["source"].iloc[0]) if "source" in d.columns and len(d) else "dairy file",
        "dairies": dairies_out,
        "assignments": sorted(assignments, key=lambda x: (x["dairy_id"], x["road_miles"])),
        "params": asdict(P), "sources": SOURCES,
    }
    if hindsight_days and assignments:  # backtest: did the window really stay open?
        dd = f.set_index("field_id")["drop_date"]
        cutoff = as_of_ts + pd.Timedelta(days=hindsight_days)
        held = sum(x["tons"] for x in assignments
                   if pd.isna(dd.get(x["field_id"])) or dd.get(x["field_id"]) >= cutoff)
        result["hindsight"] = {"days": hindsight_days,
                               "held_pct": round(100 * held / sum(x["tons"] for x in assignments), 1)}
    result["checks"] = critic(result, P)
    status = [c["status"] for c in result["checks"]]
    result["overall"] = "warn" if "warn" in status else "pass"
    result["summary"] = template_summary(result)
    return result


# ----------------------------------------------------------------------------- text

def template_summary(r: dict) -> str:
    fu, ds = r["funnel"], r["dairies"]
    req = sum(x["request_tons"] for x in ds)
    met = sum(x["matched_tons"] for x in ds)
    lines = [f"As of {r['as_of']}: {fu['candidate']} of {fu['fields']} corn fields could still be chopped "
             f"({fu['candidate_usable_tons']:,} usable tons). {len(ds)} dairies asked for {req:,} tons; "
             f"{met:,} tons matched within {r['params']['max_road_miles']:g} road miles."]
    for x in sorted(ds, key=lambda x: -x["request_tons"])[:5]:
        lines.append(f"{x['dairy_id']}: {x['matched_tons']:,} of {x['request_tons']:,} tons from {x['n_fields']} fields, "
                     f"delivered about {x['delivered_usd_per_ton'][0]:.0f} to {x['delivered_usd_per_ton'][1]:.0f} USD per ton.")
    return "\n".join(lines)


EXPLAIN_PROMPT = """You explain a feed-matching result to a dairy farmer and a NASS analyst.
Use only numbers that appear in the JSON. Write ranges as 'A to B'. Keep it under 120 words.
Say what was matched, what it costs, and the one or two checks they should look at first."""


def explain_with_nemotron(r: dict, model: str = "nvidia/nemotron-3-ultra-550b-a55b") -> str:
    import rag_pipeline as rp
    from openai import OpenAI
    slim = {k: r[k] for k in ("as_of", "funnel", "standing_price_usd_per_ton", "checks", "county_silage_share_pct")}
    slim["dairies"] = [{k: x[k] for k in ("dairy_id", "request_tons", "matched_tons", "n_fields",
                                          "delivered_usd_per_ton")} for x in r["dairies"]]
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=rp._nvidia_key())
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=600,
        messages=[{"role": "system", "content": EXPLAIN_PROMPT},
                  {"role": "user", "content": json.dumps(slim)}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("empty reply")
    return text


# ----------------------------------------------------------------------------- outputs

def to_geojson(r: dict) -> dict:
    feats = []
    for x in r["dairies"]:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [x["lon"], x["lat"]]},
                      "properties": {"kind": "dairy", **{k: v for k, v in x.items() if k not in ("lat", "lon")}}})
    dpos = {x["dairy_id"]: (x["lon"], x["lat"]) for x in r["dairies"]}
    for a in r["assignments"]:
        props = {k: v for k, v in a.items() if k not in ("lat", "lon")}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [a["lon"], a["lat"]]},
                      "properties": {"kind": "source_field", **props}})
        feats.append({"type": "Feature", "geometry": {"type": "LineString",
                                                      "coordinates": [list(dpos[a["dairy_id"]]), [a["lon"], a["lat"]]]},
                      "properties": {"kind": "haul", "dairy_id": a["dairy_id"], "field_id": a["field_id"],
                                     "tons": a["tons"], "road_miles": a["road_miles"]}})
    return {"type": "FeatureCollection", "features": feats}


def save(r: dict, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"match_{r['fips']}_{r['year']}_{r['as_of']}"
    stem.with_suffix(".json").write_text(json.dumps(r, indent=1, default=str))
    stem.with_suffix(".geojson").write_text(json.dumps(to_geojson(r), default=str))
    return stem.with_suffix(".json")


# ----------------------------------------------------------------------------- agent tool

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "recommend_sources",
        "description": ("Match dairies short on silage to nearby standing corn fields that are predicted "
                        "grain-bound and could be chopped, as of a date. Uses the county's latest field "
                        "predictions. Returns matched tons per dairy, prices, and code checks."),
        "parameters": {"type": "object", "properties": {
            "fips": {"type": "string"}, "year": {"type": "integer"},
            "as_of": {"type": "string", "description": "YYYY-MM-DD, default Sep 1"},
            "shortfall_frac": {"type": "number", "description": "share of annual need each dairy asks for"}},
            "required": ["fips", "year"]}}}


class SilageMatchToolbox:
    """Wraps any Toolbox (or CriticToolbox) and adds recommend_sources()."""

    def __init__(self, toolbox=None, params: Params | None = None):
        self._tb = toolbox
        self.params = params or Params()

    def __getattr__(self, name):
        if self._tb is None:
            raise AttributeError(name)
        return getattr(self._tb, name)

    def recommend_sources(self, fips: str, year: int, as_of: str | None = None,
                          shortfall_frac: float | None = None, policy: str = "standard") -> dict:
        P = Params(**{**asdict(self.params), **({"shortfall_frac": shortfall_frac} if shortfall_frac else {})})
        r = recommend(fips, year, as_of, P, policy)
        save(r)
        return {k: r[k] for k in ("as_of", "overall", "funnel", "county_silage_share_pct",
                                  "standing_price_usd_per_ton", "checks", "summary")} | {
            "dairies": [{k: x[k] for k in ("dairy_id", "request_tons", "matched_tons", "n_fields",
                                           "delivered_usd_per_ton")} for x in r["dairies"]]}


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fips")
    ap.add_argument("year", type=int)
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD (default Sep 1 of year)")
    ap.add_argument("--policy", default="standard")
    ap.add_argument("--shortfall", type=float, default=None, help="share of annual need requested")
    ap.add_argument("--corn-price", type=float, default=None)
    ap.add_argument("--max-miles", type=float, default=None)
    ap.add_argument("--prior", type=float, default=None, help="override silage prevalence")
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--requests", type=Path, default=None, help="CSV with dairy_id,request_tons")
    ap.add_argument("--dairies", type=Path, default=None, help="dairy CSV (default recommender/data/dairies_md.csv)")
    ap.add_argument("--llm", action="store_true", help="Nemotron writes the explanation")
    a = ap.parse_args()

    P = Params(use_prior=not a.no_prior, prior=a.prior)
    if a.shortfall is not None:
        P.shortfall_frac = a.shortfall
    if a.corn_price is not None:
        P.corn_price = a.corn_price
    if a.max_miles is not None:
        P.max_road_miles = a.max_miles
    req = pd.read_csv(a.requests, dtype={"dairy_id": str}) if a.requests else None
    r = recommend(a.fips, a.year, a.as_of, P, a.policy, requests=req, dairies_path=a.dairies)
    if a.llm:
        try:
            r["explanation"] = explain_with_nemotron(r)
            r["explained_by"] = "nemotron"
        except Exception as e:
            r["explanation"], r["explained_by"], r["explain_error"] = r["summary"], "template", str(e)[:200]
    path = save(r)
    print(r["summary"])
    print(f"\nprior: {r['prior']}  county silage share: {r['county_silage_share_pct']}")
    print("funnel:", json.dumps(r["funnel"]))
    for c in r["checks"]:
        print(f"[{c['status']}] {c['check']}: {c['why']}")
    if "explanation" in r:
        print(f"\n({r['explained_by']}) {r['explanation']}")
    print(f"\nsaved {path} (+ .geojson)")


if __name__ == "__main__":
    main()
