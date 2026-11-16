"""The dairy owner's view: where is standing corn I could still buy for silage, and what would it cost?

Built on recommender/silage_match.py. A dairy is either one from the permit file (ids only, no names)
or a location and herd size the owner types in. For that one dairy, the recommender keeps fields that,
as of the chosen date, are still standing and green, predicted grain-bound (so likely for sale as
standing corn), at least 5 acres and within reach by road, then picks the cheapest tons by haul distance
and the risk the field is already someone's silage. The result carries the recommender's own checks.

Field data exists for past seasons only, so a run is a replay of that season "as of" a date.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..settings import COUNTY_DATA, RECOMMENDER, REPO
from . import service


def _silage_match():
    if str(RECOMMENDER) not in sys.path:
        sys.path.insert(0, str(RECOMMENDER))
    import silage_match
    return silage_match


def dairies():
    """Permitted dairies from the recommender's file: id, county, herd size and location only."""
    sm = _silage_match()
    d = sm.load_dairies()
    cols = [c for c in ("dairy_id", "county", "state", "head", "lat", "lon") if c in d.columns]
    out = d[cols].copy()
    out["head"] = out["head"].astype(float).where(out["head"].notna(), None)
    return json.loads(out.to_json(orient="records"))


def seasons():
    """County-seasons with field data, the ones a dairy run can replay."""
    return [c for c in service.counties() if not c["partial"]]


@lru_cache(maxsize=16)
def _field_shapes(fips, year):
    g = json.loads((COUNTY_DATA / str(fips) / str(year) / "fields.geojson").read_text())
    return {str(f["properties"]["field_id"]): f["geometry"] for f in g["features"]}


def _ensure_predictions(fips, year):
    """The recommender reads the final module's saved predictions; classify once if they are missing."""
    sm = _silage_match()
    tb = service.toolbox(offline=True)
    policy = getattr(tb._tb, "final_policy", "standard")
    try:
        sm.prediction_path(fips, year, policy)
    except FileNotFoundError:
        tb.scout(fips, year)
        tb.extract_features(fips, year)
        tb.classify(fips, year, "standard")
    return policy


def find_silage(fips, year, as_of=None, dairy_id=None, lat=None, lon=None, head=None, tons=None,
                shortfall=None, max_miles=None, corn_price=None):
    """Matched fields for one dairy, with tons, road miles, delivered price, checks and map features."""
    sm = _silage_match()
    fips, year = str(fips), int(year)
    policy = _ensure_predictions(fips, year)

    if dairy_id:
        all_d = sm.load_dairies()
        one = all_d[all_d["dairy_id"].astype(str) == str(dairy_id)]
        if one.empty:
            raise ValueError(f"No dairy {dairy_id} in the dairy file")
        one = one.copy()
        if head:
            one["head"] = float(head)
    else:
        if lat is None or lon is None:
            raise ValueError("Give a dairy id, or a location (lat, lon)")
        one = pd.DataFrame([{"dairy_id": "YOUR-DAIRY", "county": None, "state": None, "lat": float(lat),
                             "lon": float(lon), "head": float(head) if head else None,
                             "source": "location entered by the dairy owner"}])

    params = {}
    if shortfall:
        params["shortfall_frac"] = float(shortfall)
    if max_miles:
        params["max_road_miles"] = float(max_miles)
    if corn_price:
        params["corn_price"] = float(corn_price)
    P = sm.Params(**params)
    requests = None
    if tons:
        requests = pd.DataFrame([{"dairy_id": one["dairy_id"].iloc[0], "request_tons": float(tons)}])

    r = sm.recommend(fips, year, as_of, P, policy, requests=requests, dairies=one)
    shapes = _field_shapes(fips, year)
    features = [{"type": "Feature", "geometry": shapes.get(str(a["field_id"])),
                 "properties": {k: a[k] for k in ("field_id", "tons", "road_miles", "acres", "p_silage_adj",
                                                  "ndvi_as_of", "ndvi_date", "haul_usd_per_ton")}}
                for a in r["assignments"] if shapes.get(str(a["field_id"]))]
    me = r["dairies"][0] if r["dairies"] else None
    return {
        "fips": fips, "year": year, "name": service.COUNTY_NAMES.get(fips, fips), "as_of": r["as_of"],
        "dairy": me or {"dairy_id": one["dairy_id"].iloc[0], "lat": float(one["lat"].iloc[0]),
                        "lon": float(one["lon"].iloc[0]), "matched_tons": 0, "request_tons": 0},
        "in_reach": bool(me),
        "funnel": r["funnel"], "standing_price_usd_per_ton": r["standing_price_usd_per_ton"],
        "county_silage_share_pct": r["county_silage_share_pct"],
        "fields": [{k: a[k] for k in ("field_id", "tons", "road_miles", "acres", "p_silage_adj", "ndvi_as_of",
                                      "ndvi_date", "haul_usd_per_ton", "lat", "lon")} for a in r["assignments"]],
        "map": {"type": "FeatureCollection", "features": features},
        "checks": r["checks"], "overall": r["overall"], "hindsight": r.get("hindsight"),
        "params": {k: r["params"][k] for k in ("max_road_miles", "tons_per_acre", "shortfall_frac", "corn_price",
                                               "haul_low", "haul_high", "detour")},
        "sources": r["sources"], "model": r["model"],
        "note": "A replay of a past season as of the date: model estimates, not NASS statistics. Road miles are "
                "straight-line distance x 1.3. Prices are a rule of thumb (standing silage about 7 to 8 x the corn "
                "price per wet ton, plus haul), not quotes.",
    }
