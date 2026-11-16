"""The USDA dashboard: corn supply and price from USDA, next to what our pipeline adds and how far to trust it.

Three kinds of numbers, kept apart and labelled:
  USDA     Quick Stats history and USDA's own 2026 forecast (corn_market_history.csv)
  ours     silage estimates from the pipeline, with USDA's figure beside each one
  what-if  price effect of a silage error, from models that explain a past season's price (market/README.md)
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..market import loyo_clean_check as clean
from ..market.price_impact import DATA as MARKET_CSV
from ..settings import BENCHMARK, EVALUATION, QUICKSTATS_CSV, REPO
from . import service

RESULTS = REPO / "handoff" / "results"
AGENT_RESULTS = RESULTS / "agents"


def _pct(new, old):
    return None if not old or new is None or pd.isna(new) or pd.isna(old) else round(100 * (new - old) / old, 1)


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


@lru_cache(maxsize=1)
def _market():
    return pd.read_csv(MARKET_CSV).dropna(how="all").sort_values("year").reset_index(drop=True)


@lru_cache(maxsize=1)
def _price_table():
    return clean.base_table(MARKET_CSV, QUICKSTATS_CSV)


def supply():
    """USDA: production, area, yield, stocks and price, with the latest year against the one before."""
    m = _market().set_index("year")
    y_now, y_prev = 2026, 2025
    now, prev = m.loc[y_now], m.loc[y_prev]
    stu = _price_table().set_index("year")["stu"]
    tiles = [
        {"id": "production", "label": f"Corn grain production {y_now}", "value": _num(now["prod"]),
         "unit": "bu", "change_pct": _pct(now["prod"], prev["prod"]), "vs": f"{y_prev}",
         "source": f"USDA NASS {y_now} November forecast"},
        {"id": "yield", "label": f"Grain yield {y_now}", "value": _num(now["yield"]), "unit": "bu/acre",
         "change_pct": _pct(now["yield"], prev["yield"]), "vs": f"{y_prev}", "source": "USDA NASS forecast"},
        {"id": "area", "label": f"Acres harvested for grain {y_now}", "value": _num(now["grain_ac"]), "unit": "acres",
         "change_pct": _pct(now["grain_ac"], prev["grain_ac"]), "vs": f"{y_prev}", "source": "USDA NASS"},
        {"id": "stocks", "label": f"Corn stocks, September 1 {y_now}", "value": _num(now["sep1_stocks"]),
         "unit": "bu", "change_pct": _pct(now["sep1_stocks"], prev["sep1_stocks"]), "vs": f"{y_prev}",
         "source": "USDA NASS Grain Stocks: ending stocks of the 2025/26 marketing year"},
        {"id": "stu", "label": f"Stocks-to-use, {y_prev}/{str(y_now)[2:]} marketing year",
         "value": round(100 * float(stu.loc[y_prev]), 1), "unit": "%",
         "change_pct": None, "change_pts": round(100 * float(stu.loc[y_prev] - stu.loc[y_prev - 1]), 1),
         "vs": f"{y_prev - 1}/{str(y_prev)[2:]}", "source": "computed from USDA stocks and production"},
        {"id": "price", "label": f"Price received, {y_prev}/{str(y_now)[2:]} marketing year", "value": _num(prev["price_my"]),
         "unit": "$/bu", "change_pct": _pct(prev["price_my"], m.loc[y_prev - 1, "price_my"]), "vs": f"{y_prev - 1}/{str(y_prev)[2:]}",
         "source": "USDA NASS Agricultural Prices"},
        {"id": "price_sep", "label": f"Price received, September {y_now}", "value": _num(now["p_sep"]), "unit": "$/bu",
         "change_pct": _pct(now["p_sep"], prev["p_sep"]), "vs": f"September {y_prev}",
         "source": "USDA NASS Agricultural Prices, monthly"},
        {"id": "silage", "label": f"Acres harvested for silage {y_prev}", "value": _num(prev["silage_ac"]), "unit": "acres",
         "change_pct": _pct(prev["silage_ac"], m.loc[y_prev - 1, "silage_ac"]), "vs": f"{y_prev - 1}",
         "source": f"USDA NASS {y_prev} (the {y_now} figure comes out in January)"},
    ]
    series = {
        "production": [{"year": int(y), "value": _num(r["prod"]), "forecast": int(y) == y_now}
                       for y, r in m.iterrows() if not pd.isna(r["prod"])],
        "price": [{"year": int(y), "value": _num(r["price_my"])} for y, r in m.iterrows() if not pd.isna(r["price_my"])],
        "stocks": [{"year": int(y), "value": _num(r["sep1_stocks"])} for y, r in m.iterrows() if not pd.isna(r["sep1_stocks"])],
        "silage_acres": [{"year": int(y), "value": _num(r["silage_ac"])} for y, r in m.iterrows() if not pd.isna(r["silage_ac"])],
    }
    return {"tiles": tiles, "series": series}


@lru_cache(maxsize=1)
def price_models():
    """The clean leave-one-year-out table, with last year's price as the bar to beat."""
    return clean.evaluate(_price_table())


def what_if(points=1.0):
    """Price effect of `points` more national silage share in 2025, by each explaining model."""
    points = float(points)
    shock = clean.silage_plus_one_point(_price_table(), points=points)
    pi = json.loads((AGENT_RESULTS / "price_impact.json").read_text()) if (AGENT_RESULTS / "price_impact.json").is_file() else {}
    per_point = {k: v["price_change_pct"] for k, v in (pi.get("silage_plus_1pt_2025") or {}).items()}
    models = [{"model": k, "change_pct": v["change_pct"], "change_usd": v["change_usd"], "price_before": v["price_before"]}
              for k, v in shock["models"].items()]
    pcts = [m["change_pct"] for m in models]
    return {"points": points, "acres": shock["acres"], "grain_bushels": shock["grain_bushels"],
            "stu_before_pct": round(100 * shock["stu_before"], 2), "stu_after_pct": round(100 * shock["stu_after"], 2),
            "models": models, "range_pct": [min(pcts), max(pcts)] if pcts else None,
            "other_models_per_point_pct": per_point,
            "note": "A what-if, not a forecast: these models explain a finished season's price from its ending stocks. "
                    "The effect exists only if the true silage area differs from USDA's, and the answer depends on the model."}


def _read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def silage():
    """Our silage estimates beside USDA's, everywhere we have both."""
    out = {"states_2025": [], "maryland_2025": None, "frederick_2022": None, "field_test": None}
    st = _read(AGENT_RESULTS / "states_2025_comparison.json")
    if st:
        out["states_2025"] = [{"state": r["state"], "ours_pct": r["headline_pct"], "usda_pct": r["usda_2025_silage_pct"],
                               "ours_acres": r["model_silage_acres"], "usda_acres": r["usda_2025_silage_acres"],
                               "fields": r["fields"], "clear_fields": r["clear_fields"]}
                              for r in st["shares"] if r["model"] == "eu2_logistic_v1"]
    md = _read(RESULTS / "state_estimate.json")
    if md:
        out["maryland_2025"] = {"ours_pct": round(100 * md["headline_share"], 2),
                                "range_pct": [round(100 * x, 1) for x in md["headline_range_5_95"]],
                                "usda_pct": 8.9, "usda_source": "NASS 2025 (40,000 of 450,000 acres)",
                                "ours_acres": round(md["acres"]["silage_acres"]), "fields": md["n_fields_scored"],
                                "double_crop_acres_not_scored": round(md["acres"]["double_crop_corn_acres"])}
    fr = _read(AGENT_RESULTS / "frederick_2022_county_check.json")
    if fr:
        out["frederick_2022"] = {"ours_pct": round(100 * fr["module_v2"]["silage_share_headline"], 1),
                                 "census_pct": fr["census_silage_pct"], "fields": fr["rows"][0]["fields"]}
    la = _read(AGENT_RESULTS / "lancaster_2023_county_check.json")
    if la:   # USDA has no 2023 county figure: compared with the 2022 Census
        out["lancaster_2023"] = {"ours_pct": round(100 * la["module_v2"]["silage_share_headline"], 1),
                                 "census_pct": la["census_silage_pct"], "census_year": la["census_year"],
                                 "fields": la["fields"], "clear_fields": la["module_v2"]["clear_tier_fields"]}
    out["field_test"] = {"right": 42, "of": 44, "silage_found": 11, "silage_total": 12, "always_grain": 32,
                         "source": "handoff/results/REPORT.md (Beltsville, 44 labelled field-years, scored once)"}
    return out


def _latest_md(folder):
    files = sorted(Path(folder).glob("*.md"), key=lambda p: p.stat().st_mtime) if Path(folder).is_dir() else []
    files = [f for f in files if f.name != "limitations.md"]
    return files[-1].read_text() if files else None


def agent_quality():
    return {"evaluation_md": _latest_md(EVALUATION), "benchmark_md": _latest_md(BENCHMARK),
            "limitations_md": (Path(BENCHMARK) / "limitations.md").read_text() if (Path(BENCHMARK) / "limitations.md").is_file() else None,
            "gaps_md": _gaps_md()}


def _gaps_md():
    """The seven known gaps, checked against results on disk (fieldvision_agents.validation.gap_checks)."""
    path = REPO / "handoff" / "results" / "agents" / "gap_checks.md"
    return path.read_text().split("\n", 2)[2] if path.is_file() else None


def build():
    return {"supply": supply(), "price_models": price_models(), "what_if": what_if(1.0), "silage": silage(),
            "runs": service.runs(limit=12), "agent_quality": agent_quality()}
