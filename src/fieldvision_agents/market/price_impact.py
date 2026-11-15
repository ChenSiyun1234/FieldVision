"""How much does the corn price move with grain supply and stocks, and what does our silage estimate imply?

Data: USDA NASS Quick Stats, national, 2000-2025 (data/corn_market_history.csv). Target: the
marketing-year price farmers received for corn (Sep Y to Aug Y+1). Four models, all ordinary least
squares on logs, compared on the same years:

    no_change     price_Y = price_Y-1                                   (baseline)
    supply        dlog price = a + b * dlog grain production
    stocks        log price = a + b * log stocks-to-use
    stocks_prev   log price = a + b * log stocks-to-use + c * log price_Y-1

Stocks-to-use for marketing year Y: ending stocks (Sep 1 of Y+1) / use, where use = beginning
stocks + production - ending stocks. Ending stocks are only known after the season, so the stocks
models explain a season's price; at harvest time they would need USDA's projected stocks (WASDE).

Checks: every year 2001-2024 is predicted from a model fitted without it (leave-one-year-out), and
2025 from a model fitted on 2001-2024 only.

Link to our results: acres cut for silage are not harvested for grain. If our satellite estimate of
silage acres differs from the reported figure by D acres, grain production and ending stocks differ
by about -D * yield, and each model turns that into a price effect.

This is a research estimate of supply-price relationships from public data. It is not a trading
signal or investment advice.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).with_name("data") / "corn_market_history.csv"
TRAIN_YEARS = range(2001, 2025)

MODELS = {                      # name: feature columns (log scale); target is log price unless delta
    "no_change": None,
    "supply": ["dlog_prod"],
    "stocks": ["log_stu"],
    "stocks_prev": ["log_stu", "log_price_prev"],
}
DELTA_TARGET = {"supply"}       # these models predict the change of log price, not its level


def load(path=DATA):
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
    df["log_price"] = np.log(df["price_my"])
    df["log_price_prev"] = df["log_price"].shift(1)
    df["dlog_price"] = df["log_price"].diff()
    df["dlog_prod"] = np.log(df["prod"]).diff()
    df["end_stocks"] = df["sep1_stocks"].shift(-1)
    df["use"] = df["sep1_stocks"] + df["prod"] - df["end_stocks"]
    df["stu"] = df["end_stocks"] / df["use"]
    df["log_stu"] = np.log(df["stu"])
    df["silage_share"] = df["silage_ac"] / (df["silage_ac"] + df["grain_ac"])
    return df


def fit(X, y):
    """OLS with an intercept. Returns coefficients [a, b, ...] and the residual standard deviation."""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, float(np.sqrt((resid ** 2).sum() / max(1, len(y) - X.shape[1])))


def predict_log_price(model, coef, row):
    """Predicted log price for one year's row."""
    if MODELS[model] is None:
        return row["log_price_prev"]
    z = coef[0] + sum(c * row[f] for c, f in zip(coef[1:], MODELS[model]))
    return row["log_price_prev"] + z if model in DELTA_TARGET else z


def train(df, model, years):
    if MODELS[model] is None:
        return None
    d = df[df["year"].isin(list(years))].dropna(subset=MODELS[model] + ["log_price", "dlog_price"])
    y = d["dlog_price"] if model in DELTA_TARGET else d["log_price"]
    return fit(d[MODELS[model]].to_numpy(), y.to_numpy())[0]


def leave_one_year_out(df, years=TRAIN_YEARS):
    rows = []
    years = list(years)
    for model in MODELS:
        for yr in years:
            row = df[df["year"] == yr].iloc[0]
            coef = train(df, model, [y for y in years if y != yr])
            p = float(np.exp(predict_log_price(model, coef, row)))
            prev = float(np.exp(row["log_price_prev"]))
            rows.append({"model": model, "year": int(yr), "actual": float(row["price_my"]), "predicted": round(p, 2),
                         "actual_change_pct": round(100 * (row["price_my"] / prev - 1), 1),
                         "predicted_change_pct": round(100 * (p / prev - 1), 1)})
    out = pd.DataFrame(rows)
    out["abs_err_pct"] = (100 * (out["predicted"] / out["actual"] - 1)).abs().round(1)
    out["direction_right"] = np.sign(out["predicted_change_pct"]) == np.sign(out["actual_change_pct"])
    return out


def summary(cv):
    s = cv.groupby("model").agg(mae_pct=("abs_err_pct", "mean"), direction_right=("direction_right", "sum"),
                                years=("year", "size"))
    s["mae_pct"] = s["mae_pct"].round(1)
    return s.loc[list(MODELS)]


def silage_effect(df, model, coef, year, delta_silage_acres):
    """Price change (%) in `year` if silage acres were delta higher than reported (those acres leave grain)."""
    row = df[df["year"] == year].iloc[0].copy()
    dprod = -float(delta_silage_acres) * float(row["yield"])
    base = predict_log_price(model, coef, row)
    if model == "supply":
        row["dlog_prod"] = np.log((row["prod"] + dprod) / df.loc[df["year"] == year - 1, "prod"].iloc[0])
    elif model in ("stocks", "stocks_prev"):
        row["log_stu"] = np.log((row["end_stocks"] + dprod) / row["use"])  # less grain, same use: lower stocks
    else:
        return 0.0, round(dprod)
    return round(100 * float(np.expm1(predict_log_price(model, coef, row) - base)), 2), round(dprod)


def states_delta(comparison_json, model="eu2_logistic_v1"):
    """Sum over the four checked states of (our silage acres - USDA 2025 silage acres)."""
    c = json.loads(Path(comparison_json).read_text())
    rows = [r for r in c["shares"] if r["model"] == model]
    return sum(r["model_silage_acres"] - r["usda_2025_silage_acres"] for r in rows), rows


def run(path=DATA, states_json=None, out_dir=None):
    df = load(path)
    cv = leave_one_year_out(df)
    result = {"leave_one_year_out_2001_2024": summary(cv).reset_index().to_dict("records"),
              "test_2025": {}, "coefficients": {}, "silage_plus_1pt_2025": {}}
    row25 = df[df["year"] == 2025].iloc[0]
    total_ac = float(row25["grain_ac"] + row25["silage_ac"])
    for model in MODELS:
        coef = train(df, model, TRAIN_YEARS)
        p = float(np.exp(predict_log_price(model, coef, row25)))
        result["test_2025"][model] = {"predicted": round(p, 2), "actual": float(row25["price_my"]),
                                      "error_pct": round(100 * (p / row25["price_my"] - 1), 1)}
        if coef is not None:
            result["coefficients"][model] = dict(zip(["intercept"] + MODELS[model], [round(float(c), 3) for c in coef]))
            eff, dprod = silage_effect(df, model, coef, 2025, 0.01 * total_ac)
            result["silage_plus_1pt_2025"][model] = {"acres": round(0.01 * total_ac), "grain_bushels": dprod,
                                                     "price_change_pct": eff}
    result["stocks_to_use_2025"] = round(float(row25["stu"]), 3)
    if states_json and Path(states_json).exists():
        delta, rows = states_delta(states_json)
        result["our_four_states_2025"] = {"silage_acres_ours_minus_usda": int(delta),
                                          "states": [{k: r[k] for k in ("state", "model_silage_acres", "usda_2025_silage_acres")} for r in rows]}
        for model in ("supply", "stocks", "stocks_prev"):
            eff, dprod = silage_effect(df, model, train(df, model, TRAIN_YEARS), 2025, delta)
            result["our_four_states_2025"][model] = {"grain_bushels": dprod, "price_change_pct": eff}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cv.to_csv(out / "leave_one_year_out.csv", index=False)
        (out / "price_impact.json").write_text(json.dumps(result, indent=1, default=float))
    return result, cv
