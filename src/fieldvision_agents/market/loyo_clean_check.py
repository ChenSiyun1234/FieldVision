"""Clean leave-one-year-out check of the stock/price models (review of stocks/stock_model_prediction.py).

The original fits the yield trend, the PCA embedding and the 2000-2011 fill-in on every year before the
leave-one-year-out loop. Here all three are refit inside each fold, on the training years only, so the
held-out year never shapes its own features. "Last year's price" is scored the same way.

Two feature sets:
  explain   uses this season's ending stocks (known only after the season), as the original does;
  forecast  uses last season's stocks-to-use instead, which is known when the season starts.

Also: the price effect of +1 point of national silage share in 2025 under each model, and where the
README's +$0.05-0.08 comes from.

    python loyo_clean_check.py <corn_market_history.csv> <corn_grain_silage_acres_2012_2025.csv>
"""

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REG_FEATURES = ["cb_silage_share", "db_silage_share", "pl_silage_share", "dairy_forage_ratio",
                "cornbelt_grain_dominance"]
CORN_BELT = ["IOWA", "ILLINOIS", "INDIANA", "OHIO", "MISSOURI"]
DAIRY_BELT = ["WISCONSIN", "MICHIGAN", "MINNESOTA", "NEW YORK", "PENNSYLVANIA"]
PLAINS = ["NEBRASKA", "KANSAS", "SOUTH DAKOTA", "NORTH DAKOTA"]


def _num(v):
    s = str(v).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def regional_features(quickstats_csv):
    """The original's five regional shares per year (before PCA), same rollup."""
    qs = pd.read_csv(quickstats_csv, low_memory=False)
    qs["v"] = qs["value"].map(_num)
    roll = qs[qs["domain"] == "TOTAL"].groupby(["year", "state_name", "crop_use"])["v"].sum() \
        .unstack("crop_use").fillna(0)
    rows = []
    for y in sorted(roll.index.get_level_values("year").unique()):
        sub = roll.loc[y]

        def tot(states, col):
            return sub.loc[sub.index.isin(states), col].sum()

        rows.append({"year": y,
                     "cb_silage_share": tot(CORN_BELT, "silage") / (tot(CORN_BELT, "grain") + tot(CORN_BELT, "silage") + 1e-6),
                     "db_silage_share": tot(DAIRY_BELT, "silage") / (tot(DAIRY_BELT, "grain") + tot(DAIRY_BELT, "silage") + 1e-6),
                     "pl_silage_share": tot(PLAINS, "silage") / (tot(PLAINS, "grain") + tot(PLAINS, "silage") + 1e-6),
                     "dairy_forage_ratio": tot(DAIRY_BELT, "silage") / (sub["silage"].sum() + 1e-6),
                     "cornbelt_grain_dominance": tot(CORN_BELT, "grain") / (sub["grain"].sum() + 1e-6)})
    return pd.DataFrame(rows)


def base_table(market_csv, quickstats_csv):
    """Columns that need no fitting: prices, stocks-to-use (this season and last), shares."""
    df = pd.read_csv(market_csv).dropna(how="all").sort_values("year").reset_index(drop=True)
    df["log_price"] = np.log(df["price_my"])
    df["log_price_prev"] = df["log_price"].shift(1)
    df["end_stocks"] = df["sep1_stocks"].shift(-1)
    df["use"] = df["sep1_stocks"] + df["prod"] - df["end_stocks"]
    df["stu"] = df["end_stocks"] / df["use"]
    df["log_stu"], df["inv_stu"] = np.log(df["stu"]), 1 / df["stu"]
    df["stu_prev"] = df["stu"].shift(1)                     # last season's: known when this season starts
    df["log_stu_prev"], df["inv_stu_prev"] = np.log(df["stu_prev"]), 1 / df["stu_prev"]
    df["silage_share"] = df["silage_ac"] / (df["silage_ac"] + df["grain_ac"])
    df["carryin_pressure"] = df["sep1_stocks"] / df["prod"]
    return df.merge(regional_features(quickstats_csv), on="year", how="left")


def fold_features(df, train_years):
    """Yield trend, PCA embedding and fill-in, all fitted on train_years only, applied to every row."""
    out = df.copy()
    tr = out["year"].isin(train_years)
    trend = np.poly1d(np.polyfit(out.loc[tr, "year"], out.loc[tr, "yield"], 1))(out["year"])
    out["yield_dev"] = (out["yield"] - trend) / trend

    has_reg = out[REG_FEATURES].notna().all(axis=1)
    fit_rows = tr & has_reg
    scaler = StandardScaler().fit(out.loc[fit_rows, REG_FEATURES])
    pca = PCA(n_components=2, random_state=42).fit(scaler.transform(out.loc[fit_rows, REG_FEATURES]))
    out["qs_emb1"] = np.nan
    out.loc[has_reg, "qs_emb1"] = pca.transform(scaler.transform(out.loc[has_reg, REG_FEATURES]))[:, 0]

    proj = ["silage_share", "carryin_pressure", "yield_dev"]
    known = tr & out["qs_emb1"].notna()
    fill = Ridge(alpha=0.1).fit(out.loc[known, proj], out.loc[known, "qs_emb1"])
    missing = out["qs_emb1"].isna() & out[proj].notna().all(axis=1)
    out.loc[missing, "qs_emb1"] = fill.predict(out.loc[missing, proj])
    return out


MODELS = {
    # name: (features, estimator); the first three are the original's, the forecast ones swap in last season's stocks
    "explain: linear STU": (["log_stu", "log_price_prev"], "ridge"),
    "explain: convex STU": (["log_stu", "inv_stu", "log_price_prev"], "ridge"),
    "explain: regional embedding": (["log_stu", "inv_stu", "log_price_prev", "yield_dev", "qs_emb1"], "ridge"),
    "forecast: convex STU (last season's)": (["log_stu_prev", "inv_stu_prev", "log_price_prev"], "ridge"),
    "forecast: regional embedding (last season's STU)":
        (["log_stu_prev", "inv_stu_prev", "log_price_prev", "yield_dev", "qs_emb1"], "ridge"),
}


def fit(feats, kind, data):
    est = Ridge(alpha=0.4) if kind == "ridge" else LinearRegression()
    return est.fit(data[feats], data["log_price"])


def evaluate(df, years=range(2001, 2025), test_year=2025):
    rows = []
    for name, (feats, kind) in {"no change (last year's price)": (None, None), **MODELS}.items():
        errs, ape, right, n_dir = [], [], 0, 0
        for y in years:
            train = [t for t in years if t != y]
            f = fold_features(df, train) if feats else df
            val = f[f["year"] == y].iloc[0]
            prev, actual = np.exp(val["log_price_prev"]), val["price_my"]
            if feats:
                pred = float(np.exp(fit(feats, kind, f[f["year"].isin(train)]).predict(val[feats].to_frame().T)[0]))
                right += np.sign(pred - prev) == np.sign(actual - prev)
                n_dir += 1
            else:
                pred = prev
            errs.append(abs(pred - actual))
            ape.append(abs(pred - actual) / actual * 100)
        f = fold_features(df, list(years)) if feats else df
        row25 = f[f["year"] == test_year].iloc[0]
        if feats:
            p25 = float(np.exp(fit(feats, kind, f[f["year"].isin(years)]).predict(row25[feats].to_frame().T)[0]))
        else:
            p25 = float(np.exp(row25["log_price_prev"]))
        rows.append({"model": name, "mae_usd": round(float(np.mean(errs)), 3), "mape_pct": round(float(np.mean(ape)), 2),
                     "direction_right_pct": round(100 * right / n_dir, 1) if n_dir else None,
                     "pred_2025": round(p25, 2), "actual_2025": round(float(row25["price_my"]), 2),
                     "error_2025_pct": round((p25 - row25["price_my"]) / row25["price_my"] * 100, 2)})
    return rows


def silage_plus_one_point(df, years=range(2001, 2025), year=2025, points=1.0):
    """+points of national silage share in `year`: those acres leave grain, production and ending stocks
    fall by acres x yield, stocks-to-use is recomputed; everything else held. Only 'explain' models react,
    since the forecast ones use last season's stocks."""
    f = fold_features(df, list(years))
    r = f[f["year"] == year].iloc[0].copy()
    acres = (r["grain_ac"] + r["silage_ac"]) * points / 100
    bushels = acres * r["yield"]
    shocked = r.copy()
    end = r["end_stocks"] - bushels
    use = r["use"]                                   # use unchanged; the shortfall comes out of stocks
    shocked["log_stu"], shocked["inv_stu"] = np.log(end / use), use / end
    out = {"acres": round(acres), "grain_bushels": round(-bushels), "stu_before": round(r["stu"], 4),
           "stu_after": round(end / use, 4), "models": {}}
    for name, (feats, kind) in MODELS.items():
        if not name.startswith("explain"):
            continue
        m = fit(feats, kind, f[f["year"].isin(years)])
        base = float(np.exp(m.predict(r[feats].to_frame().T)[0]))
        new = float(np.exp(m.predict(shocked[feats].to_frame().T)[0]))
        out["models"][name] = {"price_before": round(base, 3), "price_after": round(new, 3),
                               "change_usd": round(new - base, 3), "change_pct": round((new - base) / base * 100, 2)}
    return out


if __name__ == "__main__":
    market_csv, quickstats_csv = sys.argv[1], sys.argv[2]
    df = base_table(market_csv, quickstats_csv)
    result = {"leave_one_year_out_2001_2024_clean": evaluate(df), "silage_plus_1pt_2025": silage_plus_one_point(df)}
    print(json.dumps(result, indent=1))
