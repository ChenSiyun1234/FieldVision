"""
Optimized Corn Stock & Price Impact Model with Regional QuickStats Embeddings
-----------------------------------------------------------------------------
1. Aggregates QuickStats across USDA Farm Production Regions (Corn Belt, Dairy Belt, Plains).
2. Fixes Census year county rollups (2012, 2017, 2022) to provide unbroken 2012-2025 coverage.
3. Extracts dense 2D regional agricultural embeddings (qs_emb1, qs_emb2) via PCA.
4. Uses convex stock-to-use elasticity (log STU + inverse STU) + carry-in pressure.
5. Evaluates model accuracy via Leave-One-Year-Out (2001-2024) and 2025 out-of-sample test.
6. Projects 3-year ending stocks, stocks-to-use, and price trajectory under ML silage shift.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


from ..settings import QUICKSTATS_CSV

# The workspace copies read DATA_DIR, which one never defined and the other pointed at a missing folder,
# so neither ran. The market history sits next to this file.
# Read README.md in this folder before quoting any number from here: the PCA, yield trend and fill-in
# are fitted on all years before the leave-one-year-out loop (loyo_clean_check.py refits them per fold),
# and the +0.8 point silage shift in project_3_years is an assumed what-if, not a model output.
MARKET_CSV = Path(__file__).with_name("data") / "corn_market_history.csv"

def clean_numeric(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def extract_regional_embeddings():
    qs = pd.read_csv(QUICKSTATS_CSV, low_memory=False)
    qs["clean_val"] = qs["value"].apply(clean_numeric)
    
    # Filter to total area records (avoid double-counting irrigated subcategories)
    qs_tot = qs[qs["domain"] == "TOTAL"].copy()
    
    # Roll up to state-year level (handles both county-level Census years and state survey years)
    state_rollup = qs_tot.groupby(["year", "state_name", "crop_use"])["clean_val"].sum().unstack("crop_use").fillna(0)
    
    # USDA Regional Clusters
    corn_belt = ["IOWA", "ILLINOIS", "INDIANA", "OHIO", "MISSOURI"]
    dairy_belt = ["WISCONSIN", "MICHIGAN", "MINNESOTA", "NEW YORK", "PENNSYLVANIA"]
    plains = ["NEBRASKA", "KANSAS", "SOUTH DAKOTA", "NORTH DAKOTA"]
    
    reg_records = []
    for y in sorted(state_rollup.index.get_level_values("year").unique()):
        sub = state_rollup.loc[y]
        cb_grain = sub.loc[sub.index.isin(corn_belt), "grain"].sum() if "grain" in sub.columns else 0
        cb_sil = sub.loc[sub.index.isin(corn_belt), "silage"].sum() if "silage" in sub.columns else 0
        db_grain = sub.loc[sub.index.isin(dairy_belt), "grain"].sum() if "grain" in sub.columns else 0
        db_sil = sub.loc[sub.index.isin(dairy_belt), "silage"].sum() if "silage" in sub.columns else 0
        pl_grain = sub.loc[sub.index.isin(plains), "grain"].sum() if "grain" in sub.columns else 0
        pl_sil = sub.loc[sub.index.isin(plains), "silage"].sum() if "silage" in sub.columns else 0
        
        tot_sil = sub["silage"].sum() if "silage" in sub.columns else 0
        tot_gr = sub["grain"].sum() if "grain" in sub.columns else 0
        
        reg_records.append({
            "year": y,
            "cb_silage_share": cb_sil / (cb_grain + cb_sil + 1e-6),
            "db_silage_share": db_sil / (db_grain + db_sil + 1e-6),
            "pl_silage_share": pl_sil / (pl_grain + pl_sil + 1e-6),
            "dairy_forage_ratio": db_sil / (tot_sil + 1e-6),
            "cornbelt_grain_dominance": cb_grain / (tot_gr + 1e-6)
        })
        
    reg_df = pd.DataFrame(reg_records)
    reg_features = ["cb_silage_share", "db_silage_share", "pl_silage_share", "dairy_forage_ratio", "cornbelt_grain_dominance"]
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(reg_df[reg_features])
    pca = PCA(n_components=2, random_state=42)
    embs = pca.fit_transform(scaled)
    
    reg_df["qs_emb1"] = embs[:, 0]
    reg_df["qs_emb2"] = embs[:, 1]
    
    return reg_df[["year", "qs_emb1", "qs_emb2"]]

def prepare_market_dataset():
    df = pd.read_csv(MARKET_CSV).sort_values("year").reset_index(drop=True)
    df["log_price"] = np.log(df["price_my"])
    df["log_price_prev"] = df["log_price"].shift(1)
    df["end_stocks"] = df["sep1_stocks"].shift(-1)
    df["use"] = df["sep1_stocks"] + df["prod"] - df["end_stocks"]
    df["stu"] = df["end_stocks"] / df["use"]
    df["log_stu"] = np.log(df["stu"])
    df["inv_stu"] = 1.0 / df["stu"]
    df["silage_share"] = df["silage_ac"] / (df["silage_ac"] + df["grain_ac"])
    df["carryin_pressure"] = df["sep1_stocks"] / df["prod"]
    
    # Detrend yield
    years = df["year"].values
    trend_p = np.polyfit(years[:25], df["yield"].iloc[:25], 1)
    trend_y = np.poly1d(trend_p)(years)
    df["yield_dev"] = (df["yield"] - trend_y) / trend_y
    
    # Merge Regional QuickStats Embeddings
    reg_emb = extract_regional_embeddings()
    df = df.merge(reg_emb, on="year", how="left")
    
    # Learn historical projection for 2000-2011 to avoid zero-filling
    known = df.dropna(subset=["qs_emb1"])
    proj_feats = ["silage_share", "carryin_pressure", "yield_dev"]
    m_p1 = Ridge(alpha=0.1).fit(known[proj_feats], known["qs_emb1"])
    m_p2 = Ridge(alpha=0.1).fit(known[proj_feats], known["qs_emb2"])
    
    hist_missing = df["qs_emb1"].isna() & (df["year"] <= 2025)
    df.loc[hist_missing, "qs_emb1"] = m_p1.predict(df.loc[hist_missing, proj_feats])
    df.loc[hist_missing, "qs_emb2"] = m_p2.predict(df.loc[hist_missing, proj_feats])
    
    return df

def evaluate_models(df):
    train = df[(df["year"] >= 2001) & (df["year"] <= 2024)].copy()
    
    models = {
        "Baseline Linear STU": ["log_stu", "log_price_prev"],
        "Convex STU (Non-linear)": ["log_stu", "inv_stu", "log_price_prev"],
        "Optimized Regional Embedding Model": ["log_stu", "inv_stu", "log_price_prev", "yield_dev", "qs_emb1"]
    }
    
    summary = {}
    for name, feats in models.items():
        maes, rmses, mapes, dirs = [], [], [], []
        for yr in train["year"]:
            tr = train[train["year"] != yr]
            val = train[train["year"] == yr].iloc[0]
            
            m = Ridge(alpha=0.4).fit(tr[feats], tr["log_price"])
            pred = np.exp(m.predict([val[feats]])[0])
            actual = val["price_my"]
            prev = np.exp(val["log_price_prev"])
            
            maes.append(abs(pred - actual))
            rmses.append((pred - actual)**2)
            mapes.append(abs(pred - actual) / actual * 100)
            dirs.append(np.sign(pred - prev) == np.sign(actual - prev))
            
        # 2025 out-of-sample check
        row25 = df[df["year"] == 2025].iloc[0]
        m_full = Ridge(alpha=0.4).fit(train[feats], train["log_price"])
        p25 = np.exp(m_full.predict([row25[feats]])[0])
        err25 = abs(p25 - row25["price_my"]) / row25["price_my"] * 100
        
        summary[name] = {
            "MAE": round(np.mean(maes), 3),
            "RMSE": round(np.sqrt(np.mean(rmses)), 3),
            "MAPE": round(np.mean(mapes), 2),
            "Directional_Accuracy": round(np.mean(dirs) * 100, 1),
            "2025_Test_Error": round(err25, 2),
            "2025_Pred_Price": round(p25, 2),
            "model_obj": m_full,
            "feats": feats
        }
    return summary

def project_3_years(df, best_model, feats, ml_silage_loss_pct=0.8):
    last = df[df["year"] == 2025].iloc[0]
    
    projected_years = [2026, 2027, 2028]
    yields = [181.5, 183.5, 185.5]
    planted_acres = [90.5e6, 90.0e6, 91.0e6]
    base_uses = [15.1e9, 15.2e9, 15.35e9]
    
    projections = []
    curr_prev_price = last["price_my"]
    curr_sep1_stocks = last["end_stocks"]
    
    # Use 2024-2025 mean for future regional embedding
    steady_emb = df["qs_emb1"].iloc[-2:].mean()
    
    for i, yr in enumerate(projected_years):
        pl_ac = planted_acres[i]
        yd = yields[i]
        use = base_uses[i]
        
        # Official Baseline Path
        off_silage_share = 0.064
        off_silage_ac = pl_ac * off_silage_share
        off_grain_ac = pl_ac - off_silage_ac
        off_prod = off_grain_ac * yd
        off_end_stocks = curr_sep1_stocks + off_prod - use
        off_stu = off_end_stocks / use
        
        # ML Silage Corrected Path (+0.8% silage undercount)
        ml_silage_share = off_silage_share + (ml_silage_loss_pct / 100.0)
        ml_silage_ac = pl_ac * ml_silage_share
        ml_grain_ac = pl_ac - ml_silage_ac
        ml_prod = ml_grain_ac * yd
        delta_grain_prod = ml_prod - off_prod # -131M to -135M bu
        ml_end_stocks = curr_sep1_stocks + ml_prod - use
        ml_stu = ml_end_stocks / use
        
        # Construct feature vector
        # feats: ["log_stu", "inv_stu", "log_price_prev", "yield_dev", "qs_emb1"]
        x_off = [np.log(off_stu), 1.0 / off_stu, np.log(curr_prev_price), 0.0, steady_emb]
        p_off = float(np.exp(best_model.predict([x_off])[0]))
        
        x_ml = [np.log(ml_stu), 1.0 / ml_stu, np.log(curr_prev_price), 0.0, steady_emb]
        p_ml = float(np.exp(best_model.predict([x_ml])[0]))
        
        projections.append({
            "year": yr,
            "planted_ac_m": round(pl_ac / 1e6, 2),
            "yield": yd,
            "baseline_end_stocks_mbu": round(off_end_stocks / 1e6, 0),
            "baseline_stu_pct": round(off_stu * 100, 2),
            "baseline_price": round(p_off, 2),
            "ml_end_stocks_mbu": round(ml_end_stocks / 1e6, 0),
            "ml_stu_pct": round(ml_stu * 100, 2),
            "ml_price": round(p_ml, 2),
            "delta_grain_mbu": round(delta_grain_prod / 1e6, 0),
            "price_impact_pct": round((p_ml - p_off) / p_off * 100, 2)
        })
        
        curr_prev_price = p_ml
        curr_sep1_stocks = ml_end_stocks
        
    return projections

if __name__ == "__main__":
    df = prepare_market_dataset()
    results = evaluate_models(df)
    
    print("\n================== ACCURACY COMPARISON ==================")
    for name, r in results.items():
        print(f"\nModel: {name}")
        print(f"  MAE:                  ${r['MAE']} / bu")
        print(f"  RMSE:                 ${r['RMSE']} / bu")
        print(f"  MAPE:                 {r['MAPE']}%")
        print(f"  Directional Accuracy: {r['Directional_Accuracy']}%")
        print(f"  2025 Test Error:      {r['2025_Test_Error']}% (Pred: ${r['2025_Pred_Price']} vs Actual: $4.16)")
        
    best = results["Optimized Regional Embedding Model"]
    proj = project_3_years(df, best["model_obj"], best["feats"])
    
    print("\n================== 3-YEAR PROJECTIONS (2026-2028) ==================")
    for p in proj:
        print(f"\nYear {p['year']} (Planted: {p['planted_ac_m']}M ac, Yield: {p['yield']} bu/ac):")
        print(f"  Baseline:     Stocks = {p['baseline_end_stocks_mbu']:,.0f} M bu | STU = {p['baseline_stu_pct']}% | Price = ${p['baseline_price']:.2f}/bu")
        print(f"  ML-Corrected: Stocks = {p['ml_end_stocks_mbu']:,.0f} M bu | STU = {p['ml_stu_pct']}% | Price = ${p['ml_price']:.2f}/bu")
        print(f"  --> Impact:   Grain Shift = {p['delta_grain_mbu']:+,.0f} M bu | Price Effect = {p['price_impact_pct']:+.2f}%")
