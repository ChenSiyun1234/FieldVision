"""USDA Quick Stats corn acreage table from the starter kit: one clean record per
geography, year, crop use and measurement.

    python -m fieldvision.model.quickstats --csv corn_grain_silage_acres_2012_2025.csv [--out dir]

Selection rule (worked out on the 2012-2025 file; summing raw rows double counts):

1. series: short_description is one of
   "CORN, GRAIN - ACRES HARVESTED", "CORN, SILAGE - ACRES HARVESTED", "CORN - ACRES PLANTED"
   (this drops the irrigated / non-irrigated / organic breakdowns);
2. domain == "TOTAL" (drops water source, irrigation and organic sub-domains);
3. reference_period == "YEAR" (drops the March/June/August... acreage and forecast releases);
4. county level: source == "CENSUS" (Census years 2012, 2017, 2022). The county SURVEY rows
   are a separate, sparser estimate for the same years; mixing the two double counts. Rows
   named "OTHER (COMBINED) COUNTIES" have no county code and are several per state; they are
   dropped (they are not a county);
5. state level: source == "SURVEY" (the file only has state rows for non-Census years).

After this every (level, state, county, year, crop use, measurement) has exactly one record;
`select` raises if not. Values: thousands separators removed; "(D)" (withheld to avoid
disclosing single operations) and other codes become NaN with `suppressed` = True and are
never treated as zero.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SERIES = ["CORN, GRAIN - ACRES HARVESTED", "CORN, SILAGE - ACRES HARVESTED",
          "CORN - ACRES PLANTED"]
KEY = ["aggregation_level", "state_ansi", "county_ansi", "year", "crop_use", "measurement"]


def load(path):
    d = pd.read_csv(path, dtype=str)
    for c in d.columns:
        d[c] = d[c].str.strip()
    raw = d["value"].fillna("")
    d["value_num"] = pd.to_numeric(raw.str.replace(",", "", regex=False), errors="coerce")
    d["suppressed"] = d["value_num"].isna()
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    return d


def select(d):
    """Apply the selection rule; one record per KEY, checked."""
    f = d[d["short_description"].isin(SERIES) & (d["domain"] == "TOTAL")
          & (d["reference_period"] == "YEAR")]
    county = f[(f["aggregation_level"] == "COUNTY") & (f["source"] == "CENSUS")
               & f["county_ansi"].notna() & (f["county_ansi"] != "")]
    state = f[(f["aggregation_level"] == "STATE") & (f["source"] == "SURVEY")]
    out = pd.concat([county, state], ignore_index=True)
    dup = out.duplicated(KEY, keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} rows still duplicated after the selection rule: "
                         f"{out.loc[dup, KEY + ['source']].head().to_dict('records')}")
    out["fips"] = np.where(out["aggregation_level"] == "COUNTY",
                           out["state_ansi"].str.zfill(2) + out["county_ansi"].fillna("").str.zfill(3),
                           out["state_ansi"].str.zfill(2))
    return out


def shares(sel, level, year):
    """Grain and silage harvested acres and the silage share per geography for one year.

    The share is NaN where either number is withheld. Planted acres are kept for reference.
    """
    s = sel[(sel["aggregation_level"] == level) & (sel["year"] == year)].copy()
    s["col"] = s["crop_use"] + "_" + s["measurement"]
    piv = s.set_index(["fips", "col"])["value_num"].unstack("col")
    sup = s.set_index(["fips", "crop_use"])["suppressed"].unstack("crop_use")
    names = s.groupby("fips")[["state_name", "county_name"]].first()
    out = names.join(piv).reset_index()
    if level == "STATE":
        out = out.drop(columns="county_name")
    for c in ("grain_harvested", "silage_harvested", "all_corn_planted"):
        if c not in out.columns:
            out[c] = np.nan
    tot = out["grain_harvested"] + out["silage_harvested"]
    out["silage_share"] = np.where(tot > 0, out["silage_harvested"] / tot, np.nan)
    out["silage_withheld"] = out["fips"].map(sup.get("silage", pd.Series(dtype=bool))).fillna(True)
    out["year"] = year
    return out


def national_share(sel, year):
    """Silage share of harvested corn summed over geographies where both numbers are disclosed.

    County level for Census years, state level otherwise. Coverage says how much of the
    disclosed grain area that sum covers; withheld counties are left out, not zeroed.
    """
    level = "COUNTY" if (sel[(sel["aggregation_level"] == "COUNTY")]["year"] == year).any() else "STATE"
    s = shares(sel, level, year)
    both = s["grain_harvested"].notna() & s["silage_harvested"].notna()
    g, sl = s.loc[both, "grain_harvested"].sum(), s.loc[both, "silage_harvested"].sum()
    return {"year": int(year), "level": level, "n_units": int(len(s)), "n_both_disclosed": int(both.sum()),
            "grain_acres": float(g), "silage_acres": float(sl),
            "silage_share": float(sl / (g + sl)) if g + sl > 0 else float("nan"),
            "grain_area_covered": float(g / s["grain_harvested"].sum())
            if s["grain_harvested"].sum() > 0 else float("nan")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Clean the Quick Stats corn acreage table.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None, help="directory for clean CSVs")
    args = ap.parse_args(argv)
    d = load(args.csv)
    sel = select(d)
    print(f"{len(d)} rows -> {len(sel)} after the selection rule "
          f"({int(sel['suppressed'].sum())} withheld values kept as NaN)")
    years = sorted(int(y) for y in sel["year"].dropna().unique())
    rows = [national_share(sel, y) for y in years]
    nat = pd.DataFrame(rows)
    print(nat.to_string(index=False))
    ia = sel[(sel["state_name"] == "IOWA") & (sel["crop_use"] == "grain")]
    ia_tot = ia.groupby(["aggregation_level", "year"])["value_num"].sum(min_count=1)
    print("Iowa grain harvested acres (state rows, or sum of disclosed counties):")
    print(ia_tot.to_string())
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        sel.to_csv(out / "quickstats_selected.csv", index=False)
        nat.to_csv(out / "national_silage_share.csv", index=False)
        for y in (2012, 2017, 2022):
            if y in years:
                shares(sel, "COUNTY", y).to_csv(out / f"county_shares_{y}.csv", index=False)
        st = pd.concat([shares(sel, "STATE", y) for y in years
                        if (sel[(sel["aggregation_level"] == "STATE")]["year"] == y).any()])
        st.to_csv(out / "state_shares.csv", index=False)
    return sel, nat


if __name__ == "__main__":
    main()
