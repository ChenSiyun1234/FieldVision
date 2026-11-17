"""Build the 2023 survey and 2022 Census county tables, then the candidate list."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import append_log
from .config import (
    FOCUS_STATES,
    GRAIN_DESC,
    MIN_TOTAL_ACRES,
    NASS_CENSUS_CSV,
    NASS_SURVEY_CSV,
    PROCESSED,
    SILAGE_DESC,
    SILAGE_SHARE_MAX,
    SILAGE_SHARE_MIN,
    TOP_STATES,
)

ACRE_COLUMNS = ["state", "county", "fips", "grain_acres", "silage_acres", "silage_share"]


def parse_acres(value) -> float:
    text = str(value).strip().replace(",", "")
    if text in {"(D)", "(Z)", "", "nan", "None", "NaN"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _prepare(frame: pd.DataFrame, year: str, source: str) -> pd.DataFrame:
    """One grain row and one silage row per county, TOTAL domain only."""
    kept = frame.loc[
        (frame["aggregation_level"] == "COUNTY")
        & (frame["domain"] == "TOTAL")
        & (frame["year"] == year)
        & (frame["source"] == source)
        & (frame["short_description"].isin([GRAIN_DESC, SILAGE_DESC]))
    ].copy()
    kept["crop"] = np.where(kept["short_description"] == GRAIN_DESC, "grain", "silage")
    # "OTHER COUNTIES" has no county ANSI code. It is a state remainder, not a county.
    missing_fips = kept["county_ansi"].isna() | (kept["county_ansi"].str.strip() == "")
    if missing_fips.any():
        print(f"{year} {source}: excluded {int(missing_fips.sum())} rows with no county FIPS (OTHER COUNTIES).")
        kept = kept.loc[~missing_fips].copy()
    kept["fips"] = kept["state_ansi"].str.zfill(2) + kept["county_ansi"].str.zfill(3)
    kept["acres"] = kept["value"].map(parse_acres)
    duplicated = kept.duplicated(["fips", "crop"], keep=False)
    if duplicated.any():
        print(f"{year} {source}: duplicate county rows, keeping the first:")
        print(
            kept.loc[duplicated, ["fips", "state_name", "county_name", "crop", "acres"]]
            .sort_values(["fips", "crop"])
            .to_string(index=False)
        )
        kept = kept.drop_duplicates(["fips", "crop"], keep="first")
    counts = kept.groupby(["fips", "crop"]).size()
    if (counts > 1).any():
        raise AssertionError(f"{year} {source} still has more than one row per county and crop.")
    wide = kept.pivot(index=["state_name", "county_name", "fips"], columns="crop", values="acres")
    wide = wide.reset_index().rename(columns={"state_name": "state", "county_name": "county"})
    for column in ("grain", "silage"):
        if column not in wide.columns:
            wide[column] = np.nan
    wide = wide.rename(columns={"grain": "grain_acres", "silage": "silage_acres"})
    total = wide["grain_acres"] + wide["silage_acres"]
    wide["silage_share"] = wide["silage_acres"] / total
    return wide[ACRE_COLUMNS].sort_values(["state", "county"]).reset_index(drop=True)


def load_county_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    survey_raw = pd.read_csv(NASS_SURVEY_CSV, dtype=str)
    census_raw = pd.read_csv(NASS_CENSUS_CSV, dtype=str)
    survey = _prepare(survey_raw, "2023", "SURVEY")
    census = _prepare(census_raw, "2022", "CENSUS")
    return survey, census


def both_reported(table: pd.DataFrame) -> pd.DataFrame:
    return table.dropna(subset=["grain_acres", "silage_acres"])


def candidates_from(survey: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """Prefer 2023 survey acres when both classes exist, otherwise the 2022 Census."""
    merged = survey.merge(census, on="fips", how="outer", suffixes=("_2023", "_2022"))
    use_survey = merged["grain_acres_2023"].notna() & merged["silage_acres_2023"].notna()
    use_census = ~use_survey & merged["grain_acres_2022"].notna() & merged["silage_acres_2022"].notna()
    merged = merged.loc[use_survey | use_census].copy()
    use_survey = use_survey.loc[merged.index].to_numpy()
    merged["share_source"] = np.where(use_survey, "SURVEY_2023", "CENSUS_2022")
    merged["state"] = np.where(use_survey, merged["state_2023"], merged["state_2022"])
    merged["county"] = np.where(use_survey, merged["county_2023"], merged["county_2022"])
    merged["grain_acres"] = np.where(use_survey, merged["grain_acres_2023"], merged["grain_acres_2022"])
    merged["silage_acres"] = np.where(use_survey, merged["silage_acres_2023"], merged["silage_acres_2022"])
    merged["silage_share"] = np.where(use_survey, merged["silage_share_2023"], merged["silage_share_2022"])
    merged["total_acres"] = merged["grain_acres"] + merged["silage_acres"]
    keep = merged["silage_share"].between(SILAGE_SHARE_MIN, SILAGE_SHARE_MAX) & (
        merged["total_acres"] >= MIN_TOTAL_ACRES
    )
    columns = [
        "state",
        "county",
        "fips",
        "grain_acres",
        "silage_acres",
        "total_acres",
        "silage_share",
        "share_source",
        "silage_share_2023",
        "silage_share_2022",
        "grain_acres_2023",
        "silage_acres_2023",
        "grain_acres_2022",
        "silage_acres_2022",
    ]
    out = merged.loc[keep, columns].copy()
    return out.sort_values(["state", "total_acres"], ascending=[True, False]).reset_index(drop=True)


def _print_both(survey: pd.DataFrame) -> None:
    both = both_reported(survey)
    print(f"Counties with both grain and silage reported in the 2023 survey: {len(both)}")
    for state in FOCUS_STATES:
        count = int((both["state"].str.upper() == state).sum())
        print(f"  {state.title()}: {count}")


def _print_top(candidates: pd.DataFrame) -> None:
    for state in TOP_STATES:
        part = candidates.loc[candidates["state"].str.upper() == state].head(10)
        print(f"\nTop {min(10, len(part))} candidate counties in {state.title()} ({len(candidates.loc[candidates['state'].str.upper() == state])} passed):")
        if part.empty:
            print("  none")
            continue
        show = part[["county", "fips", "total_acres", "silage_share", "share_source"]]
        print(show.to_string(index=False))


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    survey, census = load_county_tables()
    survey_path = PROCESSED / "nass_county_2023.csv"
    census_path = PROCESSED / "nass_county_2022.csv"
    candidate_path = PROCESSED / "candidate_counties.csv"
    survey.to_csv(survey_path, index=False)
    census.to_csv(census_path, index=False)
    print(f"Wrote {len(survey):,} counties to {survey_path}")
    print(f"Wrote {len(census):,} counties to {census_path}")
    _print_both(survey)
    chosen = candidates_from(survey, census)
    chosen.to_csv(candidate_path, index=False)
    print(f"\nWrote {len(chosen):,} candidate counties to {candidate_path}")
    _print_top(chosen)
    append_log(
        "NASS county selection",
        [
            ("survey file", NASS_SURVEY_CSV),
            ("census file", NASS_CENSUS_CSV),
            ("filters", "COUNTY, domain TOTAL, exact grain/silage harvested descriptions, source SURVEY for 2023 and CENSUS for 2022"),
            ("withheld values", "(D) and (Z) set to missing; commas stripped"),
            ("2023 survey counties", len(survey)),
            ("2023 both classes", len(both_reported(survey))),
            ("2022 census counties", len(census)),
            ("candidate rule", f"silage share {SILAGE_SHARE_MIN}-{SILAGE_SHARE_MAX} and total corn >= {MIN_TOTAL_ACRES}; 2023 survey when both classes exist, else 2022 Census"),
            ("candidates", len(chosen)),
            ("outputs", f"{survey_path}, {census_path}, {candidate_path}"),
        ],
    )


if __name__ == "__main__":
    main()
