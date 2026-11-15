"""Train silage-versus-grain on BARC, then score counties two ways.

BARC is the only field-level ground truth, so it is the only training set.
Counties have no field labels. Their NASS silage share is a county total, so
it is used in one of the two outputs and kept out of the other:

  Independent estimate. The BARC model is applied unchanged. No county NASS
  number enters the score. The acre share this calls silage is compared with
  NASS. That comparison is the evidence that the model transfers.

  NASS-calibrated map. Two views of the same ranking. A prior shift moves the
  probabilities so their acre-weighted mean equals the county NASS share, and
  a rank cutoff flags the highest-scoring fields until the flagged acres match
  that share. The satellites still decide which fields. Neither number can
  also be used to check that county.

    python -m src.train

BARC accuracy is leave-one-year-out: each season is scored by a model that
never saw it. A second pass holds out every year of one field at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import append_log
from .config import PROCESSED
from .features import MODEL_FEATURES, field_features, load_region, region_areas

ACRES_PER_HA = 2.47105
SILAGE = "Silage"


def _model() -> Pipeline:
    # No class weighting. The probabilities keep BARC's own silage rate, so a
    # county score is a real transfer of that model rather than a rebalanced one.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )


def _labelled(frame: pd.DataFrame) -> pd.DataFrame:
    kept = frame.loc[frame["label"].isin(["Silage", "Grain"])].copy()
    kept["y"] = (kept["label"] == SILAGE).astype(int)
    return kept


def _scores(y_true: np.ndarray, probability: np.ndarray) -> dict:
    call = probability >= 0.5
    truth = y_true.astype(bool)
    positive = int(truth.sum())
    negative = int((~truth).sum())
    true_positive = int((call & truth).sum())
    false_positive = int((call & ~truth).sum())
    return {
        "n": int(len(y_true)),
        "silage": positive,
        "grain": negative,
        "precision": true_positive / (true_positive + false_positive) if (true_positive + false_positive) else np.nan,
        "recall": true_positive / positive if positive else np.nan,
        "false_positive_rate": false_positive / negative if negative else np.nan,
    }


def _cross_validate(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Out-of-fold silage probabilities. `group` is 'year' or 'field_id'."""
    features = frame[MODEL_FEATURES].to_numpy(dtype="float64")
    target = frame["y"].to_numpy()
    groups = frame[group].to_numpy()
    probability = np.full(len(frame), np.nan)
    for train_index, test_index in LeaveOneGroupOut().split(features, target, groups):
        if len(np.unique(target[train_index])) < 2:
            continue
        model = _model()
        model.fit(features[train_index], target[train_index])
        probability[test_index] = model.predict_proba(features[test_index])[:, 1]
    out = frame[["field_id", "year", "label"]].copy()
    out["p_silage"] = probability
    out["fold"] = group
    return out


def _fit(frame: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    model = _model()
    model.fit(frame[MODEL_FEATURES].to_numpy(dtype="float64"), frame["y"].to_numpy())
    coefficients = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "coefficient": model.named_steps["clf"].coef_.ravel(),
        }
    )
    coefficients["direction"] = np.where(
        coefficients["coefficient"] > 0, "higher means more silage", "higher means more grain"
    )
    return model, coefficients.sort_values("coefficient", key=np.abs, ascending=False)


def _prior_shift(probability: np.ndarray, area: np.ndarray, target_share: float) -> np.ndarray:
    """Move every probability by one logit offset so the acre-weighted mean hits the target.

    The ranking of fields is unchanged. Only the county total moves.
    """
    logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1 - 1e-6))

    def share(offset: float) -> float:
        shifted = 1 / (1 + np.exp(-(logit + offset)))
        return float(np.average(shifted, weights=area))

    low, high = -12.0, 12.0
    for _ in range(60):
        mid = (low + high) / 2
        if share(mid) < target_share:
            low = mid
        else:
            high = mid
    return 1 / (1 + np.exp(-(logit + (low + high) / 2)))


def _nass(fips: str) -> dict | None:
    """2023 survey when both classes were published, otherwise the 2022 Census."""
    survey = pd.read_csv(PROCESSED / "nass_county_2023.csv", dtype={"fips": str})
    hit = survey.loc[survey["fips"] == fips]
    if len(hit) and pd.notna(hit["silage_acres"].iloc[0]) and pd.notna(hit["grain_acres"].iloc[0]):
        row = hit.iloc[0]
        return {
            "source": "NASS 2023 survey",
            "share": float(row["silage_share"]),
            "silage_acres": float(row["silage_acres"]),
            "grain_acres": float(row["grain_acres"]),
        }
    census = pd.read_csv(PROCESSED / "nass_county_2022.csv", dtype={"fips": str})
    hit = census.loc[census["fips"] == fips]
    if len(hit) and pd.notna(hit["silage_acres"].iloc[0]) and pd.notna(hit["grain_acres"].iloc[0]):
        row = hit.iloc[0]
        return {
            "source": "NASS 2022 Census (no 2023 silage acres)",
            "share": float(row["silage_share"]),
            "silage_acres": float(row["silage_acres"]),
            "grain_acres": float(row["grain_acres"]),
        }
    return None


def _rank_cutoff(probability: np.ndarray, area: np.ndarray, target_share: float) -> float:
    """Lowest score to keep so the flagged acres reach the target share.

    Fields stay in the satellite ranking. The cutoff only moves the county total.
    """
    order = np.argsort(-probability)
    ranked_area = area[order]
    total = ranked_area.sum()
    if total <= 0:
        return 0.5
    reached = np.cumsum(ranked_area) / total
    index = int(np.searchsorted(reached, target_share, side="left"))
    index = min(index, len(probability) - 1)
    return float(probability[order][index])


def _acre_share(call: pd.Series, area: pd.Series) -> float:
    usable = area.notna() & call.notna()
    if not usable.any() or area[usable].sum() <= 0:
        return np.nan
    return float(np.average(call[usable], weights=area[usable]))


def _regions() -> list[tuple[str, Path]]:
    found = []
    for folder in sorted(PROCESSED.iterdir()):
        if folder.is_dir() and (folder / "s2_timeseries.csv").exists():
            found.append((folder.name, folder))
    return found


def _print_scores(title: str, scores: dict) -> None:
    print(
        f"{title}: {scores['n']} field-years ({scores['silage']} silage, {scores['grain']} grain), "
        f"precision {scores['precision']:.2f}, recall {scores['recall']:.2f}, "
        f"false positive rate {scores['false_positive_rate']:.2f}"
    )


def main() -> None:
    regions = _regions()
    if not any(name == "barc" for name, _ in regions):
        raise SystemExit(f"No BARC Sentinel-2 table under {PROCESSED}")

    tables = []
    for name, folder in regions:
        optical, radar = load_region(folder)
        features = field_features(optical, radar)
        features.insert(0, "region", name)
        areas = region_areas(folder)
        if len(areas):
            features = features.merge(areas, on="field_id", how="left")
        else:
            features["fips"] = ""
            features["area_ha"] = np.nan
        tables.append(features)
        print(f"{name}: {len(features)} field-years with a harvest drop, from {optical['field_id'].nunique()} fields")
    everything = pd.concat(tables, ignore_index=True)
    out_dir = PROCESSED / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    everything.to_csv(out_dir / "features.csv", index=False)

    barc = _labelled(everything.loc[everything["region"] == "barc"])
    if barc["y"].nunique() < 2:
        raise SystemExit("BARC features need both Silage and Grain labels.")

    by_year = _cross_validate(barc, "year")
    by_field = _cross_validate(barc, "field_id")
    held_out = pd.concat([by_year, by_field], ignore_index=True)
    held_out.to_csv(out_dir / "barc_held_out.csv", index=False)
    print()
    _print_scores("Leave one year out", _scores(barc["y"].to_numpy(), by_year["p_silage"].to_numpy()))
    for year, part in by_year.groupby("year"):
        matched = barc.loc[barc["year"] == year, "y"].to_numpy()
        _print_scores(f"  {int(year)}", _scores(matched, part["p_silage"].to_numpy()))
    _print_scores("Leave one field out", _scores(barc["y"].to_numpy(), by_field["p_silage"].to_numpy()))

    model, coefficients = _fit(barc)
    coefficients.to_csv(out_dir / "coefficients.csv", index=False)
    print("\nFull BARC fit, effect of one standard deviation:")
    for row in coefficients.itertuples(index=False):
        print(f"  {row.feature:28} {row.coefficient:+.2f}  {row.direction}")

    summaries = []
    for name, _folder in regions:
        if name == "barc":
            continue
        county = everything.loc[everything["region"] == name].copy()
        if county.empty:
            continue
        county["p_silage"] = model.predict_proba(county[MODEL_FEATURES].to_numpy(dtype="float64"))[:, 1]
        county["pred_independent"] = np.where(county["p_silage"] >= 0.5, "Silage", "Grain")
        fips = str(county["fips"].dropna().iloc[0]) if county["fips"].notna().any() else ""
        reference = _nass(fips) if fips else None
        county["p_calibrated"] = np.nan
        county["pred_calibrated"] = ""
        independent = _acre_share(county["p_silage"], county["area_ha"])
        hard = _acre_share((county["p_silage"] >= 0.5).astype(float), county["area_ha"])
        summary = {
            "region": name,
            "fips": fips,
            "fields_scored": int(len(county)),
            "acres_scored": county["area_ha"].sum() * ACRES_PER_HA,
            "independent_soft_share": independent,
            "independent_share": hard,
            "nass_source": "",
            "nass_share": np.nan,
            "calibrated_share": np.nan,
        }
        if reference is None:
            print(f"\n{name}: no NASS silage total, so only the independent estimate is written")
        else:
            usable = county["area_ha"].notna()
            probability = county.loc[usable, "p_silage"].to_numpy()
            area = county.loc[usable, "area_ha"].to_numpy()
            county.loc[usable, "p_calibrated"] = _prior_shift(probability, area, reference["share"])
            cutoff = _rank_cutoff(probability, area, reference["share"])
            county["pred_calibrated"] = np.where(county["p_silage"] >= cutoff, "Silage", "Grain")
            summary["nass_source"] = reference["source"]
            summary["nass_share"] = reference["share"]
            summary["calibrated_share"] = _acre_share((county["p_silage"] >= cutoff).astype(float), county["area_ha"])
            print(
                f"\n{name} ({fips}), {reference['source']} silage share {reference['share']:.1%}"
            )
            print(f"  independent, acre-weighted probability: {independent:.1%}")
            print(f"  independent, fields at probability >= 0.5: {hard:.1%} of acres")
            print(f"  calibrated map, top fields until the acres match: {summary['calibrated_share']:.1%} of acres")
            print("  The calibrated total was set to NASS, so it is not a check of that county.")
        keep = [
            "field_id", "year", "fips", "area_ha", "p_silage", "pred_independent",
            "p_calibrated", "pred_calibrated", *MODEL_FEATURES,
        ]
        county[keep].to_csv(PROCESSED / name / "silage_predictions.csv", index=False)
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(out_dir / "county_summary.csv", index=False)
    append_log(
        "Silage model",
        [
            ("training", "BARC only, logistic regression on " + ", ".join(MODEL_FEATURES)),
            ("barc validation", "leave one year out, and leave one field out"),
            ("independent estimate", "BARC model applied with no county NASS input"),
            ("calibrated map", "prior shift plus a rank cutoff so the flagged acres match NASS"),
            ("features", out_dir / "features.csv"),
            ("county summary", out_dir / "county_summary.csv"),
        ],
    )


if __name__ == "__main__":
    main()
