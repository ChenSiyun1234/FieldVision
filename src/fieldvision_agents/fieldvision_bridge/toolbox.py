"""FieldVisionToolbox: the team's FieldVision toolbox, set up the way the agent should use it.

What it adds on top of the team's toolbox (same method names, so the supervisor needs no changes):
  - the final model module (config.FINAL_POLICY, classifier config.STANDARD_MODEL) behind the
    agent's "standard" route, with superseded routes marked unavailable;
  - county totals from the final module's own county step (counted share of the clearly seen
    fields) whenever it gives one; otherwise:
  - county totals with probabilities shifted from the 50/50 training balance to the county's prior
    silage share (Quick Stats, years before the target), next to the unadjusted totals, but only when
    the fields cover enough of the county's corn to stand for it (config.MIN_COUNTY_COVERAGE);
  - the critic (rag.make_rag_critic, the team's RAG critic, or a clear "unchecked" warning);
  - NASS report search (rag.make_report_search) for the field Q&A;
  - per-field access (fields, field, planner_frame) for the reporter, Q&A and planner.

load_team_toolbox() imports the team's Toolbox itself; county_check uses it directly.
"""

import sys
from pathlib import Path

from ..settings import FIELDVISION_DEPS, FIELDVISION_FINAL, LANDSAT
from .config import (COUNTY_DATA, FINAL_POLICY, MIN_COUNTY_COVERAGE, MODEL_DECISIONS, QUICKSTATS_CSV,
                     STANDARD_MODEL, SUPERSEDED_POLICIES)
from .prevalence import county_corn_acres, shift_probability, silage_share_prior, simulate_range
from .results import attach_area, field_areas, load_field_results


def load_team_toolbox(root=FIELDVISION_FINAL, deps=FIELDVISION_DEPS, **kwargs):
    """Import the team's Toolbox the way FieldVision/docs/toolbox.md describes.

    Code and models come from `root` (the final FieldVision, installed by
    fieldvision_bridge/20_install_final_model.ipynb); county data and pyarrow from the team copy.
    """
    root = Path(root)
    for p in (root / "src", Path(deps)):             # deps ends up first, as in the team's docs
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise RuntimeError(f"pyarrow is missing. Run once:  pip install --target {deps} pyarrow") from None
    from fieldvision.toolbox import Toolbox
    kwargs.setdefault("data_root", str(COUNTY_DATA))
    kwargs.setdefault("registry", str(root / "registry"))
    kwargs.setdefault("out_root", str(root / "results"))
    return Toolbox(**kwargs)


class FieldVisionToolbox:
    def __init__(self, inner=None, critic=None, standard_model=STANDARD_MODEL, county_data=COUNTY_DATA,
                 quickstats_csv=QUICKSTATS_CSV, prior=None, county_corn=None, min_coverage=MIN_COUNTY_COVERAGE,
                 reports=None, final_policy=FINAL_POLICY, landsat=LANDSAT):
        self.inner = inner if inner is not None else load_team_toolbox(standard_model=standard_model,
                                                                        module_model=standard_model,
                                                                        landsat=landsat)
        self.final_policy = final_policy   # what the agent's "standard" route runs
        self.critic = critic               # critic(fips, year, county_result, policy)
        self.reports = reports             # search(query, fips, year=None, k=3), or None
        self.standard_model = standard_model
        self.county_data = Path(county_data)
        self.quickstats_csv = Path(quickstats_csv)
        self._fixed_prior = prior          # tests or a deliberate override: {"share": ..., "source": ...}
        self._fixed_corn = county_corn     # likewise: {"acres": ..., "source": ...}
        self.min_coverage = min_coverage
        self._priors, self._fields, self._county, self._adjust, self._policy = {}, {}, {}, {}, {}
        self._inner_policy = {}

    # --- passthrough tools -------------------------------------------------

    def __getattr__(self, name):
        """Any other public team tool (e.g. a teammate's recommend_sources) passes straight through."""
        inner = self.__dict__.get("inner")
        if name.startswith("_") or inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def scout(self, fips, year):
        result = self.inner.scout(fips, year)
        has = (self.county_data / str(fips) / str(year) / "timeseries_landsat.parquet").is_file()
        return {**result, "landsat": ("Landsat 8/9 rows added to the Sentinel-2 series" if has and
                                      getattr(self.inner, "landsat", False) else "Sentinel-2 only")}

    def route_options(self, fips, year):
        """The team's routes, with "standard" running the final module and superseded routes marked."""
        result = self.inner.route_options(fips, year)
        options = dict(result.get("options") or {})
        for pol, why in SUPERSEDED_POLICIES.items():
            if pol in options:
                options[pol] = {**options[pol], "note": f"not available: {why}"}
        final = options.get(self.final_policy)
        if final and "standard" in options:
            options["standard"] = {**final, "note": f"final model module ({self.final_policy}): {final.get('note', '')}"}
        table = "\n".join(f"{p}: {o.get('note', '')}" for p, o in options.items())
        return {**result, "options": options, "routing_table": table, "model_decisions": MODEL_DECISIONS,
                "standard_model": self.standard_model, "standard_runs": self.final_policy if final else "standard"}

    def extract_features(self, fips, year):
        return self.inner.extract_features(fips, year)

    def abstain_region(self, fips, year, reason):
        return self.inner.abstain_region(fips, year, reason)

    # --- classify and aggregate --------------------------------------------

    def classify(self, fips, year, policy):
        inner_policy = self.final_policy if policy == "standard" and self.final_policy else policy
        try:
            result = self.inner.classify(fips, year, inner_policy)
        except ValueError:                 # an older FieldVision without the final module
            inner_policy = policy
            result = self.inner.classify(fips, year, policy)
        result = {**result, "policy": policy, "runs": inner_policy}
        df = load_field_results(result["results_path"], year)
        geojson = self.county_data / str(fips) / str(year) / "fields.geojson"
        if geojson.exists():
            df = attach_area(df, field_areas(geojson))
        self._fields[(str(fips), int(year))] = df
        self._adjust.pop((str(fips), int(year)), None)
        self._policy[(str(fips), int(year))] = policy
        self._inner_policy[(str(fips), int(year))] = inner_policy
        return result

    def prior(self, fips, year):
        key = (str(fips), int(year))
        if key not in self._priors:
            self._priors[key] = self._fixed_prior or silage_share_prior(self.quickstats_csv, fips, year)
        return self._priors[key]

    def adjustment(self, fips, year):
        """Whether the county prior applies: the fields must cover enough of the county's harvested corn.

        The prior is a whole-county average. A small set of fields, such as one research farm, has its
        own mix, so shifting it to the county share would be wrong; it stays at the model's output.
        """
        key = (str(fips), int(year))
        if key in self._adjust:
            return self._adjust[key]
        if (self._county.get(key) or {}).get("silage_share_headline") is not None:
            out = {"applied": False, "prior": self.prior(fips, year), "field_acres": None, "county_corn_acres": None,
                   "county_corn_source": None, "county_coverage": None,
                   "reason": "none: the final module's county step gives the total"}
            self._adjust[key] = out
            return out
        df = self._fields.get(key)
        if df is None:
            raise RuntimeError(f"classify {fips} {year} first")
        prior = self.prior(fips, year)
        field_acres = float(df["area_acres"].sum()) if "area_acres" in df and df["area_acres"].notna().any() else None
        corn = self._fixed_corn or county_corn_acres(self.quickstats_csv, fips, year)
        out = {"applied": False, "prior": prior, "field_acres": field_acres,
               "county_corn_acres": corn and corn["acres"], "county_corn_source": corn and corn["source"],
               "county_coverage": None}
        if field_acres is None:
            out["reason"] = "none: per-field areas unavailable, totals are at the 50/50 training balance"
        elif not corn or not corn["acres"]:
            out["reason"] = ("none: the county's corn acres are not published, so it is unknown whether these "
                             "fields stand for the county; totals are at the 50/50 training balance")
        else:
            coverage = field_acres / corn["acres"]
            out["county_coverage"] = round(coverage, 4)
            if coverage >= self.min_coverage:
                out["applied"] = True
                out["reason"] = (f"field probabilities shifted from the 50/50 training balance to the prior silage "
                                 f"share: these fields are {coverage:.0%} of the county's corn")
            else:
                out["reason"] = (f"none: these fields are {coverage:.1%} of the county's corn, too few to stand for "
                                 f"the county (needs {self.min_coverage:.0%}), so the county prior would misstate "
                                 f"them; totals are at the 50/50 training balance")
        self._adjust[key] = out
        return out

    def aggregate(self, fips, year):
        """County acres, shifted to the county's prior silage share when the fields stand for the county."""
        raw = self.inner.aggregate(fips, year)
        df = self._fields.get((str(fips), int(year)))
        if df is None:
            return {**raw, "adjustment": "none: no classify results loaded, totals are at the 50/50 training balance"}
        if raw.get("silage_share_headline") is not None:
            return self._final_headline(fips, year, raw, df)
        adj = self.adjustment(fips, year)
        prior = adj["prior"]
        if raw.get("headline_method"):         # the final module ran but gave no county figure
            adj = {**adj, "reason": f"{adj['reason']} (final module: {raw['headline_method']})"}
        context = {"silage_acres_unadjusted": raw.get("silage_acres"),
                   "prior_silage_share": round(prior["share"], 4), "prior_source": prior["source"],
                   "prior_applied": adj["applied"], "county_coverage": adj["county_coverage"],
                   "county_corn_acres": adj["county_corn_acres"], "adjustment": adj["reason"]}
        if not adj["applied"]:
            result = {**raw, **context}
            self._county[(str(fips), int(year))] = result
            return result
        decided = df[~df["abstain"] & df["area_acres"].notna()]
        p_adj = shift_probability(decided["p_silage"], prior["share"])
        area = decided["area_acres"].to_numpy(float)
        silage = float((area * p_adj).sum())
        low, high = simulate_range(area, p_adj)
        result = {**raw,
                  "silage_acres": round(silage, 1), "silage_acres_low": round(low, 1),
                  "silage_acres_high": round(high, 1), "grain_acres": round(float(area.sum()) - silage, 1),
                  **context}
        self._county[(str(fips), int(year))] = result
        return result

    def _final_headline(self, fips, year, raw, df):
        """County acres from the final module's headline share, applied to every field's area.

        The module counts silage on the clearly seen fields (harvest bracket of 12 days or less);
        summing probabilities reads far too high where silage is rare (FieldVision HANDOFF.md).
        """
        share = float(raw["silage_share_headline"])
        total = float(df["area_acres"].sum()) if "area_acres" in df else float("nan")
        bounds = [raw.get("clear_tier_lower_bound"), share]
        upper = [share, raw.get("clear_wide_share_called_silage"), raw.get("window_weighted_upper")]
        prior = self.prior(fips, year)
        result = {**raw,
                  "silage_acres": round(share * total, 1),
                  "silage_acres_low": round(min(b for b in bounds if b is not None) * total, 1),
                  "silage_acres_high": round(max(u for u in upper if u is not None) * total, 1),
                  "grain_acres": round((1 - share) * total, 1),
                  "silage_share_pct": round(100 * share, 1),
                  "silage_acres_unadjusted": raw.get("silage_acres"),
                  "prior_silage_share": round(prior["share"], 4), "prior_source": prior["source"],
                  "prior_applied": False, "county_coverage": None, "county_corn_acres": None,
                  "adjustment": f"none: {raw.get('headline_method')} (the final module's county step)"}
        self._county[(str(fips), int(year))] = result
        self._adjust.pop((str(fips), int(year)), None)
        return result

    def run_critic(self, fips, year):
        if self.critic is None:
            return {"overall": "warn", "suggested_action": "accept",
                    "flags": [{"check": "critic_connected", "status": "warn",
                               "why": "The critic is not connected yet, so this result is unchecked.", "evidence": []}],
                    "summary": "Unchecked: critic not connected."}
        key = (str(fips), int(year))
        county = self._county.get(key) or self.aggregate(fips, year)
        return self.critic(fips, year, county, policy=self._policy.get(key, "standard"))

    # --- per-field access for the reporter, Q&A and planner -----------------

    def fields(self, fips, year):
        """Latest classify's fields with p_silage (the model's own, 50/50), p_adjusted (what county totals use:
        shifted to the county prior when it applies, else the same as p_silage), area and evidence."""
        df = self._fields.get((str(fips), int(year)))
        if df is None:
            raise RuntimeError(f"classify {fips} {year} first")
        adj = self.adjustment(fips, year)
        out = df.copy()
        out["p_adjusted"] = (shift_probability(out["p_silage"], adj["prior"]["share"]) if adj["applied"]
                             else out["p_silage"].astype(float))
        out.attrs["prior_applied"] = adj["applied"]
        return out

    def field(self, fips, year, field_id):
        df = self.fields(fips, year)
        match = df[(df["field_id"].astype(str) == str(field_id)) |
                   (df.get("source_field_id", df["field_id"]).astype(str) == str(field_id))]
        if match.empty:
            return None
        row = match.iloc[0]
        out = {"field_id": row["field_id"], "model": row.get("model"), "abstain": bool(row["abstain"]),
               "call": row.get("label"), "p_silage_model": round(float(row["p_silage"]), 3),
               "area_acres": None if row.get("area_acres") is None or row["area_acres"] != row["area_acres"]
               else round(float(row["area_acres"]), 1),
               "evidence": row.get("evidence") or {}}
        if df.attrs.get("prior_applied"):
            out["p_silage_county_prior"] = round(float(row["p_adjusted"]), 3)   # for county totals only
        return out

    def planner_frame(self, fips, year):
        """Columns the field-visit planner expects, with the probabilities county totals use.

        attrs["abstain_prior"] is the county prior when it applies, so abstained fields count at the same
        share as the totals; otherwise None and the planner uses these fields' own average.
        """
        df = self.fields(fips, year)
        adj = self.adjustment(fips, year)
        out = df[["field_id", "area_acres", "abstain"]].copy()
        out["p_silage"] = df["p_adjusted"]
        out = out.dropna(subset=["area_acres"])
        out.attrs["abstain_prior"] = adj["prior"]["share"] if adj["applied"] else None
        return out
