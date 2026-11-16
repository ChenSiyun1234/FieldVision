"""A small stand-in for FieldVision's Toolbox, its result files and a Quick Stats CSV."""

import csv
import json
from pathlib import Path

from ..config import ACRES_PER_HECTARE

FIPS, YEAR = "24033", 2022

# field_id, source_field_id, area_ha, p_silage (50/50 balance), abstain, evidence
FIELDS = [
    ("BARC-1", "barc-1-2022", 4.0, 0.9, False, {"drop_date": "2022-08-29", "ndvi_before": 0.81, "ndvi_after": 0.24}),
    ("BARC-2", "barc-2-2022", 8.0, 0.2, False, {"drop_date": "2022-10-12", "ndvi_before": 0.41, "ndvi_after": 0.30}),
    ("BARC-3", "barc-3-2022", 2.0, 0.5, False, {"drop_date": "2022-09-20"}),
    ("BARC-4", "barc-4-2022", 6.0, 0.7, True, {"abstain_reason": "harvest bracket over 20 days"}),
]


def make_county(root, fips=FIPS, year=YEAR):
    folder = Path(root) / "counties" / fips / str(year)
    folder.mkdir(parents=True)
    features = [{"type": "Feature", "geometry": None,
                 "properties": {"field_id": src, "year": year, "area_ha": ha}} for _, src, ha, *_ in FIELDS]
    (folder / "fields.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return Path(root) / "counties"


def make_quickstats(path, county_rows=(), state_rows=()):
    """county_rows / state_rows: (year, grain, silage); a value may be '(D)'."""
    cols = ["year", "aggregation_level", "state_ansi", "county_ansi", "crop_use", "measurement",
            "value", "source", "domain", "reference_period"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for level, rows, county, source in (("COUNTY", county_rows, "033", "CENSUS"), ("STATE", state_rows, "", "SURVEY")):
            for year, grain, silage in rows:
                for use, v in (("grain", grain), ("silage", silage)):
                    w.writerow([year, level, "24", county, use, "harvested", v, source, "TOTAL", "YEAR"])
    return path


class FakeTeamToolbox:
    """Same method names and return shapes as fieldvision.toolbox.Toolbox."""

    def __init__(self, out_dir, headline=None):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.headline = headline           # module_v2 county headline share, or None (too few clear fields)

    def scout(self, fips, year):
        return {"fips": fips, "year": year, "n_fields": len(FIELDS), "fields_with_clear_series": 3,
                "fields_with_cloud_gap_radar_ok": 0, "fields_with_little_data": 1}

    def route_options(self, fips, year):
        std = {"models_used": {"eu_logistic_v1": 4}, "n_abstain": 1, "note": "every field to eu_logistic_v1"}
        return {"fips": fips, "year": year, "options": {
            "standard": std, "radar_first": {**std, "note": "identical to standard: no radar model"},
            "regional_threshold": {"models_used": {"eu_timing_rule_v1+regional_shift": 4}, "n_abstain": 1},
            "in_site": {"note": "not available: eu_barc_logistic_v1 is not in the registry"},
            "module_v1": {"models_used": {"composite_v1": 4}, "n_abstain": 1, "note": "module_v1: composite_v1"},
            "module_v2": {"models_used": {"eu2_logistic_v1": 4}, "n_abstain": 1,
                          "note": "module_v2: evidence tiers, eu2_logistic_v1, county total on the clear tier"}}}

    def extract_features(self, fips, year):
        return {"fips": fips, "year": year, "rows": len(FIELDS)}

    def classify(self, fips, year, policy="standard"):
        if policy not in ("standard", "radar_first", "regional_threshold", "in_site", "module_v1", "module_v2"):
            raise ValueError(f"unknown policy {policy!r}")
        self.calls.append(policy)
        path = self.out / f"MD_{year}_{fips}_{policy}.parquet"
        rows = [{"field_id": f, "year": year, "source_field_id": s, "model": "eu_logistic_v1", "p_silage": p,
                 "label": "silage" if p >= 0.5 else "grain", "abstain": a, "evidence": ev}
                for f, s, _, p, a, ev in FIELDS]
        path.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return {"fips": fips, "year": year, "policy": policy, "n_fields": len(rows),
                "models_used": {"eu_logistic_v1": len(rows)}, "n_abstain": 1, "results_path": str(path)}

    def aggregate(self, fips, year):
        decided = [(ha * ACRES_PER_HECTARE, p) for _, _, ha, p, a, _ in FIELDS if not a]
        silage = sum(a * p for a, p in decided)
        return {"fips": fips, "year": year, "silage_acres": round(silage, 1), "silage_acres_low": 0.0,
                "silage_acres_high": round(sum(a for a, _ in decided), 1),
                "grain_acres": round(sum(a for a, _ in decided) - silage, 1),
                "abstained_acres": round(6.0 * ACRES_PER_HECTARE, 1), "n_fields": len(FIELDS),
                **({"silage_share_headline": self.headline, "headline_method": "module_v2: counted share of the clear tier",
                    "clear_tier_lower_bound": self.headline / 2, "clear_wide_share_called_silage": self.headline * 1.5}
                   if self.headline is not None and self.calls and self.calls[-1] == "module_v2" else {})}

    def abstain_region(self, fips, year, reason):
        return {"fips": fips, "year": year, "abstained": True, "reason": reason}
