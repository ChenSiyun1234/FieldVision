"""A real toolbox for the team's supervisor agent, with the same method names and return keys
as their `StubToolbox` (agents/react/tools_stub.py), without the `stub` flag. Every number
is computed here from field polygons, Sentinel-2 series and frozen registry models.

Data layout (one folder per county and season):

    <data_root>/<fips>/<year>/fields.geojson      field polygons (labels, if any, are ignored)
    <data_root>/<fips>/<year>/timeseries.parquet  the extractor's long table

`make_pseudo_county` builds such a folder from the Beltsville files (FIPS 24033, Prince
George's County, MD) with the label columns removed, so the interface can be shown on real
numbers before a real county is extracted.

What we do not have, said plainly in `notes`: no radar model, so `radar_first` uses the same
assignment as `standard`, and `fields_with_cloud_gap_radar_ok` is always 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features.build import build_features
from .features.loading import load_fields, load_timeseries
from .model.classify import results as classify_rows
from .model.registry import load as load_model

HA_TO_ACRE = 2.47105
REPO_ROOT = Path(__file__).resolve().parents[2]      # src/fieldvision/toolbox.py -> repo root


def _from_repo(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
POLICIES = ("standard", "radar_first", "regional_threshold")
STATE_BY_FIPS = {"01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
                 "12": "FL", "13": "GA", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS",
                 "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
                 "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ",
                 "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR",
                 "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
                 "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
LABEL_COLUMNS = ("label", "source_label", "y", "Corn_2021", "Corn_2022", "Corn_2023")


def make_pseudo_county(fields_path, timeseries_path, data_root, fips="24033", year=None):
    """Copy one season of Beltsville fields (labels stripped) and its series into the layout."""
    gj = json.loads(Path(fields_path).read_text(encoding="utf-8"))
    for f in gj["features"]:
        for k in LABEL_COLUMNS:
            f["properties"].pop(k, None)
        f["properties"]["county_fips"] = fips
    year = year or int(gj["features"][0]["properties"]["year"])
    d = Path(data_root) / str(fips) / str(year)
    d.mkdir(parents=True, exist_ok=True)
    (d / "fields.geojson").write_text(json.dumps(gj), encoding="utf-8")
    pd.read_parquet(timeseries_path).to_parquet(d / "timeseries.parquet", index=False)
    return d


class Toolbox:
    """Same interface as the team's StubToolbox; numbers from fieldvision code only."""

    def __init__(self, data_root="data/counties", registry="registry", out_root="results",
                 standard_model="composite_v1",
                 rule_model="eu_timing_rule_v1", n_resample=2000, seed=0):
        """Relative paths are taken from the repo root, so Toolbox() works from anywhere."""
        self.data_root = _from_repo(data_root)
        self.registry = _from_repo(registry)
        self.out_root = _from_repo(out_root)
        self.standard_model = standard_model
        self.rule_model = rule_model
        self.n_resample = n_resample
        self.seed = seed
        self._cache = {}
        self._last = {}

    # ------------------------------------------------------------------ helpers
    def _folder(self, fips, year):
        d = self.data_root / str(fips) / str(year)
        if not (d / "fields.geojson").exists() or not (d / "timeseries.parquet").exists():
            raise FileNotFoundError(f"no fields/time series for county {fips} season {year} under "
                                    f"{self.data_root} (expected {d})")
        return d

    def _data(self, fips, year):
        key = (str(fips), int(year))
        if key not in self._cache:
            d = self._folder(fips, year)
            fields = load_fields(d / "fields.geojson")
            fields = fields.drop(columns=[c for c in ("label", "y") if c in fields], errors="ignore")
            fields["year"] = int(year)
            ts, _ = load_timeseries(d / "timeseries.parquet", field_ids=fields["field_id"])
            _, _, cfg = load_model(self.registry, self.standard_model)
            X, info = build_features(ts, fields, cfg)
            self._cache[key] = (fields, ts, X, cfg)
        return self._cache[key]

    def _state(self, fips):
        return STATE_BY_FIPS.get(str(fips)[:2], str(fips)[:2])

    def _regional_rule(self, X):
        """Timing rule with its threshold moved by the region's median drop date minus the
        training region's median drop date (a season-relative variant)."""
        model, mj, cfg = load_model(self.registry, self.rule_model)
        base = getattr(model, "base_", model)
        ref = self.registry / self.rule_model / "training_reference.json"
        if mj.get("training_median_timing") is not None:
            train_median = float(mj["training_median_timing"])
        elif ref.exists():
            train_median = float(json.loads(ref.read_text(encoding="utf-8"))
                                 ["training_median_harv_mid_doy"])
        else:
            raise FileNotFoundError(f"{self.rule_model} has no training median drop day "
                                    "(training_median_timing in model.json or training_reference.json)")
        here = float(np.nanmedian(X[base.timing].to_numpy(float)))
        shift = here - train_median if np.isfinite(here) else 0.0
        Xs = X.copy()
        Xs[base.timing] = Xs[base.timing] - shift       # same as moving the threshold by +shift
        return model, mj, Xs, shift, train_median, here

    # ------------------------------------------------------------------ tools
    def scout(self, fips, year):
        fields, ts, X, cfg = self._data(fips, year)
        n_use = X["q_n_usable_season"].to_numpy(float)
        little = (n_use < cfg.low_evidence_min_usable) | (X["q_n_rows"].to_numpy(float) == 0)
        clear = ~X["low_evidence"].to_numpy(bool)
        gaps = X["q_max_gap_days"].to_numpy(float)
        return {
            "fips": str(fips), "year": int(year), "n_fields": int(len(fields)),
            "fields_with_clear_series": int(clear.sum()),
            "fields_with_cloud_gap_radar_ok": 0,
            "fields_with_little_data": int(little.sum()),
            "median_clear_obs_jul_nov": float(np.nanmedian(n_use)) if len(n_use) else float("nan"),
            "largest_optical_gap_days_p90": float(np.nanpercentile(gaps, 90))
            if np.isfinite(gaps).any() else float("nan"),
            "notes": "no radar data or radar model in this toolbox, so no field is counted as "
                     "rescued by radar; 'clear series' = not low evidence under the optical rule "
                     f"(>= {cfg.low_evidence_min_usable} usable dates Jul-Nov, harvest bracket <= "
                     f"{cfg.low_evidence_max_interval_days:g} days, drop confirmed)",
        }

    def _policy_predictions(self, fips, year, policy):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
        fields, ts, X, cfg = self._data(fips, year)
        if policy in ("standard", "radar_first"):
            model, mj, _ = load_model(self.registry, self.standard_model)
            rows, _ = classify_rows(fields, ts, model, self.standard_model, cfg, mj["threshold"])
            name = self.standard_model
            note = ("identical to 'standard': no radar model exists, so no field is routed to one"
                    if policy == "radar_first" else f"every field to {name}")
        else:
            model, mj, Xs, shift, t_med, here = self._regional_rule(X)
            p = model.predict_proba(Xs)
            rows, _ = classify_rows(fields, ts, _Fixed(p, model), self.rule_model + "+regional_shift",
                                    cfg, mj["threshold"])
            name = self.rule_model + "+regional_shift"
            note = (f"{self.rule_model} with its drop-date threshold moved by {shift:+.1f} days: this "
                    f"county's median drop day {here:.1f} minus the training median {t_med:.1f}")
        return rows, name, note

    def route_options(self, fips, year):
        options, lines = {}, []
        for pol in POLICIES:
            rows, name, note = self._policy_predictions(fips, year, pol)
            n_abs = int(sum(r["abstain"] for r in rows))
            options[pol] = {"models_used": {name: len(rows)}, "n_abstain": n_abs, "note": note}
            lines.append(f"{pol}: {note}; abstain on {n_abs} of {len(rows)} fields")
        return {"fips": str(fips), "year": int(year), "routing_table": "\n".join(lines),
                "options": options,
                "notes": "radar_first cannot differ from standard here (no radar model)"}

    def extract_features(self, fips, year):
        fields, ts, X, cfg = self._data(fips, year)
        out = self.out_root / "features"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{fips}_{year}.parquet"
        X.to_parquet(path, index=False)
        # nothing is imputed into the table (missing stays NaN); the only values computed
        # across a cloud gap are the greenness integrals, which interpolate between dates
        gi = X[["gi_ndvi_aug_oct", "gi_rel_after_peak"]].notna().any(axis=1)
        gap = X["q_max_gap_days"].to_numpy(float) > 15
        return {"fips": str(fips), "year": int(year), "rows": int(len(X)), "path": str(path),
                "quality": {"rows_with_gap_filled": int((gi.to_numpy() & gap).sum())}}

    def classify(self, fips, year, policy="standard"):
        rows, name, note = self._policy_predictions(fips, year, policy)
        out = self.out_root
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self._state(fips)}_{year}_{fips}_{policy}.parquet"
        df = pd.DataFrame(rows)
        table = df.drop(columns="evidence").copy()
        table["evidence"] = [json.dumps(e) for e in df["evidence"]]
        table.to_parquet(path, index=False)
        with open(path.with_suffix(".jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        n_abs = int(df["abstain"].sum())
        self._last[(str(fips), int(year))] = (policy, df)
        return {"fips": str(fips), "year": int(year), "policy": policy, "n_fields": int(len(df)),
                "models_used": {name: int(len(df))}, "n_abstain": n_abs, "results_path": str(path)}

    def aggregate(self, fips, year):
        key = (str(fips), int(year))
        if key not in self._last:
            self.classify(fips, year, "standard")
        policy, df = self._last[key]
        fields, _, _, _ = self._data(fips, year)
        area = fields.set_index("field_id")["area_ha"].astype(float) * HA_TO_ACRE
        a = df["source_field_id"].map(area).to_numpy(float)
        p = df["p_silage"].to_numpy(float)
        ab = df["abstain"].to_numpy(bool)
        ok = ~ab & np.isfinite(p) & np.isfinite(a)
        silage = float(np.sum(a[ok] * p[ok]))
        grain = float(np.sum(a[ok] * (1 - p[ok])))
        rng = np.random.default_rng(self.seed)
        draws = rng.random((self.n_resample, int(ok.sum()))) < p[ok][None, :]
        sims = draws.astype(float) @ a[ok]
        lo, hi = (np.percentile(sims, [5, 95]) if ok.any() else (0.0, 0.0))
        return {"fips": str(fips), "year": int(year), "silage_acres": round(silage, 1),
                "silage_acres_low": round(float(lo), 1), "silage_acres_high": round(float(hi), 1),
                "grain_acres": round(grain, 1),
                "abstained_acres": round(float(np.nansum(a[ab])), 1),
                "n_fields": int(len(df)), "models_used": {df["model"].iloc[0]: int(len(df))}}

    def abstain_region(self, fips, year, reason):
        fields, _, _, _ = self._data(fips, year)
        acres = float(fields["area_ha"].astype(float).sum() * HA_TO_ACRE)
        rec = {"fips": str(fips), "year": int(year), "abstained": True, "reason": str(reason),
               "n_fields": int(len(fields)), "abstained_acres": round(acres, 1)}
        out = self.out_root
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "abstained_regions.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec


class _Fixed:
    """Wraps precomputed probabilities so the shared classify code can format them."""

    def __init__(self, p, model):
        self.p = np.asarray(p, float)
        self.model = model
        self.feature_names = model.feature_names

    def predict_proba(self, X):
        return self.p

    def explain(self, X):
        return None
