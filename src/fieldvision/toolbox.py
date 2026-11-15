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


PARQUET_HINT = ("no Parquet engine (pyarrow or fastparquet) is importable; install one with "
                "`pip install --target /workspace/FieldVision/.deps pyarrow` and put that folder "
                "first on sys.path")


def _require_parquet_engine():
    import importlib.util
    if not any(importlib.util.find_spec(m) for m in ("pyarrow", "fastparquet")):
        raise ImportError(PARQUET_HINT)
POLICIES = ("standard", "radar_first", "regional_threshold", "in_site", "module_v1", "module_v2")
MODULE_V1_MODEL = "composite_v1"
# module_v2's field classifier, chosen by the rule logged at 17:02 (see docs/module.md)
MODULE_V2_MODEL = "eu_logistic_v1"
CLEAR_BRACKET_DAYS = 12.0
# module_v2's county step on the clear tier: "local_threshold", "reference", "europe" or "counted"
MODULE_V2_COUNTY_STEP = "local_threshold"     # fixed by the rule logged at 17:30, see docs/module.md


def evidence_tier(row):
    """'clear' (bracket <= 12 days), 'wide' (up to the abstain limit) or 'abstain'."""
    if row["abstain"]:
        return "abstain"
    d = row["evidence"].get("days_to_drop")
    return "clear" if d is not None and d <= CLEAR_BRACKET_DAYS else "wide"
ANOMALY_MODEL = "anomaly_v1"            # second opinion (layer 3); used when present in the registry
IN_SITE_MODEL = "eu_barc_logistic_v1"   # Austria + Beltsville, fitted on these fields (fieldvision.beltsville)
IN_SITE_FIPS = {"24033"}
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
                 standard_model=MODULE_V1_MODEL, module_model=MODULE_V2_MODEL,
                 rule_model="eu_timing_rule_v1", n_resample=2000, seed=0):
        """Relative paths are taken from the repo root, so Toolbox() works from anywhere."""
        _require_parquet_engine()
        self.data_root = _from_repo(data_root)
        self.registry = _from_repo(registry)
        self.out_root = _from_repo(out_root)
        self.standard_model = standard_model
        self.module_model = module_model
        self.rule_model = rule_model
        self.n_resample = n_resample
        self.seed = seed
        self._cache = {}
        self._last = {}
        # reference counties for layer 4: {target fips: {"fips", "year", "census_share"}};
        # read from <data_root>/references.json, overridable with set_reference()
        self.references = {}
        ref_file = self.data_root / "references.json"
        if ref_file.exists():
            self.references.update(json.loads(ref_file.read_text(encoding="utf-8")))

    def set_reference(self, fips, ref_fips, ref_year, census_share):
        """Use `ref_fips` (nearly all grain, Census silage share `census_share`) for `fips`."""
        self.references[str(fips)] = {"fips": str(ref_fips), "year": int(ref_year),
                                      "census_share": float(census_share)}

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
            cfg_model = (self.standard_model if (self.registry / self.standard_model).exists()
                         else MODULE_V1_MODEL)
            _, _, cfg = load_model(self.registry, cfg_model)
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
            "notes": self._partial_note(fips, year) +
                     "no radar data or radar model in this toolbox, so no field is counted as "
                     "rescued by radar; 'clear series' = not low evidence under the optical rule "
                     f"(>= {cfg.low_evidence_min_usable} usable dates Jul-Nov, harvest bracket <= "
                     f"{cfg.low_evidence_max_interval_days:g} days, drop confirmed)",
        }

    def _partial_note(self, fips, year):
        """'PARTIAL county: ...' when the county folder carries a PARTIAL.json, else ''."""
        p = self.data_root / str(fips) / str(year) / "PARTIAL.json"
        if not p.exists():
            return ""
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            return (f"PARTIAL county: {info.get('fields_included')} of {info.get('fields_total')} "
                    f"fields, {info.get('acres_included')} of {info.get('acres_total')} acres; "
                    "totals from aggregate cover only these fields. ")
        except (ValueError, OSError):
            return "PARTIAL county (PARTIAL.json unreadable); totals cover only the fields present. "

    def _policy_predictions(self, fips, year, policy):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
        if policy == "in_site" and str(fips) not in IN_SITE_FIPS:
            raise ValueError(f"policy 'in_site' exists only for FIPS {sorted(IN_SITE_FIPS)}: "
                             f"{IN_SITE_MODEL} was fitted on those fields and is untested elsewhere")
        fields, ts, X, cfg = self._data(fips, year)
        if policy == "in_site":
            model, mj, _ = load_model(self.registry, IN_SITE_MODEL)
            rows, _ = classify_rows(fields, ts, model, IN_SITE_MODEL, cfg, mj["threshold"])
            name = IN_SITE_MODEL
            note = (f"every field to {IN_SITE_MODEL}, fitted on these same Beltsville fields: these "
                    "calls are in-sample; its out-of-fold accuracy is in its model card")
        elif policy in ("standard", "radar_first"):
            # unchanged for the team's wrapper: every field to composite_v1
            model, mj, _ = load_model(self.registry, self.standard_model)
            rows, _ = classify_rows(fields, ts, model, self.standard_model, cfg, mj["threshold"])
            name = self.standard_model
            note = ("identical to 'standard': no radar model exists, so no field is routed to one"
                    if policy == "radar_first" else f"every field to {name}")
        elif policy == "module_v2":
            name = self.module_model
            model, mj, _ = load_model(self.registry, name)
            rows, _ = classify_rows(fields, ts, model, name, cfg, mj["threshold"])
            for r in rows:
                r["evidence"]["evidence_tier"] = evidence_tier(r)
            note = (f"module_v2: evidence tiers (clear <= {CLEAR_BRACKET_DAYS:g} days, wide up to "
                    f"{cfg.low_evidence_max_interval_days:g}, else abstain), {name} as the field "
                    "classifier (no calendar-date rule in the vote), county total on the clear tier")
        elif policy == "module_v1":
            model, mj, _ = load_model(self.registry, MODULE_V1_MODEL)
            rows, _ = classify_rows(fields, ts, model, MODULE_V1_MODEL, cfg, mj["threshold"])
            name = MODULE_V1_MODEL
            if policy == "module_v1":
                tiers, second = self._tiers(X, rows)
                for r, t in zip(rows, tiers):
                    r["evidence"]["tier"] = t
                    r["evidence"]["second_opinion"] = second
                note = (f"module_v1: evidence check, {name} as the field classifier, "
                        f"second opinion: {second}; county total calibrated on a reference county "
                        "when one is configured")
        else:
            model, mj, Xs, shift, t_med, here = self._regional_rule(X)
            p = model.predict_proba(Xs)
            rows, _ = classify_rows(fields, ts, _Fixed(p, model), self.rule_model + "+regional_shift",
                                    cfg, mj["threshold"])
            name = self.rule_model + "+regional_shift"
            note = (f"{self.rule_model} with its drop-date threshold moved by {shift:+.1f} days: this "
                    f"county's median drop day {here:.1f} minus the training median {t_med:.1f}")
        return rows, name, note

    def _tiers(self, X, rows):
        """Layer 3: confidence tier per field from the anomaly scorer, when it is registered.

        The scorer has learned what normal grain looks like; a field is 'not grain-like' when
        its score reaches the scorer's threshold. Tier: 'agree' when that view and the
        classifier's call point the same way, 'disagree' when not, 'abstained' when the
        evidence check failed. Without a scorer: 'classifier_only'.
        """
        folder = self.registry / ANOMALY_MODEL
        abstained = [bool(r["abstain"]) for r in rows]
        if not (folder / "model.json").exists():
            return (["abstained" if ab else "classifier_only" for ab in abstained],
                    f"not available ({ANOMALY_MODEL} is not in the registry yet)")
        import joblib
        mj = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        scorer = joblib.load(folder / "model.joblib")
        score = np.asarray(scorer.score(X), float)
        odd = score >= float(mj["threshold"])
        tiers = []
        for r, ab, o in zip(rows, abstained, odd):
            if ab:
                tiers.append("abstained")
            else:
                tiers.append("agree" if (r["label"] == "silage") == bool(o) else "disagree")
        return tiers, f"{ANOMALY_MODEL} (score >= {float(mj['threshold']):.3f} means not grain-like)"

    def route_options(self, fips, year):
        options, lines = {}, []
        for pol in POLICIES:
            missing = not (self.registry / IN_SITE_MODEL / "model.json").exists()
            if pol == "in_site" and (str(fips) not in IN_SITE_FIPS or missing):
                note = (f"not available: {IN_SITE_MODEL} is not in the registry" if missing else
                        f"not available: the in-site model exists only for FIPS {sorted(IN_SITE_FIPS)}")
                options[pol] = {"models_used": {}, "n_abstain": 0, "note": note}
                lines.append(f"{pol}: {note}")
                continue
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
        out = {"fips": str(fips), "year": int(year), "silage_acres": round(silage, 1),
               "silage_acres_low": round(float(lo), 1), "silage_acres_high": round(float(hi), 1),
               "grain_acres": round(grain, 1),
               "abstained_acres": round(float(np.nansum(a[ab])), 1),
               "n_fields": int(len(df)), "models_used": {df["model"].iloc[0]: int(len(df))}}
        called = (df["label"].to_numpy() == "silage")
        out.update(self._adjusted(df["model"].iloc[0], a, called, ok, silage))
        # headline rule, fixed before any full-county total (see docs/toolbox.md)
        out["silage_share_headline"] = out["silage_share_counted"]
        out["headline_method"] = "counted share of the non-abstained area"
        ps = self._prior_shift(p, a, ok)
        out.update(ps)
        cal = self._reference_calibrated(fips, year, df["model"].iloc[0], a, called, ok)
        out.update(cal)
        if out["module_headline_share"] is None and out.get("silage_share_adjusted") is not None:
            # module_v1 rule: without a reference county, the Europe-rate adjusted share
            out["module_headline_share"] = out["silage_share_adjusted"]
            out["module_headline_method"] = "Europe-rate adjusted count (no reference county)"
        if out.get("notes") and cal.get("calibration_note"):
            out["notes"] += " " + cal["calibration_note"]
        if out.get("notes"):
            out["notes"] += (f" Headline: counted share {out['silage_share_counted']:.1%}. Prior-shift "
                             f"(EM) re-estimate from the county's own probabilities: "
                             + (f"{ps['silage_share_prior_shift']:.1%}" if ps["prior_shift_converged"]
                                else "did not converge") + ".")
        if policy == "module_v2":
            out.update(self._clear_tier(fips, year, df, a))
        return out

    def _tier_frame(self, fips, year, policy="module_v2"):
        """Per-field area, score, call and evidence tier for one county under module_v2."""
        rows, _, _ = self._policy_predictions(fips, year, policy)
        f, _, _, _ = self._data(fips, year)
        area = f.set_index("field_id")["area_ha"].astype(float) * HA_TO_ACRE
        d = pd.DataFrame(rows)
        return pd.DataFrame({"a": d["source_field_id"].map(area).to_numpy(float),
                             "p": d["p_silage"].to_numpy(float),
                             "call": (d["label"] == "silage").to_numpy(),
                             "tier": [e["evidence_tier"] for e in d["evidence"]]})

    def _clear_tier(self, fips, year, df, a):
        """module_v2 county step: estimates on the clear tier, with the wide tier beside it."""
        tier = np.array([e.get("evidence_tier", "abstain") for e in df["evidence"]])
        p = df["p_silage"].to_numpy(float)
        call = (df["label"] == "silage").to_numpy()
        fin = np.isfinite(a)
        clear, wide = (tier == "clear") & fin, (tier == "wide") & fin
        total = float(np.sum(a[fin]))
        ac = float(np.sum(a[clear]))
        out = {"clear_tier_share_of_area": round(ac / total, 4) if total else None,
               "clear_tier_acres": round(ac, 1),
               "wide_tier_acres": round(float(np.sum(a[wide])), 1),
               "wide_tier_share_called_silage": round(float(np.sum(a[wide] * call[wide]) /
                                                            np.sum(a[wide])), 4) if wide.any() else None}
        counted = float(np.sum(a[clear] * call[clear]) / ac) if ac else float("nan")
        out["clear_silage_share_counted"] = round(counted, 4) if ac else None
        base = df["model"].iloc[0].split("+")[0]
        rp = self.registry / base / "rates.json"
        doc = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
        rates = None
        if doc:
            rates = doc["sets"].get(doc.get("primary_clear", ""), doc["sets"][doc["primary"]])

        def adj(share, tpr, fpr):
            den = tpr - fpr
            return float(np.clip((share - fpr) / den, 0, 1)) if den > 0.05 else None
        out["clear_silage_share_europe_adjusted"] = (adj(counted, rates["tpr"], rates["fpr"])
                                                     if rates and ac else None)
        ref = self.references.get(str(fips))
        out["clear_silage_share_reference_calibrated"] = None
        out["clear_silage_share_local_threshold"] = None
        out["clear_reference"] = None
        out["clear_silage_share_local_flagged"] = None
        out["local_per_tier_share"] = None
        out["local_per_tier_detail"] = None
        local = (doc or {}).get("local") or {}
        # the reference must be from the same season: the local error rate moves with the year
        if ref and str(ref["fips"]) != str(fips) and int(ref["year"]) == int(year) and rates and ac:
            r_all = self._tier_frame(ref["fips"], ref["year"])
            r_all = r_all[np.isfinite(r_all["a"])]
            r = r_all[r_all["tier"] == "clear"]
            if len(r) >= 50:
                s_ref = float(ref["census_share"])
                ref_counted = float(np.sum(r["a"] * r["call"]) / np.sum(r["a"]))
                fpr_ref = float(np.clip((ref_counted - rates["tpr"] * s_ref) / (1 - s_ref), 0, 0.95))
                out["clear_silage_share_reference_calibrated"] = adj(counted, rates["tpr"], fpr_ref)
                t = float(np.quantile(r["p"], 0.90))
                ref_flag = float(np.sum(r["a"] * (r["p"] >= t)) / np.sum(r["a"]))
                flag = float(np.sum(a[clear] * (p[clear] >= t)) / ac)
                # matched recall: per source domain when rates.json has it, else the pooled set
                mt = (local.get("clear") or {}).get("matched_tpr", rates.get("matched_tpr_alpha_0.10"))
                out["clear_silage_share_local_flagged"] = round(flag, 4)
                if mt is not None:
                    out["clear_silage_share_local_threshold"] = adj(flag, mt, ref_flag)
                # secondary figure: the local threshold per tier on clear + wide, combined by area
                # (added after seeing Frederick and Queen Anne's; never the headline)
                out["local_per_tier_share"], out["local_per_tier_detail"] = self._per_tier_local(
                    r_all, tier, p, a, local, mt)
                out["clear_reference"] = {"fips": ref["fips"], "year": ref["year"],
                                          "census_share": s_ref, "clear_fields": int(len(r)),
                                          "counted_share": round(ref_counted, 4),
                                          "local_threshold": round(t, 4),
                                          "flagged_share_reference": round(ref_flag, 4),
                                          "flagged_share_target": round(flag, 4),
                                          "matched_tpr": mt, "tpr": rates["tpr"]}
        key = {"local_threshold": "clear_silage_share_local_threshold",
               "reference": "clear_silage_share_reference_calibrated",
               "europe": "clear_silage_share_europe_adjusted",
               "counted": "clear_silage_share_counted"}.get(MODULE_V2_COUNTY_STEP)
        if key and out.get(key) is not None:
            out["silage_share_headline"] = out[key]
            out["headline_method"] = f"module_v2 county step '{MODULE_V2_COUNTY_STEP}' on the clear tier"
        else:
            out["silage_share_headline"] = out["clear_silage_share_counted"]
            out["headline_method"] = "module_v2: counted share of the clear tier (county step not set or not available)"
        out["module_headline_share"] = out["silage_share_headline"]
        out["module_headline_method"] = out["headline_method"]
        return out

    @staticmethod
    def _per_tier_local(r_all, tier, p, a, local, clear_tpr, alpha=0.10, min_fields=50):
        """Local threshold inside each tier (clear, wide): threshold and flagged share from that
        tier of the reference county, matched recall for that tier, adjusted shares combined by
        the target's tier areas. None when a tier is too small in the reference."""
        from .anomaly.local import adjusted_share
        detail, num, den = {}, 0.0, 0.0
        for k in ("clear", "wide"):
            rk = r_all[r_all["tier"] == k]
            mk = (tier == k) & np.isfinite(a)
            tpr = (local.get(k) or {}).get("matched_tpr", clear_tpr if k == "clear" else None)
            if len(rk) < min_fields or not mk.any() or tpr is None:
                return None, {"reason": f"tier {k}: {len(rk)} reference fields or no recall"}
            t = float(np.quantile(rk["p"], 1 - alpha))
            ref_flag = float(np.sum(rk["a"] * (rk["p"] >= t)) / np.sum(rk["a"]))
            flag = float(np.sum(a[mk] * (p[mk] >= t)) / np.sum(a[mk]))
            share = adjusted_share(flag, ref_flag, tpr)
            if not np.isfinite(share):
                return None, {"reason": f"tier {k}: recall not above the reference flagged share"}
            area = float(np.sum(a[mk]))
            detail[k] = {"threshold": round(t, 4), "flagged_share_reference": round(ref_flag, 4),
                         "flagged_share_target": round(flag, 4), "matched_tpr": tpr,
                         "share": round(share, 4), "target_acres": round(area, 1)}
            num += share * area
            den += area
        return round(num / den, 4), detail

    def _reference_calibrated(self, fips, year, model_name, a, called, ok):
        """Layer 4: counted share corrected with the false-positive rate measured in a nearly
        all-grain reference county of the same region and season.

        FPR_ref = (counted share in the reference - TPR x its Census share) / (1 - its share),
        TPR from European held-out fields; share = (counted - FPR_ref) / (TPR - FPR_ref),
        clipped to [0, 1]. The range resamples the target's fields, the reference's fields
        and TPR. Validated on European pseudo-counties (fieldvision.model.county_validation).
        """
        empty = {"silage_share_calibrated": None, "silage_share_calibrated_low": None,
                 "silage_share_calibrated_high": None, "silage_acres_calibrated": None,
                 "calibration_reference": None, "module_headline_share": None,
                 "module_headline_method": None}
        ref = self.references.get(str(fips))
        base = model_name.split("+")[0]
        rates_path = self.registry / base / "rates.json"
        if not ref or str(ref["fips"]) == str(fips) or not rates_path.exists():
            out = dict(empty)
            if out["module_headline_method"] is None:
                out["module_headline_share"] = None
            out["calibration_note"] = ("No reference county configured for this county, so no "
                                       "calibrated total." if not ref else
                                       "No reference-calibrated total (reference is this county "
                                       "or the model has no European error rates).")
            return out
        ref_policy = "module_v1" if base == MODULE_V1_MODEL else "standard"
        rows, _, _ = self._policy_predictions(ref["fips"], ref["year"], ref_policy)
        rf, _, _, _ = self._data(ref["fips"], ref["year"])
        rarea = rf.set_index("field_id")["area_ha"].astype(float) * HA_TO_ACRE
        rdf = pd.DataFrame(rows)
        ra = rdf["source_field_id"].map(rarea).to_numpy(float)
        rok = ~rdf["abstain"].to_numpy(bool) & np.isfinite(ra)
        rcalled = (rdf["label"].to_numpy() == "silage")
        doc = json.loads(rates_path.read_text(encoding="utf-8"))
        r = doc["sets"][doc["primary"]]
        s_ref = float(ref["census_share"])

        def share_of(aa, cc, rra, rcc, tpr):
            counted = np.sum(aa * cc) / np.sum(aa)
            counted_ref = np.sum(rra * rcc) / np.sum(rra)
            fpr = np.clip((counted_ref - tpr * s_ref) / (1 - s_ref), 0, 0.95)
            den = tpr - fpr
            v = (counted - fpr) / den if den > 0.05 else np.nan
            return v, fpr, counted_ref

        aa, cc = a[ok], called[ok].astype(float)
        rra, rcc = ra[rok], rcalled[rok].astype(float)
        v, fpr, counted_ref = share_of(aa, cc, rra, rcc, r["tpr"])
        if not np.isfinite(v):
            out = dict(empty)
            out["calibration_note"] = "Reference-calibrated total not defined (TPR too close to FPR)."
            return out
        clipped = bool(v < 0 or v > 1)
        v = float(np.clip(v, 0, 1))
        rng = np.random.default_rng(self.seed + 3)
        n = max(self.n_resample // 4, 200)
        boots = []
        for _ in range(n):
            i = rng.integers(0, len(aa), len(aa))
            j = rng.integers(0, len(rra), len(rra))
            tpr_b = rng.beta(r["tp"] + 1, r["fn"] + 1)
            vb, _, _ = share_of(aa[i], cc[i], rra[j], rcc[j], tpr_b)
            if np.isfinite(vb):
                boots.append(float(np.clip(vb, 0, 1)))
        lo, hi = np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan)
        area_ok = float(np.sum(aa))
        note = (f"Reference-calibrated (module_v1 county step): the reference county {ref['fips']} "
                f"(Census silage share {s_ref:.1%}) has {counted_ref:.1%} of its area called silage, "
                f"which puts the local false-positive rate at {fpr:.2f}; with the European TPR "
                f"{r['tpr']:.2f} this county's share is {v:.1%} [{lo:.1%}, {hi:.1%}]"
                + (" (clipped to 0-1)" if clipped else "") + ". This estimate uses the reference "
                "county's Census share, so it is not independent of Census figures.")
        return {"silage_share_calibrated": round(v, 4),
                "silage_share_calibrated_low": round(float(lo), 4),
                "silage_share_calibrated_high": round(float(hi), 4),
                "silage_acres_calibrated": round(v * area_ok, 1),
                "calibration_reference": {"fips": ref["fips"], "year": ref["year"],
                                          "census_share": s_ref, "counted_share": round(float(counted_ref), 4),
                                          "local_fpr": round(float(fpr), 4), "tpr": round(r["tpr"], 3),
                                          "clipped": clipped},
                "module_headline_share": round(v, 4),
                "module_headline_method": "reference-calibrated count (module_v1)",
                "calibration_note": note}

    def _prior_shift(self, p, a, ok, train_prior=0.5, tol=1e-6, max_iter=500):
        """Re-estimate the silage share from the probabilities alone (EM prior shift).

        Probabilities calibrated at `train_prior` are moved to a new prior, the new prior is
        the area-weighted mean of the moved probabilities, and so on until it settles.
        Started from 0.5. Needs no error rates; reported beside the headline, not instead of it.
        """
        def em(pp, aa):
            pi = 0.5
            for it in range(1, max_iter + 1):
                num = pp * pi / train_prior
                den = num + (1 - pp) * (1 - pi) / (1 - train_prior)
                q = num / den
                new = float(np.sum(aa * q) / np.sum(aa))
                if abs(new - pi) < tol:
                    return new, True, it
                pi = new
            return pi, False, max_iter

        pp = np.clip(p[ok], 1e-6, 1 - 1e-6)
        aa = a[ok]
        if len(pp) == 0:
            return {"silage_share_prior_shift": None, "silage_share_prior_shift_low": None,
                    "silage_share_prior_shift_high": None, "prior_shift_converged": False,
                    "prior_shift_iterations": 0}
        share, conv, it = em(pp, aa)
        rng = np.random.default_rng(self.seed + 2)
        boots = []
        for _ in range(max(self.n_resample // 10, 50)):
            idx = rng.integers(0, len(pp), len(pp))
            s, c, _ = em(pp[idx], aa[idx])
            if c:
                boots.append(s)
        lo, hi = (np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan))
        return {"silage_share_prior_shift": round(share, 4),
                "silage_share_prior_shift_low": None if not boots else round(float(lo), 4),
                "silage_share_prior_shift_high": None if not boots else round(float(hi), 4),
                "prior_shift_converged": bool(conv), "prior_shift_iterations": int(it)}

    def _adjusted(self, model_name, a, called, ok, raw_silage):
        """Counted silage area corrected with the model's European held-out error rates.

        share_adj = (counted share - FPR) / (TPR - FPR), clipped to [0, 1] (and said so).
        The range draws fields with replacement and TPR/FPR from their Beta posteriors.
        """
        base = model_name.split("+")[0]
        path = self.registry / base / "rates.json"
        area_ok = float(np.sum(a[ok]))
        counted = float(np.sum(a[ok & called]) / area_ok) if area_ok > 0 else float("nan")
        empty = {"silage_acres_adjusted": None, "silage_acres_adjusted_low": None,
                 "silage_acres_adjusted_high": None, "grain_acres_adjusted": None,
                 "silage_share_counted": round(counted, 4) if np.isfinite(counted) else None,
                 "silage_share_adjusted": None, "adjustment_method": None,
                 "adjustment_clipped": None, "adjusted_share_by_rates": {}}
        if not path.exists() or not np.isfinite(counted):
            empty["notes"] = (f"silage_acres sums area x p_silage, which assumes half the fields are "
                              f"silage (the training balance); no European error rates are stored for "
                              f"{base}, so no adjusted total is given")
            return empty
        doc = json.loads(path.read_text(encoding="utf-8"))
        sets, primary = doc["sets"], doc["primary"]

        def adjust(share, tpr, fpr):
            den = tpr - fpr
            if den < 0.05:
                return None, None
            v = (share - fpr) / den
            return float(min(max(v, 0.0), 1.0)), bool(v < 0 or v > 1)

        by = {}
        for k, r in sets.items():
            v, c = adjust(counted, r["tpr"], r["fpr"])
            by[k] = {"share": None if v is None else round(v, 4), "clipped": c,
                     "tpr": round(r["tpr"], 3), "fpr": round(r["fpr"], 3)}
        r = sets[primary]
        share, clipped = adjust(counted, r["tpr"], r["fpr"])
        if share is None:
            empty["notes"] = f"TPR and FPR of {base} are too close to correct the count"
            empty["adjusted_share_by_rates"] = by
            return empty
        rng = np.random.default_rng(self.seed + 1)
        aa, cc = a[ok], called[ok].astype(float)
        idx = rng.integers(0, len(aa), size=(self.n_resample, len(aa)))
        shares_b = (aa[idx] * cc[idx]).sum(axis=1) / aa[idx].sum(axis=1)
        tpr_b = rng.beta(r["tp"] + 1, r["fn"] + 1, self.n_resample)
        fpr_b = rng.beta(r["fp"] + 1, r["tn"] + 1, self.n_resample)
        den = tpr_b - fpr_b
        adj_b = np.clip((shares_b - fpr_b) / np.where(den > 0.05, den, np.nan), 0, 1)
        lo, hi = np.nanpercentile(adj_b, [5, 95])
        others = [v["share"] for v in by.values() if v["share"] is not None]
        note = (f"silage_acres sums area x p_silage and so assumes half the fields are silage, the "
                f"balance of the training data; it reads high where silage is rare. "
                f"silage_acres_adjusted counts the area called silage ({counted:.1%} of the area not "
                f"abstained) and corrects it with {base}'s error rates on held-out European fields "
                f"({primary}: TPR {r['tpr']:.2f}, FPR {r['fpr']:.2f}): adjusted share {share:.1%}"
                + (" (clipped to the 0-1 range)" if clipped else "")
                + f"; the range covers resampled fields and uncertain rates. Rates from the other "
                f"European validation designs give {min(others):.1%} to {max(others):.1%}. Rates "
                f"measured in Europe may not hold here. Abstained acres are in neither total.")
        return {"silage_acres_adjusted": round(share * area_ok, 1),
                "silage_acres_adjusted_low": round(float(lo) * area_ok, 1),
                "silage_acres_adjusted_high": round(float(hi) * area_ok, 1),
                "grain_acres_adjusted": round((1 - share) * area_ok, 1),
                "silage_share_counted": round(counted, 4),
                "silage_share_adjusted": round(share, 4),
                "adjustment_method": "adjusted classify-and-count, European held-out TPR/FPR",
                "adjustment_clipped": clipped, "adjusted_share_by_rates": by, "notes": note}

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
