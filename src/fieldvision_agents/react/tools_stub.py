"""Stub tools shaped like the team's data contracts.

They let the supervisor loop run end to end before real data exists. Every result
carries "stub": True so fake numbers can never be mistaken for results. Swap in a
real toolbox with the same method names and return shapes later.
"""

import hashlib

from .prompts import POLICIES

AVG_FIELD_ACRES = 25.0


def _seed(fips, year):
    return int(hashlib.md5(f"{fips}-{year}".encode()).hexdigest()[:8], 16)


class StubToolbox:
    """critic_script: what each successive run_critic call returns ("accept", "warn", "warn_regional", "fail")."""

    def __init__(self, critic_script=("warn", "accept")):
        self.critic_script = list(critic_script)
        self._critic_calls = 0
        self._last_policy = None

    def _coverage(self, fips, year):
        s = _seed(fips, year)
        n = 600 + s % 400
        clear = int(n * 0.70)
        cloudy = int(n * 0.22)
        return n, clear, cloudy, n - clear - cloudy

    def scout(self, fips, year):
        n, clear, cloudy, sparse = self._coverage(fips, year)
        return {"stub": True, "fips": fips, "year": year, "n_fields": n,
                "fields_with_clear_series": clear,
                "fields_with_cloud_gap_radar_ok": cloudy,
                "fields_with_little_data": sparse,
                "median_clear_obs_jul_nov": 11,
                "largest_optical_gap_days_p90": 24}

    def _assign(self, fips, year, policy):
        n, clear, cloudy, sparse = self._coverage(fips, year)
        if policy == "standard":
            models = {"ts_model_v1": int(clear * 0.6), "timing_rule_v1": clear - int(clear * 0.6),
                      "radar_rule_v1": cloudy, "prior_v1": sparse // 3}
        elif policy == "radar_first":
            moved = int(clear * 0.15)
            models = {"ts_model_v1": int(clear * 0.6), "timing_rule_v1": clear - int(clear * 0.6) - moved,
                      "radar_rule_v1": cloudy + moved, "prior_v1": sparse // 3}
        else:
            models = {"ts_model_v1": int(clear * 0.6), "timing_rule_regional_v1": clear - int(clear * 0.6),
                      "radar_rule_v1": cloudy, "prior_v1": sparse // 3}
        n_abstain = n - sum(models.values())
        return n, models, n_abstain

    def route_options(self, fips, year):
        options = {}
        for p in POLICIES:
            n, models, n_abstain = self._assign(fips, year, p)
            options[p] = {"models_used": models, "n_abstain": n_abstain}
        return {"stub": True, "fips": fips, "year": year,
                "routing_table": "first eligible row: ts_model > timing_rule > radar_rule > prior",
                "options": options}

    def extract_features(self, fips, year):
        n, *_ = self._coverage(fips, year)
        return {"stub": True, "fips": fips, "year": year, "rows": n,
                "path": f"features/joined_{fips}_{year}.parquet",
                "quality": {"rows_with_gap_filled": int(n * 0.18)}}

    def classify(self, fips, year, policy):
        self._last_policy = policy
        n, models, n_abstain = self._assign(fips, year, policy)
        return {"stub": True, "fips": fips, "year": year, "policy": policy, "n_fields": n,
                "models_used": models, "n_abstain": n_abstain,
                "results_path": f"results/{fips}_{year}_{policy}.parquet"}

    def aggregate(self, fips, year):
        n, models, n_abstain = self._assign(fips, year, self._last_policy or "standard")
        share = {"standard": 0.30, "radar_first": 0.27, "regional_threshold": 0.28}[self._last_policy or "standard"]
        decided_acres = (n - n_abstain) * AVG_FIELD_ACRES
        silage = round(decided_acres * share)
        return {"stub": True, "fips": fips, "year": year,
                "silage_acres": silage,
                "silage_acres_low": round(silage * 0.9),
                "silage_acres_high": round(silage * 1.1),
                "grain_acres": round(decided_acres - silage),
                "abstained_acres": round(n_abstain * AVG_FIELD_ACRES),
                "n_fields": n, "models_used": models}

    def run_critic(self, fips, year):
        verdict = self.critic_script[min(self._critic_calls, len(self.critic_script) - 1)]
        self._critic_calls += 1
        if verdict == "accept":
            return {"stub": True, "overall": "pass", "suggested_action": "accept",
                    "flags": [], "summary": "Silage share is inside the county's 2012-2021 range."}
        if verdict == "warn_regional":
            # Demo path: a warning that asks for a different policy, so a rerun can fix it.
            return {"stub": True, "overall": "warn", "suggested_action": "rerun_with_regional_threshold",
                    "flags": [{"check": "silage_share_in_history", "status": "warn",
                               "why": "Share is above the county's 2012-2021 range; timing thresholds may not fit this region.",
                               "evidence": []}],
                    "summary": "Share high; try regional timing thresholds."}
        if verdict == "warn":
            return {"stub": True, "overall": "warn", "suggested_action": "rerun_with_radar",
                    "flags": [{"check": "silage_share_in_history", "status": "warn",
                               "why": "Share is above the historical range; many fields had cloud gaps at harvest.",
                               "evidence": []}],
                    "summary": "Share high; cloudy fields may be misrouted."}
        return {"stub": True, "overall": "fail", "suggested_action": "rerun_with_regional_threshold",
                "flags": [{"check": "harvested_le_planted", "status": "fail",
                           "why": "Harvested corn acres exceed planted acres.", "evidence": []}],
                "summary": "Result is not plausible."}

    def abstain_region(self, fips, year, reason):
        return {"stub": True, "fips": fips, "year": year, "abstained": True, "reason": reason}
