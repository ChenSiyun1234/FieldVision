"""Offline tests for inspect_gaps: the tool record and the supervisor's rules around it.
Run: python -m fieldvision_agents.gap.test_gap_tool"""
import tempfile
from pathlib import Path

import pandas as pd

from fieldvision_agents.gap.gap import gap_tool_result
from fieldvision_agents.gap.test_gap import AREA, ROWS
from fieldvision_agents.react.prompts import SUPERVISOR_PROMPT, TOOL_SPECS
from fieldvision_agents.react.supervisor import RunState, _dispatch


class _Inner:
    """Just enough of the FieldVision toolbox for the gap tool."""
    landsat = False

    def __init__(self, folder):
        self.folder = folder

    def _policy_predictions(self, fips, year, policy):
        return ROWS, None, None

    def _data(self, fips, year):
        return (pd.DataFrame({"field_id": list(AREA), "area_ha": list(AREA.values())}), None, None, None)

    def _folder(self, fips, year):
        return self.folder


def test_tool_record():
    with tempfile.TemporaryDirectory() as d:
        r = gap_tool_result(_Inner(d), "36121", 2022)
        assert r["abstained_fields"] == 3 and r["abstained_fields_with_cloud_gap"] == 2
        assert r["abstained_cloud_gap_share_of_corn_pct"] == 33.3 and r["cloud_gap_at_harvest"] is True
        assert r["estimate_without_gap_fields_is_likely"] == "low"      # both gap fields read silage
        assert r["silage_share_called_pct"] == 33.3 and r["silage_share_if_gap_fields_counted_pct"] == 60.0
        assert (r["range_with_gap_fields_low_pct"], r["range_with_gap_fields_high_pct"]) == (50.0, 66.7)
        assert r["main_gap"]["from"] == "2022-09-10" and r["main_gap"]["abstained_fields"] == 2
        assert r["landsat_rows_on_disk"] is False and "vision" not in r
        assert r["not_judged_share_of_corn_pct"] == 50.0 and r["reads_low_warning"] is True
        assert r["silage_share_if_all_not_judged_were_silage_pct"] == 66.7      # c1 plus g1, g2, u1 of six
        assert r["silage_share_interval_formula_pct"] == 33.3     # too few judged fields: the rest take pi
        assert "p_silage_from_timing" not in r["main_gap"]
        (Path(d) / "timeseries_landsat.parquet").write_bytes(b"")
        assert gap_tool_result(_Inner(d), "36121", 2022)["landsat_rows_on_disk"] is True


class _Box:
    def __init__(self):
        self.calls = []

    def inspect_gaps(self, fips, year, vision_fields=0):
        self.calls.append(vision_fields)
        return {"cloud_gap_at_harvest": True, "range_with_gap_fields_low_pct": 64.6}


def test_supervisor_rules():
    box, st, a = _Box(), RunState(), {"fips": "36121", "year": 2022}
    assert "error" in _dispatch("inspect_gaps", a, box, st)             # not before classify
    st.classify_runs.append("standard")
    assert _dispatch("inspect_gaps", {**a, "vision_fields": 99}, box, st)["range_with_gap_fields_low_pct"] == 64.6
    assert box.calls == [12]                                            # capped
    assert "already ran" in _dispatch("inspect_gaps", a, box, st)["error"]   # once per run
    st2 = RunState()
    st2.classify_runs.append("standard")
    assert "no inspect_gaps" in _dispatch("inspect_gaps", a, object(), st2)["error"]


def test_critic_points_to_the_gap_tool():
    class Box(_Box):
        def __init__(self, overall):
            super().__init__()
            self.overall = overall

        def run_critic(self, fips, year):
            return {"overall": self.overall, "suggested_action": "accept"}

    a = {"fips": "36121", "year": 2022}
    for overall, hinted in (("fail", True), ("warn", True), ("pass", False)):
        st = RunState()
        st.classify_runs.append("standard")
        st.aggregated_after_classify = True
        assert ("gap_hint" in _dispatch("run_critic", a, Box(overall), st)) is hinted
    st = RunState()                                     # no hint once the gaps were inspected
    st.classify_runs.append("standard")
    st.aggregated_after_classify = True
    box = Box("fail")
    _dispatch("inspect_gaps", a, box, st)
    assert "gap_hint" not in _dispatch("run_critic", a, box, st)


def test_prompt_and_schema():
    assert "inspect_gaps" in SUPERVISOR_PROMPT
    names = [t["function"]["name"] for t in TOOL_SPECS]
    assert "inspect_gaps" in names and names.index("inspect_gaps") < names.index("recommend_sources")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} passed")
