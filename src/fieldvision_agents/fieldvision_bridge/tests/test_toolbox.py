"""The bridge toolbox: adjusted totals, per-field access, and a full supervisor run."""

import tempfile
from pathlib import Path

import numpy as np

from ...planner.planner import county_status
from ...react.supervisor import run_county
from ...react.test_supervisor import ScriptedLLM, call, summary_from_aggregate
from ..config import ACRES_PER_HECTARE, FINAL_POLICY, MODEL_DECISIONS, STANDARD_MODEL
from ..prevalence import shift_probability
from ..toolbox import FieldVisionToolbox
from .fakes import FIELDS, FIPS, YEAR, FakeTeamToolbox, make_county

PRIOR = {"share": 0.1, "level": "county", "source": "test prior"}
WHOLE_COUNTY = {"acres": 60.0, "source": "test county corn"}       # the 4 fake fields are 49 acres: 82 %
BIG_COUNTY = {"acres": 10000.0, "source": "test county corn"}      # 0.5 %: one farm in a big county


def make_toolbox(tmp, critic=None, county_corn=WHOLE_COUNTY, headline=None):
    return FieldVisionToolbox(inner=FakeTeamToolbox(Path(tmp) / "results", headline=headline), critic=critic,
                              county_data=make_county(tmp), prior=PRIOR, county_corn=county_corn)


def test_adjusted_totals_follow_the_prior():
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp)
        tb.classify(FIPS, YEAR, "standard")
        agg = tb.aggregate(FIPS, YEAR)
    decided = [(ha * ACRES_PER_HECTARE, p) for _, _, ha, p, a, _ in FIELDS if not a]
    expected = sum(a * q for (a, _), q in zip(decided, shift_probability([p for _, p in decided], 0.1)))
    assert np.isclose(agg["silage_acres"], round(expected, 1))
    assert agg["silage_acres"] < agg["silage_acres_unadjusted"]          # a 10 % county reads lower than 50/50
    assert agg["silage_acres_low"] <= agg["silage_acres"] <= agg["silage_acres_high"]
    assert agg["prior_silage_share"] == 0.1 and agg["prior_source"] == "test prior"
    assert agg["prior_applied"] and agg["county_coverage"] > 0.5


def test_one_farm_in_a_big_county_is_not_adjusted():
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp, county_corn=BIG_COUNTY)
        tb.classify(FIPS, YEAR, "standard")
        agg = tb.aggregate(FIPS, YEAR)
        one = tb.field(FIPS, YEAR, "BARC-1")
        frame = tb.planner_frame(FIPS, YEAR)
    assert not agg["prior_applied"] and agg["silage_acres"] == agg["silage_acres_unadjusted"], agg
    assert agg["county_coverage"] < 0.01 and "too few" in agg["adjustment"]
    assert one["p_silage_model"] == 0.9 and "p_silage_county_prior" not in one
    assert np.allclose(frame["p_silage"], [p for *_, p, _, _ in FIELDS]) and frame.attrs["abstain_prior"] is None


def test_planner_counts_abstained_fields_at_the_county_prior():
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp)
        tb.classify(FIPS, YEAR, "standard")
        status = county_status(tb.planner_frame(FIPS, YEAR))
    assert status["prior_for_abstained"] == 0.1 and status["n_abstained_fields"] == 1


def test_field_access_and_planner_frame():
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp)
        tb.classify(FIPS, YEAR, "standard")
        one = tb.field(FIPS, YEAR, "BARC-1")
        by_source = tb.field(FIPS, YEAR, "barc-1-2022")
        frame = tb.planner_frame(FIPS, YEAR)
    assert one == by_source and one["p_silage_county_prior"] == 0.5 and one["evidence"]["drop_date"] == "2022-08-29"
    assert one["call"] == "silage" and one["p_silage_model"] == 0.9
    assert one["area_acres"] == round(4.0 * ACRES_PER_HECTARE, 1)
    assert list(frame.columns) == ["field_id", "area_acres", "abstain", "p_silage"] and len(frame) == 4


def test_route_options_explain_model_choices():
    with tempfile.TemporaryDirectory() as tmp:
        routes = make_toolbox(tmp).route_options(FIPS, YEAR)
    assert routes["standard_model"] == STANDARD_MODEL == "eu2_logistic_v1" and routes["model_decisions"] == MODEL_DECISIONS
    assert routes["standard_runs"] == FINAL_POLICY and "final model module" in routes["options"]["standard"]["note"]
    assert routes["options"]["module_v1"]["note"].startswith("not available")


def test_supervisor_end_to_end_on_the_bridge():
    steps = [
        [call("scout")], [call("route_options")], [call("extract_features")],
        [call("classify", policy="in_site", reason="local model")],              # marked unavailable: refused
        [call("classify", policy="standard", reason="validated default model")],
        [call("aggregate")], [call("run_critic")],
        lambda msgs: [call("finish", status="accepted_with_warnings", policy_used="standard",
                           summary=summary_from_aggregate(msgs))],
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp)
        result = run_county(FIPS, YEAR, tb, ScriptedLLM(steps), log_dir=tmp)
        assert tb.inner.calls == [FINAL_POLICY]            # the agent's "standard" runs the final module
    assert result["status"] == "accepted_with_warnings" and result["policies"] == ["standard"], result


def test_county_total_comes_from_the_final_module_when_it_gives_one():
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp, headline=0.2)
        out = tb.classify(FIPS, YEAR, "standard")
        agg = tb.aggregate(FIPS, YEAR)
        frame = tb.planner_frame(FIPS, YEAR)
    total = sum(ha for _, _, ha, *_ in FIELDS) * ACRES_PER_HECTARE
    assert out["runs"] == FINAL_POLICY and out["policy"] == "standard"
    assert np.isclose(agg["silage_acres"], round(0.2 * total, 1)) and agg["silage_share_pct"] == 20.0
    assert agg["silage_acres_low"] < agg["silage_acres"] < agg["silage_acres_high"]
    assert not agg["prior_applied"] and "final module" in agg["adjustment"]
    assert frame.attrs["abstain_prior"] is None             # no prior shift alongside the module's own total


def test_without_a_final_headline_the_bridge_says_why():
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp, county_corn=BIG_COUNTY)      # headline None: too few clear fields
        tb.classify(FIPS, YEAR, "standard")
        agg = tb.aggregate(FIPS, YEAR)
    assert agg["silage_acres"] == agg["silage_acres_unadjusted"] and "too few" in agg["adjustment"]


def test_unconnected_critic_warns_but_accepts():
    with tempfile.TemporaryDirectory() as tmp:
        verdict = make_toolbox(tmp).run_critic(FIPS, YEAR)
    assert verdict["overall"] == "warn" and verdict["suggested_action"] == "accept"
