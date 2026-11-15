"""Offline tests for the field verification planner and supply impact. No Nemotron or network calls.

Run in the workspace terminal:  python -m fieldvision_agents.planner.test_planner
"""

import math

import pandas as pd

from ..reporter.test_reporter import FakeLLM
from ..react.test_supervisor import call
from .planner import Z90, apply_reports, ask_planner, county_status, demo_fields, plan_visits, supply_impact, uncertainty_curve


def three_fields():
    # var terms: A 10^2*.5*.5 = 25, B 20^2*.9*.1 = 36, C 5^2*.02*.98 = 0.49
    return pd.DataFrame({"field_id": ["A", "B", "C"], "area_acres": [10.0, 20.0, 5.0],
                         "p_silage": [0.5, 0.9, 0.02], "abstain": [False, False, False]})


def test_status_matches_hand_math():
    s = county_status(three_fields())
    assert s["silage_acres"] == round(10 * 0.5 + 20 * 0.9 + 5 * 0.02)
    assert s["plus_minus"] == round(Z90 * math.sqrt(25 + 36 + 0.49))


def test_best_visit_is_largest_variance_term():
    plan = plan_visits(three_fields(), k=1)
    assert plan["fields"][0]["field_id"] == "B"            # 36 beats 25 even though A is a coin flip
    assert plan["plus_minus_after"] == round(Z90 * math.sqrt(25 + 0.49))
    assert plan_visits(three_fields(), k=3)["plus_minus_after"] == 0


def test_abstained_field_uses_prior_and_says_why():
    df = three_fields()
    df.loc[0, ["p_silage", "abstain"]] = [float("nan"), True]
    plan = plan_visits(df, k=3)
    reasons = {f["field_id"]: f["why"] for f in plan["fields"]}
    assert "no satellite read" in reasons["A"] and plan["fields"][[f["field_id"] for f in plan["fields"]].index("A")]["p_silage"] is None


def test_reports_remove_uncertainty():
    df = demo_fields(200, seed=1)
    df["p_silage"], df["abstain"] = df["p_silage"].fillna(0.3), False   # no prior shifts, so the math is exact
    plan = plan_visits(df, k=10)
    reports = {f["field_id"]: "silage" for f in plan["fields"]}
    after = county_status(apply_reports(df, reports))
    assert after["plus_minus"] == plan["plus_minus_after"] and after["demo_data"] is True


def test_curve_is_decreasing():
    pts = [p["plus_minus"] for p in uncertainty_curve(demo_fields(300), max_visits=30)["points"]]
    assert pts == sorted(pts, reverse=True) and pts[0] > pts[-1]


def test_supply_impact_math():
    status = {"silage_acres": 100, "range_low": 80, "range_high": 120, "corn_acres": 400}
    s = supply_impact(status, grain_yield=(180.0, "test"))
    assert (s["bushels_not_in_grain_supply"], s["bushels_low"], s["bushels_high"]) == (18000, 14400, 21600)
    assert s["silage_share_of_corn_pct"] == 25.0 and s["silage_tons"] == round(100 * 21.8)


def test_planner_agent_grounded_and_uses_budget():
    df = three_fields()
    plan = plan_visits(df, k=2)
    answer = (f"With 2 visits the range narrows from ±{plan['plus_minus_before']} to ±{plan['plus_minus_after']} acres. "
              f"Visit B and A.")
    llm = FakeLLM([[call("plan_visits", visits=2)], answer])
    out = ask_planner(df, "We have 2 visits. Where should we go?", llm=llm)
    assert out["grounded"] and out["tools_used"] == ["plan_visits"], out


def test_planner_agent_retries_invented_number():
    df = three_fields()
    plan = plan_visits(df, k=1)
    llm = FakeLLM([[call("plan_visits", visits=1)], "This saves 31337 acres of doubt.",
                   f"One visit narrows the range to ±{plan['plus_minus_after']} acres."])
    out = ask_planner(df, "Where should one visit go?", llm=llm)
    assert out["grounded"] and "31337" not in out["answer"], out


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


def test_confident_large_field_is_not_called_unsure():
    df = pd.DataFrame({"field_id": ["big", "small"], "area_acres": [100.0, 5.0],
                       "p_silage": [0.9, 0.5], "abstain": [False, False]})
    reasons = {f["field_id"]: f["why"] for f in plan_visits(df, k=2)["fields"]}
    assert reasons["big"].startswith("large field") and "unsure" not in reasons["big"]
    assert reasons["small"].startswith("model unsure")
