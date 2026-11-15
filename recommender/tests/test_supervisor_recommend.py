"""recommend_sources inside the supervisor loop, with the team's stub toolbox and a scripted LLM.

Run from the repository root:  python recommender/tests/test_supervisor_recommend.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fieldvision_agents.react.supervisor import run_county  # noqa: E402
from fieldvision_agents.react.test_supervisor import ScriptedLLM, call, last_tool_result, summary_from_aggregate  # noqa: E402
from fieldvision_agents.react.tools_stub import StubToolbox  # noqa: E402

FIPS, YEAR = "24021", 2022
seen = {}


class FakeMatch(StubToolbox):
    """Stub toolbox plus a canned recommend_sources (the real one is tested in test_silage_match.py)."""

    def recommend_sources(self, fips, year, as_of=None, shortfall_frac=None, policy="standard"):
        seen["policy"] = policy
        return {"as_of": as_of or f"{year}-09-01", "overall": "pass",
                "dairies": [{"dairy_id": "MDE-1", "request_tons": 13065, "matched_tons": 13065}],
                "summary": "8 dairies asked for 13,065 tons; 13,065 tons matched."}


def note(name, then):
    """Record the last recommend_sources result, then make the next scripted move."""
    def step(messages):
        seen[name] = last_tool_result(messages, "recommend_sources")
        return then(messages) if callable(then) else then
    return step


def finish(messages):
    critic = last_tool_result(messages, "run_critic")
    status = "accepted" if critic.get("suggested_action") == "accept" else "accepted_with_warnings"
    tons = seen["after_critic"]["dairies"][0]["matched_tons"]
    return [call("finish", status=status, policy_used="standard",
                 summary=summary_from_aggregate(messages) + f" Dairies were matched with {tons} tons of standing corn.")]


def run(steps):
    seen.clear()
    with tempfile.TemporaryDirectory() as d:
        return run_county(FIPS, YEAR, FakeMatch(), ScriptedLLM(steps), log_dir=d)


BASE = [[call("scout", fips=FIPS, year=YEAR)], [call("route_options", fips=FIPS, year=YEAR)],
        [call("extract_features", fips=FIPS, year=YEAR)],
        [call("classify", fips=FIPS, year=YEAR, policy="standard")], [call("aggregate", fips=FIPS, year=YEAR)]]


def test_needs_critic_first_and_runs_once():
    steps = BASE + [
        [call("recommend_sources", fips=FIPS, year=YEAR)],
        note("before_critic", [call("run_critic", fips=FIPS, year=YEAR)]),
        [call("recommend_sources", fips=FIPS, year=YEAR, as_of="2022-09-15")],
        note("after_critic", [call("recommend_sources", fips=FIPS, year=YEAR)]),
        note("second", finish)]
    r = run(steps)
    assert "error" in seen["before_critic"], seen["before_critic"]
    assert seen["after_critic"]["dairies"][0]["matched_tons"] == 13065
    assert "already ran" in seen["second"].get("error", ""), seen["second"]
    assert seen["policy"] == "standard"
    assert r.get("status") in ("accepted", "accepted_with_warnings"), r


def test_toolbox_without_tool_is_refused():
    steps = BASE + [[call("run_critic", fips=FIPS, year=YEAR)],
                    [call("recommend_sources", fips=FIPS, year=YEAR)],
                    note("missing", [call("abstain_region", fips=FIPS, year=YEAR, reason="test")])]
    seen.clear()
    with tempfile.TemporaryDirectory() as d:
        run_county(FIPS, YEAR, StubToolbox(), ScriptedLLM(steps), log_dir=d)
    assert "no recommend_sources" in seen["missing"].get("error", ""), seen["missing"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
