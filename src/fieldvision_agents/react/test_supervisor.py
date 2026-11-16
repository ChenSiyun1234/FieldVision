"""Offline tests for the supervisor loop. No Nemotron calls, no credits.

A scripted fake LLM plays the agent so we can check the guardrails:
step order, one rerun, the critic's policy, abstaining, and grounded numbers.

Run in the workspace terminal:  python -m fieldvision_agents.react.test_supervisor
"""

import json
import tempfile
import uuid

from .supervisor import run_county
from .tools_stub import StubToolbox

FIPS, YEAR = "24021", 2022


def call(name, **args):
    return {"id": f"call-{uuid.uuid4().hex[:6]}", "name": name, "arguments": json.dumps(args)}


def last_tool_result(messages, tool_name):
    """Find the most recent result of a given tool in the conversation."""
    ids = {}
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            ids[tc["id"]] = tc["function"]["name"]
    for m in reversed(messages):
        if m.get("role") == "tool" and ids.get(m["tool_call_id"]) == tool_name:
            return json.loads(m["content"])
    return {}


def summary_from_aggregate(messages):
    a = last_tool_result(messages, "aggregate")
    return (f"Silage {a['silage_acres']} acres (range {a['silage_acres_low']} to {a['silage_acres_high']}), "
            f"abstained {a['abstained_acres']} acres.")


class ScriptedLLM:
    """Returns one scripted step per chat() call. A step is a list of calls, or a function of messages."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.seen = []

    def chat(self, messages, tools):
        step = self.steps.pop(0) if self.steps else []
        calls = step(messages) if callable(step) else step
        self.seen.append(messages[-1])
        return {"content": "scripted", "tool_calls": calls, "usage": None, "cached": False}


def run(steps, critic_script):
    with tempfile.TemporaryDirectory() as tmp:
        return run_county(FIPS, YEAR, StubToolbox(critic_script), ScriptedLLM(steps), log_dir=tmp), None


def test_warn_then_rerun_then_accept():
    steps = [
        [call("scout")], [call("route_options")], [call("extract_features")],
        [call("classify", policy="standard")], [call("aggregate")], [call("run_critic")],
        [call("classify", policy="radar_first")], [call("aggregate")], [call("run_critic")],
        lambda msgs: [call("finish", status="accepted", policy_used="radar_first",
                           summary=summary_from_aggregate(msgs))],
    ]
    result, _ = run(steps, ("warn", "accept"))
    assert result["status"] == "accepted", result
    assert result["policies"] == ["standard", "radar_first"], result
    assert result["critic"] == ["warn", "pass"], result


def test_second_rerun_blocked_then_abstain():
    steps = [
        [call("scout")], [call("extract_features")],
        [call("classify", policy="standard")], [call("aggregate")], [call("run_critic")],
        [call("classify", policy="regional_threshold")], [call("aggregate")], [call("run_critic")],
        [call("classify", policy="radar_first")],                       # third classify: must be refused
        [call("abstain_region", reason="Critic rejected the result after one rerun.")],
        [call("finish", status="abstained", policy_used="none", summary="County abstained after the rerun.")],
    ]
    result, _ = run(steps, ("fail", "fail"))
    assert result["status"] == "abstained", result
    assert result["policies"] == ["standard", "regional_threshold"], result


def test_order_and_policy_guardrails():
    steps = [
        [call("classify", policy="standard")],                          # before scout/features: refused
        [call("scout")], [call("extract_features")],
        [call("classify", policy="standard")], [call("aggregate")], [call("run_critic")],
        [call("classify", policy="regional_threshold")],                # critic said radar: refused
        [call("classify", policy="radar_first")], [call("aggregate")], [call("run_critic")],
        lambda msgs: [call("finish", status="accepted", policy_used="radar_first",
                           summary=summary_from_aggregate(msgs))],
    ]
    result, _ = run(steps, ("warn", "accept"))
    assert result["status"] == "accepted", result
    assert result["policies"] == ["standard", "radar_first"], result


def test_invented_number_is_refused():
    steps = [
        [call("scout")], [call("extract_features")],
        [call("classify", policy="standard")], [call("aggregate")], [call("run_critic")],
        [call("finish", status="accepted", policy_used="standard",
              summary="Silage was 123456 acres.")],                    # made-up number: refused
        lambda msgs: [call("finish", status="accepted", policy_used="standard",
                           summary=summary_from_aggregate(msgs))],
    ]
    result, _ = run(steps, ("accept",))
    assert result["status"] == "accepted", result
    assert "123456" not in result["summary"], result


def test_cannot_claim_accepted_after_warning():
    steps = [
        [call("scout")], [call("extract_features")],
        [call("classify", policy="standard")], [call("aggregate")], [call("run_critic")],
        lambda msgs: [call("finish", status="accepted", policy_used="standard",
                           summary=summary_from_aggregate(msgs))],     # critic warned: refused
        lambda msgs: [call("finish", status="accepted_with_warnings", policy_used="standard",
                           summary=summary_from_aggregate(msgs))],
    ]
    result, _ = run(steps, ("warn",))
    assert result["status"] == "accepted_with_warnings", result


def test_same_policy_rerun_refused():
    steps = [
        [call("scout")], [call("extract_features")],
        [call("classify", policy="radar_first")], [call("aggregate")], [call("run_critic")],
        [call("classify", policy="radar_first")],                       # critic says radar, already used: refused
        [call("abstain_region", reason="Critic warned and the suggested rerun was already used.")],
        [call("finish", status="abstained", policy_used="none", summary="County abstained.")],
    ]
    result, _ = run(steps, ("warn",))
    assert result["status"] == "abstained", result
    assert result["policies"] == ["radar_first"], result


def test_abstain_requires_critic_on_latest():
    steps = [
        [call("scout")], [call("extract_features")],
        [call("classify", policy="standard")], [call("aggregate")], [call("run_critic")],
        [call("classify", policy="radar_first")], [call("aggregate")],
        [call("abstain_region", reason="skip the critic")],             # critic not run on rerun: refused
        [call("run_critic")],
        lambda msgs: [call("finish", status="accepted", policy_used="radar_first",
                           summary=summary_from_aggregate(msgs))],
    ]
    result, _ = run(steps, ("warn", "accept"))
    assert result["status"] == "accepted", result
    assert result["critic"] == ["warn", "pass"], result


def test_abstain_then_silence_auto_finishes():
    steps = [
        [call("scout")], [call("extract_features")],
        [call("classify", policy="standard")], [call("aggregate")], [call("run_critic")],
        [call("abstain_region", reason="Result not plausible.")],
        [], [], [],                                                     # model goes quiet
    ]
    result, _ = run(steps, ("fail",))
    assert result["status"] == "abstained", result
    assert result.get("auto_finished") is True, result


class ExtraRouteToolbox(StubToolbox):
    """A toolbox that offers an extra 'in_site' route and marks 'radar_first' unavailable."""

    def route_options(self, fips, year):
        r = super().route_options(fips, year)
        r["options"]["in_site"] = {**r["options"]["standard"], "note": "trained on local labels"}
        r["options"]["radar_first"] = {"note": "not available: no radar model"}
        return r

    def classify(self, fips, year, policy):
        return {**super().classify(fips, year, "standard" if policy == "in_site" else policy), "policy": policy}


def test_routes_come_from_the_toolbox():
    steps = [
        [call("scout")], [call("route_options")], [call("extract_features")],
        [call("classify", policy="radar_first")],                       # marked unavailable: refused
        [call("classify", policy="in_site")], [call("aggregate")], [call("run_critic")],
        lambda msgs: [call("finish", status="accepted", policy_used="in_site", summary=summary_from_aggregate(msgs))],
    ]
    with tempfile.TemporaryDirectory() as tmp:
        result = run_county(FIPS, YEAR, ExtraRouteToolbox(("accept",)), ScriptedLLM(steps), log_dir=tmp)
    assert result["status"] == "accepted" and result["policies"] == ["in_site"], result


def test_failed_result_cannot_be_accepted_with_warnings():
    steps = [
        [call("scout")], [call("extract_features")],
        [call("classify", policy="standard")], [call("aggregate")], [call("run_critic")],
        lambda msgs: [call("finish", status="accepted_with_warnings", policy_used="standard",
                           summary=summary_from_aggregate(msgs))],     # critic failed: refused
        [call("abstain_region", reason="Critic failed the result.")],
        [call("finish", status="abstained", policy_used="none", summary="Abstained.")],
    ]
    result, _ = run(steps, ("fail",))
    assert result["status"] == "abstained", result


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
