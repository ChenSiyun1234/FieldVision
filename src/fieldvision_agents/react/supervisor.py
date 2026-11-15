"""Nemotron supervisor: a ReAct loop that runs one county-season.

Nemotron chooses the steps and the routing policy; code enforces the rules:
step order, one rerun at most, a valid policy, and numbers in the final summary
that come from tool results. Every LLM turn and tool call goes to the audit log.

Usage (in the workspace):
    from fieldvision_agents.react.supervisor import run_county, NemotronClient
    from fieldvision_agents.react.tools_stub import StubToolbox
    result = run_county("24021", 2022, StubToolbox(), NemotronClient())
"""

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

from ..settings import ENV_FILE, LLM_CACHE, LOGS, WORKSPACE  # noqa: F401  (WORKSPACE: used by planner)
from .prompts import FINISH_STATUSES, POLICIES, RERUN_POLICY, SUPERVISOR_PROMPT, TOOL_SPECS

BASE_URL = "https://integrate.api.nvidia.com/v1"
# Models our key can use (probed 2026-11-14): Lightning runs the loop, Ultra for critic/reporter,
# Super VL for images. nemotron-3-super-120b-a12b returns 404 for this account.
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
JUDGE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
VISION_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def read_env(path):
    """KEY=value lines of a .env file (same reader as the workspace's Nemotron notebook)."""
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and not name.startswith("#"):
                values[name.strip()] = value.strip().strip('"').strip("'")
    return values


class NemotronClient:
    """Chat calls with retries, plus an on-disk cache so runs can be replayed without new API cost."""

    def __init__(self, model=DEFAULT_MODEL, temperature=0, max_tokens=2048,  # 900 left Lightning's thinking no room: empty replies
                 cache_dir=LLM_CACHE, use_cache=True):
        from openai import OpenAI
        key = os.environ.get("NVIDIA_API_KEY") or read_env(ENV_FILE).get("NVIDIA_API_KEY", "")
        if not key:
            raise RuntimeError(f"Add NVIDIA_API_KEY=... to {ENV_FILE}, or run with --offline")
        self.client = OpenAI(base_url=BASE_URL, api_key=key)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.cache_dir = Path(cache_dir) if use_cache else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def chat(self, messages, tools=None):
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            payload["tools"] = tools
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        cached = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cached and cached.exists():
            return {**json.loads(cached.read_text()), "cached": True}

        r, last_error = None, None
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(**payload, timeout=120)
                break
            except Exception as e:
                last_error = e
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and status != 429:
                    break  # 4xx such as 404/401 never succeed; only rate limits, timeouts and 5xx are retried
                time.sleep(2 ** attempt)
        if r is None:
            raise RuntimeError(f"Nemotron call failed ({self.model}): {last_error}")

        m = r.choices[0].message
        out = {"content": m.content or "",
               "tool_calls": [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
                              for tc in (m.tool_calls or [])],
               "usage": r.usage.model_dump() if getattr(r, "usage", None) else None}
        if cached:
            cached.write_text(json.dumps(out))
        return {**out, "cached": False}


class AuditLog:
    def __init__(self, run_id, log_dir=LOGS):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(log_dir) / f"audit_{run_id}.jsonl"
        self.run_id = run_id

    def write(self, step, **fields):
        line = {"run_id": self.run_id, "step": step, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(line, default=str) + "\n")


class RunState:
    """What has happened so far in this run, so code can refuse out-of-order or extra steps."""

    def __init__(self):
        self.scouted = False
        self.features = False
        self.classify_runs = []        # policies used, in order
        self.aggregated_after_classify = False
        self.critic_results = []
        self.critic_on_latest = False
        self.abstained = False
        self.abstain_reason = None
        self.available_policies = None  # set from route_options; None means the default POLICIES
        self.finished = None
        self.recommended = False       # recommend_sources runs at most once
        self.tool_numbers = set()      # every number seen in tool args and results

    def rerun_allowed_policy(self):
        if not self.critic_results:
            return None
        return RERUN_POLICY.get(self.critic_results[-1].get("suggested_action"))


_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _numbers_in(obj, out):
    """Collect every number in a JSON-like object, as normalised strings."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(_norm(str(obj)))
    elif isinstance(obj, str):
        for n in _NUM.findall(obj):
            out.add(_norm(n))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)


def _norm(s):
    s = s.replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def ungrounded_numbers(text, allowed):
    """Numbers in text that never appeared in a tool result. Small counting words (0-10) are ignored."""
    bad = []
    for raw in _NUM.findall(text or ""):
        n = _norm(raw)
        try:
            if 0 <= float(n) <= 10:
                continue
        except ValueError:
            pass
        if n not in allowed:
            bad.append(raw)
    return bad


def _policy_available(option):
    """A route_options entry is usable unless the toolbox marks it unavailable."""
    if not isinstance(option, dict):
        return False
    return option.get("available", True) is not False and \
        not str(option.get("note", "")).lower().startswith("not available")


def _dispatch(name, args, toolbox, state):
    """Run one tool after checking the rules. Returns a JSON-able dict."""
    fips, year = args.get("fips"), args.get("year")

    if name == "scout":
        state.scouted = True
        return toolbox.scout(fips, year)

    if name == "route_options":
        if not state.scouted:
            return {"error": "Call scout first."}
        result = toolbox.route_options(fips, year)
        offered = [p for p, opt in (result.get("options") or {}).items() if _policy_available(opt)]
        state.available_policies = offered or None
        return result

    if name == "extract_features":
        if not state.scouted:
            return {"error": "Call scout first."}
        state.features = True
        return toolbox.extract_features(fips, year)

    if name == "classify":
        policy = args.get("policy")
        allowed = state.available_policies or list(POLICIES)
        if policy not in allowed:
            return {"error": f"policy must be one of {allowed}"}
        if not state.features:
            return {"error": "Call extract_features first."}
        if len(state.classify_runs) >= 2:
            return {"error": "Only one rerun is allowed. Call finish, or abstain_region."}
        if state.classify_runs:
            wanted = state.rerun_allowed_policy()
            if wanted is None:
                return {"error": "A rerun is only allowed when the critic suggests one."}
            if wanted == state.classify_runs[-1]:
                return {"error": f"Policy '{wanted}' was already used; rerunning it changes nothing. "
                                 "Call finish with accepted_with_warnings, or abstain_region."}
            if policy != wanted:
                return {"error": f"The critic's suggestion maps to policy '{wanted}'. Use that for the rerun."}
        state.classify_runs.append(policy)
        state.aggregated_after_classify = False
        state.critic_on_latest = False
        return toolbox.classify(fips, year, policy)

    if name == "aggregate":
        if not state.classify_runs:
            return {"error": "Call classify first."}
        state.aggregated_after_classify = True
        state.critic_on_latest = False
        return toolbox.aggregate(fips, year)

    if name == "run_critic":
        if not state.aggregated_after_classify:
            return {"error": "Call aggregate on the latest classify result first."}
        if state.critic_on_latest:
            return {"error": "The critic already checked this result."}
        result = toolbox.run_critic(fips, year)
        state.critic_results.append(result)
        state.critic_on_latest = True
        if result.get("suggested_action") != "accept" and len(state.classify_runs) >= 2:
            result = {**result, "note": "Rerun already used. Call abstain_region" +
                      ("." if result.get("overall") == "fail" else ", or finish with a warning status.")}
        return result

    if name == "abstain_region":
        if not args.get("reason"):
            return {"error": "Give a reason."}
        if state.classify_runs and not state.critic_on_latest:
            return {"error": "Run aggregate and run_critic on the latest classify result before abstaining."}
        state.abstained = True
        state.abstain_reason = args["reason"]
        return {**toolbox.abstain_region(fips, year, args["reason"]),
                "next": "Call finish with status 'abstained' and a short summary."}

    if name == "recommend_sources":
        if not state.critic_on_latest:
            return {"error": "Run run_critic on the latest result first."}
        if state.critic_results[-1].get("overall") == "fail":
            return {"error": "The critic failed this result, so do not recommend sources from it."}
        if state.recommended:
            return {"error": "recommend_sources already ran in this run."}
        if not hasattr(toolbox, "recommend_sources"):
            return {"error": "This toolbox has no recommend_sources tool; skip it and finish."}
        state.recommended = True
        policy = state.classify_runs[-1] if state.classify_runs else "standard"
        return toolbox.recommend_sources(fips, year, as_of=args.get("as_of"),
                                         shortfall_frac=args.get("shortfall_frac"), policy=policy)

    if name == "finish":
        status = args.get("status")
        if status not in FINISH_STATUSES:
            return {"error": f"status must be one of {list(FINISH_STATUSES)}"}
        if status == "abstained" and not state.abstained:
            return {"error": "Call abstain_region before finishing as abstained."}
        if status != "abstained":
            if not state.critic_on_latest:
                return {"error": "Run the critic on the latest result before finishing."}
            last = state.critic_results[-1]
            if last.get("overall") == "fail":
                return {"error": "The critic failed this result, so it cannot be accepted in any form. "
                                 "Rerun with the policy the critic suggests if a rerun is left, otherwise call abstain_region."}
            if status == "accepted" and last.get("suggested_action") != "accept":
                return {"error": "The critic did not accept. Use accepted_with_warnings or abstain."}
        bad = ungrounded_numbers(args.get("summary", ""), state.tool_numbers)
        if bad:
            return {"error": f"These numbers are not in any tool result: {bad}. Copy numbers exactly or remove them."}
        state.finished = args
        return {"ok": True}

    return {"error": f"Unknown tool {name}"}


def _text_tool_call(content):
    """Fallback when the model writes a tool call as text, e.g. {"tool": "scout", "args": {...}}."""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("tool") in {s["function"]["name"] for s in TOOL_SPECS}:
        return {"id": f"text-{uuid.uuid4().hex[:8]}", "name": obj["tool"],
                "arguments": json.dumps(obj.get("args", {}))}
    return None


def run_county(fips, year, toolbox, llm, max_steps=16, run_id=None, log_dir=LOGS):
    """Run one county-season. Returns the final status, summary, audit path and token usage."""
    run_id = run_id or f"{fips}-{year}-{uuid.uuid4().hex[:6]}"
    audit = AuditLog(run_id, log_dir)
    state = RunState()
    _numbers_in({"fips": fips, "year": year}, state.tool_numbers)
    messages = [{"role": "system", "content": SUPERVISOR_PROMPT},
                {"role": "user", "content": f"Run the county-season fips='{fips}', year={year}."}]
    usage_total, nudges = 0, 0

    for step in range(max_steps):
        reply = llm.chat(messages, TOOL_SPECS)
        if reply.get("usage"):
            usage_total += reply["usage"].get("total_tokens", 0)
        calls = reply["tool_calls"] or ([c] if (c := _text_tool_call(reply["content"])) else [])
        audit.write(step, kind="llm", reason=reply["content"][:800], cached=reply.get("cached"),
                    tool_calls=[{"name": c["name"], "arguments": c["arguments"]} for c in calls])

        messages.append({"role": "assistant", "content": reply["content"],
                         **({"tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in calls]} if calls else {})})
        if not calls:
            nudges += 1
            if nudges > 2:
                break
            hint = ("You abstained. Call finish with status 'abstained' and a short summary."
                    if state.abstained else "Continue by calling a tool. End the run with finish.")
            messages.append({"role": "user", "content": hint})
            continue

        for c in calls:
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args, result = {}, {"error": "Arguments were not valid JSON."}
            else:
                if c["name"] != "finish":
                    # The run is pinned to one county-season, whatever the model passes.
                    args["fips"], args["year"] = fips, year
                try:
                    result = _dispatch(c["name"], args, toolbox, state)
                except Exception as e:  # a failing tool should not crash the run
                    result = {"error": f"{type(e).__name__}: {e}"}
            if "error" not in result:
                _numbers_in(args, state.tool_numbers)
                _numbers_in(result, state.tool_numbers)
            audit.write(step, kind="tool", tool=c["name"], args=args, result=result)
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(result, default=str)})

        if state.finished:
            break

    if not state.finished and state.abstained:
        # The model abstained but never called finish: close the run from the recorded reason.
        state.finished = {"status": "abstained", "policy_used": "none",
                          "summary": f"Abstained: {state.abstain_reason}", "auto_finished": True}
    final = state.finished or {"status": "incomplete", "policy_used": "none",
                               "summary": "The agent stopped before calling finish."}
    audit.write(max_steps, kind="final", **final, policies=state.classify_runs,
                critic=[r.get("overall") for r in state.critic_results], tokens=usage_total)
    return {"run_id": run_id, **final, "policies": state.classify_runs,
            "critic": [r.get("overall") for r in state.critic_results],
            "audit_log": str(audit.path), "tokens": usage_total}
