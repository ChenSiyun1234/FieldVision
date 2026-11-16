"""HTML views for the analyst screen. Pure functions, no widgets, so they can be tested offline."""

import html
import json

from ..reporter.reporter import build_evidence, load_audit

STATUS_STYLE = {
    "accepted": ("#1b5e20", "#e8f5e9", "Accepted"),
    "accepted_with_warnings": ("#8a5300", "#fff4e0", "Accepted with warnings"),
    "abstained": ("#5f2120", "#fdecea", "Abstained"),
    "incomplete": ("#424242", "#eeeeee", "Incomplete"),
}

BOX = ("border:1px solid #c9c9c9;border-radius:10px;padding:14px 16px;margin:8px 0;"
       "font-family:system-ui,sans-serif;font-size:14px;line-height:1.5")


def _e(x):
    return html.escape(str(x))


def _fmt(n):
    return f"{n:,}" if isinstance(n, (int, float)) else "—"


def status_badge(status):
    fg, bg, label = STATUS_STYLE.get(status, STATUS_STYLE["incomplete"])
    return (f'<span style="background:{bg};color:{fg};border:1px solid {fg};border-radius:12px;'
            f'padding:2px 10px;font-weight:600;font-size:13px">{_e(label)}</span>')


def summary_card(result, county_name="", seconds=None):
    """The headline: status, acres with range, how the agent got there, cost."""
    ev = build_evidence(result)
    est = ev.get("estimate") or {}
    stub = ('<div style="color:#8a5300;margin-top:6px">Placeholder test data: tools are stubs.</div>'
            if ev.get("is_stub") else "")
    timing = f" · {seconds:.0f} s" if seconds else ""
    tokens = f" · {result['tokens']:,} tokens" if result.get("tokens") else ""
    policies = " → ".join(ev["run"]["policy_sequence"]) or "none"
    critic = " → ".join(ev["run"]["critic_sequence"]) or "not run"
    return f"""
<div style="{BOX}">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
    <div style="font-size:17px;font-weight:600">{_e(county_name or ev['county']['fips'])} · {_e(ev['county']['year'])}</div>
    {status_badge(result['status'])}
  </div>
  <div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:12px">
    <div><div style="color:#666;font-size:12px">Silage acres (model estimate)</div>
         <div style="font-size:24px;font-weight:600">{_fmt(est.get('silage_acres'))}</div>
         <div style="color:#666;font-size:12px">range {_fmt(est.get('silage_acres_low'))} – {_fmt(est.get('silage_acres_high'))}</div></div>
    <div><div style="color:#666;font-size:12px">Grain acres</div>
         <div style="font-size:24px;font-weight:600">{_fmt(est.get('grain_acres'))}</div></div>
    <div><div style="color:#666;font-size:12px">Abstained acres</div>
         <div style="font-size:24px;font-weight:600">{_fmt(est.get('abstained_acres'))}</div></div>
  </div>
  <div style="margin-top:10px;color:#444">Routing: <b>{_e(policies)}</b> · Critic: <b>{_e(critic)}</b></div>
  <div style="color:#888;font-size:12px;margin-top:4px">Run {_e(result['run_id'])}{timing}{tokens}</div>
  {stub}
</div>"""


def _result_note(tool, result):
    """One short line per tool result, in words an analyst can read."""
    if not isinstance(result, dict):
        return ""
    if "error" in result:
        return f'<span style="color:#b00020">Refused by guardrail: {_e(result["error"])}</span>'
    if tool == "scout":
        return (f"{_fmt(result.get('n_fields'))} fields · {_fmt(result.get('fields_with_clear_series'))} clear · "
                f"{_fmt(result.get('fields_with_cloud_gap_radar_ok'))} cloudy (radar ok) · "
                f"{_fmt(result.get('fields_with_little_data'))} little data")
    if tool == "route_options":
        return "Compared policies: " + ", ".join(_e(p) for p in (result.get("options") or {}))
    if tool == "extract_features":
        return f"{_fmt(result.get('rows'))} feature rows"
    if tool == "classify":
        models = ", ".join(f"{_e(k)} {_fmt(v)}" for k, v in (result.get("models_used") or {}).items())
        return f"{models} · {_fmt(result.get('n_abstain'))} abstain"
    if tool == "aggregate":
        return (f"silage {_fmt(result.get('silage_acres'))} "
                f"({_fmt(result.get('silage_acres_low'))}–{_fmt(result.get('silage_acres_high'))}) · "
                f"grain {_fmt(result.get('grain_acres'))}")
    if tool == "run_critic":
        why = "; ".join(f.get("why", "") for f in result.get("flags") or [])
        return f"<b>{_e(result.get('overall'))}</b> → {_e(result.get('suggested_action'))}" + (f" · {_e(why)}" if why else "")
    if tool == "abstain_region":
        return f"Abstained: {_e(result.get('reason'))}"
    if tool == "finish":
        return "Run closed"
    return _e(json.dumps(result)[:120])


def steps_timeline(audit_path):
    """Every tool call in order, with the agent's stated reason where it gave one."""
    rows = []
    for s in load_audit(audit_path):
        if s.get("kind") != "tool":
            continue
        args = s.get("args") or {}
        why = args.get("reason") or ""
        what = s["tool"] + (f" <b>{_e(args['policy'])}</b>" if args.get("policy") else "")
        if s["tool"] == "finish":
            what += f" <b>{_e(args.get('status'))}</b>"
        rows.append(f"""
  <tr style="border-top:1px solid #e3e3e3;vertical-align:top">
    <td style="padding:6px 8px;color:#888">{len(rows) + 1}</td>
    <td style="padding:6px 8px;white-space:nowrap">{what}</td>
    <td style="padding:6px 8px">{_result_note(s['tool'], s.get('result'))}
        {f'<div style="color:#555;font-style:italic;margin-top:2px">Why: {_e(why)}</div>' if why else ''}</td>
  </tr>""")
    return f"""
<div style="{BOX}">
  <div style="font-weight:600;margin-bottom:6px">What the agent did</div>
  <table style="border-collapse:collapse;width:100%">{''.join(rows)}</table>
</div>"""


def answer_card(question, answer):
    """One Q&A exchange, with the evidence sections it used and whether every number checked out."""
    ok = answer.get("grounded")
    badge = ('<span style="color:#1b5e20">✔ every number checked</span>' if ok
             else '<span style="color:#b00020">⚠ contains [unverified] numbers</span>')
    sources = ", ".join(answer.get("sections_used") or []) or "none"
    text = _e(answer.get("answer", "")).replace("**", "")
    return f"""
<div style="{BOX}">
  <div style="color:#666;font-size:12px">Question</div>
  <div style="font-weight:600;margin-bottom:6px">{_e(question)}</div>
  <div>{text}</div>
  <div style="color:#888;font-size:12px;margin-top:6px">Evidence used: {_e(sources)} · {badge}</div>
</div>"""
