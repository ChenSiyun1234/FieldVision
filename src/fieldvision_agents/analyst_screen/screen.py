"""Analyst screen: a NASS analyst picks a county, runs the agent, reads the result and asks questions.

In a notebook:
    from fieldvision_agents.analyst_screen.screen import AnalystScreen
    AnalystScreen().show()                      # stub tools
    AnalystScreen(real_toolbox=FieldVisionToolbox()).show()   # real data (fieldvision_agents.fieldvision_bridge)
"""

import time
from pathlib import Path

import ipywidgets as w
from IPython.display import HTML, Markdown, display

from ..settings import LOGS
from ..react.supervisor import NemotronClient, run_county
from ..react.tools_stub import StubToolbox
from ..reporter.reporter import ask, result_from_log, write_report
from .views import answer_card, steps_timeline, summary_card

# Maryland counties near the BARC labels and with dairy (FIPS -> name).
COUNTIES = {
    "24021": "Frederick County, MD",
    "24043": "Washington County, MD",
    "24013": "Carroll County, MD",
    "24029": "Kent County, MD",
    "24025": "Harford County, MD",
    "24033": "Prince George's County, MD",
}

# Stub critic behaviour, for demos before the real critic is plugged in.
SCENARIOS = {
    "Critic warns, rerun fixes it": ("warn_regional", "accept"),
    "Critic accepts first time": ("accept",),
    "Critic warns, no useful rerun": ("warn",),
    "Critic fails twice (abstain)": ("fail", "fail"),
}

SAMPLE_QUESTIONS = [
    "How many silage acres, and how sure are we?",
    "Why did the agent choose that routing policy?",
    "What did the critic flag, and what happened next?",
]


class AnalystScreen:
    def __init__(self, real_toolbox=None, log_dir=LOGS):
        self.real_toolbox = real_toolbox
        self.log_dir = log_dir
        self.result = None
        self._llm = None

        title = w.HTML('<div style="font-family:system-ui,sans-serif">'
                       '<div style="font-size:20px;font-weight:600">Silage or grain · county desk</div>'
                       '<div style="color:#666">Run a county-season, read the estimate, ask why.</div></div>')
        self.county = w.Dropdown(options=[(f"{n} ({f})", f) for f, n in COUNTIES.items()],
                                 description="County", layout=w.Layout(width="330px"))
        self.year = w.Dropdown(options=[2023, 2022, 2021], value=2022, description="Season",
                               layout=w.Layout(width="170px"))
        self.source = w.ToggleButtons(options=["Stub data", "Real data"], value="Stub data",
                                      disabled=real_toolbox is None,
                                      tooltips=["Placeholder tools", "Team's real tools"])
        self.scenario = w.Dropdown(options=list(SCENARIOS), description="Critic",
                                   layout=w.Layout(width="330px"))
        self.run_btn = w.Button(description="Run county", button_style="primary", icon="play")
        self.past = w.Dropdown(description="Past runs", layout=w.Layout(width="520px"))
        self.load_btn = w.Button(description="Open", icon="folder-open")
        self.report_btn = w.Button(description="Write full report (Ultra, ~1 min)", icon="file-text",
                                   layout=w.Layout(width="290px"), disabled=True)
        self.question = w.Combobox(options=SAMPLE_QUESTIONS, placeholder="Ask about this run…",
                                   ensure_option=False, layout=w.Layout(width="520px"))
        self.ask_btn = w.Button(description="Ask", icon="comment", disabled=True)
        self.status = w.HTML()
        self.out_summary, self.out_steps = w.Output(), w.Output()
        self.out_report, self.out_answers = w.Output(), w.Output()

        self.run_btn.on_click(self._run)
        self.load_btn.on_click(self._load)
        self.report_btn.on_click(self._report)
        self.ask_btn.on_click(self._ask)
        self.source.observe(lambda c: setattr(self.scenario.layout, "display",
                                              "none" if c["new"] == "Real data" else ""), "value")
        self._refresh_past()

        self.ui = w.VBox([
            title,
            w.HBox([self.county, self.year]),
            w.HBox([self.source, self.scenario]),
            w.HBox([self.run_btn, self.status]),
            w.HBox([self.past, self.load_btn]),
            self.out_summary, self.out_steps,
            w.HBox([self.question, self.ask_btn]),
            self.out_answers,
            self.report_btn, self.out_report,
        ])

    def show(self):
        display(self.ui)

    # --- helpers ---------------------------------------------------------

    def _llm_client(self):
        if self._llm is None:
            self._llm = NemotronClient()
        return self._llm

    def _say(self, text, color="#555"):
        self.status.value = f'<span style="color:{color};font-family:system-ui,sans-serif">{text}</span>'

    def _refresh_past(self):
        logs = sorted(Path(self.log_dir).glob("audit_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        options = []
        for p in logs[:30]:
            try:
                r = result_from_log(p)
            except Exception:
                continue  # unfinished or unreadable log
            fips = r["run_id"].split("-")[0]
            when = time.strftime("%H:%M", time.localtime(p.stat().st_mtime))
            options.append((f"{when} · {COUNTIES.get(fips, fips)} · {r['status']}", str(p)))
        self.past.options = options or [("No runs yet", "")]

    def _show_result(self, result, seconds=None):
        self.result = result
        fips = result["run_id"].split("-")[0]
        for out, content in ((self.out_summary, summary_card(result, COUNTIES.get(fips, fips), seconds)),
                             (self.out_steps, steps_timeline(result["audit_log"]))):
            out.clear_output()
            with out:
                display(HTML(content))
        self.out_answers.clear_output()
        self.out_report.clear_output()
        self.report_btn.disabled = self.ask_btn.disabled = False

    # --- buttons ---------------------------------------------------------

    def _run(self, _):
        self.run_btn.disabled = True
        self._say("Running the agent… (about 20–60 seconds)")
        try:
            toolbox = (self.real_toolbox if self.source.value == "Real data"
                       else StubToolbox(SCENARIOS[self.scenario.value]))
            t0 = time.perf_counter()
            result = run_county(self.county.value, self.year.value, toolbox, self._llm_client(),
                                log_dir=self.log_dir)
            self._show_result(result, time.perf_counter() - t0)
            self._say("Done.", "#1b5e20")
            self._refresh_past()
        except Exception as e:  # show the problem on screen instead of a traceback
            self._say(f"Run failed: {type(e).__name__}: {e}", "#b00020")
        finally:
            self.run_btn.disabled = False

    def _load(self, _):
        if not self.past.value:
            return
        try:
            self._show_result(result_from_log(self.past.value))
            self._say("Loaded a past run.", "#1b5e20")
        except Exception as e:
            self._say(f"Could not open that run: {e}", "#b00020")

    def _report(self, _):
        if not self.result:
            return
        self.report_btn.disabled = True
        self._say("Writing the report with Nemotron Ultra… (about a minute)")
        try:
            rep = write_report(self.result)
            self.out_report.clear_output()
            with self.out_report:
                display(Markdown(rep["report_md"]))
                ok = rep["grounding"]["grounded"]
                display(HTML(f'<div style="color:{"#1b5e20" if ok else "#b00020"};font-family:system-ui">'
                             f'{"✔ every number checked against the evidence" if ok else "⚠ some numbers marked [unverified]"}'
                             f' · saved to {rep["path"]}</div>'))
            self._say("Report ready.", "#1b5e20")
        except Exception as e:
            self._say(f"Report failed: {type(e).__name__}: {e}", "#b00020")
        finally:
            self.report_btn.disabled = False

    def _ask(self, _):
        q = (self.question.value or "").strip()
        if not self.result or not q:
            self._say("Run or open a county, then type a question.", "#8a5300")
            return
        self.ask_btn.disabled = True
        self._say("Thinking…")
        try:
            a = ask(self.result, q, llm=self._llm_client())
            with self.out_answers:
                display(HTML(answer_card(q, a)))
            self._say("Answered.", "#1b5e20")
        except Exception as e:
            self._say(f"Question failed: {type(e).__name__}: {e}", "#b00020")
        finally:
            self.ask_btn.disabled = False
