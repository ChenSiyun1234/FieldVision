"""The team's RAG critic and report search, wired into the bridge (fake modules, no index or key)."""

import tempfile
from types import SimpleNamespace

from ...react.test_supervisor import call
from ...reporter.test_reporter import FakeLLM
from ..field_qa import ask_fields
from ..rag import make_rag_critic, make_report_search
from .fakes import FIPS, YEAR
from .test_toolbox import make_toolbox


class FakeCriticToolbox:
    """Reads the toolbox the way critic_toolbox.CriticToolbox does: aggregate() and data_root."""

    def __init__(self, toolbox, use_llm=True):
        self.tb, self.use_llm, self._agg, self._policy = toolbox, use_llm, {}, {}

    def run_critic(self, fips, year):
        key = (str(fips), int(year))
        agg = self._agg.get(key) or self.tb.aggregate(fips, year)
        return {"overall": "warn", "suggested_action": "accept_with_warnings", "seen": agg,
                "policy": self._policy.get(key), "data_root": str(self.tb.data_root), "flags": [], "summary": "ok"}


FAKE_CRITIC = SimpleNamespace(CriticToolbox=FakeCriticToolbox, STATE_FIPS={"24": "Maryland"})


class FakeRag:
    def __init__(self):
        self.calls = []

    def retrieve_reports(self, query, state=None, year=None, k=5):
        self.calls.append((query, state, year, k))
        return [{"source": "MD_crop_progress_2022.pdf", "page": 2, "state": "Maryland", "year": 2022,
                 "text": "Corn for silage harvested 85 percent, near the 5-year average. " * 30, "score": 0.7}]


def test_critic_sees_the_bridge_totals_and_policy():
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp, critic=make_rag_critic(critic_module=FAKE_CRITIC, county_data=tmp))
        tb.classify(FIPS, YEAR, "regional_threshold")
        agg = tb.aggregate(FIPS, YEAR)
        verdict = tb.run_critic(FIPS, YEAR)
    assert verdict["seen"] == agg and verdict["policy"] == "regional_threshold" and verdict["data_root"] == tmp


def test_critic_aggregates_first_when_needed():
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp, critic=make_rag_critic(critic_module=FAKE_CRITIC, county_data=tmp))
        tb.classify(FIPS, YEAR, "standard")
        verdict = tb.run_critic(FIPS, YEAR)
    assert verdict["seen"]["prior_applied"] is True and verdict["policy"] == "standard"


def test_report_search_filters_to_the_state_and_trims():
    rag = FakeRag()
    hits = make_report_search(rag_module=rag, max_chars=100)("silage harvest pace", FIPS, YEAR, k=9)
    assert rag.calls == [("silage harvest pace", "Maryland", YEAR, 5)]
    assert len(hits[0]["text"]) == 100 and hits[0]["page"] == 2 and "score" not in hits[0]


def test_field_qa_can_cite_reports():
    with tempfile.TemporaryDirectory() as tmp:
        tb = make_toolbox(tmp)
        tb.reports = make_report_search(rag_module=FakeRag())
        tb.classify(FIPS, YEAR, "standard")
        llm = FakeLLM([[call("search_reports", query="Maryland silage harvest pace 2022")],
                       "Maryland silage harvest was 85 percent done, near the 5-year average "
                       "(MD_crop_progress_2022.pdf, page 2; search_reports)."])
        out = ask_fields(tb, FIPS, YEAR, None, "How was the silage harvest going in Maryland?", llm=llm)
    assert out["grounded"] and out["tools_used"] == ["search_reports"], out
