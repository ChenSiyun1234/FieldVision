"""Field-level Q&A with FieldVision evidence."""

import tempfile

from ...react.test_supervisor import call
from ...reporter.test_reporter import FakeLLM
from ..field_qa import ask_fields, list_fields
from .fakes import FIPS, YEAR
from .test_toolbox import make_toolbox


def classified(tmp):
    tb = make_toolbox(tmp)
    tb.classify(FIPS, YEAR, "standard")
    return tb


def test_list_fields_by_kind():
    with tempfile.TemporaryDirectory() as tmp:
        df = classified(tmp).fields(FIPS, YEAR)
    assert [f["field_id"] for f in list_fields(df, "abstained")["fields"]] == ["BARC-4"]
    assert "BARC-2" in [f["field_id"] for f in list_fields(df, "grain_like")["fields"]]
    assert list_fields(df, "most_uncertain")["fields"][0]["field_id"] == "BARC-3"   # calls use the model's own p
    silage = list_fields(df, "silage_like")["fields"]
    assert [f["field_id"] for f in silage] == ["BARC-1", "BARC-3"] and silage[0]["p_silage_model"] == 0.9


def test_field_answer_is_grounded():
    with tempfile.TemporaryDirectory() as tmp:
        tb = classified(tmp)
        llm = FakeLLM([[call("field_detail", field_id="BARC-1")],
                       "BARC-1 dropped on 2022-08-29 from NDVI 0.81 to 0.24; P(silage) 0.5 at the county prior (field_detail)."])
        out = ask_fields(tb, FIPS, YEAR, None, "Why is BARC-1 silage-like?", llm=llm)
    assert out["grounded"] and out["tools_used"] == ["field_detail"], out


def test_invented_number_gets_one_retry():
    with tempfile.TemporaryDirectory() as tmp:
        tb = classified(tmp)
        llm = FakeLLM([[call("field_detail", field_id="BARC-2")], "BARC-2 covers 123 acres.",
                       "BARC-2 looks grain-like: it browned before a late drop (field_detail)."])
        out = ask_fields(tb, FIPS, YEAR, None, "Tell me about BARC-2", llm=llm)
    assert out["grounded"] and "123" not in out["answer"], out


def test_empty_answer_gets_one_nudge():
    with tempfile.TemporaryDirectory() as tmp:
        tb = classified(tmp)
        llm = FakeLLM([[call("field_detail", field_id="BARC-1")], "",
                       "BARC-1 dropped on 2022-08-29 (field_detail)."])
        out = ask_fields(tb, FIPS, YEAR, None, "When did BARC-1 drop?", llm=llm)
    assert out["answer"].startswith("BARC-1 dropped") and out["grounded"], out


def test_last_step_must_answer():
    class Recording(FakeLLM):
        def chat(self, messages, tools=None):
            self.tools_seen = getattr(self, "tools_seen", []) + [tools is not None]
            return super().chat(messages, tools)

    with tempfile.TemporaryDirectory() as tmp:
        tb = classified(tmp)
        llm = Recording([[call("list_fields", kind="largest")], [call("list_fields", kind="abstained")],
                         "The run record does not say (list_fields)."])
        out = ask_fields(tb, FIPS, YEAR, None, "Anything?", llm=llm, max_steps=3)
    assert llm.tools_seen == [True, True, False] and out["answer"].startswith("The run record"), out


def test_unanswered_report_question_lists_what_was_checked():
    from .test_rag import FakeRag
    from ..rag import make_report_search
    with tempfile.TemporaryDirectory() as tmp:
        tb = classified(tmp)
        tb.reports = make_report_search(rag_module=FakeRag())
        llm = FakeLLM([[call("search_reports", query="silage pace")], "", ""])
        out = ask_fields(tb, FIPS, YEAR, None, "Silage pace?", llm=llm, max_steps=3)
    assert "MD_crop_progress_2022.pdf p.2" in out["answer"] and "do not answer" in out["answer"], out
