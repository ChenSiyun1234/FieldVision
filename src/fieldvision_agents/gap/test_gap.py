"""Offline tests for the gap path: no network, no key. Run: python -m fieldvision_agents.gap.test_gap"""
import tempfile

from fieldvision_agents.gap.gap import (agreement, evidence_record, explain, gap_summary, gap_table,
                                        harvest_curves, interval_probability, interval_share, interval_window,
                                        numbers_ok, prior_shifted_share, run_vision, template_reason,
                                        vision_status, vision_windows)


def _row(fid, tier, label, p, a="2022-09-10", b="2022-10-05", days=25.0, abstain=False):
    return {"field_id": fid, "source_field_id": fid, "p_silage": p, "label": label, "abstain": abstain,
            "evidence": {"evidence_tier": tier, "last_standing_date": a, "drop_date": b,
                         "days_to_drop": days, "ndvi_before": 0.83, "ndvi_after": 0.27,
                         "greenness_before_share_of_peak": 0.95, "residue_ndti_after": 0.19,
                         "abstain_reason": "wide_harvest_interval" if abstain else None}}


ROWS = [
    _row("c1", "clear", "silage", 0.9, "2022-09-01", "2022-09-06", 5.0),
    _row("c2", "clear", "grain", 0.1, "2022-10-20", "2022-10-28", 8.0),
    _row("w1", "wide", "grain", 0.3, "2022-10-05", "2022-10-23", 18.0),
    _row("g1", "abstain", "silage", 0.7, abstain=True),
    _row("g2", "abstain", "silage", 0.8, abstain=True),
    _row("u1", "abstain", "grain", 0.5, a=None, b=None, days=None, abstain=True),
]
AREA = {"c1": 10, "c2": 10, "w1": 10, "g1": 10, "g2": 10, "u1": 10}   # hectares


def test_table_and_summary():
    df = gap_table(ROWS, AREA)
    assert list(df["kind"]) == ["clear", "clear", "gap", "gap", "gap", "unseen"]
    s = gap_summary(df)
    assert s["kinds"]["gap"]["fields"] == 3 and s["kinds"]["unseen"]["fields"] == 1
    assert s["abstained_with_bracket"] == 2
    sh = s["silage_share"]
    assert sh["clear_only"] == 0.5                      # c1 of c1, c2
    assert sh["called"] == round(1 / 3, 4)              # c1 of c1, c2, w1
    assert sh["with_gap_fields"] == 0.6                 # c1, g1, g2 of five seen fields
    assert sh["lower"] == 0.5 and sh["upper"] == round(4 / 6, 4)   # of all six, then plus unseen
    top = s["main_gaps"][0]
    assert (top["from"], top["to"], top["fields"], top["days"]) == ("2022-09-10", "2022-10-05", 2, 25)


def test_prior_shift():
    df = gap_table(ROWS, AREA)
    # at the training balance nothing moves: mean of 0.9, 0.1, 0.3, 0.7, 0.8 and 0.5 for the unseen field
    assert prior_shifted_share(df, 0.5) == 0.55
    low, high = prior_shifted_share(df, 0.05), prior_shifted_share(df, 0.6)
    assert low < 0.55 < high and 0.05 < low                 # moves toward the prior, never past it


def _county(n=12):
    """Silage cut 1-6 Sep, grain 20-28 Oct, all judged; plus abstained fields with wide brackets."""
    rows = [_row(f"s{i}", "clear", "silage", 0.9, "2022-09-01", "2022-09-06", 5.0) for i in range(n)]
    rows += [_row(f"g{i}", "clear", "grain", 0.1, "2022-10-20", "2022-10-28", 8.0) for i in range(n)]
    rows += [_row("early", "abstain", "grain", 0.5, "2022-08-25", "2022-09-20", 26.0, abstain=True),
             _row("late", "abstain", "silage", 0.6, "2022-10-10", "2022-11-05", 26.0, abstain=True),
             _row("both", "abstain", "silage", 0.6, "2022-08-25", "2022-11-05", 72.0, abstain=True)]
    return gap_table(rows, {r["field_id"]: 10 for r in rows})


def test_interval_probability():
    df = _county()
    days, fs, fg, ns, ng = harvest_curves(df)
    assert (ns, ng) == (12, 12) and abs(fs.sum() - 1) < 1e-9 and abs(fg.sum() - 1) < 1e-9
    p, info = interval_probability(df)
    by = dict(zip(df["field_id"], p))
    assert info["used"] and info["pi"] == 0.5
    assert info["median_silage_harvest_doy"] < info["median_grain_harvest_doy"]
    assert by["early"] > 0.9          # bracket covers the silage weeks only, whatever the model said
    assert by["late"] < 0.1           # bracket covers the grain weeks only
    assert abs(by["both"] - 0.5) < 0.02    # bracket covers everything: timing says nothing, p = pi
    lo, _ = interval_probability(df, prior=0.1)
    assert dict(zip(df["field_id"], lo))["both"] < 0.15      # a low prior pulls the uninformative field down
    share, info = interval_share(df)
    assert info["abstained_fields_given_interval_p"] == 3 and abs(share - (12 + by["early"] + by["late"] + by["both"]) / 27) < 1e-3


def test_interval_window_and_reason():
    df = _county()
    w = interval_window(df, "2022-08-25", "2022-09-20")
    assert w["silage_harvest_in_gap_pct"] > 90 and w["grain_harvest_in_gap_pct"] < 10
    assert w["p_silage_from_timing"] > 0.9 and w["judged_fields_called_silage_pct"] == 50.0
    row = df[df["field_id"] == "early"].iloc[0]
    rec = evidence_record(row, None, w)
    text = template_reason(rec)
    assert "on timing alone" in text and numbers_ok(text, rec)
    assert interval_window(_county(n=4), "2022-08-25", "2022-09-20") is None


def test_interval_needs_both_classes():
    df = _county(n=4)                 # too few judged fields per class: no curves, fields fall back to pi
    p, info = interval_probability(df)
    assert not info["used"] and all(x != x for x in p)
    share, _ = interval_share(df)
    assert abs(share - (4 + 3 * 0.5) / 11) < 1e-3


def test_windows():
    df = gap_table(ROWS, AREA)
    before, after = vision_windows(df.iloc[3])
    assert before == ("2022-09-09", "2022-09-11") and after == ("2022-10-04", "2022-10-06")


def test_vision_status_and_agreement():
    bare = {"hint": "silage-like", "confidence": 0.9}
    assert vision_status("silage", bare) == "supported"
    assert vision_status("grain", bare) == "conflict"
    assert vision_status("silage", {"hint": "silage-like", "confidence": 0.3}) == "unclear"
    assert vision_status("silage", {"hint": "no hint", "confidence": 0.9}) == "unclear"
    a = agreement([{"status": "supported"}, {"status": "supported"}, {"status": "conflict"},
                   {"status": "unclear"}])
    assert a == {"checked": 4, "readable": 3, "supported": 2, "conflict": 1, "unclear": 1,
                 "agreement": 0.667}


def test_run_vision_resumes_and_survives_errors():
    df = gap_table(ROWS, AREA)
    calls = []

    def chip(geom, start, end, max_cloud=100):
        if geom == "bad":
            raise RuntimeError("no scene")
        return b"png", start, 1.0

    def ask(b, a, llm):
        calls.append(1)
        return {"before_canopy": "green", "after_ground": "bare_soil", "confidence": 0.8,
                "reason": "dark soil", "hint": "silage-like"}

    geoms = {"w1": "ok", "g1": "ok", "g2": "bad"}
    with tempfile.TemporaryDirectory() as d:
        r1 = run_vision(df, geoms, "gap", 3, 0, d, None, chip, ask, log=lambda *_: None)
        assert len(r1) == 3 and len(calls) == 2
        by = {r["field_id"]: r["status"] for r in r1}
        assert by == {"w1": "conflict", "g1": "supported", "g2": "unclear"}
        r2 = run_vision(df, geoms, "gap", 3, 0, d, None, chip, ask, log=lambda *_: None)
        assert len(r2) == 3 and len(calls) == 2          # nothing asked twice


class _LLM:
    def __init__(self, text):
        self.text = text

    def chat(self, messages, tools=None):
        return {"content": self.text}


def test_reasons_are_checked():
    df = gap_table(ROWS, AREA)
    v = {"before_canopy": "green", "after_ground": "bare_soil", "confidence": 0.8, "reason": "dark soil",
         "hint": "silage-like", "before_date": "2022-09-10", "after_date": "2022-10-05",
         "status": "supported"}
    rec = evidence_record(df.iloc[3], v)
    assert rec["gap_days"] == 25 and rec["picture_status"] == "supported"
    assert numbers_ok(template_reason(rec), rec)
    good = "Cloud hid the field from 2022-09-10 to 2022-10-05, 25 days. Greenness fell from 0.83 to 0.27."
    assert explain(rec, _LLM(good)) == {"reason": good, "written_by": "nemotron"}
    bad = explain(rec, _LLM("About 60 percent of state silage was cut in that window."))
    assert bad["written_by"] == "template" and bad["rejected"] == "a number not in the record"
    assert explain(rec, _LLM(""))["rejected"] == "empty reply"
    assert explain(rec)["written_by"] == "template"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} passed")
