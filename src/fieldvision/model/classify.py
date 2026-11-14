"""Classify fields with one frozen registry model, in the team's result shape.

    python -m fieldvision.model.classify --registry C:\\t\\fv\\data\\registry --model ts_model_v1 \
        --timeseries <...> --fields <...> --state MD --out C:\\t\\fv\\data\\results

One result per field and year:
    {"field_id": "BARC-112", "year": 2023, "model": "timing_rule_v1", "p_silage": 0.91,
     "label": "silage", "abstain": false,
     "evidence": {"drop_date": "2023-08-29", "ndvi_before": 0.81, "ndvi_after": 0.24,
                  "days_to_drop": 6, ...}}

Written to <out>/<state>_<year>.parquet (evidence as a JSON string column plus flat columns)
and <out>/<state>_<year>.jsonl. Labels in the field file are never read here. Numbers come
only from the frozen model and the feature code; the abstain rate is printed and stored.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from .artifacts import predictions_frame, write_json
from .registry import DEFAULT_REGISTRY, load

BARC_ID = re.compile(r"^barc-(\d+)-(\d{4})$", re.IGNORECASE)


def team_field_id(field_id):
    """barc-112-2023 -> BARC-112; anything else unchanged."""
    m = BARC_ID.match(str(field_id))
    return f"BARC-{m.group(1)}" if m else str(field_id)


def _num(v, digits=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def _date(v):
    return None if pd.isna(v) else pd.Timestamp(v).strftime("%Y-%m-%d")


def results(fields, ts, model, name, cfg, threshold):
    X, info = build_features(ts, fields, cfg)
    p = model.predict_proba(X)
    pf = predictions_frame(X, fields.drop(columns=["label", "y"], errors="ignore"), p, threshold, name)
    # composite models also abstain when their members disagree strongly
    disagree = model.disagreement(X) if hasattr(model, "disagreement") else np.zeros(len(X), bool)
    member_p = model.member_proba(X) if hasattr(model, "member_proba") else None
    rows = []
    for i, r in pf.iterrows():
        x = X.iloc[i]
        start, end = r["harvest_start"], r["harvest_end"]
        ev = {
            "drop_date": _date(end),
            "last_standing_date": _date(start),
            "days_to_drop": _num(r["harvest_interval_days"], 0),
            "ndvi_before": _num(x["pre_ndvi"]),
            "ndvi_after": _num(x["post_ndvi"]),
            "ndmi_before": _num(x["pre_ndmi"]),
            "greenness_before_share_of_peak": _num(x.get("pre_ndvi_over_peak")),
            "days_peak_to_drop": _num(x["harv_days_since_peak"], 1),
            "residue_ndti_after": _num(x["post_ndti"]),
            "n_usable_dates": int(r["n_usable_dates"]),
            "largest_gap_days": _num(r["max_gap_days"], 0),
            "abstain_reason": r["low_evidence_reason"] or None,
        }
        if member_p is not None:
            ev["members_p_silage"] = {k: _num(v, 3) for k, v in member_p.iloc[i].items()}
            if disagree[i]:
                ev["abstain_reason"] = "+".join(filter(None, [ev["abstain_reason"],
                                                              "members_disagree"]))
        year = r.get("year")
        rows.append({"field_id": team_field_id(r["field_id"]),
                     "year": int(year) if pd.notna(year) else None,
                     "source_field_id": r["field_id"], "model": name,
                     "p_silage": _num(r["p_silage"], 4), "label": r["predicted_label"],
                     "abstain": bool(r["low_evidence"] or disagree[i]), "evidence": ev})
    return rows, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Classify fields with a frozen registry model.")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY),
                    help="registry folder (default: the repo's registry/)")
    ap.add_argument("--model", required=True, help="registry folder name, e.g. ts_model_v1")
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--state", required=True, help="two-letter state code for the output file name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model, mj, cfg = load(a.registry, a.model)
    fields = load_fields(a.fields, keep_columns=["OBJECTID"])
    ts, _ = load_timeseries(a.timeseries, field_ids=fields["field_id"])
    rows, info = results(fields, ts, model, a.model, cfg, mj["threshold"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    flat = pd.json_normalize(df["evidence"]).add_prefix("evidence_")
    table = pd.concat([df.drop(columns="evidence"), flat], axis=1)
    table["evidence"] = [json.dumps(e) for e in df["evidence"]]
    summary = {}
    for year, g in table.groupby("year", dropna=False):
        stem = f"{a.state}_{int(year) if pd.notna(year) else 'unknown'}"
        # one file per state and year holds every model's results; replace only this model's rows
        path = out / f"{stem}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            g = pd.concat([old[old["model"] != a.model], g], ignore_index=True)
        g.to_parquet(path, index=False)
        lines = []
        jpath = out / f"{stem}.jsonl"
        if jpath.exists():
            lines = [ln for ln in jpath.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and json.loads(ln).get("model") != a.model]
        lines += [json.dumps(rec) for rec in df[df["year"] == year].to_dict("records")]
        jpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        g = g[g["model"] == a.model]
        summary[stem] = {"n": int(len(g)), "abstain_rate": float(g["abstain"].mean()),
                         "n_silage": int((g["label"] == "silage").sum())}
    write_json(out / f"{a.state}_{a.model}_summary.json",
               {"model": a.model, "files": summary,
                "seconds_per_1000_fields": info["seconds_per_1000_fields"]})
    for k, s in summary.items():
        print(f"{k}: {s['n']} fields, {s['n_silage']} called silage, abstain rate {s['abstain_rate']:.2f}")
    return table


if __name__ == "__main__":
    main()
