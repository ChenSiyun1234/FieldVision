"""What the command line and the web console both call: one place that wires the pipeline together.

    toolbox = FieldVisionToolbox(critic=RAG critic, reports=NASS report search)    code: the numbers
    agent   = NemotronClient()  or  RuleAgent() offline                            routes, checks, explains
    run_county(fips, year, toolbox, agent)  ->  audit log in runs/logs
    write_report / ask  (or template_report / evidence_answer offline)  ->  runs/reports
"""

import json
import os
import time
from functools import lru_cache
from pathlib import Path

from ..reporter.reporter import ask as nemotron_ask
from ..reporter.reporter import build_evidence, load_audit, result_from_log, write_report
from ..react.supervisor import DEFAULT_MODEL, NemotronClient, run_county
from ..settings import COUNTY_DATA, LOGS, QUICKSTATS_CSV, RAG_PIPELINE, REGISTRY, REPORTS, RECOMMENDER
from .offline import RuleAgent, evidence_answer, template_report

# Counties with field data in the team's hand-over, and the ones the analyst screen offered (FIPS -> name).
COUNTY_NAMES = {
    "24033": "Prince George's County, MD (Beltsville)", "24021": "Frederick County, MD",
    "24035": "Queen Anne's County, MD", "24029": "Kent County, MD", "24013": "Carroll County, MD",
    "24043": "Washington County, MD", "24025": "Harford County, MD",
    "26023": "Branch County, MI", "26057": "Gratiot County, MI",
    "42071": "Lancaster County, PA", "55027": "Dodge County, WI",
}


def has_key():
    return bool(os.environ.get("NVIDIA_API_KEY"))


def counties(root=COUNTY_DATA):
    """County-seasons with field outlines and a Sentinel-2 time series on disk."""
    out = []
    for fields in sorted(Path(root).glob("*/*/fields.geojson")):
        folder = fields.parent
        if (folder / "timeseries.parquet").is_file():
            fips, year = folder.parent.name, int(folder.name)
            out.append({"fips": fips, "year": year, "name": COUNTY_NAMES.get(fips, fips),
                        "partial": (folder / "PARTIAL.json").is_file()})
    return out


def status():
    """What this machine can run, for the console header and `fieldvision doctor`."""
    from ..fieldvision_bridge.rag import embeddings_available
    index = RAG_PIPELINE / "rag_index" / "chunks.jsonl"
    return {
        "nemotron": {"ok": has_key(), "model": DEFAULT_MODEL,
                     "note": "NVIDIA_API_KEY found" if has_key() else "no NVIDIA_API_KEY: offline mode only"},
        "report_search": {"ok": index.is_file(),
                          "mode": "embeddings" if embeddings_available() else "keyword",
                          "note": "index present" if index.is_file() else f"missing {index}"},
        "quickstats": {"ok": Path(QUICKSTATS_CSV).is_file(), "path": str(QUICKSTATS_CSV)},
        "registry": {"ok": (REGISTRY / "eu2_logistic_v1" / "model.json").is_file(), "path": str(REGISTRY)},
        "county_data": {"ok": bool(counties()), "path": str(COUNTY_DATA), "n": len(counties())},
        "recommender": {"ok": (RECOMMENDER / "silage_match.py").is_file()},
    }


@lru_cache(maxsize=4)
def toolbox(offline=False, recommend=False):
    """The real tools: final model module, RAG critic (Nemotron explains only when online), report search."""
    from ..fieldvision_bridge import FieldVisionToolbox, make_rag_critic, make_report_search
    tb = FieldVisionToolbox(critic=make_rag_critic(use_llm=not offline), reports=make_report_search())
    if recommend:
        import sys
        sys.path.insert(0, str(RECOMMENDER))
        from silage_match import SilageMatchToolbox
        tb = SilageMatchToolbox(tb)
    return tb


def agent(offline=False):
    if offline:
        return RuleAgent()
    if not has_key():
        raise RuntimeError("No NVIDIA_API_KEY in the environment or .env. Add it, or run offline.")
    return NemotronClient()


def run(fips, year, offline=False, recommend=False, log_dir=LOGS):
    t0 = time.perf_counter()
    result = run_county(str(fips), int(year), toolbox(offline, recommend), agent(offline), log_dir=log_dir)
    result["seconds"] = round(time.perf_counter() - t0, 1)
    result["mode"] = "offline" if offline else "nemotron"
    return result


def runs(log_dir=LOGS, limit=50):
    """Past runs, newest first."""
    out = []
    for path in sorted(Path(log_dir).glob("audit_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            r = result_from_log(path)
        except (StopIteration, json.JSONDecodeError, KeyError):
            continue                                          # unfinished or damaged log
        fips = r["run_id"].split("-")[0]
        out.append({"run_id": r["run_id"], "status": r["status"], "fips": fips,
                    "year": int(r["run_id"].split("-")[1]) if r["run_id"].count("-") >= 2 else None,
                    "name": COUNTY_NAMES.get(fips, fips), "policies": r.get("policies", []),
                    "critic": r.get("critic", []), "offline": "offline" in (r.get("summary") or "").lower(),
                    "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))})
    return out


def load(run_id, log_dir=LOGS):
    path = Path(log_dir) / f"audit_{run_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No run {run_id} in {log_dir}")
    return result_from_log(path)


def detail(run_id, log_dir=LOGS):
    """Everything the console shows for one run: result, evidence pack, steps, saved report."""
    result = load(run_id, log_dir)
    steps = [{"tool": s["tool"], "args": {k: v for k, v in (s.get("args") or {}).items() if k not in ("fips", "year")},
              "result": s.get("result")} for s in load_audit(result["audit_log"]) if s.get("kind") == "tool"]
    reasons = [s.get("reason") for s in load_audit(result["audit_log"]) if s.get("kind") == "llm"]
    report = Path(REPORTS) / f"report_{run_id}.md"
    return {"result": result, "evidence": build_evidence(result), "steps": steps, "reasons": reasons,
            "name": COUNTY_NAMES.get(result["run_id"].split("-")[0], ""),
            "report_md": report.read_text() if report.is_file() else None}


def report(run_id, offline=False):
    result = load(run_id)
    if offline:
        return template_report(result)
    if not has_key():
        raise RuntimeError("No NVIDIA_API_KEY: write the report offline instead.")
    return write_report(result)


def ask(run_id, question, offline=False):
    result = load(run_id)
    if offline:
        return evidence_answer(result, question)
    if not has_key():
        raise RuntimeError("No NVIDIA_API_KEY: ask offline instead.")
    return nemotron_ask(result, question)
