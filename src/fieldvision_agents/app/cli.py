"""fieldvision: run the silage/grain agent pipeline from a terminal, or serve the web console.

    fieldvision doctor                         what this machine can run
    fieldvision counties                       county-seasons with data on disk
    fieldvision run 24021 2022 [--offline]     one county-season through the agent; prints the result
    fieldvision runs                           past runs, newest first
    fieldvision report RUN_ID [--offline]      the county report (Nemotron writer + editor, or a template)
    fieldvision ask RUN_ID "question" [--offline]
    fieldvision test                           every offline test (no network, no key)
    fieldvision serve [--port 8765]            the web console at http://127.0.0.1:8765

Without NVIDIA_API_KEY (in the environment or .env), --offline is implied: a rule-based agent runs the
same tools under the same guardrails, and reports and answers are laid out from the run record.
"""

import argparse
import json
import sys

from . import service


def _offline(args):
    if args.offline:
        return True
    if not service.has_key():
        print("No NVIDIA_API_KEY found: running offline.", file=sys.stderr)
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fieldvision", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="what this machine can run")
    sub.add_parser("counties", help="county-seasons with data on disk")
    p = sub.add_parser("run", help="run one county-season through the agent")
    p.add_argument("fips")
    p.add_argument("year", type=int)
    p.add_argument("--offline", action="store_true", help="rule-based agent, no language model")
    p.add_argument("--recommend", action="store_true", help="let the agent call recommend_sources (online only)")
    p.add_argument("--report", action="store_true", help="also write the county report")
    sub.add_parser("runs", help="past runs, newest first")
    p = sub.add_parser("report", help="write the county report for a run")
    p.add_argument("run_id")
    p.add_argument("--offline", action="store_true")
    p = sub.add_parser("ask", help="ask a question about a run")
    p.add_argument("run_id")
    p.add_argument("question")
    p.add_argument("--offline", action="store_true")
    sub.add_parser("test", help="every offline test")
    p = sub.add_parser("serve", help="the web console")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    if args.cmd == "doctor":
        for name, s in service.status().items():
            print(f"{'ok ' if s['ok'] else '-- '} {name:14s} {s.get('note') or s.get('mode') or s.get('path') or ''}"
                  + (f" ({s['n']})" if 'n' in s else "") + (f", {s['mode']}" if name == "report_search" else ""))
        return 0
    if args.cmd == "counties":
        for c in service.counties():
            print(f"{c['fips']} {c['year']}  {c['name']}" + ("  (partial)" if c["partial"] else ""))
        return 0
    if args.cmd == "run":
        offline = _offline(args)
        result = service.run(args.fips, args.year, offline=offline, recommend=args.recommend and not offline)
        print(json.dumps(result, indent=1, default=str))
        if args.report:
            print(service.report(result["run_id"], offline)["report_md"])
        return 0 if result["status"] != "incomplete" else 1
    if args.cmd == "runs":
        for r in service.runs():
            print(f"{r['when']}  {r['run_id']:22s} {r['status']:24s} {' → '.join(r['policies'])}")
        return 0
    if args.cmd == "report":
        rep = service.report(args.run_id, offline=_offline(args))
        print(rep["report_md"])
        print(f"\nsaved to {rep['path']}", file=sys.stderr)
        return 0
    if args.cmd == "ask":
        a = service.ask(args.run_id, args.question, offline=_offline(args))
        print(a["answer"])
        print(f"\nsections: {', '.join(a['sections_used'])}; every number checked: {a['grounded']}", file=sys.stderr)
        return 0
    if args.cmd == "test":
        from ..run_tests import run_all
        return 0 if run_all() else 1
    if args.cmd == "serve":
        from .server import serve
        serve(args.host, args.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
