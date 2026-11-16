"""Run the top silage counties through the pipeline in one batch, on the pod (not the Mac).

    python scripts/pod_batch_counties.py list --top 50                          # writes runs/batch/counties.csv
    python scripts/pod_batch_counties.py run <extract_root> 2023 [--nemotron]   # convert, run, collect

The satellite pull is the slow part and stays in Earth Engine / Colab: one extraction folder per county under
<extract_root>, named <fips> or <ST>_<fips> (fields.gpkg + s2_timeseries.csv, as for Lancaster). This script
picks the counties, converts each finished extraction with colab_to_county.py, runs the agent (offline rule
agent by default; --nemotron uses the key in .env, about 30,000 tokens a county), and appends one row per
county to runs/batch/results.csv. Counties already in the results file are skipped, so it can be re-run
as more extractions land. A county that fails is logged and the batch moves on.

Model estimates, not NASS statistics.
"""
import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from fieldvision_agents.settings import COUNTY_DATA, RUNS  # noqa: E402

OUT = RUNS / "batch"
COLS = ["fips", "name", "year", "status", "our_silage_pct", "census_2022_pct", "fields", "not_judged_pct",
        "critic", "run_id", "seconds", "error"]


def top_counties(n):
    from fieldvision_agents.app import geo
    a = geo._acres()
    c = a[(a["level"] == "county") & (a["source"] == "CENSUS") & (a["year"] == 2022)].dropna(subset=["silage"])
    c = c.sort_values("silage", ascending=False).head(n)
    total = a[(a["level"] == "county") & (a["source"] == "CENSUS") & (a["year"] == 2022)]["silage"].sum()
    rows = [{"fips": r["fips"], "name": f"{str(r['county_name']).title()}, {geo.STATE_NAMES.get(r['state'], r['state'])}",
             "census_2022_silage_acres": round(r["silage"]), "census_2022_pct": round(r["silage_share_pct"], 1)}
            for _, r in c.iterrows()]
    return rows, round(100 * c["silage"].sum() / total, 1)


def find_extract(root, fips):
    root = Path(root)
    for p in [root / fips, *sorted(root.glob(f"*_{fips}"))]:
        if (p / "fields.gpkg").is_file() and (p / "s2_timeseries.csv").is_file():
            return p
    return None


def run_one(fips, name, year, extract_root, nemotron, census_pct):
    from colab_to_county import convert
    from fieldvision_agents.app import service
    t0 = time.perf_counter()
    row = {"fips": fips, "name": name, "year": year, "census_2022_pct": census_pct}
    if not (COUNTY_DATA / fips / str(year) / "timeseries.parquet").is_file():
        src = find_extract(extract_root, fips)
        if src is None:
            return None                                    # extraction not there yet: try again next time
        convert(src, year, COUNTY_DATA)
    r = service.run(fips, year, offline=not nemotron)
    est = (service.detail(r["run_id"]).get("evidence") or {}).get("estimate") or {}
    s, g, a = (float(est.get(k) or 0) for k in ("silage_acres", "grain_acres", "abstained_acres"))
    row.update(status=r["status"], run_id=r["run_id"], critic=" → ".join(r.get("critic") or []),
               our_silage_pct=round(100 * s / (s + g), 1) if s + g else None,
               not_judged_pct=round(100 * a / (s + g + a), 1) if s + g + a else None,
               fields=est.get("n_fields"), seconds=round(time.perf_counter() - t0, 1))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--top", type=int, default=50)
    p = sub.add_parser("run")
    p.add_argument("extract_root")
    p.add_argument("year", type=int)
    p.add_argument("--nemotron", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    counties = OUT / "counties.csv"

    if a.cmd == "list":
        rows, share = top_counties(a.top)
        with counties.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{counties}: top {len(rows)} counties, {share}% of reported county silage acres (2022 Census)")
        return

    todo = list(csv.DictReader(counties.open()))
    results = OUT / "results.csv"
    done = {(r["fips"], r["year"]) for r in csv.DictReader(results.open())} if results.is_file() else set()
    new = not results.is_file()
    with results.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        waiting = 0
        for c in todo:
            if (c["fips"], str(a.year)) in done:
                continue
            try:
                row = run_one(c["fips"], c["name"], a.year, a.extract_root, a.nemotron, c["census_2022_pct"])
            except Exception as e:                         # one bad county must not stop the batch
                row = {"fips": c["fips"], "name": c["name"], "year": a.year, "status": "error", "error": repr(e)[:300]}
                (OUT / f"error_{c['fips']}.txt").write_text(traceback.format_exc())
            if row is None:
                waiting += 1
                continue
            w.writerow(row)
            f.flush()
            print(f"{row['fips']} {row['name']}: {row['status']} {row.get('our_silage_pct')}% "
                  f"(Census 2022 {row.get('census_2022_pct')}%)")
    print(f"{results}; {waiting} counties still waiting for their extraction")


if __name__ == "__main__":
    main()
