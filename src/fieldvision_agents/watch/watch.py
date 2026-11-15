"""watch: follow a county through the season in 5-day steps, as imagery arrives.

Sentinel-2 passes every 5 days. At each step the code looks only at images taken up to that date,
decides whether enough new clear looks arrived to be worth re-classifying, and reports what is known
so far as a range:

    at least   silage found so far, as a share of all corn acres
    at most    that, plus all corn not judged yet (still standing, or hidden by cloud)

The range narrows as fields are harvested and seen. Nothing here uses a USDA figure; one can be drawn
on the chart for comparison. Two modes:

    replay   python -m fieldvision_agents.watch.watch 36121 2022 --data <counties dir> --out <dir>
             a past season, cut off at each step (the time series on disk is truncated, nothing is fetched)
    poll     python -m fieldvision_agents.watch.watch 24021 2026 --data <counties dir> --poll 2026-11-01
             asks the public catalog which Sentinel-2 scenes over the county are newer than a date

Rules fixed in advance: step 5 days; a usable look has clear_fraction >= 0.8; re-classify when at
least 5% of fields got a new usable look since the last classification; call the agent again when the
"at least" figure moved by 1 point or more since the agent last ran.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..gap.gap import gap_table

STEP_DAYS = 5
CLEAR = 0.8
MIN_NEW_LOOK_SHARE = 0.05
AGENT_MOVE_POINTS = 1.0


def cutoffs(start, end, step=STEP_DAYS):
    """Dates from start to end every step days, the end date always included."""
    d = list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{int(step)}D"))
    if not d or d[-1] != pd.Timestamp(end):
        d.append(pd.Timestamp(end))
    return [x.strftime("%Y-%m-%d") for x in d]


def new_look_share(ts, after, upto, n_fields, clear=CLEAR):
    """Share of fields with at least one usable look in (after, upto]."""
    d = pd.to_datetime(ts["date"])
    m = (ts["clear_fraction"] >= clear) & (d <= pd.Timestamp(upto))
    if after is not None:
        m &= d > pd.Timestamp(after)
    return float(ts.loc[m, "field_id"].nunique()) / n_fields if n_fields else 0.0


def decide(new_share, min_new=MIN_NEW_LOOK_SHARE):
    return "classify" if new_share >= min_new else "skip"


def step_metrics(df):
    """What is known at one step, in percent of all corn acres."""
    a = df["acres"].to_numpy(float)
    total = float(np.nansum(a))
    judged = (~df["abstain"]).to_numpy() & (df["kind"] != "unseen").to_numpy()
    sil = judged & (df["call"] == "silage").to_numpy()
    gr = judged & (df["call"] == "grain").to_numpy()
    pct = lambda m: round(100 * float(np.nansum(a[m])) / total, 1) if total else None
    low = pct(sil)
    return {"silage_at_least_pct": low, "grain_found_pct": pct(gr), "not_judged_pct": pct(~judged),
            "silage_at_most_pct": None if low is None else round(low + pct(~judged), 1),
            "fields_judged": int(judged.sum())}


def first_within(rows, width):
    """First date from which the range stays no wider than width points to the end."""
    out = None
    for r in rows:
        w = r["silage_at_most_pct"] - r["silage_at_least_pct"]
        if w <= width:
            out = out or r["date"]
        else:
            out = None
    return out


def truncate_county(src, dst, cutoff):
    """A copy of one county-season folder whose time series stops at the cutoff date."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name.startswith("timeseries") and f.suffix == ".parquet":
            ts = pd.read_parquet(f)
            ts[pd.to_datetime(ts["date"]) <= pd.Timestamp(cutoff)].to_parquet(dst / f.name, index=False)
        elif f.is_file():
            shutil.copy(f, dst / f.name)
    return dst


def replay(fips, year, data, out, registry="registry", start=None, end=None, step=STEP_DAYS, log=print):
    """Run the season step by step. Returns the list of step records (also written to <out>/replay.json)."""
    from fieldvision.toolbox import Toolbox
    src = Path(data) / str(fips) / str(year)
    ts = pd.read_parquet(src / "timeseries.parquet", columns=["field_id", "date", "clear_fraction"])
    n_fields = ts["field_id"].nunique()
    start = start or f"{year}-08-01"
    end = end or str(pd.to_datetime(ts["date"]).max().date())
    out = Path(out)
    work = out / "work"
    rows, last_classified, last, agent_at = [], None, None, None
    for c in cutoffs(start, end, step):
        new = new_look_share(ts, last_classified, c, n_fields)
        action = decide(new) if last is not None else "classify"
        t0 = time.perf_counter()
        if action == "classify":
            root = work / c
            truncate_county(src, root / str(fips) / str(year), c)
            try:
                tb = Toolbox(data_root=str(root), registry=registry, out_root=str(root / "results"))
                rws, _, _ = tb._policy_predictions(str(fips), int(year), "module_v2")
                area = tb._data(str(fips), int(year))[0].set_index("field_id")["area_ha"].astype(float).to_dict()
                last = step_metrics(gap_table(rws, area))
                last_classified = c
            except Exception as e:   # too little imagery this early: nothing can be said yet
                action = f"skip (not enough imagery: {str(e)[:60]})"
            shutil.rmtree(root, ignore_errors=True)
        if last is None:
            continue
        low = last["silage_at_least_pct"]
        call_agent = agent_at is None or abs(low - agent_at) >= AGENT_MOVE_POINTS
        if call_agent:
            agent_at = low
        rows.append({"date": c, "new_look_share_pct": round(100 * new, 1), "action": action,
                     "call_agent": bool(call_agent), "seconds": round(time.perf_counter() - t0, 1), **last})
        log(f"{c} new looks {100 * new:5.1f}%  {action:9s} agent {'yes' if call_agent else 'no '}  "
            f"silage {low} to {last['silage_at_most_pct']}%")
    shutil.rmtree(work, ignore_errors=True)
    summary = {
        "fips": str(fips), "year": int(year), "fields": int(n_fields), "steps": len(rows),
        "classified_steps": sum(r["action"] == "classify" for r in rows),
        "skipped_steps": sum(r["action"] != "classify" for r in rows),
        "agent_calls": sum(r["call_agent"] for r in rows),
        "classify_seconds_total": round(sum(r["seconds"] for r in rows), 1),
        "range_within_20_points_from": first_within(rows, 20), "range_within_10_points_from": first_within(rows, 10),
        "final_at_least_pct": rows[-1]["silage_at_least_pct"], "final_at_most_pct": rows[-1]["silage_at_most_pct"],
        "note": "Model estimates from satellite data. Not a NASS statistic. No USDA figure was used.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.json").write_text(json.dumps({"summary": summary, "steps": rows}, indent=1))
    pd.DataFrame(rows).to_csv(out / "replay.csv", index=False)
    return rows, summary


def chart(rows, summary, path, title=None, reference=None):
    """The range by date as one band. reference: (label, percent) drawn as a grey line, for comparison only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    blue, ink, muted, surface = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"
    d = pd.to_datetime([r["date"] for r in rows])
    lo = np.array([r["silage_at_least_pct"] for r in rows], float)
    hi = np.array([r["silage_at_most_pct"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.fill_between(d, lo, hi, color=blue, alpha=0.18, linewidth=0)
    ax.plot(d, hi, color=blue, linewidth=2, alpha=0.55)
    ax.plot(d, lo, color=blue, linewidth=2)
    agent = [i for i, r in enumerate(rows) if r["call_agent"]]
    ax.scatter(d[agent], lo[agent], s=36, color=blue, edgecolor=surface, linewidth=1.5, zorder=3)
    ax.annotate("at most (plus all corn not judged yet)", (d[1], hi[1]), xytext=(4, -14),
                textcoords="offset points", color=ink, fontsize=9, ha="left")
    ax.annotate("at least (silage found so far)", (d[-1], lo[-1]), xytext=(-6, -30), textcoords="offset points",
                color=ink, fontsize=9, ha="right")
    if reference:
        ax.axhline(reference[1], color=muted, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"{reference[0]} {reference[1]}%", (d[0], reference[1]), xytext=(4, 5),
                    textcoords="offset points", color=muted, fontsize=9)
    for w, key, y, side in ((20, "range_within_20_points_from", 70, -1), (10, "range_within_10_points_from", 56, 1)):
        if summary.get(key):
            x = pd.Timestamp(summary[key])
            ax.axvline(x, color=muted, linewidth=0.8, alpha=0.5)
            ax.annotate(f"range within {w} points\nfrom {x.strftime('%b %d').replace(' 0', ' ')}", (x, y),
                        xytext=(4 * side, 0), textcoords="offset points", color=muted, fontsize=8, va="top",
                        ha="left" if side > 0 else "right")
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Silage share of corn acres (%)", color=muted, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(colors=muted, labelsize=8, length=0)
    ax.grid(axis="y", color=muted, alpha=0.15, linewidth=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title or f"County {summary['fips']}, {summary['year']}: what we know, by date", loc="left",
                 color=ink, fontsize=12, fontweight="bold")
    fig.text(0.125, 0.01, "Dots: steps where the estimate moved a point or more and the agent would run again. "
             "Model estimates, not NASS statistics.", color=muted, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return path


def poll(fields_geojson, since, until=None):
    """Sentinel-2 scenes over the county's fields newer than a date, from the public catalog."""
    import geopandas as gpd
    import pystac_client
    box = gpd.read_file(fields_geojson).to_crs("EPSG:4326").total_bounds
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    rng = f"{since}/{until}" if until else f"{since}/.."
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=list(box), datetime=rng).items())
    days = sorted({str(i.datetime.date()) for i in items})
    cloud = [i.properties.get("eo:cloud_cover") for i in items if i.properties.get("eo:cloud_cover") is not None]
    return {"since": since, "scenes": len(items), "days_with_scenes": days,
            "median_scene_cloud_pct": None if not cloud else round(float(np.median(cloud)), 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Follow a county through the season in 5-day steps")
    ap.add_argument("fips")
    ap.add_argument("year", type=int)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/watch")
    ap.add_argument("--registry", default="registry")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--step", type=int, default=STEP_DAYS)
    ap.add_argument("--reference", nargs=2, metavar=("LABEL", "PCT"), help="comparison line for the chart only")
    ap.add_argument("--poll", metavar="SINCE", help="only list scenes newer than this date")
    a = ap.parse_args(argv)
    if a.poll:
        folder = Path(a.data) / a.fips
        years = sorted(p.name for p in folder.iterdir() if (p / "fields.geojson").exists())
        print(json.dumps(poll(folder / years[-1] / "fields.geojson", a.poll), indent=1))
        return
    out = Path(a.out) / f"{a.fips}_{a.year}"
    rows, summary = replay(a.fips, a.year, a.data, out, a.registry, a.start, a.end, a.step)
    ref = (a.reference[0], float(a.reference[1])) if a.reference else None
    chart(rows, summary, out / "replay.png", reference=ref)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
