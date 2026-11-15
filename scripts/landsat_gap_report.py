"""Deliver a merged Landsat run next to its county and report what it adds in a window.

Copies <timeseries>/ls_<fips>_<year>/timeseries.parquet to
<counties>/<fips>/<year>/timeseries_landsat.parquet (the Sentinel-2 timeseries.parquet
there is not touched), then counts, per field, the looks inside --from/--to from
Sentinel-2 alone and from Sentinel-2 plus Landsat (one look per calendar date), at two
thresholds: usable (clear_fraction >= 0.2, the pipeline's rule) and fully clear (>= 0.9),
and the largest gap between looks inside the window (window edges included).
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def looks(table, lo, hi, threshold):
    t = table[(table.n_clear >= 1) & (table.clear_fraction >= threshold)].copy()
    t["date"] = pd.to_datetime(t.date)
    t = t[(t.date >= lo) & (t.date <= hi)]
    return t.groupby("field_id").date.apply(lambda s: sorted(set(s)))


def gap(dates, lo, hi):
    edges = [lo, *dates, hi]
    return max((b - a).days for a, b in zip(edges[:-1], edges[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fips", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--from", dest="lo", required=True, help="MM-DD")
    parser.add_argument("--to", dest="hi", required=True, help="MM-DD")
    parser.add_argument("--timeseries", type=Path, default=Path(r"C:\t\fv\data\timeseries"))
    parser.add_argument("--counties", type=Path, default=Path(r"C:\t\fv\data\counties"))
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()
    run = args.timeseries / f"ls_{args.fips}_{args.year}"
    landsat = pd.read_parquet(run / "timeseries.parquet")
    target = args.counties / args.fips / str(args.year) / "timeseries_landsat.parquet"
    if not args.no_copy:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / "timeseries.parquet", target)
    s2 = pd.read_parquet(args.timeseries / f"us_{args.fips}_{args.year}" / "timeseries.parquet")
    fields = sorted(set(s2.field_id) | set(landsat.field_id))
    lo, hi = pd.Timestamp(f"{args.year}-{args.lo}"), pd.Timestamp(f"{args.year}-{args.hi}")
    both = pd.concat([s2, landsat], ignore_index=True)
    report = {"run": run.name, "delivered": str(target) if not args.no_copy else None, "window": [str(lo.date()), str(hi.date())],
              "fields": len(fields), "landsat_rows": len(landsat), "landsat_scenes": int(landsat.scene_id.nunique())}
    for name, threshold in (("usable", 0.2), ("fully_clear", 0.9)):
        a, b = looks(s2, lo, hi, threshold), looks(both, lo, hi, threshold)
        na = np.array([len(a.get(f, [])) for f in fields])
        nb = np.array([len(b.get(f, [])) for f in fields])
        ga = np.array([gap(a.get(f, []), lo, hi) for f in fields])
        gb = np.array([gap(b.get(f, []), lo, hi) for f in fields])
        report[name] = {
            "looks_per_field_median_s2": float(np.median(na)), "looks_per_field_median_both": float(np.median(nb)),
            "fields_with_no_look_s2": int((na == 0).sum()), "fields_with_no_look_both": int((nb == 0).sum()),
            "largest_gap_median_s2": float(np.median(ga)), "largest_gap_median_both": float(np.median(gb)),
            "fields_gap_over_14d_s2": int((ga > 14).sum()), "fields_gap_over_14d_both": int((gb > 14).sum()),
            "fields_gaining_a_look": int((nb > na).sum()),
            "s2_look_dates": sorted({d.date().isoformat() for v in a for d in v}),
            "landsat_look_dates": sorted({d for d in pd.to_datetime(landsat[(landsat.clear_fraction >= threshold)
                                          & (landsat.n_clear >= 1)].date).dt.date.astype(str) if lo.date().isoformat() <= d <= hi.date().isoformat()}),
        }
    out = run / "landsat_gap_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
