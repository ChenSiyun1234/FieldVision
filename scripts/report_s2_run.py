"""Check a finished extractor run and print (or append) its READY.txt line.

The line: path, fields, median usable dates per field, share of fields with at
least 8 usable dates between Jul 1 and Nov 30, source collections of usable rows,
scene failures (plausibility rejections counted separately) and timing.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


def summarize(run):
    manifest = json.loads((run / "manifest.json").read_text())
    coverage = pd.read_csv(run / "fields_coverage.csv")
    table = pd.read_parquet(run / "timeseries.parquet")
    params = manifest["parameters"]
    usable = table[(table.n_clear >= params["min_clear_pixels"]) & (table.clear_fraction >= params["min_clear_fraction"])]
    dates = pd.to_datetime(usable.date)
    late = usable[(dates >= f"{params['year']}-07-01") & (dates <= f"{params['year']}-11-30")]
    late_counts = late.groupby("field_id").date.nunique().reindex(coverage.field_id, fill_value=0)
    errors = Counter()
    scenes = set()
    for block in manifest["blocks"].values():
        scenes |= set(block.get("completed_scene_ids", []))
        for message in block.get("scene_errors", {}).values():
            errors["plausibility" if "sanity check" in message else
                   "legacy_flag" if "unreliable" in message or "ambiguous" in message else
                   # Upstream STAC items that lack band assets cannot be read on any retry.
                   "missing_asset" if message.startswith("KeyError") else "read/other"] += 1
    attempted = len(scenes) + sum(errors.values())
    sources = usable.collection.value_counts()
    return {
        "path": str(run), "status": manifest["status"], "fields": len(coverage),
        "fields_status": dict(Counter(coverage.status)),
        "small_fields": int(((coverage.n_pixels > 0) & (coverage.n_pixels < 20)).sum()),
        "min_pixels": int(coverage.n_pixels.min()),
        "median_usable_dates": float(coverage.usable_dates.median()),
        "share_ge8_jul_nov": float((late_counts >= 8).mean()),
        "sources": "; ".join(f"{name} {count / len(usable):.0%}" for name, count in sources.items()),
        "scene_block_pairs_ok": len(scenes), "scene_failures": dict(errors), "attempted": attempted,
        "started": manifest["started_at"], "ended": manifest["ended_at"],
        "failed_blocks": sum(b["status"] == "failed" for b in manifest["blocks"].values()),
        "invalid_fields": len(manifest.get("invalid_fields", {})),
    }


def usable_for_modelling(s):
    """Complete, or partial only because of scene errors a rerun cannot fix."""
    if s["status"] == "complete":
        return True
    return (s["status"] == "partial" and not s["failed_blocks"] and not s["invalid_fields"]
            and not s["scene_failures"].get("read/other"))


def line(s):
    failures = ", ".join(f"{k} {v}" for k, v in s["scene_failures"].items()) or "none"
    flag = ""
    lost = s["scene_failures"].get("plausibility", 0)
    if s["attempted"] and lost / s["attempted"] > 0.05:
        flag = f" WARNING: {lost / s['attempted']:.0%} of scene-block reads rejected by plausibility check."
    if s["status"] == "partial":
        flag += " (status partial only because of the permanent scene errors listed; all other scenes are in.)"
    started = datetime.fromisoformat(s["started"])
    ended = datetime.fromisoformat(s["ended"])
    hours = (ended - started).total_seconds() / 3600
    return (f"{ended.strftime('%Y-%m-%dT%H:%MZ')} {s['path']} status={s['status']} fields={s['fields']} "
            f"median_usable_dates={s['median_usable_dates']:g} share_ge8_usable_jul_nov={s['share_ge8_jul_nov']:.2f} "
            f"sources=[{s['sources']}] scene_failures=[{failures}] field_status={s['fields_status']} "
            f"fields_under_20px={s['small_fields']} min_pixels={s['min_pixels']} "
            f"run_hours={hours:.2f}{flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--append", type=Path, help="READY.txt to append to")
    args = parser.parse_args()
    summary = summarize(args.run)
    text = line(summary)
    print(text)
    if args.append:
        if not usable_for_modelling(summary):
            raise SystemExit("Run has failed blocks, invalid fields or retryable read errors; rerun first")
        with open(args.append, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
