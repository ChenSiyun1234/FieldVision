"""Summarize a completed extractor run and plot three field NDVI series."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fieldvision.s2.indices import add_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--field-ids", nargs="*", default=[])
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    coverage = pd.read_csv(args.run / "fields_coverage.csv")
    table = add_indices(pd.read_parquet(args.run / "timeseries.parquet"))
    params = manifest["parameters"]
    usable = table[(table.n_clear >= params["min_clear_pixels"]) &
                   (table.clear_fraction >= params["min_clear_fraction"])].copy()
    usable["date"] = pd.to_datetime(usable.date)
    catalogs = [json.loads(path.read_text()) for path in (args.run / "catalog").glob("*.json")]
    found = {item["id"] for catalog in catalogs for item in catalog["items"]}
    blocks = list(manifest["blocks"].values())
    durations = [block["wall_seconds"] for block in blocks if "wall_seconds" in block]
    summary = {
        "status": manifest["status"], "fields": len(coverage), "blocks": len(blocks),
        "scenes_found_unique": len(found), "scene_block_pairs": sum(len(c["items"]) for c in catalogs),
        "scene_failures": manifest["scene_failures"], "rows": len(table),
        "usable_dates_min_median_max": [int(coverage.usable_dates.min()), float(coverage.usable_dates.median()), int(coverage.usable_dates.max())],
        "largest_gap_days_min_median_max": [int(coverage.largest_gap_days.min()), float(coverage.largest_gap_days.median()), int(coverage.largest_gap_days.max())],
        "wall_seconds": manifest["attempt_wall_seconds"],
        "estimated_50_block_seconds": float(np.mean(durations) * 50) if durations else None,
        "band_ranges": {band: [float(table[band].min()), float(table[band].max())]
                        for band in ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")},
        "field_examples": {},
    }
    field_ids = args.field_ids or sorted(usable.field_id.unique())[:3]
    figure, axes = plt.subplots(len(field_ids), 1, figsize=(11, 3 * len(field_ids)), squeeze=False, sharex=True)
    for field_id, axis in zip(field_ids, axes[:, 0]):
        series = usable[usable.field_id == field_id].sort_values("date")
        axis.plot(series.date, series.NDVI, marker="o", linewidth=1)
        axis.set(title=field_id, ylabel="NDVI", ylim=(-0.1, 1))
        axis.grid(alpha=0.3)
        series.to_csv(args.run / f"ndvi_{field_id}.csv", index=False)
        summary["field_examples"][field_id] = [
            {"date": row.date.date().isoformat(), "NDVI": round(float(row.NDVI), 4),
             "B04": round(float(row.B04), 4), "B08": round(float(row.B08), 4), "n_clear": int(row.n_clear)}
            for row in series.itertuples()]
    figure.tight_layout()
    figure.savefig(args.run / "ndvi_examples.png", dpi=150)
    plt.close(figure)
    (args.run / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("field_examples", "band_ranges")}, indent=2))


if __name__ == "__main__":
    main()
