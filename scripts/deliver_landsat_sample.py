"""Put the Landsat rows of a state sample (three parts) next to its optical series, split by stratum.

Reads <timeseries>/ls_<st>_<year>_s1..sN/timeseries.parquet, checks each part's run status
and that its fields are disjoint from the other parts, then splits the rows by the
``stratum`` of each field in <counties>/<st>_<year>_sample/fields.geojson, like the
optical files there: stratum 'corn' -> timeseries_landsat.parquet, every other stratum ->
timeseries_landsat_dblcrop.parquet. Prints completeness only (rows, scenes, fields,
failures): these are test places, so the contents are not looked at.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--parts", type=int, default=3)
    parser.add_argument("--timeseries", type=Path, default=Path(r"C:\t\fv\data\timeseries"))
    parser.add_argument("--counties", type=Path, default=Path(r"C:\t\fv\data\counties"))
    args = parser.parse_args()
    sample = args.counties / f"{args.state}_{args.year}_sample"
    strata = {f["properties"]["field_id"]: f["properties"].get("stratum", "corn")
              for f in json.loads((sample / "fields.geojson").read_text(encoding="utf-8"))["features"]}
    tables, seen, report = [], set(), {"parts": {}}
    for k in range(1, args.parts + 1):
        run = args.timeseries / f"ls_{args.state}_{args.year}_s{k}"
        manifest = json.loads((run / "manifest.json").read_text())
        errors = Counter()
        for block in manifest["blocks"].values():
            for message in block.get("scene_errors", {}).values():
                errors["plausibility" if "sanity check" in message else "read" if "RasterioIOError" in message
                       else message[:40]] += 1
        table = pd.read_parquet(run / "timeseries.parquet")
        ids = set(table.field_id)
        if ids & seen:
            raise SystemExit(f"part {k} shares {len(ids & seen)} fields with an earlier part")
        seen |= ids
        tables.append(table)
        report["parts"][run.name] = {"status": manifest["status"], "rows": len(table),
                                     "scenes": int(table.scene_id.nunique()), "fields": len(ids),
                                     "scene_failures": dict(errors)}
    table = pd.concat(tables, ignore_index=True)
    missing = set(strata) - set(table.field_id)
    extra = set(table.field_id) - set(strata)
    if extra:
        raise SystemExit(f"{len(extra)} Landsat fields are not in the sample's fields.geojson")
    stratum = table.field_id.map(strata)
    corn, other = table[stratum == "corn"], table[stratum != "corn"]
    corn.to_parquet(sample / "timeseries_landsat.parquet", index=False)
    if len(other):
        other.to_parquet(sample / "timeseries_landsat_dblcrop.parquet", index=False)
    report.update({"sample": str(sample), "fields_in_sample": len(strata), "fields_missing_from_landsat": len(missing),
                   "corn_fields": int(corn.field_id.nunique()), "corn_rows": len(corn),
                   "other_strata_fields": int(other.field_id.nunique()), "other_strata_rows": len(other),
                   "scenes": int(table.scene_id.nunique())})
    (sample / "landsat_delivery.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
