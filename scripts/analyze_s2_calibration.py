"""Summarise per-band agreement from compare_s2_sources.py output.

Each non-reference row is paired with the reference row for the same field and
acquisition (tile + date). Reflectance is computed under two interpretations:
"offset" = DN * 1e-4 - 0.1 and "no_offset" = DN * 1e-4. The reference always
uses its documented rule (Collection 1: offset; PC: offset from baseline 04.00).
Only fields that are at least 90% clear in both rows are compared.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def reflectance(table, offset):
    return table[BANDS] * 1e-4 - offset


def group_label(row):
    flag = "" if pd.isna(row.offset_flag) else f" flag={row.offset_flag}"
    return f"{row.source} b{row.baseline}{flag}"


def compare(table, reference_mask, min_clear=0.9):
    table = table.assign(clear=table.n_clear / table.n_pixels.where(table.n_pixels > 0))
    table = table[table.clear >= min_clear].copy()
    reference = table[reference_mask(table)]
    ref_offset = np.where(reference.baseline.astype(float) >= 4.0, 0.1, 0.0)
    ref_values = reference[BANDS] * 1e-4 - ref_offset[:, None]
    ref_values[["field_id", "date"]] = reference[["field_id", "date"]]
    ref_dn = reference[["field_id", "date", *BANDS]]
    others = table[~reference_mask(table)].copy()
    others["group"] = [group_label(r) for r in others.itertuples()]
    out = []
    for group, rows in others.groupby("group"):
        paired = rows.merge(ref_values, on=["field_id", "date"], suffixes=("", "_ref"))
        paired_dn = rows.merge(ref_dn, on=["field_id", "date"], suffixes=("", "_ref"))
        if paired.empty:
            continue
        for hypothesis, offset in (("offset", 0.1), ("no_offset", 0.0)):
            record = {"group": group, "hypothesis": hypothesis, "pairs": len(paired),
                      "dates": paired.date.nunique()}
            for band in BANDS:
                diff = paired[band] * 1e-4 - offset - paired[band + "_ref"]
                record[band + "_bias"] = float(np.median(diff))
                record[band + "_mad"] = float(np.median(np.abs(diff)))
            record["median_dn_diff_B04"] = float(np.median(paired_dn["B04"] - paired_dn["B04_ref"]))
            record["median_dn_diff_B08"] = float(np.median(paired_dn["B08"] - paired_dn["B08_ref"]))
            out.append(record)
        # Per-date view, so a single odd scene is not hidden by medians.
    return pd.DataFrame(out)


def per_date(table, reference_mask, group_filter, offset, min_clear=0.9):
    table = table.assign(clear=table.n_clear / table.n_pixels.where(table.n_pixels > 0))
    table = table[table.clear >= min_clear]
    reference = table[reference_mask(table)]
    rows = table[~reference_mask(table)]
    rows = rows[[group_filter(r) for r in rows.itertuples()]]
    paired = rows.merge(reference, on=["field_id", "date"], suffixes=("", "_ref"))
    ref_offset = np.where(paired.baseline_ref.astype(float) >= 4.0, 0.1, 0.0)
    paired["red"] = paired.B04 * 1e-4 - offset
    paired["red_ref"] = paired.B04_ref * 1e-4 - ref_offset
    paired["nir_diff"] = paired.B08 * 1e-4 - offset - (paired.B08_ref * 1e-4 - ref_offset)
    paired["red_diff"] = paired.red - paired.red_ref
    return paired.groupby(["date", "scene_id", "offset_flag", "baseline"], dropna=False).agg(
        fields=("field_id", "size"), red_ref=("red_ref", "median"), red=("red", "median"),
        red_diff=("red_diff", "median"), nir_diff=("nir_diff", "median")).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--reference", choices=("c1", "pc-original"), required=True)
    args = parser.parse_args()
    table = pd.read_parquet(args.table)
    if args.reference == "c1":
        reference_mask = lambda t: t.source == "sentinel-2-c1-l2a"
    else:
        # The PC item with the lower baseline is the original ESA product.
        lowest = table[table.source == "planetary:sentinel-2-l2a"].groupby("date").baseline.transform("min")
        original = table.index.isin(lowest.index) & (table.baseline == lowest.reindex(table.index))
        reference_mask = lambda t: pd.Series(original, index=table.index).reindex(t.index).fillna(False).astype(bool)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    summary = compare(table, reference_mask)
    cols = ["group", "hypothesis", "pairs", "dates", "median_dn_diff_B04", "median_dn_diff_B08"]
    print("Bias (median other - reference, reflectance units):")
    print(summary[cols + [b + "_bias" for b in BANDS]].round(4).to_string(index=False))
    print("\nMedian absolute difference:")
    print(summary[["group", "hypothesis"] + [b + "_mad" for b in BANDS]].round(4).to_string(index=False))
    summary.to_csv(args.table.with_suffix(".summary.csv"), index=False)
    legacy = table[table.source == "sentinel-2-l2a"]
    if not legacy.empty:
        for flag in (True, False):
            offset = 0.0 if flag else 0.1  # what the extractor currently does
            view = per_date(table, reference_mask, lambda r, f=flag: r.source == "sentinel-2-l2a" and r.offset_flag == f, offset)
            if len(view):
                print(f"\nLegacy flag={flag}, per date, extractor's current rule (offset {'not ' if flag else ''}applied):")
                print(view.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
