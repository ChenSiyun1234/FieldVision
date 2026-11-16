"""The fold file the stream models train on, taken from a reference run.

    python -m fieldvision.stream.folds --reference <run>/predictions.parquet --out folds.parquet
        [--fields <subset fields.geojson ...>]

Reads the spatial-fold assignment of an earlier comparison run
(`fieldvision.foundation.compare`), so the new models are trained and
scored on exactly the folds the reference models used. With `--fields`
the table is cut down to the fields in those files.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

COLUMNS = ["field_id", "block_id", "country", "year", "low_evidence", "fold", "y"]


def folds_from_reference(preds, model="phys-logistic", field_ids=None):
    ref = preds[(preds["scheme"] == "spatial") & (preds["model"] == model)][COLUMNS]
    ref = ref.drop_duplicates("field_id").reset_index(drop=True)
    if field_ids is not None:
        ref = ref[ref["field_id"].isin(set(map(str, field_ids)))].reset_index(drop=True)
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields", nargs="*", default=[])
    args = ap.parse_args(argv)
    ids = None
    if args.fields:
        ids = []
        for path in args.fields:
            with open(path, encoding="utf-8") as fh:
                ids += [f["properties"]["field_id"] for f in json.load(fh)["features"]]
    folds = folds_from_reference(pd.read_parquet(args.reference), field_ids=ids)
    folds.to_parquet(args.out, index=False)
    print(f"{len(folds)} fields; per country {folds['country'].value_counts().to_dict()}; "
          f"per fold {folds['fold'].value_counts().sort_index().to_dict()}; "
          f"silage share {folds['y'].mean():.3f}")


if __name__ == "__main__":
    main()
