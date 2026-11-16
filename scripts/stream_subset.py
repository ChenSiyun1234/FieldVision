"""Select whole read blocks, or copy field geometries with labels removed."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.s2.extract import atomic_json
from fieldvision.s2.geometry import load_fields, make_blocks

SAFE_PROPERTIES = ("field_id", "year", "country", "block_id", "area_ha", "centroid_lon", "centroid_lat")


def parse_spec(spec, stripped=False):
    parts = spec.split(":")
    if len(parts) != (2 if stripped else 3) or not re.fullmatch(r"[a-z]+_[0-9]{4}", parts[0]):
        raise ValueError("Expected run:year for stripped fields or run:year:target for labelled fields")
    run, year = parts[0], int(parts[1])
    if run.startswith("us_") or (run.startswith("barc_") and not stripped):
        raise ValueError("This run is not allowed for labelled subset selection")
    target = None if stripped else int(parts[2])
    if target is not None and target < 1:
        raise ValueError("Target must be positive")
    return run, year, target


def select_blocks(path, year, target, seed):
    fields = load_fields(path, year)
    invalid = sum(bool(field.get("error")) for field in fields)
    if invalid:
        raise ValueError(f"Input has {invalid} invalid fields")
    groups = defaultdict(list)
    for block in make_blocks(fields, 6000):
        labels = {str(field["properties"].get("block_id", "")) for field in block["fields"]}
        if len(labels) != 1:
            raise ValueError("A read block spans multiple label blocks")
        groups[next(iter(labels))].append(block)
    rng = np.random.default_rng(seed)
    for label_block in sorted(groups):
        rng.shuffle(groups[label_block])
    chosen, count, round_index = [], 0, 0
    while count < target:
        added = False
        for label_block in sorted(groups):
            if round_index < len(groups[label_block]):
                block = groups[label_block][round_index]
                chosen.append(block)
                count += len(block["fields"])
                added = True
        if not added:
            break
        round_index += 1
    return chosen


def run(args):
    plans = [(*parse_spec(spec), False) for spec in args.plan]
    plans += [(*parse_spec(spec, True), True) for spec in args.strip_labels]
    names = [plan[0] for plan in plans]
    if len(names) != len(set(names)):
        raise ValueError("Each run must occur only once")
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, year, target, stripped in plans:
        source = args.labels_root / name / "fields.geojson"
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("type") != "FeatureCollection":
            raise ValueError("Input must be a FeatureCollection")
        if stripped:
            features = []
            for feature in data["features"]:
                properties = feature.get("properties") or {}
                clean = {key: properties[key] for key in SAFE_PROPERTIES if key in properties}
                clean["label"] = None
                features.append(dict(feature, properties=clean))
            summary = {"year": year, "fields": len(features), "labels_removed": True}
            printed = {"run": name, "fields": len(features)}
        else:
            chosen = select_blocks(source, year, target, args.seed)
            ids = {field["field_id"] for block in chosen for field in block["fields"]}
            features = [f for f in data["features"] if f["properties"]["field_id"] in ids]
            label_counts = Counter(str(f["properties"].get("label")) for f in features)
            label_blocks = sorted({str(f["properties"].get("block_id", "")) for f in features})
            summary = {"year": year, "seed": args.seed, "target": target, "fields": len(features),
                       "number_per_label": dict(sorted(label_counts.items())), "label_blocks": label_blocks,
                       "read_block_ids": [block["id"] for block in chosen]}
            printed = {"run": name, "seed": args.seed, "target": target, "fields": len(features),
                       "number_per_label": summary["number_per_label"],
                       "label_blocks": len(label_blocks), "read_blocks": len(chosen)}
        destination = args.out / name
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "fields.geojson", dict(data, features=features))
        summaries[name] = summary
        print(json.dumps(printed, sort_keys=True))
    atomic_json(args.out / "subset.json", {"runs": summaries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan", nargs="*", default=[])
    parser.add_argument("--strip-labels", nargs="*", default=[])
    try:
        run(parser.parse_args())
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
