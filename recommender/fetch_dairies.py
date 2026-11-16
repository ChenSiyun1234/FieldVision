"""Download permitted dairy operations in Maryland from MDE's public AFO/CAFO layer.

Source: Maryland Department of the Environment, Animal Feeding Operations (CAFO/MAFO) map service
https://mde.geodata.md.gov/mdedata/rest/services/LMA_Resource_Management_Program/Animal_Feeding_Operations_in_Maryland/MapServer/0

Keeps only rows with dairy animals (Sum_of_Dairy > 0). Operator names and street addresses are
dropped on purpose: the recommender only needs an id, a location and a herd size.

Usage:  python3 recommender/fetch_dairies.py            -> recommender/data/dairies_md.csv
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

LAYER = ("https://mde.geodata.md.gov/mdedata/rest/services/LMA_Resource_Management_Program/"
         "Animal_Feeding_Operations_in_Maryland/MapServer/0/query")
OUT = Path(__file__).resolve().parent / "data" / "dairies_md.csv"
COLUMNS = ["dairy_id", "county", "state", "lat", "lon", "head", "animal_type", "permit_type", "source"]


def _num(v) -> float:
    try:
        return float(str(v).strip())
    except ValueError:
        return 0.0


def fetch(raw_path: Path | None = None) -> list[dict]:
    if raw_path and raw_path.exists():
        data = json.loads(raw_path.read_text())
    else:
        q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "outSR": "4326", "f": "json"})
        with urllib.request.urlopen(f"{LAYER}?{q}", timeout=90) as r:
            data = json.loads(r.read())
    rows = []
    for f in data.get("features", []):
        a = f["attributes"]
        head = _num(a.get("Sum_of_Dairy"))
        lat, lon = _num(a.get("Latitude")), _num(a.get("Longitude"))
        if head <= 0 or not lat or not lon:
            continue
        rows.append({"dairy_id": f"MDE-{a.get('AI_Id')}", "county": a.get("County"), "state": "MD",
                     "lat": round(lat, 5), "lon": round(lon, 5), "head": int(head),
                     "animal_type": a.get("Animal_Type"), "permit_type": a.get("Permit_Typ"),
                     "source": "MDE AFO/CAFO permits (public map service)"})
    return rows


def main() -> None:
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT.parent / "mde_afo_raw.json"
    rows = fetch(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} dairies, {sum(r['head'] for r in rows):,} head -> {OUT}")


if __name__ == "__main__":
    main()
