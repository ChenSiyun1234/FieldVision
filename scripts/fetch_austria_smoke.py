"""Fetch 30 declared silage-maize parcels for the extractor smoke test."""
import argparse
import json
from pathlib import Path
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    url = "https://gis.lfrz.gv.at/api/geodata/i009501/ogc/features/v1/collections/i009501:invekos_schlaege_2024_2_polygon/items"
    response = requests.get(url, params={"f": "json", "limit": 30, "filter-lang": "cql2-text",
                                         "filter": "snar_code='109'", "bbox": "15.5,47.9,16.0,48.2"}, timeout=90)
    response.raise_for_status()
    data = response.json()
    (args.out / "austria_raw.geojson").write_text(json.dumps(data), encoding="utf-8")
    features = []
    for feature in data["features"]:
        props = feature["properties"]
        features.append({"type": "Feature", "geometry": feature["geometry"], "properties": {
            "field_id": f"AT_2024_{props['geo_id']}", "label": "silage", "year": 2024,
            "country": "AT", "source": "INVEKOS 2024", "crop_code": props["snar_code"],
            "area_ha": props["sl_flaeche_brutto_ha"]}})
    output = args.out / "fields.geojson"
    output.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    print(f"Saved {len(features)} fields to {output}")


if __name__ == "__main__":
    main()
