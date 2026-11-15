import json

import pandas as pd


def load(fips):
    t = pd.read_parquet(rf"C:\t\fv\data\diag\dairy\results\MD_2022_{fips}_module_v2.parquet")
    ev = t.evidence.map(json.loads).apply(pd.Series)
    t = pd.concat([t.drop(columns="evidence"), ev], axis=1)
    feats = json.load(open(rf"C:\t\fv\data\labels\us_{fips}_2022\fields.geojson"))["features"]
    area = {f["properties"]["field_id"]: f["properties"]["area_ha"] for f in feats}
    t["area_ha"] = t.field_id.map(area)
    t["fips"] = fips
    return t


fr, ke = load("24021"), load("24029")
print("label values", fr.label.value_counts(dropna=False).to_dict(), "| tiers", fr.evidence_tier.value_counts(dropna=False).to_dict())
print("kent tiers", ke.evidence_tier.value_counts(dropna=False).to_dict(), "| abstain", ke.abstain.value_counts().to_dict())
drop = pd.to_datetime(ke.drop_date)
groups = {
    "frederick_silage_clear": fr[(fr.evidence_tier == "clear") & (fr.label == "silage") & ~fr.abstain.astype(bool)],
    "frederick_grain_clear": fr[(fr.evidence_tier == "clear") & (fr.label == "grain") & ~fr.abstain.astype(bool)],
    "kent_silage_clear_early": ke[(ke.evidence_tier == "clear") & (ke.label == "silage") & ~ke.abstain.astype(bool)
                                  & (drop < "2022-09-08")],
    "kent_cloudy_spell": ke[(drop >= "2022-09-09") & (drop <= "2022-09-22")
                            & ((ke.evidence_tier == "wide") | ke.abstain.astype(bool))],
}
picks = {}
for name, g in groups.items():
    g = g.dropna(subset=["area_ha"])
    median = g.area_ha.median()
    row = g.iloc[(g.area_ha - median).abs().argsort().iloc[0]]
    picks[name] = {k: (None if pd.isna(v) else v) for k, v in row[["field_id", "fips", "label", "abstain", "evidence_tier",
                    "drop_date", "last_standing_date", "area_ha", "p_silage", "n_usable_dates"]].to_dict().items()}
    picks[name]["group_size"] = len(g)
    picks[name]["group_median_area_ha"] = round(float(median), 2)
    print(name, len(g), "fields, median area", round(median, 2), "->", picks[name])
json.dump(picks, open(r"C:\t\fv\data\share\strips_picks.json", "w"), indent=2, default=str)
