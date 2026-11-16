"""Four-state check for 2025: silage share, silage acres and harvest pace against USDA.

Pennsylvania, Wisconsin, Iowa and Nebraska. No field labels exist here; three USDA sources check
the models, and no model was fitted on any of them:
  1. silage share of harvested corn per state, 2025 survey (the numbers the challenge evaluates on);
  2. silage acres: our share times the state's 2025 corn area in Crop Sequence Boundaries (our own
     field inventory, not the survey), against the survey's silage acres;
  3. weekly harvest progress 2025 (Quick Stats Crop Progress): fields called grain should be cut at
     USDA's corn-for-grain pace, fields called silage at its corn-for-silage pace (PA and WI only).
The county prior uses survey years before 2025 only.

Fields: USDA NASS Crop Sequence Boundaries 2018-2025 (national file geodatabase, read inside the zip),
polygons whose CDL2025 class is corn. Per state, 10 km blocks (EPSG:5070) are drawn with probability
proportional to corn area, then a fixed number of fields at random per block, so extraction reads a
few compact areas; each field carries its inverse inclusion probability.

Models: eu2_logistic_v1 (the final model: 8,000 European fields, run from the final FieldVision on
GitHub main) and eu_logistic_v1 (the previous default, for comparison). As in the final module
(module_v2), the headline share counts silage on clearly seen fields only (harvest bracket of 12 days
or less) when there are enough of them; otherwise it is the share called silage among decided fields.

Run in the workspace (not on a laptop):  python -m fieldvision_agents.states.pipeline all
Stages: download, sample, extract, classify, compare.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..settings import DATA, FIELDVISION, FIELDVISION_DEPS, FIELDVISION_FINAL, QUICKSTATS_CSV, WORKSPACE

for _p in (FIELDVISION_DEPS, FIELDVISION / "src", WORKSPACE):       # pyarrow lives in .deps
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

OUT = DATA / "states"
CSB_URL = ("https://www.nass.usda.gov/sites/default/files/Research_and_Science/Crop-Sequence-Boundaries/"
           "datasets/NationalCSB_2018-2025_rev23.zip")
CSB_GDB, CSB_LAYER = "CSB1825.gdb", "national1825"
QUICKSTATS = QUICKSTATS_CSV
PROGRESS = OUT / "usda_progress_2025.csv"      # Quick Stats API, corn grain/silage PCT HARVESTED, weekly

YEAR = 2025
STATES = {"PA": ("42", "PENNSYLVANIA"), "WI": ("55", "WISCONSIN"), "IA": ("19", "IOWA"), "NE": ("31", "NEBRASKA")}
CORN = 1                      # CDL class 1
BLOCK_M = 10_000
N_BLOCKS, PER_BLOCK, SEED = 20, 15, 42
PRIOR_YEARS = (2021, 2023, 2024)  # survey years before 2025 in this file (it has no 2022 state survey)
MODELS = ("eu2_logistic_v1", "eu_logistic_v1")
CLEAR_BRACKET_DAYS, MIN_CLEAR_FIELDS, MIN_CLEAR_AREA_SHARE = 12.0, 100, 0.15   # module_v2's clear tier rules
CPUS = os.cpu_count() or 8
ACRES_PER_HA = 2.47105


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def status(stage, **info):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "status.json"
    s = json.loads(path.read_text()) if path.exists() else {}
    s[stage] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
    path.write_text(json.dumps(s, indent=1))


# --- 1. download (ranged, parallel: the NASS server answers range requests) --------

def download(part_mb=64, workers=16):
    path = OUT / "csb" / "NationalCSB_2018-2025_rev23.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    size = int(urllib.request.urlopen(urllib.request.Request(CSB_URL, method="HEAD"), timeout=60).headers["Content-Length"])
    if path.exists() and path.stat().st_size == size:
        log(f"have {path.name} ({size / 1e9:.2f} GB)")
        status("download", file=str(path), bytes=size)
        return
    parts_dir = path.parent / "parts"
    parts_dir.mkdir(exist_ok=True)
    step = part_mb << 20
    ranges = [(i, a, min(a + step, size) - 1) for i, a in enumerate(range(0, size, step))]

    def get(r):
        i, a, b = r
        part = parts_dir / f"{i:05d}"
        if part.exists() and part.stat().st_size == b - a + 1:
            return
        subprocess.run(["curl", "-sS", "--retry", "8", "-r", f"{a}-{b}", "-o", str(part), CSB_URL], check=True)
        if part.stat().st_size != b - a + 1:
            raise RuntimeError(f"part {i}: {part.stat().st_size} bytes, expected {b - a + 1}")

    log(f"downloading {size / 1e9:.2f} GB in {len(ranges)} parts, {workers} at a time")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(get, ranges))
    with open(path.with_suffix(".part"), "wb") as out:
        for i, _, _ in ranges:
            out.write((parts_dir / f"{i:05d}").read_bytes())
    path.with_suffix(".part").rename(path)
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    log(f"have {path.name}")
    status("download", file=str(path), bytes=size)


# --- 2. sample -----------------------------------------------------------------

def _read_state(st):
    """Every 2025 corn polygon of one state, from the geodatabase inside the zip (EPSG:5070 Albers)."""
    import pyogrio
    src = f"/vsizip/{(OUT / 'csb' / 'NationalCSB_2018-2025_rev23.zip').as_posix()}/{CSB_GDB}"
    info = pyogrio.read_info(src, layer=CSB_LAYER)
    dtypes = dict(zip(info["fields"], info["dtypes"]))
    fips = STATES[st][0]
    st_cond = f"STATEFIPS = '{fips}'" if dtypes["STATEFIPS"] == "object" else f"STATEFIPS = {int(fips)}"
    corn_cond = f"CDL{YEAR} = '{CORN}'" if dtypes[f"CDL{YEAR}"] == "object" else f"CDL{YEAR} = {CORN}"
    return pyogrio.read_dataframe(src, layer=CSB_LAYER, where=f"{st_cond} AND {corn_cond}",
                                  columns=["CSBID", "CSBACRES", "STATEFIPS", "CNTYFIPS", f"CDL{YEAR}"])


def sample():
    import pandas as pd
    rng = np.random.default_rng(SEED)
    log("reading 2025 corn polygons, four states in parallel")
    with ThreadPoolExecutor(len(STATES)) as pool:
        frames = dict(zip(STATES, pool.map(_read_state, STATES)))
    summary = {}
    for st, d in frames.items():
        d = d.to_crs("EPSG:5070")
        d["area_ha"] = d.area.to_numpy() / 1e4
        state_corn_acres = float(d["area_ha"].sum() * ACRES_PER_HA)        # all corn polygons, before the 1 ha floor
        d = d[d["area_ha"] >= 1.0]                       # same 1 ha floor as the European labels
        c = d.geometry.representative_point()
        d["block_id"] = [f"US-{st}-E{int(x // BLOCK_M)}N{int(y // BLOCK_M)}" for x, y in zip(c.x, c.y)]
        blocks = d.groupby("block_id").agg(area=("area_ha", "sum"), n=("area_ha", "size"))
        p = (blocks["area"] / blocks["area"].sum()).to_numpy()
        n_blocks = min(N_BLOCKS, len(blocks))
        picked = rng.choice(blocks.index.to_numpy(), size=n_blocks, replace=False, p=p)
        rows = []
        for b in picked:
            inb = d[d["block_id"] == b]
            take = inb.sample(n=min(PER_BLOCK, len(inb)), random_state=int(rng.integers(1 << 31)))
            # inclusion probability: block (about n_blocks * its area share) times field within block
            pi = min(1.0, n_blocks * blocks.loc[b, "area"] / blocks["area"].sum()) * len(take) / len(inb)
            rows.append(take.assign(weight=1.0 / pi))
        s = pd.concat(rows).to_crs("EPSG:4326")
        folder = OUT / st
        folder.mkdir(parents=True, exist_ok=True)
        out = s.assign(field_id=["csb-" + str(i) for i in s["CSBID"]], year=YEAR, label=None, country="US",
                       source="nass-csb-2018-2025", crop_code=str(CORN), crop_name="Corn", region=st,
                       county_fips=s["STATEFIPS"].astype(str).str.zfill(2) + s["CNTYFIPS"].astype(str).str.zfill(3))
        out[["field_id", "year", "label", "country", "source", "crop_code", "crop_name", "area_ha", "block_id",
             "region", "county_fips", "weight", "geometry"]].to_file(folder / "fields.geojson", driver="GeoJSON")
        summary[st] = {"corn_fields_in_state": int(len(d)), "corn_acres_in_state_csb": round(state_corn_acres),
                       "blocks_in_state": int(len(blocks)), "blocks_sampled": int(n_blocks),
                       "fields_sampled": int(len(out)), "sampled_ha": round(float(out["area_ha"].sum()), 1)}
        log(f"{st}: {summary[st]}")
    (OUT / "sample_summary.json").write_text(json.dumps(summary, indent=1))
    status("sample", **summary)


# --- 3. extract and 4. classify (FieldVision's own CLIs) -------------------------

def _env(code=FIELDVISION):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(code / "src"), str(FIELDVISION_DEPS), env.get("PYTHONPATH", "")])
    return env


def _parallel(cmds, logs, code=FIELDVISION):
    procs = {k: subprocess.Popen(c, cwd=code, env=_env(code), stdout=open(logs[k], "a"), stderr=subprocess.STDOUT)
             for k, c in cmds.items()}
    return {k: p.wait() for k, p in procs.items()}


def extract(workers=max(8, CPUS // len(STATES))):     # all CPUs, split evenly across the four states
    cmds = {st: [sys.executable, "-m", "fieldvision.s2.extract", "--fields", str(OUT / st / "fields.geojson"),
                 "--year", str(YEAR), "--start", "04-01", "--end", "11-30", "--out", str(OUT / st / "timeseries"),
                 "--workers", str(workers)] for st in STATES}
    log(f"extracting 4 states, {workers} workers each")
    codes = _parallel(cmds, {st: OUT / st / "extract.log" for st in STATES})
    log(f"extract exit codes: {codes}")
    # exit 2 is "partial": every block ran, but some fields were unusable (e.g. invalid CSB geometry)
    usable = {st: c == 0 or (c == 2 and (OUT / st / "timeseries" / "timeseries.parquet").exists())
              for st, c in codes.items()}
    status("extract", exit_codes=codes, usable=usable)
    if not all(usable.values()):
        raise RuntimeError(f"extraction failed: {codes} (see data/states/<ST>/extract.log)")


def classify():
    cmds, logs = {}, {}
    for st in STATES:                                    # every state x model in parallel
        for m in MODELS:
            cmds[(st, m)] = [sys.executable, "-m", "fieldvision.model.classify", "--model", m,
                             "--registry", str(FIELDVISION_FINAL / "registry"),
                             "--timeseries", str(OUT / st / "timeseries" / "timeseries.parquet"),
                             "--fields", str(OUT / st / "fields.geojson"), "--state", st,
                             "--out", str(OUT / st / "results" / m)]
            logs[(st, m)] = OUT / st / f"classify_{m}.log"
    codes = {f"{st}/{m}": c for (st, m), c in _parallel(cmds, logs, code=FIELDVISION_FINAL).items()}
    status("classify", models=list(MODELS), exit_codes=codes)
    if any(codes.values()):
        raise RuntimeError(f"classify failed: {codes}")


# --- 5. compare with USDA ---------------------------------------------------------

def _survey():
    """Harvested grain and silage acres per state and year (survey)."""
    import pandas as pd
    df = pd.read_csv(QUICKSTATS, dtype=str)
    df = df[(df.measurement == "harvested") & (df.domain == "TOTAL") & (df.reference_period == "YEAR") &
            (df.aggregation_level == "STATE") & (df.source == "SURVEY")].copy()
    df["v"] = pd.to_numeric(df.value.str.replace(",", ""), errors="coerce")
    return df.pivot_table(index=["state_name", "year"], columns="crop_use", values="v", aggfunc="first")


def _harvest_date(ev):
    ev = ev if isinstance(ev, dict) else {}
    d = ev.get("drop_date") or ev.get("harvest_end")
    return None if not d else str(d)[:10]


def _pace(frame, weeks):
    """Weighted share of these fields already harvested by each week-ending date."""
    import pandas as pd
    hd = pd.to_datetime([_harvest_date(e) for e in frame["evidence"]])
    ok = ~pd.isna(hd)
    w = (frame["weight"] * frame["area_ha"]).to_numpy()[ok]
    hd = hd[ok]
    return {wk: (round(100 * float((w * (hd <= pd.Timestamp(wk))).sum() / w.sum()), 1) if w.sum() else None)
            for wk in weeks}, int(ok.sum())


def compare():
    import geopandas as gpd
    import pandas as pd
    from fieldvision_agents.fieldvision_bridge.prevalence import shift_probability

    survey = _survey()
    share = (survey["silage"] / (survey["grain"] + survey["silage"]))
    progress = pd.read_csv(PROGRESS, dtype={"week_ending": str})
    samples = json.loads((OUT / "sample_summary.json").read_text())
    rows, pace = [], []
    for m in MODELS:
        for st, (_, name) in STATES.items():
            fields = gpd.read_file(OUT / st / "fields.geojson")[["field_id", "area_ha", "weight"]]
            res = pd.read_json(OUT / st / "results" / m / f"{st}_{YEAR}.jsonl", lines=True)
            key = "source_field_id" if "source_field_id" in res and res["source_field_id"].isin(fields["field_id"]).any() else "field_id"
            d = res.merge(fields, left_on=key, right_on="field_id", suffixes=("", "_f"))
            abstain = d["abstain"].astype(bool)
            ok = d[~abstain & d["p_silage"].notna()]
            wa = ok["weight"] * ok["area_ha"]
            called = float((wa * (ok["p_silage"] >= 0.5)).sum() / wa.sum())
            # module_v2 clear tier: harvest seen within 12 days on both sides of the cut
            days = pd.to_numeric(pd.Series([(e or {}).get("days_to_drop") for e in ok["evidence"]], index=ok.index),
                                 errors="coerce")
            clear = ok[days <= CLEAR_BRACKET_DAYS]
            wc = clear["weight"] * clear["area_ha"]
            clear_share = float((wc * (clear["p_silage"] >= 0.5)).sum() / wc.sum()) if len(clear) else None
            clear_area = float((d["weight"] * d["area_ha"])[d.index.isin(clear.index)].sum() / (d["weight"] * d["area_ha"]).sum())
            enough = len(clear) >= MIN_CLEAR_FIELDS and clear_area >= MIN_CLEAR_AREA_SHARE
            headline = clear_share if enough else called
            prior = float(np.mean([share[(name, str(y))] for y in PRIOR_YEARS]))
            corn_acres = samples[st]["corn_acres_in_state_csb"]
            rows.append({
                "model": m, "state": st,
                "usda_2025_silage_pct": round(100 * float(share[(name, str(YEAR))]), 1),
                "headline_pct": round(100 * headline, 1),
                "headline_method": "counted share of the clear tier (module_v2)" if enough
                                   else f"share called silage among decided fields (clear tier {len(clear)} fields, too few)",
                "called_silage_pct": round(100 * called, 1),
                "clear_fields": int(len(clear)),
                "clear_called_silage_pct": None if clear_share is None else round(100 * clear_share, 1),
                "mean_p_pct": round(100 * float((wa * ok["p_silage"]).sum() / wa.sum()), 1),
                "mean_p_at_prior_pct": round(100 * float((wa * shift_probability(ok["p_silage"].to_numpy(float), prior)).sum() / wa.sum()), 1),
                "prior_pct": round(100 * prior, 1),
                "usda_2025_silage_acres": int(survey.loc[(name, str(YEAR)), "silage"]),
                "model_silage_acres": int(round(headline * corn_acres, -3)),
                "csb_corn_acres": corn_acres,
                "usda_2025_corn_acres": int(survey.loc[(name, str(YEAR)), "silage"] + survey.loc[(name, str(YEAR)), "grain"]),
                "fields": int(len(d)), "abstained": int(abstain.sum()),
            })
            for use, part in (("grain", ok[ok["p_silage"] < 0.5]), ("silage", ok[ok["p_silage"] >= 0.5])):
                usda = progress[(progress["state"] == st) & (progress["use"] == use)]
                if usda.empty:
                    continue
                model_pct, n = _pace(part, usda["week_ending"].tolist())
                for _, r in usda.iterrows():
                    pace.append({"model": m, "state": st, "use": use, "week_ending": r["week_ending"],
                                 "usda_pct_harvested": float(r["pct_harvested"]),
                                 "model_pct_harvested": model_pct[r["week_ending"]], "fields_with_date": n})
    table, pace_t = pd.DataFrame(rows), pd.DataFrame(pace)
    summary = {}
    for m in MODELS:
        t, p = table[table["model"] == m], pace_t[pace_t["model"] == m]
        summary[m] = {
            "rank_usda": t.sort_values("usda_2025_silage_pct", ascending=False)["state"].tolist(),
            "rank_model": t.sort_values("headline_pct", ascending=False)["state"].tolist(),
            "share_mean_abs_error_pts": round(float((t["headline_pct"] - t["usda_2025_silage_pct"]).abs().mean()), 1),
            "share_mean_abs_error_pts_with_prior": round(float((t["mean_p_at_prior_pct"] - t["usda_2025_silage_pct"]).abs().mean()), 1),
            "silage_acres_total_usda": int(t["usda_2025_silage_acres"].sum()),
            "silage_acres_total_model": int(t["model_silage_acres"].sum()),
            "pace_mean_abs_diff_pts": {u: round(float((g["model_pct_harvested"] - g["usda_pct_harvested"]).abs().mean()), 1)
                                       for u, g in p.groupby("use")} if len(p) else {},
        }
    out = {"year": YEAR, "models": list(MODELS), "shares": rows, "harvest_pace": pace, "summary": summary,
           "sources": {"shares_and_acres": "USDA NASS Quick Stats, state survey 2025 (the challenge's evaluation numbers)",
                       "corn_area": "USDA NASS Crop Sequence Boundaries 2018-2025, CDL2025 corn polygons",
                       "pace": "USDA NASS Quick Stats Crop Progress 2025, PCT HARVESTED"},
           "note": "Model estimates from a block sample of about 300 fields per state; not NASS statistics. "
                   "Beltsville was not used. The prior uses survey years before 2025 only."}
    (OUT / "comparison.json").write_text(json.dumps(out, indent=1))
    table.to_csv(OUT / "shares.csv", index=False)
    pace_t.to_csv(OUT / "harvest_pace.csv", index=False)
    log(table.to_string(index=False))
    log(json.dumps(summary, indent=1))
    status("compare", file=str(OUT / "comparison.json"))
    return out


STAGES = {"download": download, "sample": sample, "extract": extract, "classify": classify, "compare": compare}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    args = ap.parse_args(argv)
    for name in (STAGES if args.stage == "all" else [args.stage]):
        log(f"== {name}")
        STAGES[name]()
    log("done")


if __name__ == "__main__":
    main()
