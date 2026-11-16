"""Rebuild the README figures from the checked-in result record.

Run from any directory: python scripts/make_readme_figures.py
Dependencies: matplotlib, numpy, pandas, Pillow.
Source notes are kept beside the extraction code and in each figure footer.
"""

from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import json
import re
import runpy
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/readme"
GREEN, GOLD, GRAY, BLUE, PAPER = "#2F6B4F", "#C9A227", "#333333", "#2878D6", "#FCFCFA"
MUTED, GRID, PALE = "#66716B", "#E1E6E1", "#DCE8DF"
# Competition-window definition supplied with the figure request; commit observations
# themselves are read from git log origin/main --format=%aI, without deduplication.
PACIFIC = timezone(timedelta(hours=-8))
START = datetime(2026, 11, 14, 10, 30, tzinfo=PACIFIC)
END = datetime(2026, 11, 15, 9, 0, tzinfo=PACIFIC)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": GRAY,
    "axes.labelcolor": GRAY, "xtick.color": MUTED, "ytick.color": GRAY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.facecolor": PAPER, "figure.facecolor": PAPER, "axes.titleweight": "bold",
    "axes.titlesize": 14, "axes.labelsize": 11, "savefig.facecolor": PAPER,
})


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def tables(text):
    """Read Markdown pipe tables, retaining cell text for explicit numeric parsing."""
    found, rows = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            cells = [v.strip().replace("`", "") for v in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r"[:\- ]+", v) for v in cells):
                rows.append(cells)
        elif rows:
            found.append(pd.DataFrame(rows[1:], columns=rows[0]))
            rows = []
    return found


def first_number(text):
    return float(re.search(r"[-+]?\d+(?:\.\d+)?", text).group())


def title(fig, heading, subheading):
    fig.text(.045, .944, heading, fontsize=23, weight="bold", color=GREEN, va="top")
    fig.text(.045, .885, subheading, fontsize=12, color=MUTED, va="top")


def footer(fig, note):
    fig.text(.045, .035, note, fontsize=9, color=MUTED, va="bottom", linespacing=1.5)


def grid(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=.8)
    ax.tick_params(length=0, pad=8)


def save(fig, name):
    """Save fresh PNG pixels, with no copied EXIF or software metadata."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    buffer.seek(0)
    pixels = Image.open(buffer).convert("RGB")
    clean = Image.frombytes("RGB", pixels.size, pixels.tobytes())
    dest = OUT / name
    clean.save(dest, optimize=True)
    if dest.stat().st_size >= 300_000:
        clean.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(dest, optimize=True)
    assert dest.stat().st_size < 300_000, f"PNG exceeds size limit: {name}"
    print(f"docs/readme/{name}: {clean.width} x {clean.height}, {dest.stat().st_size:,} bytes")


def pipeline(numbers):
    model = json.loads(read("registry/eu2_logistic_v1/model.json"))
    module = read("docs/module.md")
    tier_table = next(t for t in tables(module) if list(t.columns) == ["tier", "rule", "used for"])
    clear = int(first_number(tier_table.loc[tier_table.tier == "clear", "rule"].iloc[0]))
    assert "fieldvision.toolbox.Toolbox" in module
    n_blocks = numbers["MICHIGAN_2025"]["blocks"]
    fig = plt.figure(figsize=(16, 6.1))
    title(fig, "From a corn parcel to a silage-area estimate", "Public imagery, an interpretable classifier, and an explicit observation-quality gate")
    ax = fig.add_axes([.025, .24, .95, .49])
    ax.set(xlim=(0, 7), ylim=(0, 1))
    ax.axis("off")
    nodes = [
        ("USDA field\nboundaries", "Crop Sequence\nBoundaries", f"{n_blocks} sampled blocks / state"),
        ("Satellite\ntime series", "Sentinel-2 +\nLandsat 8/9", "Google Colab extraction"),
        ("Phenology\nfeatures", "Timing, moisture,\ngreenness, residue", f"{len(model['features'])} features / field"),
        ("Logistic\nregression", "Europe-trained\neu2_logistic_v1", f"{numbers['FIELDS_LABELLED_EUROPE']:,} labelled field-years"),
        ("Evidence\ntiers", "Clear / wide /\nabstain", f"Clear harvest bracket ≤ {clear} days"),
        ("County + state\nsilage share", "Area-weighted\nclear-tier calls", "Wide / unseen area reported"),
        ("Python\ninterface", "Toolbox", "fieldvision.toolbox.Toolbox"),
    ]
    for i, (heading, detail, note) in enumerate(nodes):
        ax.add_patch(FancyBboxPatch((i+.07, .30), .84, .61, boxstyle="round,pad=0.02,rounding_size=0.04",
                                   facecolor=GREEN if i == 3 else PAPER, edgecolor=GREEN, linewidth=1.5))
        ax.text(i+.49, .73, heading, ha="center", va="center", fontsize=13, weight="bold", color=PAPER if i == 3 else GREEN)
        ax.text(i+.49, .47, detail, ha="center", va="center", fontsize=10.2, color=PAPER if i == 3 else GRAY)
        ax.text(i+.49, .17, note, ha="center", va="center", fontsize=8.5, color=BLUE if i == 3 else MUTED)
        if i < len(nodes)-1:
            ax.add_patch(FancyArrowPatch((i+.93, .60), (i+1.055, .60), arrowstyle="-|>", mutation_scale=15, color=GREEN, linewidth=1.5))
    fig.text(.045, .18, "Every field keeps a call. Only clear evidence contributes to the headline area share.", fontsize=13, color=GREEN)
    footer(fig, "Sources: docs/model.md; docs/module.md, ‘The layers’; HANDOFF.md; registry/eu2_logistic_v1/model.json; docs/pitch/pitch_numbers.py\nState figures are sampled estimates. Clear evidence also requires usable dates, a detected harvest, and a confirming observation.")
    save(fig, "01_pipeline.png")


def features():
    # model.json specifies order and positive class; it does not serialize coefficients.
    # The fitted equation is published in registry/eu_logistic_v1/model_card.md, line 9.
    model = json.loads(read("registry/eu_logistic_v1/model.json"))
    equation = read("registry/eu_logistic_v1/model_card.md")
    terms = dict((key, float(value)) for value, key in re.findall(r"([+-]\d+\.\d+)\*z\((\w+)\)", equation))
    assert set(terms) == set(model["features"])
    assert model["positive_class"] == "silage"
    # Plain-English feature descriptions: docs/module.md, ‘2. Field classifier’;
    # docs/model.md, ‘Probabilities, class balance, explanations’ defines the sign.
    names = {
        "harv_days_since_peak": "Days from green peak to harvest",
        "harv_decline_days": "Length of pre-harvest decline",
        "pre_ndvi_frac": "Greenness before harvest / peak",
        "pre_ndmi": "Moisture before harvest (NDMI)",
        "sen_ndvi_slope_rel": "Relative greenness-decline slope",
        "post_ndti": "Residue after harvest (NDTI)",
    }
    vals = np.array([terms[k] for k in model["features"]])
    fig = plt.figure(figsize=(16, 8))
    title(fig, "Six measurable signals, one interpretable score", "eu_logistic_v1 · Austria-trained reference model · standardized-feature coefficients")
    ax = fig.add_axes([.30, .23, .63, .56])
    y = np.arange(len(vals))
    ax.barh(y, vals, height=.57, color=[BLUE if v > 0 else GREEN for v in vals])
    for yy, value in zip(y, vals):
        ax.text(value + (.024 if value > 0 else -.024), yy, f"{value:+.2f}", ha="left" if value > 0 else "right", va="center", weight="bold")
    ax.set_yticks(y, [names[k] for k in model["features"]])
    ax.invert_yaxis()
    ax.set_xlim(-.85, .9)
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_xlabel("Change in the uncalibrated log-odds score per one training standard deviation")
    grid(ax)
    fig.text(.35, .14, "Negative → grain", color=GREEN, weight="bold", fontsize=13)
    fig.text(.71, .14, "Positive → silage", color=BLUE, weight="bold", fontsize=13)
    footer(fig, "Sources: registry/eu_logistic_v1/model.json (feature order, positive class); model_card.md (published equation); docs/model.md; docs/module.md\nThese are fitted coefficients, not feature importance. The final Europe-trained classifier is eu2_logistic_v1; its coefficients differ.")
    save(fig, "02_features.png")


def validation(numbers):
    transfer, summary = numbers["TRANSFER"], numbers["TRANSFER_SUMMARY"]
    summary_by_name = {r[0]: r for r in summary}
    small_features=int(first_number(transfer[0][1]))
    big_features=int(first_number(transfer[1][1]))
    labels = {
        "Small model": f"Logistic · {small_features} features", "Big model": f"Boosted trees · {big_features} features",
        "Monthly band values": "Monthly bands", "Pretrained satellite model": "Presto embedding",
        "Big model + pretrained": "Trees + Presto", "Big model + band values": "Trees + bands",
        "Big model, dates dropped": "Trees · dates dropped", "Sequence model": "Sequence model",
        "Timing + moisture rule": "Timing + moisture rule", "Average of three simple models": "Mean of three models",
        "Timing rule": "Timing rule", "Always say grain": "Always grain",
    }
    candidates = [r[0] for r in transfer] + [r[0] for r in summary if r[0] not in {t[0] for t in transfer}]
    mean = [summary_by_name[k][2] if k in summary_by_name else np.mean([v[0] for v in next(t for t in transfer if t[0] == k)[2]]) for k in candidates]
    fig = plt.figure(figsize=(16, 9))
    title(fig, "Transfer first; labelled US fields as a check", "Hold out a whole country for transfer · Beltsville, Maryland, 2021–2023 for field-level checks")
    ax = fig.add_axes([.22, .23, .285, .53])
    y = np.arange(len(candidates))
    ax.barh(y, mean, color=[BLUE if k == "Small model" else PALE for k in candidates], height=.57)
    for i, value in enumerate(mean):
        ax.text(.91, i, f"{value:.3f}", va="center", fontsize=10, weight="bold" if i == 0 else "normal")
    markers = ["o", "s", "^"]
    for i, row in enumerate(transfer):
        for j, (value, _) in enumerate(row[2]):
            ax.scatter(value, i+(j-1)*.12, marker=markers[j], s=25, color=GRAY, zorder=4)
    ax.set_yticks(y, [labels[k] for k in candidates], fontsize=11)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.03), xlabel="ROC-AUC (mean in right column)")
    ax.set_title("A  Leave-one-country-out", loc="left", pad=22)
    grid(ax)
    handles = [Line2D([], [], color=GRAY, marker=m, ls="", label=k, markersize=5) for k,m in zip(numbers["TRANSFER_COUNTRIES"], markers)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.46, -.14), ncol=3, frameon=False, fontsize=9)

    scored = [(r[0], r[2]) for r in numbers["LABELLED"]]
    scored.insert(2, ("Big model, dates dropped", numbers["THINNED_BIG_MODEL_CORRECT"]))
    scored.insert(3, ("Sequence model", int(re.search(r"\d+", summary_by_name["Sequence model"][4]).group())))
    ax = fig.add_axes([.72, .23, .23, .53])
    y = np.arange(len(scored))
    ax.barh(y, [r[1] for r in scored], color=[BLUE if k == "Small model" else GREEN if k == "Always say grain" else PALE for k,_ in scored], height=.57)
    for i, (key, value) in enumerate(scored):
        ax.text(value+.7, i, f"{value}/{numbers['LABELLED_TOTAL']}", va="center", fontsize=11, weight="bold" if key == "Small model" else "normal")
    ax.set_yticks(y, [labels[k] for k,_ in scored], fontsize=10.5)
    ax.invert_yaxis()
    ax.set(xlim=(0, numbers["LABELLED_TOTAL"]+8), xlabel="Correct field-years")
    ax.set_title("B  Beltsville labelled check", loc="left", pad=22)
    grid(ax)
    footer(fig, "Sources: docs/pitch/pitch_numbers.py, TRANSFER / TRANSFER_SUMMARY / LABELLED; docs/model.md; docs/RESULTS.md\nCountry dots: Austria / France / Netherlands; bars: published means (or arithmetic means where no summary is supplied).\nBeltsville includes four separate scorings; date-thinned trees and sequence model were later checks. Final classifier: six-feature logistic.")
    save(fig, "03_validation.png")


def confusion(numbers):
    section = read("handoff/results/REPORT.md").split("Confusion matrix, `logistic`")[1]
    table = tables(section)[0]
    matrix = table[["silage", "grain"]].astype(int).to_numpy()
    expected = numbers["CONFUSION"]
    assert matrix.tolist() == [[expected["silage_found"], expected["silage_missed"]], [expected["grain_called_silage"], expected["grain_right"]]]
    total = int(matrix.sum())
    full = np.zeros((3,3), dtype=int)
    full[:2,:2], full[:2,2], full[2,:2], full[2,2] = matrix, matrix.sum(axis=1), matrix.sum(axis=0), total
    fig = plt.figure(figsize=(16, 8.5))
    title(fig, "Beltsville: the field calls behind the headline", "Europe-trained logistic regression · Evaluation 2 · rows are declared labels, columns are predicted labels")
    ax = fig.add_axes([.15, .17, .48, .62])
    ax.set(xlim=(0,3), ylim=(3,0), aspect="equal")
    for row in range(3):
        for col in range(3):
            marginal = row == 2 or col == 2
            color = PALE if marginal else GREEN if row == col else "#EDF2F7"
            ax.add_patch(plt.Rectangle((col,row), 1,1, facecolor=color, edgecolor=PAPER, linewidth=4))
            ax.text(col+.5, row+.47, str(full[row,col]), ha="center", va="center", fontsize=30, weight="bold", color=GRAY if marginal or row != col else PAPER)
            if not marginal:
                ax.text(col+.5, row+.78, ["Silage found", "Silage missed", "False silage", "Grain correct"][row*2+col], ha="center", fontsize=10, color=PAPER if row==col else GRAY)
    ax.set_xticks([.5,1.5,2.5], ["Silage", "Grain", "Total"])
    ax.set_yticks([.5,1.5,2.5], ["Silage", "Grain", "Total"])
    ax.xaxis.tick_top()
    ax.set_ylabel("Declared crop use", labelpad=15)
    ax.set_xlabel("Predicted crop use", labelpad=15)
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0, pad=10)
    fig.text(.66,.66,f"{np.trace(matrix)}/{total}", fontsize=56, color=BLUE, weight="bold")
    fig.text(.66,.60,"field-years correct", fontsize=17)
    fig.text(.66,.45,f"Silage recall      {matrix[0,0]}/{matrix[0].sum()}  ({matrix[0,0]/matrix[0].sum():.1%})", fontsize=15)
    fig.text(.66,.38,f"Silage precision  {matrix[0,0]}/{matrix[:,0].sum()}  ({matrix[0,0]/matrix[:,0].sum():.1%})", fontsize=15)
    fig.text(.66,.31,f"False-positive rate   {matrix[1,0]}/{matrix[1].sum()}  ({matrix[1,0]/matrix[1].sum():.1%})", fontsize=15)
    footer(fig, "Sources: handoff/results/REPORT.md, ‘Confusion matrix, logistic’; docs/RESULTS.md; docs/pitch/pitch_numbers.py, CONFUSION\nSmall external check on one research farm, not a statewide field-accuracy estimate. All field-years are included, regardless of evidence tier.")
    save(fig, "04_confusion.png")


def nass(numbers):
    table = tables(read("docs/RESULTS.md"))[0]
    specs = [("Michigan", "2025", "MICHIGAN_2025"), ("New York", "2025", "NEW_YORK_2025"),
             ("Lancaster", "2023", "LANCASTER_2023"), ("Wisconsin", "2025", "WISCONSIN_2025"),
             ("Pennsylvania", "2025", "PENNSYLVANIA_2025")]
    rows = []
    for place, year, key in specs:
        row = table[(table.place.str.startswith(place)) & (table.year == year)].iloc[0]
        ours = first_number(row.headline)
        official = first_number(re.search(r"(?:NASS|Census)\s+(\d+\.?\d*)", row.comparison).group(1))
        data = numbers[key]
        assert ours == data["ours"] and official == data.get("usda", data.get("census_2022"))
        interval = re.search(r"\[([\d.]+), ([\d.]+)\]", row.headline)
        if place == "Wisconsin":
            miss = float(re.search(r"(\d+\.\d+) low", row["used in tuning"]).group(1))
        else:
            miss = abs(ours-official)
        rows.append((place,year,ours,official,miss,interval))
    # State pass marks are read from the logged constants, not inferred from rounded values.
    mark = numbers["MICHIGAN_2025"]["pass_mark"]
    fig = plt.figure(figsize=(16,8.6))
    title(fig,"Five checks of the locked procedure", f"Model vs. USDA silage share · comparison line: within {mark} percentage points")
    ax = fig.add_axes([.19,.25,.55,.53])
    for i,(place,year,ours,official,miss,interval) in enumerate(rows):
        ax.plot([official-mark,official+mark],[i,i],lw=15,color=PALE,solid_capstyle="round",zorder=1)
        ax.plot([ours,official],[i,i],color=MUTED,lw=2,zorder=2)
        if interval:
            lo,hi = map(float,interval.groups())
            ax.plot([lo,hi],[i,i],color=BLUE,alpha=.38,lw=2,zorder=2)
        ax.scatter(ours,i,color=BLUE,s=100,zorder=4)
        ax.scatter(official,i,color=GRAY,marker="D",s=70,zorder=4)
        ax.text(ours,i+.23,f"{ours:.1f}%",ha="center",color=BLUE,fontsize=12,weight="bold")
        ax.text(official,i-.21,f"{official:.1f}%",ha="center",color=GRAY,fontsize=12)
        verdict = "PASS" if miss <= mark else "FAIL"
        extra = " · guard flagged" if place == "Wisconsin" else ""
        ax.text(63,i,f"{verdict}  ·  −{miss:.2f} pp{extra}" if place == "Wisconsin" else f"{verdict}  ·  −{miss:.1f} pp",va="center",fontsize=12,weight="bold" if verdict=="FAIL" else "normal",color=GREEN if verdict=="PASS" else GRAY)
    ax.set_yticks(range(len(rows)),[f"{r[0]} {r[1]}" + ("\nCounty" if r[0]=="Lancaster" else "\nState sample") for r in rows])
    ax.set(ylim=(4.6,-.6),xlim=(0,62),xlabel="Corn area harvested for silage (%)")
    grid(ax)
    handles=[Line2D([],[],marker="o",color=BLUE,ls="",label="Model"),Line2D([],[],marker="D",color=GRAY,ls="",label="USDA"),Line2D([],[],color=PALE,lw=10,label=f"USDA ±{mark} pp")]
    ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(0,-.18),ncol=3,frameon=False,fontsize=11)
    footer(fig,"Sources: docs/RESULTS.md, ‘For the deck’; docs/pitch/pitch_numbers.py (exact Wisconsin miss, logged pass marks)\nWisconsin passes by 0.02 pp, but its shift guard flags the estimate as unreliable. Thin blue lines show published sampling ranges.\nLancaster 2023 uses the 2022 Census reference; its original logged mark was within 10 pp of the official band. State references: NASS 2025.")
    save(fig,"05_nass_comparison.png")


def landsat(numbers):
    rows = numbers["LANDSAT_PAIR"]
    fig = plt.figure(figsize=(16,8.8))
    title(fig,"A second satellite recovers hidden harvests", "Development checks on previously seen counties · coverage gains do not guarantee accurate crop-use totals")
    y = np.arange(len(rows))
    ax = fig.add_axes([.17,.25,.31,.53])
    ax.barh(y-.17,[r["seen"][0] for r in rows],height=.30,color=GREEN,label="Sentinel-2")
    ax.barh(y+.17,[r["seen"][1] for r in rows],height=.30,color=BLUE,label="+ Landsat 8/9")
    for i,r in enumerate(rows):
        for j,v in enumerate(r["seen"]):
            ax.text(v+1.5,i+(-.17 if j==0 else .17),f"{v}%",va="center",fontsize=11)
    labels=[f"{r['name']}\n" + ("Wisconsin 2023" if r['name'] in ["Marathon","Racine"] else "Maryland 2022") + (" · control" if r["control"] else "") for r in rows]
    ax.set_yticks(y,labels)
    ax.set(xlim=(0,108),ylim=(3.65,-.65),xlabel="Corn area seen (%)")
    ax.set_title("A  Corn area seen",loc="left",pad=20)
    grid(ax)
    ax.legend(loc="upper left",bbox_to_anchor=(-.05,-.15),ncol=2,frameon=False,fontsize=11)

    ax = fig.add_axes([.60,.25,.34,.53])
    ax.barh(y-.17,[r["figure"][0] for r in rows],height=.30,color=GREEN)
    ax.barh(y+.17,[r["figure"][1] for r in rows],height=.30,color=BLUE)
    for i,r in enumerate(rows):
        for j,v in enumerate(r["figure"]):
            ax.text(v+.65,i+(-.17 if j==0 else .17),f"{v:.1f}%",va="center",fontsize=11)
        official=r["usda"]
        if isinstance(official,tuple):
            ax.plot(official,[i,i],lw=6,color=GRAY,solid_capstyle="butt")
            ax.text(official[1]+.7,i-.39,f"USDA {official[0]:.1f}–{official[1]:.1f}%",fontsize=9,color=GRAY)
        else:
            ax.plot([official,official],[i-.35,i+.35],color=GRAY,lw=2)
            ax.text(official+.7,i-.39,f"USDA {official:.1f}%",fontsize=9,color=GRAY)
    ax.set_yticks(y,[])
    ax.set(xlim=(0,58),ylim=(3.65,-.65),xlabel="Corn area called silage (%)")
    ax.set_title("B  Silage share vs. USDA",loc="left",pad=20)
    grid(ax)
    footer(fig,"Sources: docs/pitch/pitch_numbers.py, LANDSAT_PAIR; docs/RESULTS.md, ‘Landsat’; HANDOFF.md\nSeen area: two-week harvest-midpoint windows with sufficient clear-tier area. USDA: Census 2022 (Maryland), survey 2023 / official band (Wisconsin).\nMarathon remains far below USDA; Racine is the already-visible control and moves above its official band after adding Landsat.")
    save(fig,"06_landsat.png")


def timeline():
    raw = subprocess.check_output(["git","log","origin/main","--author=ChenSiyun1234","--format=%aI"],cwd=ROOT,text=True)
    times=[datetime.fromisoformat(v.replace("Z","+00:00")).astimezone(PACIFIC) for v in raw.splitlines()]
    hours=(END-START).total_seconds()/3600
    inside=[t for t in times if START<=t<=END]
    elapsed=np.array([(t-START).total_seconds()/3600 for t in inside])
    edges=np.r_[np.arange(0,hours,1),hours]
    counts,_=np.histogram(elapsed,bins=edges)
    # The log includes literal pipe characters in the event text. Its first two
    # columns and final column are fixed; keep internal pipes in the event text.
    entries=[]
    for line in read("handoff/results/ACCESS_LOG.md").splitlines():
        if line.startswith("|"):
            cells=[c.strip() for c in line.strip("|").split("|")]
            clock=re.match(r"^(\d{2}:\d{2})",cells[0])
            if clock:
                entries.append((clock.group(1),cells[1],"|".join(cells[2:-1]),cells[-1]))
    entries=pd.DataFrame(entries,columns=["Time","Who","What","Changed"])
    # This checked-in log ends at 19:23 on Nov 14. Later events named in pitch
    # notes are not inserted into the log or presented as log-backed milestones.
    specs=[("13:53","Plan declared in advance","Frozen model\ndeclared"),
           ("17:02","MODULE_V2, PROMPTED","Evidence-tier\nrule"),
           ("18:02","EVALUATION 2 RESULT","Beltsville\nresult"),
           ("19:11","MICHIGAN 2022 RESULT","Michigan\ncounty check"),
           ("19:23","CORRECTION","Double-crop\ncorrection")]
    events=[]
    for clock,needle,label in specs:
        rows=entries[(entries.Time==clock)&entries.iloc[:,2].str.contains(needle,regex=False,na=False)]
        assert len(rows)==1,f"Missing or ambiguous access-log event: {needle}"
        hh,mm=map(int,rows.iloc[0].Time.split(":"))
        day=START.date() if hh>=START.hour else END.date()
        when=datetime.combine(day,datetime.min.time(),tzinfo=PACIFIC).replace(hour=hh,minute=mm)
        events.append(((when-START).total_seconds()/3600,clock,label))
    last_clock=entries.Time.str.extract(r"^(\d{2}:\d{2})$").dropna().iloc[-1,0]
    fig=plt.figure(figsize=(16,8.8))
    title(fig,f"A {hours:g}-hour build window",f"{len(inside)} of {len(times)} commits fall inside the window: classifier, data pipeline, evaluation and docs · Pacific time (UTC−8)")
    ax=fig.add_axes([.08,.49,.87,.31])
    ax.bar(edges[:-1]+np.diff(edges)/2,counts,width=np.diff(edges)*.87,color=GREEN)
    for center,count in zip(edges[:-1]+np.diff(edges)/2,counts):
        if count:
            ax.text(center,count+.6,str(count),ha="center",fontsize=9)
    ax.set(xlim=(0,hours),ylim=(0,max(counts)*1.15),ylabel="Commits / bin")
    ticks=[0,3.5,7.5,11.5,15.5,19.5,hours]
    ticklabels=[(START+timedelta(hours=h)).strftime("%b %d\n%H:%M") for h in ticks]
    ax.set_xticks(ticks,ticklabels)
    grid(ax,"y")
    ax=fig.add_axes([.08,.15,.87,.26])
    ax.set(xlim=(0,hours),ylim=(-1.5,1.45))
    ax.axis("off")
    ax.plot([0,hours],[0,0],color=GRID,lw=4)
    # Labels are staggered in lanes; anchors remain at the recorded times.
    label_x=[3.38,5.65,7.65,9.7,11.7]
    for i,((x,clock,label),lx) in enumerate(zip(events,label_x)):
        top=i%2==0
        yy=.62 if top else -.65
        ax.scatter(x,0,color=BLUE,s=65,zorder=3)
        ax.annotate(f"{clock}\n{label}",xy=(x,0),xytext=(lx,yy),ha="center",va="bottom" if top else "top",fontsize=10,
                    arrowprops={"arrowstyle":"-","color":MUTED,"lw":1,"connectionstyle":"angle,angleA=90,angleB=0,rad=0"})
    ax.text(17.9,.62,f"Checked-in access log ends\nNov 14 · {last_clock}",ha="center",fontsize=12,color=MUTED)
    footer(fig,"Sources: git log origin/main --author=ChenSiyun1234 --format=%aI; handoff/results/ACCESS_LOG.md, timestamped rows\nWindow: Nov 14, 10:30 → Nov 15, 09:00, 2026. Bins start at 10:30; the final bin spans half an hour. Each commit is counted once.\nOnly milestones present in this access-log copy are labelled; later Landsat, final test and handover milestones are absent from that file.")
    save(fig,"07_timeline.png")
    print(f"Timeline: {len(times)} total commits; {len(inside)} inside window; {len(times)-len(inside)} outside")


def evidence(numbers):
    gap=numbers["GAP_EVIDENCE"]
    labelled=numbers["LABELLED_BY_EVIDENCE"]
    module=read("docs/module.md")
    tier_table=next(t for t in tables(module) if list(t.columns)==["tier","rule","used for"])
    clear_limit=int(first_number(tier_table.loc[tier_table.tier=="clear","rule"].iloc[0]))
    wide_limits=re.search(r"bracket (\d+)-(\d+)",tier_table.loc[tier_table.tier=="wide","rule"].iloc[0]).groups()
    documented=re.search(r"\(([\d.]+) % of fields at 5-10 days, ([\d.]+) % at 15-20",module)
    assert [float(v) for v in documented.groups()] == [v for _,v in gap]
    documented_tiers=re.search(r"clear tier (\d+) field-years, (\d+) wrong; wide tier (\d+), (\d+)",module)
    assert tuple(map(int,documented_tiers.groups())) == labelled["clearly_seen"]+labelled["partly_seen"]
    replay=pd.read_csv(ROOT/"handoff/results/thinning_replay.csv")
    replay=replay[(replay["set"]=="Beltsville (eu2_logistic_v1)")&(replay.tier=="all")]
    replay=replay[replay.thinning.isin(["gap 0 d","gap 28 d","gap 35 d"])]
    fig=plt.figure(figsize=(16,9))
    title(fig,"Why the observation gap matters", "Separate an uncertain view of harvest from the crop-use prediction itself")
    ax=fig.add_axes([.08,.27,.25,.47])
    values=[v for _,v in gap]
    ax.bar([0,1],values,color=[GREEN,BLUE],width=.55)
    for i,v in enumerate(values):
        ax.text(i,v+2,f"{v:.1f}%",ha="center",fontsize=15,weight="bold")
    ax.set_xticks([0,1],[r[0].replace("gap of ","").replace(" to ","–") for r in gap])
    ax.set(ylim=(0,75),ylabel="Fields called silage (%)",xlabel="Observed harvest bracket")
    ax.set_title("A  Unlabelled county",loc="left",pad=22)
    grid(ax,"y")
    fig.text(.08,.16,"Queen Anne's: more silage calls\nwith wider gaps. This is not an error rate.",fontsize=11,color=MUTED)

    ax=fig.add_axes([.405,.27,.23,.47])
    keys=["clearly_seen","partly_seen"]
    rates=[labelled[k][1]/labelled[k][0]*100 for k in keys]
    ax.bar([0,1],rates,width=.55,color=[GREEN,BLUE])
    for i,k in enumerate(keys):
        n,wrong=labelled[k]
        ax.text(i,rates[i]+.65,f"{wrong}/{n} wrong\n{rates[i]:.1f}%",ha="center",fontsize=13,weight="bold")
    ax.scatter(0,0,color=GREEN,s=45,zorder=5,clip_on=False)
    ax.set_xticks([0,1],[f"Clear\n≤{clear_limit} days",f"Wide\n{wide_limits[0]}–{wide_limits[1]} days"])
    ax.set(ylim=(0,13),ylabel="Wrong field-years (%)")
    ax.set_title("B  Labelled tier check",loc="left",pad=22)
    grid(ax,"y")
    fig.text(.405,.16,"Beltsville regression check: original\npredictions, grouped by evidence tier.",fontsize=11,color=MUTED)

    ax=fig.add_axes([.73,.27,.22,.47])
    vals=replay.grain_called_silage.to_numpy()*100
    ax.bar(np.arange(len(vals)),vals,color=[GREEN,BLUE,BLUE],width=.55)
    for i,v in enumerate(vals):
        ax.text(i,v+1.5,f"{v:.1f}%",ha="center",fontsize=13,weight="bold")
    ax.set_xticks(range(len(vals)),["None","28 days","35 days"])
    ax.set(ylim=(0,50),ylabel="Grain called silage (%)",xlabel="Observations hidden near harvest")
    ax.set_title("C  Labelled gap replay",loc="left",pad=22)
    grid(ax,"y")
    fig.text(.73,.16,"Beltsville: artificial gaps increase\nfalse silage calls on labelled grain.",fontsize=11,color=MUTED)
    covered=sum(labelled[k][0] for k in keys)
    total=numbers["LABELLED_TOTAL"]
    footer(fig,f"Sources: docs/pitch/pitch_numbers.py, GAP_EVIDENCE / LABELLED_BY_EVIDENCE; docs/module.md; handoff/results/thinning_replay.csv\nPanels have different denominators and scales: unlabelled field calls, labelled tier errors, and grain false-positive rates under artificial gaps.\nThe two tier groups shown cover {covered} of {total} labelled field-years; the other {total-covered} were abstained. Gap-replay percentages are averaged over replays.")
    save(fig,"08_evidence_tiers.png")


def banner():
    # Award placement: README.md, opening paragraph. The prize amount and judge
    # affiliations requested for the banner must also have a checked-in source.
    texts=[read(p) for p in ("README.md","HANDOFF.md","docs/RESULTS.md","docs/pitch/pitch_numbers.py")]
    record="\n".join(texts)
    award=re.search(r"Grand Prize, 1st place",record)
    amount=re.search(r"\$5,000",record)
    judges=re.search(r"Judges from USDA, NVIDIA, Niantic Spatial, UC Davis",record)
    if not (award and amount and judges):
        print("Skipped 00_banner.png: checked-in sources do not establish the requested prize amount and judge affiliations.")
        return
    fig=plt.figure(figsize=(24,7))
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,24),ylim=(0,7)); ax.axis("off")
    cmap=LinearSegmentedColormap.from_list("field_green",["#163B2A",GREEN])
    ax.imshow(np.tile(np.linspace(0,1,2400),(700,1)),extent=(0,24,0,7),cmap=cmap,aspect="auto")
    for sign in (-1,1):
        ax.add_patch(Polygon([(3.2+sign*.2,3.2),(3.2+sign*1.4,1),(3.2+sign*.7,1.25),(3.2+sign*.15,1.0),(3.2-sign*.4,3.0)],facecolor=GOLD,alpha=.8))
    ax.add_patch(Circle((3.2,4),1.7,facecolor=GOLD,edgecolor=PAPER,lw=2))
    ax.add_patch(Circle((3.2,4),1.48,fill=False,edgecolor=GREEN,lw=2))
    ax.text(3.2,4.25,re.search(r"1",award.group()).group(),ha="center",va="center",fontsize=100,weight="bold",color=GREEN)
    ax.text(3.2,3.25,"GRAND PRIZE",ha="center",fontsize=18,weight="bold",color=GREEN)
    ax.text(6.2,5.15,"FieldVision",fontsize=70,weight="bold",color=PAPER)
    ax.text(6.25,4.25,"Silage vs. grain corn from satellite time series",fontsize=27,color=PAPER)
    ax.text(6.25,3.10,"Hack the Harvest 2026 · USDA NASS challenge",fontsize=23,color=PAPER)
    ax.text(6.25,2.35,"Grand Prize · 1st place · "+amount.group(),fontsize=26,color=GOLD,weight="bold")
    ax.text(6.25,1.60,judges.group(),fontsize=21,color=PAPER)
    save(fig,"00_banner.png")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    numbers=runpy.run_path(str(ROOT/"docs/pitch/pitch_numbers.py"))
    pipeline(numbers)
    features()
    validation(numbers)
    confusion(numbers)
    nass(numbers)
    landsat(numbers)
    timeline()
    evidence(numbers)
    banner()


if __name__ == "__main__":
    main()
