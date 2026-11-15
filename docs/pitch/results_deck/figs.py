"""Charts and maps for the FieldVision deck, in the harvest theme (dark field green).

Every number comes from the repository's results: handoff/results/REPORT.md, HANDOFF.md,
handoff/results/agents/*.json, the price clean check, the planner and the field calls.
Colours: silage #14a37f, grain #c98500, USDA #4b8fe2 (validated pair-wise for colour-blind separation),
silage-share ramp #11654e..#9be3c6 (validated as an ordered scale).

    python figs.py <out_dir>
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

matplotlib.use("Agg")
REPO = Path.home() / "hack-the-harvest" / "FieldVision"
HERE = Path(__file__).parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "figs")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))

BG, PANEL, LINE, TEXT, MUTED, FAINT = "#0b1410", "#13201a", "#2a3d33", "#eef3ee", "#a7b8ad", "#73887b"
SILAGE, GRAIN, USDA, ABSTAIN, GOLD, FAIL = "#14a37f", "#c98500", "#4b8fe2", "#73887b", "#f2c14e", "#ff7a6b"
RAMP = ["#11654e", "#13845f", "#14a37f", "#4fc79f", "#9be3c6"]
BINS = [2, 5, 10, 20]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "text.color": TEXT, "axes.labelcolor": MUTED,
    "axes.edgecolor": LINE, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": PANEL,
    "figure.facecolor": PANEL, "savefig.facecolor": PANEL, "axes.titlecolor": TEXT, "axes.titleweight": "bold",
    "axes.titlesize": 16, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.8, "legend.frameon": False, "legend.labelcolor": MUTED,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", name)


def field_test():
    rows = [("Small model: 6 numbers, logistic (ours)", 42, SILAGE), ("Big model, observation dates dropped", 39, "#4fc79f"),
            ("Harvest-timing rule", 38, "#4fc79f"), ("Big model: boosted trees, 135 numbers", 37, "#4fc79f"),
            ("Always say grain", 32, ABSTAIN), ("TempDiff-SMID, published method (Agronomy 2025)", 22, ABSTAIN)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(rows))[::-1]
    for yi, (label, v, c) in zip(y, rows):
        ax.barh(yi, v, color=c, height=0.62)
        ax.text(v + 0.4, yi, f"{v} of 44", va="center", fontsize=14, fontweight="bold" if c == SILAGE else "normal",
                color=TEXT if c == SILAGE else MUTED)
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(0, 47); ax.set_xticks([0, 11, 22, 33, 44])
    ax.axvline(44, color=FAINT, lw=1, ls=":"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Labelled US field-years called right (Beltsville, 2021–2023; 12 silage, 32 grain)")
    ax.tick_params(axis="y", length=0, labelsize=13)
    save(fig, "f_field_test")


def loco():
    fam = ["Harvest-timing rule", "Timing + moisture rule", "Small model (logistic)", "Big model (boosted trees)", "Composite"]
    vals = {"Austria": [.654, .746, .754, .770, .756], "France": [.694, .772, .826, .805, .800], "Netherlands": [.638, .665, .717, .713, .724]}
    marks = {"Austria": ("o", SILAGE), "France": ("s", GRAIN), "Netherlands": ("D", USDA)}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(fam))[::-1]
    means = [.662, .728, .765, .763, .760]          # as logged at 17:49 in handoff/results/ACCESS_LOG.md
    for i, f in enumerate(fam):
        v = [vals[c][i] for c in vals]
        ax.plot([min(v), max(v)], [y[i]] * 2, color=LINE, lw=6, solid_capstyle="round", zorder=1)
        ax.text(0.86, y[i], f"mean {means[i]:.3f}", va="center", color=TEXT if i == 2 else MUTED, fontsize=13,
                fontweight="bold" if i == 2 else "normal")
    for c, (m, col) in marks.items():
        ax.scatter(vals[c], y, marker=m, s=90, color=col, edgecolor=PANEL, linewidth=1.5, zorder=3, label=f"{c} held out")
    ax.set_yticks(y, fam); ax.set_xlim(0.6, 0.92); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ROC-AUC on a country the model never saw (0.5 = guessing, 1.0 = perfect)")
    ax.legend(loc="lower right", ncol=3, bbox_to_anchor=(1, 1.0), fontsize=12)
    save(fig, "f_loco")


def counties():
    rows = [("Queen Anne's, MD", 1.3, 4.0, "used in tuning"), ("Kent, MD", 5.5, 12.9, "used in tuning"),
            ("Carroll, MD", 8.9, 3.6, "used in tuning"), ("Frederick, MD", 19.6, 13.2, "used in tuning"),
            ("Branch, MI", 1.8, 2.1, "never looked at"), ("Gratiot, MI", 24.2, 30.0, "never looked at")]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(rows))[::-1]
    for yi, (name, census, ours, tag) in zip(y, rows):
        ax.plot([census, ours], [yi, yi], color=LINE, lw=4, zorder=1)
        ax.scatter([census], [yi], s=140, color=USDA, zorder=3, edgecolor=PANEL, linewidth=1.5)
        ax.scatter([ours], [yi], s=140, color=SILAGE, zorder=3, marker="D", edgecolor=PANEL, linewidth=1.5)
        ax.text(1.02, yi, tag, va="center", fontsize=12, color=GOLD if tag == "never looked at" else FAINT,
                transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(0, 32); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Silage share of corn, % (2022)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=USDA, markersize=10, label="USDA Census 2022"),
                       Line2D([], [], marker="D", ls="", color=SILAGE, markersize=9, label="Our count of clearly seen fields")],
              loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=12)
    save(fig, "f_counties")


def states():
    s = json.loads((REPO / "handoff/results/agents/states_2025_comparison.json").read_text())
    rows = [(r["state"], r["usda_2025_silage_pct"], r["headline_pct"]) for r in s["shares"] if r["model"] == "eu2_logistic_v1"]
    rows.append(("MD", 8.9, 4.75))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(rows)); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, [r[1] for r in rows], w, color=USDA, label="USDA 2025 survey")
    b2 = ax.bar(x + w / 2 + 0.01, [r[2] for r in rows], w, color=SILAGE, label="Our model (sample of fields)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():g}%", ha="center", fontsize=12, color=MUTED)
    names = {"PA": "Pennsylvania", "WI": "Wisconsin", "IA": "Iowa", "NE": "Nebraska", "MD": "Maryland"}
    ax.set_xticks(x, [names[r[0]] for r in rows]); ax.grid(axis="x", visible=False); ax.tick_params(axis="x", length=0, labelsize=13)
    ax.set_ylabel("Silage share of harvested corn, %"); ax.set_ylim(0, 37)
    ax.legend(loc="upper right", fontsize=12)
    save(fig, "f_states")


def price():
    from fieldvision_agents.app import dashboard
    rows = dashboard.price_models()
    base = next(r for r in rows if r["model"].startswith("no change"))
    other = [r for r in rows if r is not base]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    labels = [r["model"].replace("explain: ", "Explains: ").replace("forecast: ", "Forecasts: ") for r in other]
    y = np.arange(len(other))[::-1]
    cols = [SILAGE if r["mape_pct"] < base["mape_pct"] else GRAIN for r in other]
    ax.barh(y, [r["mape_pct"] for r in other], color=cols, height=0.6)
    for yi, r in zip(y, other):
        ax.text(r["mape_pct"] + 0.2, yi, f"{r['mape_pct']}%", va="center", fontsize=13, color=MUTED)
    ax.axvline(base["mape_pct"], color=USDA, lw=2, ls="--")
    ax.text(base["mape_pct"] + 0.15, len(other) - 0.45, f"last year's price: {base['mape_pct']}%", color=USDA, fontsize=12)
    ax.set_yticks(y, labels); ax.set_xlim(0, 23); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Average error, each year 2001–2024 predicted by a model fitted without it")
    save(fig, "f_price_models")


def whatif():
    rows = [("Supply only", 0.85), ("Linear stocks-to-use", 4.2), ("Regional embedding", 4.99), ("Stocks + last price", 5.01),
            ("Convex stocks-to-use", 5.19), ("Stocks-to-use only", 9.38)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=GRAIN, height=0.6)
    for yi, (_, v) in zip(y, rows):
        ax.text(v + 0.15, yi, f"+{v}%", va="center", fontsize=13, color=MUTED)
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(0, 11); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Corn price change if national silage share were 1 point higher in 2025 (971,350 acres; 181 M bushels)")
    save(fig, "f_whatif")


def planner():
    from fieldvision_agents.app import service
    from fieldvision_agents.planner.planner import uncertainty_curve
    tb = service.toolbox(offline=True)
    for step in ("scout", "extract_features"):
        getattr(tb, step)("24021", 2022)
    tb.classify("24021", 2022, "standard")
    frame = tb._tb.planner_frame("24021", 2022)
    pts = uncertainty_curve(frame, max_visits=60, step=5, abstain_prior=frame.attrs.get("abstain_prior"))["points"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([p["visits"] for p in pts], [p["plus_minus"] for p in pts], color=SILAGE, lw=2.5, marker="o", markersize=6)
    for p in pts:
        if p["visits"] in (0, 5, 20, 60):
            ax.annotate(f"±{p['plus_minus']:,}", (p["visits"], p["plus_minus"]), textcoords="offset points", xytext=(6, 8), color=TEXT, fontsize=12)
    ax.set_xlabel("Field visits, best fields first"); ax.set_ylabel("90% range of silage acres, ±")
    ax.set_ylim(0, max(p["plus_minus"] for p in pts) * 1.15)
    save(fig, "f_planner")


def binned(v):
    return RAMP[next((i for i, b in enumerate(BINS) if v < b), len(BINS))]


def us_map():
    from fieldvision_agents.app import geo
    a = geo._acres()
    c = a[(a["level"] == "county") & (a["source"] == "CENSUS") & (a["year"] == 2022)].dropna(subset=["silage_share_pct"])
    share = dict(zip(c["fips"], c["silage_share_pct"]))
    g = gpd.read_file(HERE / "counties.geojson", on_invalid="fix")
    st = gpd.read_file(HERE / "states.geojson", on_invalid="fix")
    keep = ~g["id"].str[:2].isin(["02", "15", "72", "60", "66", "69", "78"])
    g, st = g[keep].to_crs(5070), st[~st["id"].isin(["02", "15", "72", "60", "66", "69", "78"])].to_crs(5070)
    g["color"] = [binned(share[i]) if i in share else PANEL for i in g["id"]]
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    g.plot(ax=ax, color=g["color"], edgecolor=BG, linewidth=0.15)
    st.boundary.plot(ax=ax, color=BG, linewidth=0.9)
    st[st["id"] == "24"].boundary.plot(ax=ax, color=GOLD, linewidth=2)
    ax.set_axis_off()
    labels = ["under 2%", "2–5%", "5–10%", "10–20%", "20% or more"]
    ax.legend(handles=[Patch(color=RAMP[i], label=labels[i]) for i in range(5)] + [Patch(facecolor=PANEL, edgecolor=LINE, label="withheld or no corn")],
              title="Silage share of harvested corn, Census 2022", loc="lower left", fontsize=12, title_fontsize=12, labelcolor=MUTED)
    ax.get_legend().get_title().set_color(TEXT)
    fig.savefig(OUT / "m_us_counties.png", dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor=BG)
    plt.close(fig); print("saved m_us_counties")


def frederick():
    from fieldvision_agents.app import geo
    fc = geo.field_calls("24021", 2022)
    f = gpd.GeoDataFrame.from_features(fc["features"], crs=4326).to_crs(26918)
    county = gpd.read_file(HERE / "counties.geojson", on_invalid="fix")
    county = county[county["id"] == "24021"].to_crs(26918)
    d = geo.dairy_points()
    dairies = gpd.GeoDataFrame.from_features(d["features"], crs=4326).to_crs(26918)
    dairies = dairies[dairies.within(county.buffer(15000).iloc[0])]
    colors = {"silage": SILAGE, "grain": GRAIN, "not judged": ABSTAIN}
    fig, ax = plt.subplots(figsize=(10, 11))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    county.plot(ax=ax, color="#101c16", edgecolor=LINE, linewidth=1.5)
    for call in ("grain", "not judged", "silage"):
        f[f["call"] == call].plot(ax=ax, color=colors[call], edgecolor="none")
    dairies.plot(ax=ax, color=USDA, markersize=[40 + (h or 300) / 12 for h in dairies["head"]], edgecolor=BG, linewidth=1.2)
    ax.set_axis_off()
    n = fc["counts"]
    ax.legend(handles=[Patch(color=SILAGE, label=f"called silage · {n['silage']:,}"), Patch(color=GRAIN, label=f"called grain · {n['grain']:,}"),
                       Patch(color=ABSTAIN, label=f"not judged · {n['not judged']:,}"),
                       Line2D([], [], marker="o", ls="", color=USDA, markersize=10, label="permitted dairy (size = herd)")],
              loc="lower right", fontsize=13, labelcolor=MUTED)
    fig.savefig(OUT / "m_frederick.png", dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor=BG)
    plt.close(fig); print("saved m_frederick")


if __name__ == "__main__":
    only = set(sys.argv[2:])
    for fn in (field_test, loco, counties, states, price, whatif, planner, us_map, frederick):
        if not only or fn.__name__ in only:
            fn()
