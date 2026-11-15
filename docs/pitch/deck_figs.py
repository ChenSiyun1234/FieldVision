"""Draws the figure bodies for the slide deck. The slide carries the message as its title, so
these figures have no title of their own.

    python docs/pitch/deck_figs.py --pictures <folder with the field picture strips>

Writes PNGs to docs/pitch/deck/. Numbers come from pitch_numbers.py (each with its source).
The picture strips are the true-colour chips made by the extraction work (not in git); without
--pictures only the charts are drawn. Needs matplotlib and Pillow.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

sys.dont_write_bytecode = True
import draw as D  # noqa: E402
import pitch_numbers as N  # noqa: E402

D.DECK = True
plt.rcParams.update({"font.family": "Arial"})
BG, INK, INK2, BLUE, GREY, GRID = D.BG, D.INK, D.INK2, D.BLUE, D.GREY, D.GRID
OUT = D.HERE / "deck"
STATE_BAND = "#e6eef8"  # behind the rows that are a whole state in 2025


def body(height=6.48):
    """A figure the size of a slide body: 12.8 in wide, 1920 px."""
    return plt.figure(figsize=(12.8, height), dpi=150, facecolor=BG)


def put(fig, name):
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{name}.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    # flat colours and text only: a 256-colour palette keeps the look and more than halves the file
    Image.open(out).convert("RGB").quantize(colors=256, dither=Image.Dither.NONE).save(out, compress_level=9)
    print(f"{out.name}: {out.stat().st_size // 1024} KB")


def chips(pictures, source, name):
    """The row of true-colour chips from a picture strip, without its heading and curve."""
    im = Image.open(Path(pictures) / source)
    sx, sy = im.width / 1200, im.height / 540
    im.crop((int(50 * sx), int(76 * sy), int(1186 * sx), int(300 * sy))).save(OUT / f"{name}.png")
    print(f"{name}.png: from {source}")


def county_map(maps, source, name):
    """The map part of a county map sheet (every corn field coloured by its call), without its heading."""
    im = Image.open(Path(maps) / source)
    w, h = im.size
    im.crop((int(0.085 * w), int(0.125 * h), int(0.625 * w), int(0.965 * h))).save(OUT / f"{name}.png")
    print(f"{name}.png: from {source}")


# --- every place with a figure, against USDA -----------------------------------------------------
def places():
    def pct(v):
        return f"{v:.2f}%" if round(v * 100) % 10 else f"{v:.1f}%"

    wi, pa = N.WISCONSIN_2025, N.PENNSYLVANIA_2025
    fig = body()
    ax = fig.add_axes([0.2, 0.262, 0.79, 0.638])
    D.clean(ax, grid=None)
    ax.set_xlim(-0.8, 116)
    usda_x, ours_x, beside_x = 65, 74, 76
    step = 0.78
    tuning = [r for r in N.PLACES if r[0] == "tuning"]
    tests = [r for r in N.PLACES if r[0] == "test"]
    top = 0.7 + step * (len(tuning) + len(tests) + len(N.PLACES_TO_OPEN)) + 1.9
    ax.set_ylim(0, top + 1.15)
    y_of = {}
    y = top
    ax.text(0, y, "Used while tuning: four Maryland counties, 2022", fontsize=13.5, fontweight="bold", color=INK, va="center")
    for row in tuning:
        y -= step
        y_of[row[1]] = y
    y -= 0.95
    ax.text(0, y, "Tests: never looked at before, each run once", fontsize=13.5, fontweight="bold", color=INK, va="center")
    for row in tests:
        y -= step
        y_of[row[1]] = y
    lowest = y
    y -= 0.95
    ax.text(0, y, "Opened last, once each. Our prediction, written first: the satellite alone reads low in dairy states",
            fontsize=13.5, fontweight="bold", color=INK, va="center")
    for name, usda, band, ours in N.PLACES_TO_OPEN:
        y -= step
        ax.text(-1.6, y, name, ha="right", va="center", fontsize=14.5, fontweight="bold", color=INK)
        if band:
            ax.plot(band, [y, y], color=INK, lw=8, alpha=0.16, solid_capstyle="round", zorder=1)
        ax.scatter([usda], [y], s=125, marker="D", color=INK, edgecolors=BG, linewidths=1.5, zorder=6)
        ax.text(usda_x, y, pct(usda), fontsize=14, color=INK, ha="right", va="center")
        if ours is None:
            ax.text(ours_x, y, "to come", fontsize=12.5, color=INK2, ha="right", va="center", style="italic")
        else:
            ax.plot([min(usda, ours), max(usda, ours)], [y, y], color=GREY, lw=2.2, zorder=2)
            ax.scatter([ours], [y], s=150, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=5)
            ax.text(ours_x, y, pct(ours), fontsize=14.5, fontweight="bold", color=BLUE, ha="right", va="center")
            if name == "Wisconsin 2025":
                ax.text(beside_x, y, f"{wi['points_low']:.1f} low (pass mark {wi['pass_mark']}); unreliable by our rule",
                        fontsize=12.5, color=INK, va="center")
            if name == "Pennsylvania 2025":
                ax.text(beside_x, y, f"MISSED by {pa['points_low']}; {pa['single_crop']}% without double crops", fontsize=12.5,
                        color=INK, fontweight="bold", va="center")
    ax.text(usda_x, top + 0.75, "USDA", fontsize=13, color=INK2, ha="right", va="center")
    ax.text(ours_x, top + 0.75, "Ours", fontsize=13, fontweight="bold", color=BLUE, ha="right", va="center")
    ax.text(beside_x, top + 0.75, "Beside it", fontsize=13, color=INK2, va="center")
    for _, name, usda, band, plain, window, unseen in N.PLACES:
        y = y_of[name]
        by_window = unseen > N.NOT_SEEN_SWITCH
        ours, other = (window, plain) if by_window else (plain, window)
        missed = N.PLACE_MISSES.get(name)
        ax.text(-1.6, y, name, ha="right", va="center", fontsize=14.5, fontweight="bold", color=INK)
        if band:
            ax.plot(band, [y, y], color=INK, lw=8, alpha=0.16, solid_capstyle="round", zorder=1)
        ax.plot([min(usda, ours), max(usda, ours)], [y, y], color=GREY, lw=2.2, zorder=2)
        ax.scatter([other], [y], s=95, facecolors=BG, edgecolors=GREY, linewidths=2.2, zorder=3)
        ax.scatter([ours], [y], s=150, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=5)
        ax.scatter([usda], [y], s=125, marker="D", color=INK, edgecolors=BG, linewidths=1.5, zorder=6)
        ax.text(usda_x, y, pct(usda), fontsize=14, color=INK, ha="right", va="center")
        ax.text(ours_x, y, pct(ours), fontsize=14.5, fontweight="bold", color=BLUE, ha="right", va="center")
        if by_window:
            beside = f"plain count {pct(plain)}; {unseen:.0f}% not seen"
        else:
            beside = f"by window {pct(window)}"
        if name.startswith("Maryland 2025"):
            dc = N.STATE_2025["Maryland"]["with_double_crop_share"]
            ax.scatter([dc], [y], s=150, facecolors=BG, edgecolors=BLUE, linewidths=2.4, zorder=4)
            beside += f"; {pct(dc)} with double crops"
        if missed:
            beside = f"MISSED by {missed:g} points; " + beside
        ax.text(beside_x, y, beside, fontsize=12.5, color=INK if missed else INK2, fontweight="bold" if missed else "normal",
                va="center")
    ticks = [0, 10, 20, 30, 40, 50]
    for x in ticks:
        ax.plot([x, x], [0, top - 0.45], color=GRID, lw=1, zorder=0)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks])
    handles = [
        plt.Line2D([], [], marker="D", ls="", markersize=10, markerfacecolor=INK, markeredgecolor=BG, label="USDA"),
        plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=BLUE, markeredgecolor=BG, label="ours"),
        plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor=BG, markeredgecolor=GREY, markeredgewidth=2.2,
                   label="the other way of adding up"),
        plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=BG, markeredgecolor=BLUE, markeredgewidth=2.4,
                   label="with double-crop corn, scored afterwards"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.0), ncol=4, frameon=False, fontsize=12.5,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.5)
    fig.text(0.025, 0.02, "Share of the corn area that is silage. USDA: Census 2022; Wisconsin counties, the 2023 survey; states, NASS 2025. "
             "Ours = the plain count of the clearly seen fields;\n"
             f"where cloud hid more than {N.NOT_SEEN_SWITCH}% of the corn (Queen Anne's, Kent), the count by harvest window. "
             "Grey band: the range USDA's own figures allow\n"
             "(the fields we score, in Carroll and Frederick; Census 2022 to survey 2023, in Wisconsin). "
             f"In Marathon {N.MARATHON_NOT_JUDGED}% of the fields could not be judged.\n"
             f"Wisconsin 2025: {wi['fields']:,} fields in {wi['blocks']} random blocks, {wi['seen']}% of the corn seen; a warning check "
             f"went off (two grain counties read {wi['guard']['ours']}% for USDA's {wi['guard']['census']}%).\n"
             f"Pennsylvania 2025: {pa['fields']:,} fields in {pa['blocks']} blocks, {pa['seen']}% seen, no warning. Both states were read "
             "with two satellites and include double-crop corn, by the rule written after Maryland.",
             fontsize=11.5, color=INK2, va="bottom", linespacing=1.4)
    put(fig, "s05_places")


# --- the scoreboard: every place against USDA, in plain view ----------------------------------------
def scoreboard():
    def pct(v):
        return f"{v:.2f}%" if round(v * 100) % 10 else f"{v:.1f}%"

    zones = [("Little silage", "we are close", lambda u: u < 8),
             ("Some silage", "off by 4 to 6 points", lambda u: 8 <= u < 30),
             ("Much silage", "we read far too low", lambda u: u >= 30)]
    opened = [("test", name if "," in name else name + " (state)", usda, band, ours, ours, 0.0)
              for name, usda, band, ours in N.PLACES_TO_OPEN if ours is not None]
    rows = sorted(N.PLACES + opened, key=lambda r: r[2])
    wi, pa = N.WISCONSIN_2025, N.PENNSYLVANIA_2025
    fig = body()
    ax = fig.add_axes([0.36, 0.2, 0.63, 0.73])
    D.clean(ax, grid=None)
    ax.set_xlim(-0.8, 108)
    y = 0.0
    placed, spans = [], []
    for zone_name, verdict, inside in zones:
        members = [r for r in rows if inside(r[2])]
        first = y
        for r in members:
            placed.append((y, r))
            y -= 1.0
        spans.append((zone_name, verdict, first, y + 1.0))
        y -= 0.55
    bottom = y + 0.55
    ax.set_ylim(bottom - 0.75, 0.9)
    for yy, (group, name, usda, band, plain, window, unseen) in placed:
        ours = window if unseen > N.NOT_SEEN_SWITCH else plain
        missed = name.replace(" (state)", "") in N.PLACE_MISSES
        ax.text(-1.6, yy, name.replace(" (state)", ""), ha="right", va="center", fontsize=14.5, fontweight="bold", color=INK)
        ax.plot([min(usda, ours), max(usda, ours)], [yy, yy], color=GREY, lw=2.4, zorder=2)
        ax.scatter([ours], [yy], s=170, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=5)
        ax.scatter([usda], [yy], s=140, marker="D", color=INK, edgecolors=BG, linewidths=1.5, zorder=6)
        ax.text(57, yy, f"USDA {pct(usda)}", fontsize=13.5, color=INK2, va="center")
        ax.text(74.5, yy, f"we say {pct(ours)}", fontsize=14.5, fontweight="bold", color=INK if missed else BLUE, va="center")
        ax.text(94.5, yy, "tuned here" if group == "tuning" else "test", fontsize=12, color=INK2, va="center",
                style="italic" if group == "tuning" else "normal")
        if name.endswith("(state)"):
            ax.axhspan(yy - 0.5, yy + 0.5, color=STATE_BAND, lw=0, zorder=0)
    for x in (0, 10, 20, 30, 40, 50):
        ax.plot([x, x], [bottom - 0.6, 0.6], color=GRID, lw=1, zorder=0)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    # the three zones, named down the left edge of the figure
    to_fig = lambda yy: fig.transFigure.inverted().transform(ax.transData.transform((0, yy)))[1]  # noqa: E731
    for zone_name, verdict, y_top, y_low in spans:
        mid = (to_fig(y_top) + to_fig(y_low)) / 2
        last = zone_name.startswith("Much")
        fig.text(0.02, mid + 0.028, zone_name, fontsize=17, fontweight="bold", color=INK, va="center")
        fig.text(0.02, mid - 0.028, verdict, fontsize=14, color=INK if last else BLUE, fontweight="bold", va="center")
        fig.add_artist(plt.Line2D([0.172, 0.172], [to_fig(y_low) - 0.03, to_fig(y_top) + 0.03], color=GREY, lw=2,
                                  transform=fig.transFigure))
    handles = [
        plt.Line2D([], [], marker="D", ls="", markersize=11, markerfacecolor=INK, markeredgecolor=BG, label="what USDA reports"),
        plt.Line2D([], [], marker="o", ls="", markersize=12, markerfacecolor=BLUE, markeredgecolor=BG, label="what we read from space"),
        plt.Line2D([], [], marker="s", ls="", markersize=13, markerfacecolor=STATE_BAND, markeredgecolor=STATE_BAND,
                   label="a whole state in 2025 (what the organisers score)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=3, frameon=False, fontsize=13,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.8)
    waiting = [p[0] for p in N.PLACES_TO_OPEN if p[3] is None]
    tests = {5: "five", 6: "six", 7: "seven", 8: "eight"}[sum(1 for r in rows if r[0] == "test")]
    more = f" Still to come: {'; '.join(waiting)}." if waiting else ""
    fig.text(0.02, 0.02, "Share of the corn area that is silage. The four Maryland counties (2022) were used to tune the method; "
             f"the other {tests} places were tests, each run once.\n"
             f"Wisconsin 2025 (random state sample, two satellites): {wi['points_low']:.1f} points low, right on our pass mark of "
             f"{wi['pass_mark']}; we had expected a low reading (we wrote \"below {wi['predicted_below']}%\"; it is {wi['ours_exact']}%).\n"
             f"A warning check went off there (two grain counties read {wi['guard']['ours']}% where USDA's Census says "
             f"{wi['guard']['census']}%); by the rule we wrote down first, that marks this figure as unreliable.\n"
             f"Pennsylvania 2025 (same procedure, {pa['seen']}% of the corn seen, no warning): {pa['points_low']} points low, a clear miss; "
             f"{pa['single_crop']}% without double-crop corn." + more,
             fontsize=11.5, color=INK2, va="bottom", linespacing=1.4)
    put(fig, "s06_scoreboard")


# --- a second satellite in the three counties where clouds hid the harvest, and in a control county
def landsat_pair():
    rows = N.LANDSAT_PAIR
    cloudy = [r for r in rows if not r["control"]]
    control = [r for r in rows if r["control"]]
    fig = body()
    fig.text(0.25, 0.915, "Corn area we could see", fontsize=16, fontweight="bold", color=INK, va="center")
    fig.text(0.6, 0.915, "Our silage figure, against USDA's", fontsize=16, fontweight="bold", color=INK, va="center")
    axl = fig.add_axes([0.25, 0.17, 0.27, 0.71])
    axr = fig.add_axes([0.6, 0.17, 0.38, 0.71])
    D.clean(axl, grid=None)
    D.clean(axr)
    step = 2.1
    y = step * (len(rows) - 1) + 1.5
    top = y + 1.45
    placed = []
    for group, head in ((cloudy, "Clouds hid the harvest"), (control, "Control: not hit by cloud")):
        axl.text(-5, y + 1.0, head, ha="right", va="center", fontsize=12.5, color=INK2, style="italic")
        for r in group:
            placed.append((y, r))
            y -= step
        y -= 0.8
    for yc, r in placed:
        one, two = r["seen"]
        axl.barh(yc + 0.3, one, height=0.5, color=GREY, zorder=3)
        axl.barh(yc - 0.3, two, height=0.5, color=BLUE, zorder=3)
        axl.text(one + 2.5, yc + 0.3, f"{one}%", va="center", fontsize=15, color=INK)
        axl.text(two - 2.5, yc - 0.3, f"{two}%", va="center", ha="right", fontsize=15, fontweight="bold", color="#ffffff", zorder=4)
        axl.text(-5, yc + 0.32, r["name"], ha="right", va="center", fontsize=15.5, fontweight="bold", color=INK)
        axl.text(-5, yc - 0.36, r["when"], ha="right", va="center", fontsize=10.5, color=INK2, linespacing=1.25)
        before, after = r["figure"]
        usda = r["usda"]
        if isinstance(usda, tuple):
            towards = abs(after - min(max(after, usda[0]), usda[1])) < abs(before - min(max(before, usda[0]), usda[1]))
            axr.plot(usda, [yc, yc], color=INK, lw=9, alpha=0.16, solid_capstyle="round", zorder=1)
            said = f"USDA says {usda[0]} to {usda[1]}%"
        else:
            towards = abs(after - usda) < abs(before - usda)
            axr.scatter([usda], [yc], s=150, marker="D", color=INK, edgecolors=BG, linewidths=1.5, zorder=6)
            said = f"USDA says {usda}%"
        colour = BLUE if towards else INK
        axr.annotate("", xy=(after, yc), xytext=(before, yc), zorder=3,
                     arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.6, mutation_scale=20, shrinkA=7, shrinkB=7))
        axr.scatter([before], [yc], s=120, facecolors=BG, edgecolors=GREY, linewidths=2.4, zorder=4)
        axr.scatter([after], [yc], s=160, color=colour, edgecolors=BG, linewidths=1.5, zorder=5)
        axr.text(0.8, yc + 0.6, f"{before}%  →  {after}%", fontsize=16, fontweight="bold", color=colour, va="center")
        axr.text(22.5, yc + 0.6, said, fontsize=13.5, color=INK, va="center")
        if isinstance(usda, tuple) and after > usda[1]:
            axr.text(after + 3.5, yc, f"now {after - usda[1]:.1f} points above USDA's range", fontsize=12.5, color=INK2,
                     va="center", style="italic")
    for ax, right in ((axl, 100), (axr, 54)):
        ax.set_xlim(0, right)
        ax.set_ylim(y + 0.8 + step - 1.0, top)
        ax.set_yticks([])
    axl.set_xticks([0, 50, 100])
    axl.set_xticklabels(["0%", "50%", "100%"])
    axr.set_xticks([0, 10, 20, 30, 40, 50])
    axr.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    handles = [
        plt.Line2D([], [], marker="s", ls="", markersize=11, markerfacecolor=GREY, markeredgecolor=GREY, label="one satellite (Sentinel-2)"),
        plt.Line2D([], [], marker="s", ls="", markersize=11, markerfacecolor=BLUE, markeredgecolor=BLUE,
                   label="two satellites (Sentinel-2 and Landsat)"),
        plt.Line2D([], [], marker="D", ls="", markersize=10, markerfacecolor=INK, markeredgecolor=BG, label="what USDA reports"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.24, 1.005), ncol=3, frameon=False, fontsize=13,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.8)
    fig.text(0.02, 0.015, "Checks on four counties we had already seen, so not tests. In the control county, already well seen, "
             "the figure changed little and slipped one point past USDA's range.\n"
             "In Marathon the rest of the gap is not cloud: silage in the north is cut after the crop has yellowed, "
             "and that needs labelled northern fields.", fontsize=11.5, color=INK2, va="bottom", linespacing=1.4)
    put(fig, "s_landsat")


# --- starting from USDA's last figure: which way did the pictures move it? --------------------------
def history_update():
    def pct(v):
        return f"{v:.1f}%"

    counties, states = N.HISTORY_UPDATE, N.HISTORY_UPDATE_STATES
    fig = body()
    ax = fig.add_axes([0.2, 0.225, 0.79, 0.65])
    D.clean(ax, grid=None)
    ax.set_xlim(-20, 31)
    n = len(counties) + len(states) + 1  # one empty row between the two groups
    ax.set_ylim(-0.7, n + 0.35)
    ax.plot([0, 0], [-0.7, n - 0.4], color=INK, lw=1.6, zorder=1)
    ax.text(0, n - 0.15, "the true figure", ha="center", va="bottom", fontsize=12.5, color=INK)
    for x in (-15, -10, -5, 5):
        ax.plot([x, x], [-0.7, n - 0.4], color=GRID, lw=1, zorder=0)
    ax.text(9, n - 0.15, "USDA's last figure  →  with our pictures", fontsize=12.5, color=INK2, va="bottom")
    ax.text(25.5, n - 0.15, "true", fontsize=12.5, color=INK2, va="bottom")
    ax.text(-20.6, n - 0.15, "Counties seen while tuning", ha="right", va="bottom", fontsize=12.5, color=INK2, style="italic")
    ax.text(-20.6, len(states), "Whole states, 2025", ha="right", va="center", fontsize=12.5, color=INK2, style="italic")
    placed = [(n - 1 - i, r) for i, r in enumerate(counties)] + [(len(states) - 1 - i, r) for i, r in enumerate(states)]
    for y, (name, last, updated, true, band) in placed:
        a, b = last - true, updated - true
        # "into USDA's range" only when the last figure lay outside the range and the updated one lies inside it
        inside = band is not None and band[0] <= updated <= band[1] and not band[0] <= last <= band[1]
        closer = abs(b) < abs(a)
        if band:
            ax.plot([band[0] - true, band[1] - true], [y, y], color=INK, lw=9, alpha=0.14, solid_capstyle="round", zorder=1)
        good = closer or inside
        ax.annotate("", xy=(b, y), xytext=(a, y), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=BLUE if good else INK, lw=2.6, mutation_scale=20, shrinkA=6, shrinkB=6))
        ax.scatter([a], [y], s=110, facecolors=BG, edgecolors=GREY, linewidths=2.4, zorder=4)
        ax.scatter([b], [y], s=150, color=BLUE if good else INK, edgecolors=BG, linewidths=1.5, zorder=5)
        ax.text(-20.6, y, name, ha="right", va="center", fontsize=14.5, fontweight="bold", color=INK)
        ax.text(9, y, f"{pct(last)}  →  {pct(updated)}", fontsize=14, color=INK, va="center")
        ax.text(25.5, y, pct(true), fontsize=14, color=INK, va="center", fontweight="bold")
        word = "closer" if closer else ("into USDA's range" if inside else "further away")
        ax.text(17.5, y, word, fontsize=11.5, color=BLUE if good else INK, va="center",
                fontweight="normal" if good else "bold")
    ax.set_yticks([])
    ax.set_xticks([-15, -10, -5, 0, 5])
    ax.set_xticklabels(["15 below", "10 below", "5 below", "right", "5 above"])
    handles = [
        plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor=BG, markeredgecolor=GREY, markeredgewidth=2.4,
                   label="USDA's last figure before that year"),
        plt.Line2D([], [], marker="o", ls="", markersize=12, markerfacecolor=BLUE, markeredgecolor=BG,
                   label="the same figure, updated field by field with our pictures"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=2, frameon=False, fontsize=13,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.8)
    m, wi, pa = N.HISTORY_UPDATE_MISS, N.WISCONSIN_2025, N.PENNSYLVANIA_2025
    fig.text(0.02, 0.02, "The eight counties had already been seen while tuning, so they are not a test. One satellite (Sentinel-2), "
             "as for Maryland 2025; the other two states were read with two.\n"
             "USDA's last figure: the 2017 Census for the 2022 counties, the 2022 survey for the Wisconsin counties, "
             "the 2024 survey for the three states.\n"
             f"Mean miss on the eight counties: last figure alone {m['last']} points, updated {m['updated']}; without Marathon "
             f"{m['last_no_marathon']} and {m['updated_no_marathon']}. The satellite on its own: {m['satellite']}.\n"
             "Wisconsin and Pennsylvania 2025 were tests with the rule written first: the update had to end closer than the last figure "
             f"alone. It ended {wi['updated_off']} and {pa['updated_off']} points off, against {wi['last_official_off']} and "
             f"{pa['last_official_off']}.",
             fontsize=11.5, color=INK2, va="bottom", linespacing=1.4)
    put(fig, "s_update")


# --- Marathon: the silage weeks were the cloudy weeks ---------------------------------------------
def marathon_gap():
    g = N.MARATHON_GAP
    fig = plt.figure(figsize=(8.6, 5.0), dpi=200, facecolor=BG)
    ax = fig.add_axes([0.09, 0.12, 0.89, 0.66])
    D.clean(ax, grid="y")
    w = 0.36
    xs = range(len(g["months"]))
    sep = g["months"].index("Sep")
    ax.axvspan(sep - 0.46, sep + 1.1, color=GRID, alpha=0.55, zorder=0)
    ax.text(sep + 0.32, 7.3, "silage is cut here", ha="center", va="center", color=INK2, fontsize=12)
    ax.bar([x - w / 2 - 0.02 for x in xs], g["marathon"], width=w, color=BLUE, zorder=3)
    ax.bar([x + w / 2 + 0.02 for x in xs], g["racine"], width=w, color=GREY, zorder=3)
    for x, (a, b) in enumerate(zip(g["marathon"], g["racine"])):
        if g["months"][x] in ("Aug", "Sep", "Oct"):
            ax.text(x - w / 2 - 0.02, a + 0.15, f"{a:.1f}", ha="center", va="bottom", color=INK, fontsize=13,
                    fontweight="bold" if x == sep else "normal")
            ax.text(x + w / 2 + 0.02, b + 0.15, f"{b:.1f}", ha="center", va="bottom", color=INK2, fontsize=12)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(g["months"], color=INK, fontsize=12.5)
    ax.set_ylim(0, 7.9)
    ax.set_yticks([0, 2, 4, 6])
    ax.tick_params(axis="y", labelsize=12, colors=INK2)
    ax.set_ylabel("clear looks per field in the month", color=INK2, fontsize=12)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Marathon County (half its corn is silage)"),
        plt.Rectangle((0, 0), 1, 1, color=GREY, label="Racine County (almost all grain)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.99), ncol=1, frameon=False, fontsize=12.5,
               labelcolor=INK2, handlelength=1.1, handletextpad=0.5)
    put(fig, "marathon_gap")


# --- slide 7: why simple won ---------------------------------------------------------------------
def transfer_summary():
    fig = body()
    fig.text(D.LEFT, 0.95, "Accuracy (ROC-AUC) with each of three European countries held out of training in turn. "
             "8,000 labelled fields.", fontsize=13.5, color=INK2, va="top")
    ax = fig.add_axes([0.35, 0.285, 0.42, 0.575])
    D.clean(ax)
    ax.tick_params(axis="x", labelsize=14)
    rows = N.TRANSFER_SUMMARY
    n, lo, hi = len(rows), 0.75, 0.86
    for i, (name, sub, out, loss, us) in enumerate(rows):
        y = n - 1 - i
        if loss is not None:
            ax.plot([out, out + loss], [y, y], color=GREY, lw=3, solid_capstyle="round", zorder=2)
            ax.scatter([out + loss], [y], s=170, facecolors=BG, edgecolors=GREY, linewidths=2.6, zorder=3)
        ax.scatter([out], [y], s=200, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=4)
        ax.text(lo + 0.002, y + 0.33, f"{out:.3f}", fontsize=19, color=INK, va="center",
                fontweight="bold" if i == 0 else "normal")
        ax.text(lo + 0.019, y + 0.33, f"loses {loss:.3f}" if loss is not None else "loss not measured here",
                fontsize=14, color=INK2, va="center")
        ax.text(lo - 0.004, y + 0.13, name, fontsize=18, fontweight="bold", color=INK, ha="right", va="center")
        ax.text(lo - 0.004, y - 0.2, sub, fontsize=12.5, color=INK2, ha="right", va="center")
        ax.text(hi + 0.008, y, us, fontsize=20, color=INK, va="center", fontweight="bold" if i == 0 else "normal")
    ax.text(hi + 0.008, n - 0.3, "On the 44 labelled\nUS field-years", fontsize=13.5, color=INK2, va="center", linespacing=1.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.55, n - 0.05)
    ax.set_yticks([])
    ax.set_xticks([0.76, 0.78, 0.80, 0.82, 0.84])
    handles = [
        plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=BLUE, markeredgecolor=BG,
                   label="in a country it never saw (number shown)"),
        plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=BG, markeredgecolor=GREY,
                   markeredgewidth=2.4, label="in a country it trained on"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.34, 0.15), ncol=2, frameon=False, fontsize=13.5,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.6)
    pairs = "; ".join(f"{name} {small:g}% against {seq:g}%" for name, small, seq in N.SEQUENCE_PARTLY_SEEN)
    fig.text(D.LEFT, 0.02, "On the counties the two read equally well, so the small model was kept: it can be explained in one sentence.\n"
             "Where pictures are missing around the harvest, the small model calls too much silage and the sequence model does not:\n"
             f"{pairs} (share of such fields called silage).", fontsize=12, color=INK2, va="bottom", linespacing=1.4)
    put(fig, "s07_why_simple_won")


# --- slide 9: tuning 1, look at the evidence -------------------------------------------------------
def gap_evidence():
    fig = body()
    fig.text(0.05, 0.93, "Queen Anne's County: share of fields called silage", fontsize=16, fontweight="bold", color=INK, va="center")
    fig.text(0.05, 0.875, "by the gap between the last image before the cut and the first image after it", fontsize=12.5,
             color=INK2, va="center")
    ax = fig.add_axes([0.07, 0.14, 0.36, 0.66])
    D.clean(ax, grid="y")
    for i, (label, value) in enumerate(N.GAP_EVIDENCE):
        ax.bar(i, value, width=0.56, color=BLUE if i == 0 else GREY, zorder=3)
        ax.text(i, value + 2, f"{value:g}%", ha="center", va="bottom", fontsize=24, fontweight="bold", color=INK)
    ax.set_xticks(range(len(N.GAP_EVIDENCE)))
    ax.set_xticklabels([g[0] for g in N.GAP_EVIDENCE], fontsize=14, color=INK)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels(["0%", "20%", "40%", "60%"], fontsize=12.5, color=INK2)
    ax.set_ylim(0, 74)
    ax.set_xlim(-0.6, 1.6)

    bx = fig.add_axes([0.5, 0.0, 0.5, 1.0])
    bx.set_xlim(0, 10)
    bx.set_ylim(0, 10)
    bx.axis("off")
    bx.text(0.2, 9.3, "So every field is sorted by that gap", fontsize=16, fontweight="bold", color=INK, va="center")
    tiers = [("Clearly seen", "12 days or less", "counted"), ("Partly seen", "12 to 20 days", "shown beside the count"),
             ("Not judged", "over 20 days, or no harvest found", "left out")]
    for i, (name, rule, use) in enumerate(tiers):
        y = 7.9 - i * 1.45
        bx.add_patch(FancyBboxPatch((0.2, y - 0.42), 2.9, 0.84, boxstyle="round,pad=0,rounding_size=0.18",
                                    fc=BLUE if i == 0 else BG, ec=BLUE if i == 0 else INK2, lw=1.5))
        bx.text(1.65, y, name, fontsize=15, fontweight="bold", color="#ffffff" if i == 0 else INK, ha="center", va="center")
        bx.text(3.4, y + 0.22, rule, fontsize=14.5, color=INK, va="center")
        bx.text(3.4, y - 0.26, use, fontsize=12.5, color=INK2, va="center")
    r = N.REPLAY
    bx.text(0.2, 3.0, "The same thing on fields whose answer we know", fontsize=16, fontweight="bold", color=INK, va="center")
    bx.text(0.2, 1.85, f"Hide four to five weeks of pictures around the cut, and grain\ncalled silage rises from {r['full']}% to "
            f"{r['hidden'][0]}-{r['hidden'][1]}%. The few fields still\nclearly seen stay at {r['clear'][0]}-{r['clear'][1]}%.",
            fontsize=14.5, color=INK, va="center", linespacing=1.45)
    put(fig, "s09_gap_evidence")


# --- slide 10: tuning 2, the honest test ------------------------------------------------------------
def maryland_test():
    fig = body()
    fig.text(D.LEFT, 0.95, "Limits written down before either county was opened. Each county run once. "
             "Share of the county's corn area that is silage.", fontsize=13.5, color=INK2, va="top")
    ax = fig.add_axes([0.14, 0.26, 0.56, 0.5])
    D.clean(ax)
    ys = [1, 0]
    for (name, census, read, verdict, passband, weak), y in zip(N.MARYLAND_TEST, ys):
        ax.barh(y, passband[1] - passband[0], left=passband[0], height=0.5, color=BLUE, alpha=0.22, zorder=1)
        for a, b in weak:
            ax.barh(y, b - a, left=a, height=0.5, color=GREY, alpha=0.38, zorder=1)
        ax.scatter([census], [y], s=190, marker="D", color=INK, edgecolors=BG, linewidths=1.5, zorder=5)
        ax.scatter([read], [y], s=260, marker="X", color=INK, edgecolors=BG, linewidths=1.5, zorder=6)
        ax.text(-0.7, y, name, ha="right", va="center", fontsize=18, fontweight="bold", color=INK)
        ax.text(27.5, y + 0.12, f"read {read:.1f}%,  Census {census:.1f}%", fontsize=16, color=INK, va="center", clip_on=False)
        ax.text(27.5, y - 0.2, verdict, fontsize=16, fontweight="bold", color=INK, va="center", clip_on=False)
    ax.set_xlim(-0.5, 26)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xticklabels(["0%", "5%", "10%", "15%", "20%", "25%"])
    handles = [
        plt.Line2D([], [], marker="D", ls="", markersize=12, markerfacecolor=INK, markeredgecolor=BG, label="USDA Census 2022"),
        plt.Line2D([], [], marker="X", ls="", markersize=14, markerfacecolor=INK, markeredgecolor=BG, label="what our county step read"),
        plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.22, label="pass"),
        plt.Rectangle((0, 0), 1, 1, fc=GREY, alpha=0.38, label="weak"),
        plt.Rectangle((0, 0), 1, 1, fc=BG, ec=GRID, label="fail: anywhere else"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.13, 0.09), ncol=5, frameon=False, fontsize=13,
               labelcolor=INK2, handletextpad=0.4, columnspacing=1.6)
    fig.text(D.LEFT, 0.83, "The step was a local correction: a cut-off taken from a nearly all-grain county next door.",
             fontsize=13.5, color=INK2, va="center")
    put(fig, "s10_maryland_test")


# --- slide 11: tuning 3, adding up by harvest window ------------------------------------------------
def window_fix():
    fig = body(3.4)
    ax = fig.add_axes([0.13, 0.31, 0.83, 0.52])
    D.clean(ax, grid=None)
    n = len(N.WINDOW_FIX)
    for i, (name, census, plain, window, cloudy) in enumerate(N.WINDOW_FIX):
        y = n - 1 - i
        ax.plot([min(census, plain, window), max(census, plain, window)], [y, y], color=GRID, lw=2.4, zorder=1)
        ax.scatter([plain], [y], s=120, facecolors=BG, edgecolors=GREY, linewidths=2.4, zorder=3)
        ax.scatter([window], [y], s=135, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=4)
        ax.scatter([census], [y], s=110, marker="D", color=INK, edgecolors=BG, linewidths=1.5, zorder=5)
        ax.text(-0.5, y, name, ha="right", va="center", fontsize=14.5, fontweight="bold", color=INK)
        note = f"Census {census:.1f}%:  {plain:.1f}%  →  {window:.1f}%"
        if cloudy:
            note += f"     ({cloudy}% cut in the cloudy fortnight)"
        ax.text(23, y, note, fontsize=13, color=INK, va="center")
    for x in (0, 5, 10, 15, 20):
        ax.plot([x, x], [-0.5, n - 0.5], color=GRID, lw=1, zorder=0)
    ax.set_xlim(-0.3, 48)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels(["0%", "5%", "10%", "15%", "20%"])
    handles = [
        plt.Line2D([], [], marker="D", ls="", markersize=10, markerfacecolor=INK, markeredgecolor=BG, label="USDA Census 2022"),
        plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=BG, markeredgecolor=GREY, markeredgewidth=2.4,
                   label="plain count of the clearly seen fields"),
        plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=BLUE, markeredgecolor=BG,
                   label="added up by two-week harvest window"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.0), ncol=3, frameon=False, fontsize=12.5,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.6)
    other = " / ".join(f"{v:g}" for v in N.WINDOW_FIX_OTHER_DATE)
    fig.text(0.025, 0.085, f"Found on these four counties after Kent failed, so it is not a test. With the harvest date taken as the middle of the gap,\n"
             f"the same idea reads {other}%. So it is shown beside the count, not instead of it.", fontsize=12, color=INK2,
             va="center", linespacing=1.4)
    put(fig, "s11_window_fix")


# --- slide 12: tuning 4, what are Kent's early green fields? ----------------------------------------
def crop_history():
    fig = body(3.2)
    ax = fig.add_axes([0.17, 0.2, 0.36, 0.62])
    D.clean(ax)
    n = len(N.KENT_CALLS_HISTORY)
    for i, (crop, share) in enumerate(N.KENT_CALLS_HISTORY):
        y = n - 1 - i
        hay = crop.startswith("Hay")
        ax.barh(y, share, height=0.56, color=BLUE if hay else GREY, zorder=3)
        ax.text(share + 1.5, y, f"{share}%", va="center", fontsize=16, fontweight="bold", color=INK)
        ax.text(-1.5, y, crop, va="center", ha="right", fontsize=14.5, fontweight="bold", color=INK)
    lo, hi = N.FREDERICK_CALLS_HAY
    ax.barh(0, hi - lo, left=lo, height=0.56, color=BLUE, alpha=0.25, zorder=2)
    ax.text(hi + 3, 0, f"Frederick's silage calls: {lo} to {hi}%", va="center", fontsize=12.5, color=INK2)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    fig.text(0.025, 0.93, "Kent's early fields that we call silage: what else grew on that ground (share of their acres)",
             fontsize=14, fontweight="bold", color=INK, va="center")
    early, total = N.KENT_CALLS_CUT_EARLY
    fig.text(0.6, 0.66, f"{early} of {total} were cut by 2 September.", fontsize=15, color=INK, va="center")
    fig.text(0.6, 0.47, "Soybean and wheat ground with almost no hay\nis grain ground, not dairy ground.", fontsize=15, color=INK,
             va="center", linespacing=1.4)
    fig.text(0.6, 0.22, "Most likely grain cut early while still green.\nNo labelled set contains such fields, and the\n"
             "picture alone cannot tell them from silage.", fontsize=13, color=INK2, va="center", linespacing=1.4)
    put(fig, "s12_crop_history")


# --- slide 13: the clean test ------------------------------------------------------------------------
def michigan():
    fig = body()
    fig.text(D.LEFT, 0.95, "Two Michigan counties, 2022, never looked at before. Rule written first: Gratiot at least 10 points above "
             "Branch,\nneither more than 10 points off the Census. Each county run once.", fontsize=13.5, color=INK2, va="top",
             linespacing=1.45)
    ax = fig.add_axes([0.14, 0.2, 0.52, 0.52])
    D.clean(ax)
    rows = N.MICHIGAN_2022
    for i, (name, census, count, seen) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([census, count], [y, y], color=GREY, lw=3, zorder=2)
        ax.scatter([census], [y], s=220, marker="D", color=INK, edgecolors=BG, linewidths=1.5, zorder=5)
        ax.scatter([count], [y], s=260, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=4)
        ax.text(-1.2, y + 0.08, name, ha="right", va="center", fontsize=18, fontweight="bold", color=INK)
        ax.text(-1.2, y - 0.2, "little silage" if census < 10 else "much silage", ha="right", va="center", fontsize=12.5, color=INK2)
        ax.text(35.5, y + 0.13, f"we read {count:.1f}%", fontsize=22, fontweight="bold", color=BLUE, va="center", clip_on=False)
        ax.text(35.5, y - 0.2, f"Census {census:.1f}%   ({seen}% of the corn clearly seen)", fontsize=13.5, color=INK2, va="center",
                clip_on=False)
    ax.set_xlim(-0.5, 34)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(["0%", "10%", "20%", "30%"])
    ax.set_xlabel("share of the county's corn area that is silage", fontsize=12.5, color=INK2, labelpad=10)
    handles = [
        plt.Line2D([], [], marker="D", ls="", markersize=12, markerfacecolor=INK, markeredgecolor=BG, label="USDA Census 2022"),
        plt.Line2D([], [], marker="o", ls="", markersize=13, markerfacecolor=BLUE, markeredgecolor=BG,
                   label="plain count of the clearly seen fields"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.13, 0.02), ncol=2, frameon=False, fontsize=13,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.8)
    fig.text(0.72, 0.1, f"Result: {N.MICHIGAN_GAP} points apart. Passed.", fontsize=15, fontweight="bold", color=INK, va="center")
    put(fig, "s13_michigan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pictures", help="folder with strip_*.png (true-colour chips with the NDVI curve)")
    ap.add_argument("--maps", help="folder with map_<fips>_<year>.png (county maps of the field calls)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    D.module_diagram()
    D.labelled_fields()
    D.state_2025()
    places()
    scoreboard()
    landsat_pair()
    history_update()
    marathon_gap()
    transfer_summary()
    gap_evidence()
    maryland_test()
    window_fix()
    crop_history()
    michigan()
    if args.pictures:
        # the title slide gets the larger rendering of its strip when the maps folder has one
        big = Path(args.maps or args.pictures) / "strip_frederick_silage_clear.png"
        src = big.parent if big.exists() and big.stat().st_size > 20_000 else args.pictures
        chips(src, "strip_frederick_silage_clear.png", "pic_frederick_silage")
        chips(args.pictures, "strip_frederick_grain_clear.png", "pic_frederick_grain")
        chips(args.pictures, "strip_kent_cloudy_spell.png", "pic_kent_cloudy")
        chips(args.pictures, "strip_kent_silage_clear_early.png", "pic_kent_early")
    if args.maps:
        for fips, name in (("24029", "map_kent"), ("24021", "map_frederick")):
            if (Path(args.maps) / f"map_{fips}_2022.png").exists():
                county_map(args.maps, f"map_{fips}_2022.png", name)
        sheet = Path(args.maps) / "before_after_24021_2022.png"
        if sheet.exists():  # twelve fields just before and just after the cut: keep the chips, drop the sheet's own labels
            im = Image.open(sheet)
            w, h = im.size
            crop = im.crop((int(0.177 * w), int(0.148 * h), int(0.979 * w), int(0.977 * h))).convert("RGB")
            crop.save(OUT / "pic_before_after.jpg", quality=90)  # a photograph: JPEG keeps it small
            print("pic_before_after.jpg: from", sheet.name)
