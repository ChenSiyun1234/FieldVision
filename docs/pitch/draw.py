"""Draws the module diagram and the result figures of the pitch pack, all in one style.

    python docs/pitch/draw.py              every figure, written next to this file
    python docs/pitch/draw.py counties     one figure by name (see FIGURES at the bottom)

The numbers are in pitch_numbers.py, each with its source. A None there is a result that is not
logged yet; it is drawn as a dashed "to come" box. Needs matplotlib only. Each PNG is 1920 x 1080.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.dont_write_bytecode = True  # keep the folder free of cache files
import pitch_numbers as N  # noqa: E402

HERE = Path(__file__).resolve().parent
BG, INK, INK2, BLUE, GREY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#b4b2a9", "#e4e2dc"
LEFT = 0.035
DECK = False  # set by deck_figs.py: the slide carries the title, so the figure leaves it out

plt.rcParams.update({"font.family": "DejaVu Sans"})


def new_fig(title, subtitle=""):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor=BG)
    if not DECK:
        fig.text(LEFT, 0.928, title, fontsize=22, fontweight="bold", color=INK, ha="left", va="center")
    if subtitle:
        fig.text(LEFT, 0.878, subtitle, fontsize=13.5, color=INK2, ha="left", va="top", linespacing=1.45)
    return fig


def foot(fig, text):
    fig.text(LEFT, 0.03, text, fontsize=11.5, color=INK2, ha="left", va="bottom", linespacing=1.45)


def clean(ax, grid="x"):
    ax.set_facecolor(BG)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(axis="x", colors=INK2, length=0, labelsize=12.5)
    ax.tick_params(axis="y", colors=INK, length=0, labelsize=14)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)


def to_come(ax, x0, x1, y, text, half=0.27, size=12.5):
    """A marked place for a number that is not logged yet."""
    ax.add_patch(FancyBboxPatch((x0, y - half), x1 - x0, 2 * half, boxstyle="square,pad=0", fc=BG, ec=INK2,
                                lw=1.4, ls=(0, (4, 3)), zorder=3))
    ax.text((x0 + x1) / 2, y, text, ha="center", va="center", color=INK2, fontsize=size, style="italic", zorder=4)


def save(fig, name):
    out = (HERE / "deck" if DECK else HERE) / f"{name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    if DECK:  # cut the empty title band off the top
        from PIL import Image
        im = Image.open(out)
        im.crop((0, int(im.height * 0.1), im.width, im.height)).save(out)
    print(f"{out.name}: {out.stat().st_size // 1024} KB")


# --- the module ------------------------------------------------------------------------------------
def module_diagram():
    fig = new_fig("The model module: from one field to county and state acres")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, head, lines, fill=BG, edge=INK, head_color=INK, text_color=INK2, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.16", fc=fill, ec=edge,
                                    lw=1.6, ls=(0, (4, 3)) if dashed else "-", zorder=2))
        ax.text(x + 0.28, y + h - 0.42, head, fontsize=16, fontweight="bold", color=head_color, va="center", zorder=3)
        ax.text(x + 0.28, y + h - 0.86, lines, fontsize=13.5, color=text_color, va="top", linespacing=1.5, zorder=3)

    def arrow(x0, y0, x1, y1, color=INK):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, lw=2, color=color,
                                     shrinkA=0, shrinkB=0, zorder=1))

    top, h = 4.5, 3.2
    box(0.5, top, 2.75, h, "In", "Field boundary\n\nSatellite images,\nApril to November")
    box(3.85, top, 3.4, h, "1  Evidence check", "Did the satellite see\nthe harvest?\n\nIf not, the field is\nnot judged.")
    box(7.85, top, 3.45, h, "2  Classifier", "Six measurements of the\ncrop at harvest\n\nSilage or grain, with\na probability",
        fill=BLUE, edge=BLUE, head_color="#ffffff", text_color="#ffffff")
    box(11.9, top, 3.6, h, "3  Add up", "Silage and grain acres,\nby county and by state\n\nThe silage share is\n"
        "measured on the clearly\nseen fields.")
    mid = top + h / 2
    for x0, x1 in ((3.25, 3.85), (7.25, 7.85), (11.3, 11.9)):
        arrow(x0 + 0.06, mid, x1 - 0.06, mid)

    # what the evidence check sorts fields into
    arrow(5.55, top - 0.06, 5.55, 3.98, color=INK2)
    ax.text(3.85, 3.8, "Sorted by the gap between the last image before\nthe cut and the first image after it:",
            fontsize=13, color=INK2, va="top", linespacing=1.45)
    rows = [("Clearly seen", "12 days or less"), ("Partly seen", "12 to 20 days"),
            ("Not judged", "over 20 days, or no harvest found")]
    for i, (name, rule) in enumerate(rows):
        y = 2.4 - i * 0.7
        ax.add_patch(FancyBboxPatch((3.85, y - 0.26), 2.0, 0.52, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=BLUE if i == 0 else BG, ec=BLUE if i == 0 else INK2, lw=1.4, zorder=2))
        ax.text(4.85, y, name, fontsize=13.5, fontweight="bold", color="#ffffff" if i == 0 else INK,
                ha="center", va="center", zorder=3)
        ax.text(6.05, y, rule, fontsize=13, color=INK2, va="center")

    # what is shown beside every total
    arrow(13.7, top - 0.06, 13.7, 3.88, color=INK2)
    box(10.6, 0.95, 4.9, 2.85, "Shown beside every total",
        "- Corn not clearly seen\n   (share of the corn area)\n- Double-crop corn\n   (acres, kept separate)",
        dashed=True, edge=INK2)

    ax.text(0.5, 0.42, "The classifier is a logistic regression trained on 8,000 labelled fields in Austria, France "
            "and the Netherlands. It has seen no US field.", fontsize=12, color=INK2, va="center")
    save(fig, "module_diagram")


# --- 1. what the model keys on -----------------------------------------------------------------------
def signal():
    fig = new_fig("What tells silage from grain: moisture just before harvest",
                  "How much accuracy the small model loses when one of its six measurements is scrambled.\n"
                  "European training data (Austria 2024, 2,400 fields), scored on areas held out of training.")
    ax = fig.add_axes([0.385, 0.2, 0.545, 0.53])
    clean(ax)
    n = len(N.FEATURES)
    for i, (name, sub, mean, spread) in enumerate(N.FEATURES):
        y = n - 1 - i
        ax.barh(y, max(mean, 0.0008), height=0.56, color=BLUE if i == 0 else GREY, zorder=3)
        ax.plot([max(mean - spread, 0), mean + spread], [y, y], color=INK2, lw=1.6, zorder=4)
        ax.text(mean + spread + 0.003, y, f"{mean:.3f}", va="center", ha="left", fontsize=14, color=INK,
                fontweight="bold" if i == 0 else "normal")
        ax.text(-0.004, y + 0.14, name, va="center", ha="right", fontsize=14.5, color=INK, fontweight="bold")
        ax.text(-0.004, y - 0.2, sub, va="center", ha="right", fontsize=11.5, color=INK2)
    ax.set_xlim(0, 0.14)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.05, 0.10])
    ax.axvline(0, color=GREY, lw=1, zorder=4)
    ax.set_xlabel("drop in ROC-AUC when the measurement is scrambled (bigger = more useful)", fontsize=12.5,
                  color=INK2, labelpad=10)
    foot(fig, "Silage is cut while the plant is still green and wet; grain is left to dry. Thin lines: spread over five held-out folds.\n"
              "The harvest date on its own adds nothing once the other five are in the model.")
    save(fig, "fig1_signal")


# --- 2. a country the model never saw ----------------------------------------------------------------
def new_country():
    fig = new_fig("In a new country the small model loses least",
                  "It loses at most 0.015 of ROC-AUC; the bigger models lose 0.02 to 0.19. Three of the bigger models are still\n"
                  "ahead when Austria is the new country; in France and the Netherlands none of them is.")
    n = len(N.TRANSFER)
    lo, hi = 0.62, 0.90
    for p, country in enumerate(N.TRANSFER_COUNTRIES):
        ax = fig.add_axes([0.335 + p * 0.215, 0.235, 0.195, 0.495])
        clean(ax)
        for i, (name, sub, cells) in enumerate(N.TRANSFER):
            out, loss = cells[p]
            inside = out + loss
            y = n - 1 - i
            ax.plot([out, inside], [y, y], color=GREY, lw=2.2, solid_capstyle="round", zorder=2)
            ax.scatter([inside], [y], s=95, facecolors=BG, edgecolors=GREY, linewidths=2.2, zorder=3)
            ax.scatter([out], [y], s=105, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=4)
            ax.text(lo + 0.004, y + 0.3, f"{out:.3f}", fontsize=13.5, color=INK, va="center",
                    fontweight="bold" if i == 0 else "normal")
            ax.text(lo + 0.076, y + 0.3, f"({-loss:+.3f})", fontsize=12, color=INK2, va="center")
            if p == 0:
                ax.text(lo - 0.012, y + 0.12, name, fontsize=14.5, fontweight="bold", color=INK, ha="right", va="center")
                ax.text(lo - 0.012, y - 0.2, sub, fontsize=11.5, color=INK2, ha="right", va="center")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.55, n - 0.2)
        ax.set_yticks([])
        ax.set_xticks([0.65, 0.75, 0.85])
        ax.set_title(country, fontsize=14.5, color=INK, loc="left", pad=10, fontweight="bold")
    fig.text(0.323, 0.757, "New country:", fontsize=12.5, color=INK2, ha="right", va="center")
    handles = [
        plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor=BLUE, markeredgecolor=BG,
                   label="tested in the new country (number shown)"),
        plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor=BG, markeredgecolor=GREY,
                   markeredgewidth=2.2, label="same model tested in a country it trained on (change in brackets)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.325, 0.07), ncol=1, frameon=False, fontsize=12,
               labelcolor=INK2, handletextpad=0.3)
    fig.text(0.335 + 0.215 + 0.0975, 0.175, "ROC-AUC  (0.5 = guessing, 1.0 = perfect)", fontsize=12.5, color=INK2,
             ha="center", va="center")
    foot(fig, "8,000 labelled maize fields: Austria 2022-2024, France 2022, Netherlands 2025. Each country left out of training in turn.")
    save(fig, "fig2_new_country")


# --- 3. the organisers' labelled fields --------------------------------------------------------------
def labelled_fields():
    best = N.LABELLED[0][2]
    thinned = N.THINNED_BIG_MODEL_CORRECT
    fig = new_fig(f"On the organisers' labelled fields the small model gets {best} of {N.LABELLED_TOTAL}",
                  "44 field-years at Beltsville, Maryland, 2021-2023. Every model was trained on European fields only and frozen before the test.")
    ax = fig.add_axes([0.385, 0.2, 0.545, 0.55])
    if thinned is None:
        extra_sub = "one more test, set in advance"
    else:
        extra_sub = "scored later, once: " + ("passed its line of 37" if thinned > 37 else "did not pass its line of 37")
    rows = list(N.LABELLED) + [("Big model, dates dropped in training", extra_sub, thinned, "other")]
    n = len(rows)
    ax.set_xlim(0, 46)
    ax.set_ylim(-0.6, n - 0.4)
    clean(ax)
    for i, (name, sub, correct, kind) in enumerate(rows):
        y = n - 1 - i
        ax.text(-0.9, y + (0.14 if sub else 0), name, va="center", ha="right", fontsize=14.5, color=INK, fontweight="bold")
        if sub:
            ax.text(-0.9, y - 0.2, sub, va="center", ha="right", fontsize=11.5, color=INK2)
        if correct is None:
            to_come(ax, 0.4, 30, y, "to come: passes if it gets more than 37")
            continue
        if kind == "baseline":
            ax.barh(y, correct, height=0.56, color=BG, edgecolor=GREY, linewidth=2, zorder=3)
        else:
            ax.barh(y, correct, height=0.56, color=BLUE if kind == "main" else GREY, zorder=3)
        ax.text(correct - 0.7, y, f"{correct} of {N.LABELLED_TOTAL}", va="center", ha="right", fontsize=14.5,
                color="#ffffff" if kind == "main" else INK, fontweight="bold" if kind == "main" else "normal", zorder=5)
    ax.axvline(0, color=GREY, lw=1, zorder=4)
    ax.set_yticks([])
    ax.set_xticks([0, 11, 22, 33, 44])
    ax.set_xlabel("field-years called correctly, out of 44", fontsize=12.5, color=INK2, labelpad=10)
    first = N.LABELLED_FIRST_RUN
    times = {2: "twice", 3: "three times", 4: "four times"}[getattr(N, "LABELLED_SCORINGS", 3 if thinned else 2)]
    foot(fig, f"44 field-years is a small test: one field is 2.3 points. The small model's two misses: one silage field missed, one grain field called silage.\n"
              f"The set was scored {times}, each time under a rule written beforehand. First run (models trained on Austria only): "
              f"small model {first['small']}, big model {first['big']}.")
    save(fig, "fig3_labelled_fields")


# --- 4. dropping observation dates in training -------------------------------------------------------
def date_dropping():
    base, thin = N.AUGMENT[0][1], [a for a in N.AUGMENT if a[2]][0][1]
    fig = new_fig("Dropping observation dates in training helps the big model travel",
                  f"Mean ROC-AUC in a country the model never saw goes from {base:.3f} to {thin:.3f}. "
                  "The other changes do not help on their own.")
    ax = fig.add_axes([0.385, 0.235, 0.545, 0.515])
    clean(ax)
    n = len(N.AUGMENT)
    x0 = 0.70
    for i, (name, value, strong) in enumerate(N.AUGMENT):
        y = n - 1 - i
        ax.barh(y, value - x0, left=x0, height=0.56, color=BLUE if strong else GREY, zorder=3)
        ax.text(value - 0.0012, y, f"{value:.3f}", va="center", ha="right", fontsize=14,
                color="#ffffff" if strong else INK, fontweight="bold" if strong else "normal", zorder=5)
        ax.text(x0 - 0.0015, y, name, va="center", ha="right", fontsize=14, color=INK,
                fontweight="bold" if strong or i == 0 else "normal")
    # the small model, for comparison: an outlined bar on top and a line behind the other bars
    small = N.AUGMENT_SMALL_MODEL
    ax.barh(n + 0.25, small - x0, left=x0, height=0.56, color=BG, edgecolor=INK, linewidth=1.6, zorder=3)
    ax.text(small - 0.0012, n + 0.25, f"{small:.3f}", va="center", ha="right", fontsize=14, color=INK, zorder=5)
    ax.text(x0 - 0.0015, n + 0.25, "Small model (6 features), for comparison", va="center", ha="right", fontsize=14, color=INK)
    ax.plot([small, small], [-0.6, n - 0.03], color=INK2, lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.set_xlim(x0, 0.80)
    ax.set_ylim(-0.6, n + 0.7)
    ax.set_yticks([])
    ax.set_xticks([0.70, 0.72, 0.74, 0.76, 0.78, 0.80])
    ax.set_xlabel("mean ROC-AUC over three held-out countries (axis starts at 0.70)", fontsize=12.5, color=INK2, labelpad=10)
    loss = N.AUGMENT_LOSS
    if N.THINNED_BIG_MODEL_CORRECT is None:
        check = "One confirmation was set in advance, on the 44 labelled US field-years (passes above 37 of 44): TO COME."
    else:
        verdict = "passed" if N.THINNED_BIG_MODEL_CORRECT > 37 else "did not pass"
        check = (f"One confirmation was set in advance, on the 44 labelled US field-years (passes above 37 of 44): "
                 f"{N.THINNED_BIG_MODEL_CORRECT} of 44, {verdict}. The small model has 42.")
    foot(fig, f"135-feature boosted trees, 8,000 labelled fields in Austria, France and the Netherlands; changes made to training fields only.\n"
              f"It still loses {loss['big_dates_dropped']:.3f} out of country (before: {loss['big']:.3f}; small model: {loss['small']:.3f}). "
              f"This arm was picked after seeing six.\n{check}")
    save(fig, "fig4_date_dropping")


# --- 5. county totals against the Census -------------------------------------------------------------
def counties():
    verdict = N.MICHIGAN_VERDICT
    if verdict == "passed":
        title = "County totals: a failure in Maryland, then a pass in Michigan"
        second = "then passed a test on two Michigan counties it had never seen. "
    elif verdict is None:
        title = "County totals: our first method failed its test in Maryland"
        second = "gets its clean test on two Michigan counties: still to come. "
    else:
        title = "County totals: our first method failed its test in Maryland"
        second = f"had its test on two Michigan counties it had never seen: {verdict}. "
    fig = new_fig(title, "The first method failed its test on Kent County. The plain count (share of the clearly seen corn called silage)\n"
                  + second + f"In Maryland it misses single counties by up to {N.PLAIN_COUNT_WORST_MISS_MD} points.")
    ax = fig.add_axes([0.16, 0.2, 0.8, 0.5])
    ax.set_xlim(-1.6, 61.4)
    rows = [("md", r) for r in N.MARYLAND_2022] + [("mi", r) for r in N.MICHIGAN_2022]
    ys = [6.3, 5.3, 4.3, 3.3, 1.4, 0.4]
    ax.set_ylim(-0.3, 7.4)
    clean(ax)
    name_x, note_x = -2.6, 35.5

    def note(y, line1, line2=""):
        ax.text(note_x, y + (0.17 if line2 else 0), line1, ha="left", va="center", fontsize=13, color=INK)
        if line2:
            ax.text(note_x, y - 0.21, line2, ha="left", va="center", fontsize=11.5, color=INK2)

    for (group, row), y in zip(rows, ys):
        name, census = row[0], row[1]
        if group == "md":
            _, _, band, first, count = row
        else:
            _, _, count, _seen = row
            band, first = None, None
        ax.text(name_x, y, name, ha="right", va="center", fontsize=14.5, color=INK, fontweight="bold")
        if band:
            ax.plot(band, [y, y], color=INK, lw=8, alpha=0.16, solid_capstyle="round", zorder=1)
        if first:
            ax.scatter([first[0]], [y], s=150, marker="X", color=INK2, edgecolors=BG, linewidths=1.2, zorder=4)
            ax.text(first[0], y + 0.36, f"first method {first[0]:.1f}%: {first[1]}", ha="center" if first[0] > 5 else "left",
                    va="center", fontsize=12, color=INK2)
        if count is None:
            to_come(ax, note_x, 60, y, "to come: clean test, run once")
        else:
            ax.plot([min(census, count), max(census, count)], [y, y], color=GREY, lw=2.2, zorder=2)
            ax.scatter([count], [y], s=135, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=5)
            line2 = (f"grey band: Census range for scored fields, {band[0]}-{band[1]}%" if band
                     else f"miss {count - census:+.1f} points")
            note(y, f"Census {census:.1f}%,  plain count {count:.1f}%", line2)
        ax.scatter([census], [y], s=120, marker="D", color=INK, edgecolors=BG, linewidths=1.5, zorder=6)
    ax.text(name_x, 7.05, "Maryland 2022", ha="right", va="center", fontsize=12.5, color=INK2)
    ax.text(0.3, 7.05, "seen while the method was built", ha="left", va="center", fontsize=12.5, color=INK2)
    ax.text(name_x, 2.15, "Michigan 2022", ha="right", va="center", fontsize=12.5, color=INK2)
    ax.text(0.3, 2.15, "the clean test, each county run once" + (f": {verdict}" if verdict else ""), ha="left",
            va="center", fontsize=12.5, color=INK2)
    ax.set_yticks([])
    ticks = [0, 5, 10, 15, 20, 25, 30]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks])
    ax.grid(False)
    for x in ticks:
        ax.plot([x, x], [-0.3, 6.8], color=GRID, lw=1, zorder=0)
    ax.set_xlabel("share of the county's corn area that is silage", fontsize=12.5, color=INK2, labelpad=10, loc="left")
    handles = [
        plt.Line2D([], [], marker="D", ls="", markersize=10, markerfacecolor=INK, markeredgecolor=BG, label="USDA Census 2022"),
        plt.Line2D([], [], marker="X", ls="", markersize=12, markerfacecolor=INK2, markeredgecolor=BG,
                   label="first method (a local cut-off), tested on Kent and Carroll"),
        plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=BLUE, markeredgecolor=BG,
                   label="plain count of the clearly seen fields"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.775), ncol=3, frameon=False, fontsize=12.5,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.8)
    rule = "Michigan rule, written first: Gratiot at least 10 points above Branch, neither more than 10 points off the Census."
    if verdict:
        rule += f" Result: {N.MICHIGAN_GAP} points apart."
    foot(fig, rule + f"\nKent: the first method read 21.0% against 5.5% (fail line {N.KENT_FAIL_LINE}%). "
              "The field classifier was trained on European fields only.")
    save(fig, "fig5_counties")


# --- state acres, 2025 -------------------------------------------------------------------------------
def state_2025():
    md = N.STATE_2025["Maryland"]
    have = md["our_silage_acres"] is not None
    low = have and md["our_silage_share_range"][1] < md["nass_silage_share"]
    if not have:
        title = "Maryland 2025 state acres: our number is still to come"
    elif low:
        title = "Maryland 2025: our silage estimate reads low"
    else:
        title = "Maryland 2025: our state acres beside the USDA figure"
    fig = new_fig(title, f"Random sample: {md['blocks_drawn']} of {md['blocks_total']} image blocks, {md['fields_scored']:,} corn fields scored. "
                  "Steps written down before any 2025 image was opened.")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    cols = (5.6, 9.6)
    ax.text(cols[0], 7.0, "Ours, from satellite", fontsize=14.5, color=BLUE, fontweight="bold", ha="center", va="center")
    ax.text(cols[1], 7.0, "USDA NASS 2025", fontsize=14.5, color=INK, fontweight="bold", ha="center", va="center")

    def acres(v):
        return f"{v:,.0f}"

    def share(v):
        return f"{v:g}%"

    table = [
        ("Silage acres", md["our_silage_acres"], md.get("our_silage_acres_range"), acres, acres(md["nass_silage_acres"])),
        ("Grain acres", md["our_grain_acres"], None, acres, acres(md["nass_grain_acres"])),
        ("Silage share of corn", md["our_silage_share"], md.get("our_silage_share_range"), share, share(md["nass_silage_share"])),
    ]
    for i, (label, ours, rng, fmt, nass) in enumerate(table):
        y = 6.15 - i * 0.95
        ax.plot([0.6, 11.4], [y - 0.5, y - 0.5], color=GRID, lw=1)
        ax.text(0.6, y, label, fontsize=16, color=INK, fontweight="bold", va="center")
        if ours is None:
            to_come(ax, cols[0] - 1.35, cols[0] + 1.35, y, "to come", half=0.3, size=14)
        else:
            ax.text(cols[0], y + (0.1 if rng else 0), fmt(ours), fontsize=22, color=BLUE, fontweight="bold", ha="center", va="center")
            if rng:
                ax.text(cols[0], y - 0.3, f"range {fmt(rng[0]).rstrip('%')} to {fmt(rng[1])}", fontsize=11.5, color=INK2,
                        ha="center", va="center")
        ax.text(cols[1], y, nass, fontsize=22, color=INK, ha="center", va="center")

    ax.text(0.6, 3.38, "Shown beside it", fontsize=14.5, color=INK, fontweight="bold", va="center")
    with_dc = None
    if md.get("with_double_crop_silage_acres"):
        with_dc = (md["with_double_crop_silage_acres"], md["with_double_crop_share"], md["with_double_crop_range"])
    window = (md["window_variant_share"], md["window_variant_range"]) if md.get("window_variant_share") else None
    beside = [
        ("Corn area clearly seen at harvest (the share is measured there)", md.get("clearly_seen_share_of_area"), lambda v: f"{v}%"),
        ("The same count weighted by harvest window", window, lambda v: f"{v[0]:g}%  (range {v[1][0]:g} to {v[1][1]:g}%)"),
        ("Double-crop corn, outside the figure above", md["double_crop_acres_not_scored"], lambda v: f"{v:,} acres"),
        ("With double-crop corn scored too (added after the result was seen)", with_dc,
         lambda v: f"{v[0]:,} acres, {v[1]:g}%  (range {v[2][0]:g} to {v[2][1]:g}%)"),
        ("Upper bound: every double-crop acre counted as silage", md.get("upper_bound_silage_acres"),
         lambda v: f"{v:,} acres, {md['share_if_all_double_crop_is_silage']:.1f}%"),
        (f"With a second satellite too, after the fact ({md.get('two_satellites_seen')}% of the corn seen)",
         md.get("two_satellites_after_the_fact"),
         lambda v: f"{v[0]:g}% with double crops  (range {v[1][0]:g} to {v[1][1]:g}%)"),
    ]
    for i, (label, value, fmt) in enumerate(beside):
        y = 2.92 - i * 0.37
        ax.text(0.6, y, label, fontsize=13.5, color=INK2, va="center")
        if value is None:
            to_come(ax, 9.15, 11.15, y, "to come", half=0.19, size=12)
        else:
            ax.text(9.15, y, fmt(value), fontsize=13.5, color=INK, va="center", fontweight="bold")

    ax.add_patch(FancyBboxPatch((12.1, 2.95), 3.35, 4.4, boxstyle="round,pad=0,rounding_size=0.16", fc=BG, ec=INK2,
                                lw=1.4, ls=(0, (4, 3))))
    ax.text(12.4, 7.0, "Other states, 2025", fontsize=14.5, color=INK, fontweight="bold", va="center")
    for i, state in enumerate(("Michigan", "New York", "Wisconsin", "Pennsylvania")):
        y = 6.25 - i * 0.87
        s = N.STATE_2025[state]["our_silage_share"]
        ax.text(12.4, y, state, fontsize=14, color=INK, va="center")
        usda = N.STATE_2025[state].get("nass_silage_share")
        ax.text(12.4, y - 0.42, "to come" if s is None else f"{s:g}% silage" + (f" (USDA {usda:g}%)" if usda else ""),
                fontsize=13, color=INK2, va="center", style="italic" if s is None else "normal")
    if low:
        first = ("USDA's figure lies outside our range. The range covers sampling only, not the method's own miss "
                 "(about 7 points on single counties in 2022).")
    else:
        first = "The range covers sampling only, not the method's own miss (about 7 points on single counties in 2022)."
    foot(fig, first + "\nThe 2025 USDA figure was already known to us when the steps were written; no 2025 field had been scored.")
    save(fig, "state_2025")


FIGURES = {
    "module": module_diagram,
    "signal": signal,
    "new_country": new_country,
    "labelled_fields": labelled_fields,
    "date_dropping": date_dropping,
    "counties": counties,
    "state": state_2025,
}

if __name__ == "__main__":
    for key in sys.argv[1:] or FIGURES:
        FIGURES[key]()
