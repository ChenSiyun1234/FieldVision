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

plt.rcParams.update({"font.family": "DejaVu Sans"})


def new_fig(title, subtitle=""):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor=BG)
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
    out = HERE / f"{name}.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
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
    box(11.9, top, 3.6, h, "3  Add up", "Silage and grain acres,\nby county and by state\n\nSilage is counted only\n"
        "where the harvest was\nclearly seen.")
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
        "- Corn we could not see\n   (share of the corn area)\n- Double-crop corn\n   (acres; not scored)",
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
                  "It loses at most 0.015 of ROC-AUC; the bigger models lose 0.02 to 0.19. Two of the bigger models are still\n"
                  "ahead when Austria is the new country; in France and the Netherlands none of them is.")
    n = len(N.TRANSFER)
    lo, hi = 0.62, 0.90
    for p, country in enumerate(N.TRANSFER_COUNTRIES):
        ax = fig.add_axes([0.335 + p * 0.215, 0.265, 0.195, 0.455])
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
    fig.text(0.323, 0.747, "New country:", fontsize=12.5, color=INK2, ha="right", va="center")
    handles = [
        plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor=BLUE, markeredgecolor=BG,
                   label="tested in the new country (number shown)"),
        plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor=BG, markeredgecolor=GREY,
                   markeredgewidth=2.2, label="same model tested in a country it trained on (change in brackets)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.325, 0.085), ncol=1, frameon=False, fontsize=12,
               labelcolor=INK2, handletextpad=0.3)
    fig.text(0.335 + 0.215 + 0.0975, 0.2, "ROC-AUC  (0.5 = guessing, 1.0 = perfect)", fontsize=12.5, color=INK2,
             ha="center", va="center")
    foot(fig, "8,000 labelled maize fields: Austria 2022-2024, France 2022, Netherlands 2025. Each country left out of training in turn.")
    save(fig, "fig2_new_country")


# --- 3. the organisers' labelled fields --------------------------------------------------------------
def labelled_fields():
    best = N.LABELLED[0][2]
    fig = new_fig(f"On the organisers' labelled fields the small model gets {best} of {N.LABELLED_TOTAL}",
                  "44 field-years at Beltsville, Maryland, 2021-2023. Every model was trained on European fields only and frozen before the test.")
    ax = fig.add_axes([0.385, 0.2, 0.545, 0.55])
    rows = list(N.LABELLED) + [("Big model, dates dropped in training", "one test, set in advance", N.THINNED_BIG_MODEL_CORRECT, "extra")]
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
    foot(fig, f"44 field-years is a small test: one field is 2.3 points. The small model's two misses: one silage field missed, one grain field called silage.\n"
              f"The set was scored twice, by a plan written before the first run. First run (models trained on Austria only): "
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
                 f"{N.THINNED_BIG_MODEL_CORRECT} of 44, {verdict}.")
    foot(fig, f"135-feature boosted trees, 8,000 labelled fields in Austria, France and the Netherlands; changes made to training fields only.\n"
              f"It still loses {loss['big_dates_dropped']:.3f} out of country (before: {loss['big']:.3f}; small model: {loss['small']:.3f}). "
              f"This arm was picked after seeing six.\n{check}")
    save(fig, "fig4_date_dropping")


# --- 5. county totals against the Census -------------------------------------------------------------
def counties():
    tested = N.MICHIGAN_VERDICT is not None
    title = "County totals: our first method failed its test in Maryland"
    sub = ("Second method: count silage only where the harvest was seen. It was built on these four counties after the failure,\n"
           "so they do not test it. Its clean test is two Michigan counties"
           + (f": {N.MICHIGAN_VERDICT}." if tested else ", still to come."))
    fig = new_fig(title, sub)
    ax = fig.add_axes([0.16, 0.2, 0.8, 0.5])
    ax.set_xlim(-1.6, 50.4)
    rows = [("md", r) for r in N.MARYLAND_2022] + [("mi", r) for r in N.MICHIGAN_2022]
    ys = [6.3, 5.3, 4.3, 3.3, 1.4, 0.4]
    ax.set_ylim(-0.3, 7.4)
    clean(ax)
    name_x, note_x = -2.4, 28.5

    def note(y, line1, line2=""):
        ax.text(note_x, y + (0.17 if line2 else 0), line1, ha="left", va="center", fontsize=13, color=INK)
        if line2:
            ax.text(note_x, y - 0.21, line2, ha="left", va="center", fontsize=11.5, color=INK2)

    for (group, row), y in zip(rows, ys):
        name, census = row[0], row[1]
        ax.text(name_x, y, name, ha="right", va="center", fontsize=14.5, color=INK, fontweight="bold")
        if group == "md":
            _, _, first, verdict, second, unseen = row
            if name == "Frederick":
                ax.plot(N.FREDERICK_SCORED_RANGE, [y, y], color=INK, lw=8, alpha=0.16, solid_capstyle="round", zorder=1)
            ax.plot([min(census, second), max(census, second)], [y, y], color=GREY, lw=2.2, zorder=2)
            if first is not None:
                ax.scatter([first], [y], s=150, marker="X", color=INK2, edgecolors=BG, linewidths=1.2, zorder=4)
                ax.text(first, y + 0.36, f"first method {first:.1f}%: {verdict}", ha="center" if first > 5 else "left",
                        va="center", fontsize=12, color=INK2)
            ax.scatter([second], [y], s=135, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=5)
            line2 = f"{unseen}% of the corn area not seen at harvest" if unseen else ""
            if name == "Frederick":
                lo_r, hi_r = N.FREDERICK_SCORED_RANGE
                line2 = f"grey band: Census range for scored fields, {lo_r}-{hi_r}%"
            note(y, f"Census {census:.1f}%,  second method {second:.1f}%", line2)
        else:
            _, _, ours, unseen = row
            if ours is None:
                to_come(ax, note_x, 49.5, y, "to come: clean test, run once")
            else:
                ax.plot([min(census, ours), max(census, ours)], [y, y], color=GREY, lw=2.2, zorder=2)
                ax.scatter([ours], [y], s=135, color=BLUE, edgecolors=BG, linewidths=1.5, zorder=5)
                note(y, f"Census {census:.1f}%,  second method {ours:.1f}%",
                     f"{unseen}% of the corn area not seen at harvest" if unseen is not None else "")
        ax.scatter([census], [y], s=120, marker="D", color=INK, edgecolors=BG, linewidths=1.5, zorder=6)
    ax.text(name_x, 7.05, "Maryland 2022", ha="right", va="center", fontsize=12.5, color=INK2)
    ax.text(0.3, 7.05, "seen while the method was built", ha="left", va="center", fontsize=12.5, color=INK2)
    ax.text(name_x, 2.15, "Michigan 2022", ha="right", va="center", fontsize=12.5, color=INK2)
    ax.text(0.3, 2.15, "the clean test: one county with much silage, one with little", ha="left", va="center",
            fontsize=12.5, color=INK2)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xticklabels(["0%", "5%", "10%", "15%", "20%", "25%"])
    ax.grid(False)
    for x in (0, 5, 10, 15, 20, 25):
        ax.plot([x, x], [-0.3, 6.8], color=GRID, lw=1, zorder=0)
    ax.set_xlabel("share of the county's corn area that is silage", fontsize=12.5, color=INK2, labelpad=10, loc="left")
    handles = [
        plt.Line2D([], [], marker="D", ls="", markersize=10, markerfacecolor=INK, markeredgecolor=BG, label="USDA Census 2022"),
        plt.Line2D([], [], marker="X", ls="", markersize=12, markerfacecolor=INK2, markeredgecolor=BG,
                   label="first method, tested once on Kent and Carroll"),
        plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=BLUE, markeredgecolor=BG,
                   label="second method (not yet confirmed)" if not tested else "second method"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.775), ncol=3, frameon=False, fontsize=12.5,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.8)
    foot(fig, f"Kent was a test with limits written down first: the first method said 21.0% where the Census has 5.5% (fail line {N.KENT_FAIL_LINE}%).\n"
              f"Second method on the four seen counties: worst miss {N.SECOND_METHOD_WORST_MISS} points. Field classifier trained on European fields only.")
    save(fig, "fig5_counties")


# --- state acres, 2025 -------------------------------------------------------------------------------
def state_2025():
    md = N.STATE_2025["Maryland"]
    have = md["our_silage_acres"] is not None
    fig = new_fig("Maryland 2025: our state acres beside the USDA figure" if have
                  else "Maryland 2025 state acres: our number is still to come",
                  f"Random sample: {md['blocks_drawn']} of {md['blocks_total']} image blocks, {md['fields_drawn']:,} fields drawn. "
                  "Steps written down before any 2025 image was opened.")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    cols = (5.6, 9.6)
    ax.text(cols[0], 6.95, "Ours, from satellite", fontsize=14.5, color=BLUE, fontweight="bold", ha="center", va="center")
    ax.text(cols[1], 6.95, "USDA NASS 2025", fontsize=14.5, color=INK, fontweight="bold", ha="center", va="center")

    def fmt_acres(v):
        return f"{v:,.0f}"

    table = [
        ("Silage acres", md["our_silage_acres"], fmt_acres, fmt_acres(md["nass_silage_acres"])),
        ("Grain acres", md["our_grain_acres"], fmt_acres, fmt_acres(md["nass_grain_acres"])),
        ("Silage share of corn", md["our_silage_share"], lambda v: f"{v:.1f}%", f"{md['nass_silage_share']:.1f}%"),
    ]
    for i, (label, ours, fmt, nass) in enumerate(table):
        y = 6.15 - i * 0.85
        ax.plot([0.6, 11.4], [y - 0.425, y - 0.425], color=GRID, lw=1)
        ax.text(0.6, y, label, fontsize=16, color=INK, fontweight="bold", va="center")
        if ours is None:
            to_come(ax, cols[0] - 1.35, cols[0] + 1.35, y, "to come", half=0.3, size=14)
        else:
            ax.text(cols[0], y, fmt(ours), fontsize=22, color=BLUE, fontweight="bold", ha="center", va="center")
        ax.text(cols[1], y, nass, fontsize=22, color=INK, ha="center", va="center")

    ax.text(0.6, 3.55, "Shown beside it", fontsize=14.5, color=INK, fontweight="bold", va="center")
    beside = [
        ("Range of the silage share, from resampling the blocks", md["our_range_share"], lambda v: f"{v[0]:.1f}% to {v[1]:.1f}%"),
        ("Corn area in harvest windows we could not see", md["unseen_share"], lambda v: f"{v:.0f}%"),
        ("Upper figure: unseen windows filled from the window before", md["upper_share"], lambda v: f"{v:.1f}%"),
        ("Double-crop corn, not scored", md["double_crop_acres_not_scored"], lambda v: f"{v:,} acres"),
    ]
    for i, (label, value, fmt) in enumerate(beside):
        y = 3.0 - i * 0.52
        ax.text(0.6, y, label, fontsize=13.5, color=INK2, va="center")
        if value is None:
            to_come(ax, 8.6, 10.6, y, "to come", half=0.19, size=12)
        else:
            ax.text(8.6, y, fmt(value), fontsize=13.5, color=INK, va="center", fontweight="bold")

    ax.add_patch(FancyBboxPatch((12.1, 4.02), 3.35, 3.3, boxstyle="round,pad=0,rounding_size=0.16", fc=BG, ec=INK2,
                                lw=1.4, ls=(0, (4, 3))))
    ax.text(12.4, 6.95, "Other states, 2025", fontsize=14.5, color=INK, fontweight="bold", va="center")
    for i, state in enumerate(("Pennsylvania", "Wisconsin")):
        y = 6.15 - i * 1.1
        share = N.STATE_2025[state]["our_silage_share"]
        ax.text(12.4, y, state, fontsize=14, color=INK, va="center")
        ax.text(12.4, y - 0.42, "to come" if share is None else f"{share:.1f}% silage", fontsize=13, color=INK2,
                va="center", style="italic" if share is None else "normal")
    foot(fig, "The 2025 USDA figure was already known to us when the steps were written; no 2025 field had been scored.\n"
              "Corn in harvest windows we could not see is counted as grain, so the silage figure can read low; the upper figure shows by how much.")
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
