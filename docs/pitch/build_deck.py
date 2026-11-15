"""Builds the slide deck (PowerPoint, 16:9): twelve plain slides and three backup slides.

    python docs/pitch/deck_figs.py --pictures <folder with the field picture strips> --maps <folder with the county maps>
    python docs/pitch/build_deck.py --out <folder>/FieldVision_model_story.pptx

Every slide has an editable title (the one idea of the slide), a picture or editable boxes, and
speaker notes: two or three short English sentences, then the same in Chinese. Needs python-pptx.
Numbers shown in editable text come from pitch_numbers.py.
"""
import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.dont_write_bytecode = True
import pitch_numbers as N  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "deck"
BG, INK, INK2, BLUE, GREY, WHITE = "FCFCFB", "0B0B0B", "52514E", "2A78D6", "B4B2A9", "FFFFFF"
FONT = "Arial"
W, H = 13.333, 7.5
MX = 0.6  # side margin
FULL = W - 2 * MX


def rgb(hex6):
    return RGBColor.from_string(hex6)


def new_slide(prs, title, size=30, top=0.32, height=1.0):
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # "Title Only"
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(MX), Inches(top), Inches(FULL), Inches(height)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = 0
    tf.text = title
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.LEFT
        for r in p.runs:
            r.font.name, r.font.size, r.font.bold = FONT, Pt(size), True
            r.font.color.rgb = rgb(INK)
    return slide


def text(slide, x, y, w, h, body, size=16, bold=False, color=INK2, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.1):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(body.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
        r.font.color.rgb = rgb(color)
    return box


def card(slide, x, y, w, h, fill=WHITE, line=GREY):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1.25)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def bar(slide, x, y, w, h, color):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(color)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def tile(slide, x, y, w, h, big, label, big_color=BLUE, big_size=54, label_size=16, fill=WHITE):
    """A card with one big figure and one plain sentence under it."""
    light = fill == BLUE
    card(slide, x, y, w, h, fill=fill, line=None if light else GREY)
    text(slide, x + 0.3, y + 0.2, w - 0.6, big_size / 60, big, size=big_size, bold=True, color=WHITE if light else big_color)
    text(slide, x + 0.3, y + 0.35 + big_size / 60, w - 0.6, h - 0.5 - big_size / 60, label, size=label_size,
         color=WHITE if light else INK, spacing=1.15)


def picture(slide, name, x, y, w):
    return slide.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), width=Inches(w))


def body_picture(slide, name, top=1.32, height=6.0):
    """A full figure body (1920 x 972), centred under the title."""
    width = height * 1920 / 972
    return picture(slide, name, (W - width) / 2, top, width)


def notes(slide, english, chinese):
    slide.notes_slide.notes_text_frame.text = english + "\n\n" + chinese


def build(out):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    half = (FULL - 0.4) / 2
    third = (FULL - 2 * 0.3) / 3

    # title ---------------------------------------------------------------------------------
    s = new_slide(prs, "Chopped for silage or combined for grain?\nWe read it for every corn field, from space",
                  size=36, top=0.75, height=1.9)
    text(s, MX, 2.8, FULL, 0.5, "FieldVision: the model module. Hack the Harvest.", size=20)
    picture(s, "pic_frederick_silage.png", MX, 3.75, FULL)
    text(s, MX, 6.3, FULL, 0.6, "One corn field in Frederick County, Maryland, through the 2022 season (Sentinel-2 true colour, 10 m pixels). "
         "The model calls it silage: cut while still green, around 2 September.", size=13)
    notes(s, "This is the model part of FieldVision. For each corn field we ask one question: silage or grain? "
             "We answer it from free satellite images.",
          "这是 FieldVision 的模型部分。对每一块玉米地，我们只问一个问题：青贮还是籽粒？我们用免费的卫星影像来回答。")

    # the problem and the catch --------------------------------------------------------------
    s = new_slide(prs, "A hard task, and the US has only 44 labelled fields")
    traps = [
        ("Clouds", "The harvest can fall between two clear images."),
        ("Early is not silage", "Some grain is cut early too. The date alone misleads."),
        ("Few silage fields", "Most US corn is grain. Always saying grain looks right."),
        ("Double crops", "Corn sown after a small grain follows another calendar."),
    ]
    cw = (FULL - 3 * 0.3) / 4
    for i, (head, line) in enumerate(traps):
        x = MX + i * (cw + 0.3)
        card(s, x, 1.6, cw, 2.4)
        text(s, x + 0.22, 1.82, cw - 0.44, 0.45, head, size=18, bold=True, color=INK)
        text(s, x + 0.22, 2.45, cw - 0.44, 1.4, line, size=15)
    text(s, MX, 4.15, FULL, 0.35, "The traps the organisers warned about", size=13)
    card(s, MX, 4.75, FULL, 2.1, fill=BLUE, line=None)
    text(s, MX + 0.4, 4.95, 2.4, 1.7, "44", size=80, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(s, MX + 3.0, 4.95, FULL - 3.5, 1.7, "The catch: the only labelled US fields are 44, all at one research farm in "
         "Beltsville, Maryland.\nToo few to train on. We kept them for testing only.", size=20, color=WHITE,
         anchor=MSO_ANCHOR.MIDDLE, spacing=1.15)
    notes(s, "The task has traps: clouds, early harvest, few silage fields, double crops. "
             "And the US has only 44 labelled fields, all at one farm. That is too few to train on.",
          "这个任务有几个陷阱：云、早收、青贮地块少、双季作物。而且美国带标签的地块只有 44 个，全在一个农场。这点数据不够训练。")

    # what the satellite sees ----------------------------------------------------------------
    s = new_slide(prs, "Silage is cut green. Grain is left to dry first")
    if (FIG / "pic_before_after.jpg").exists():
        ph = 5.05
        from PIL import Image
        with Image.open(FIG / "pic_before_after.jpg") as im:
            pw = ph * im.width / im.height
        px = W - MX - pw
        picture(s, "pic_before_after.jpg", px, 1.4, pw)
        lw = px - MX - 0.2
        for frac, label in ((0.125, "last look standing"), (0.381, "first look after the cut"),
                            (0.638, "last look standing"), (0.894, "first look after the cut")):
            text(s, MX, 1.4 + frac * ph - 0.05, lw, 0.4, label, size=13)
        text(s, MX, 1.4 + 0.125 * ph - 0.6, lw, 0.5, "Called silage", size=20, bold=True, color=BLUE)
        text(s, MX, 1.4 + 0.638 * ph - 0.6, lw, 0.5, "Called grain", size=20, bold=True, color=INK)
        text(s, MX, 6.6, FULL, 0.6, "Twelve real corn fields in Frederick County, Maryland, 2022, picked by rule, not by eye "
             "(two of the silage picks were cut late and look more like grain; the rule kept them). "
             "Sentinel-2 true colour, 10 m pixels. The calls are the model's: these fields have no label.", size=12)
    else:
        bw = 11.2
        bx = (W - bw) / 2
        text(s, bx, 1.42, bw, 0.35, "Called silage: still green on 31 August, cut around 2 September", size=16, bold=True, color=INK)
        picture(s, "pic_frederick_silage.png", bx, 1.8, bw)
        text(s, bx, 4.12, bw, 0.35, "Called grain: brown by late September, cut by 7 October", size=16, bold=True, color=INK)
        picture(s, "pic_frederick_grain.png", bx, 4.5, bw)
        text(s, bx, 6.85, bw, 0.4, "Two corn fields in Frederick County, Maryland, 2022. Sentinel-2 true colour, 10 m pixels. "
             "The calls are the model's: these fields have no label.", size=12)
    notes(s, "These are real satellite pictures of Maryland corn fields. Silage fields are still green just before the cut. "
             "Grain fields are already brown.",
          "这些是马里兰玉米地的真实卫星图片。青贮地在收割前还是绿的，籽粒地在收割前已经变黄。")

    # the model and its test -----------------------------------------------------------------
    best, total = N.LABELLED[0][2], N.LABELLED_TOTAL
    s = new_slide(prs, f"Field by field: a small model, trained in Europe, gets {best} of {total} right")
    lw = 5.6
    card(s, MX, 1.5, lw, 5.4)
    text(s, MX + 0.35, 1.7, lw - 0.7, 0.5, "How it works", size=20, bold=True, color=INK)
    steps = [
        ("1", "Did the satellite see the harvest? If not, the field is not judged."),
        ("2", "Six numbers describe the crop at harvest. The key one: how wet it is just before the cut."),
        ("3", "A simple formula (logistic regression) says silage or grain. It learned from 8,000 fields in Europe "
              "and from no US field."),
    ]
    for i, (num, line) in enumerate(steps):
        y = 2.45 + i * 1.45
        text(s, MX + 0.35, y - 0.08, 0.6, 0.8, num, size=40, bold=True, color=BLUE)
        text(s, MX + 1.05, y, lw - 1.4, 1.3, line, size=16, color=INK, spacing=1.15)
    rx, rw = MX + lw + 0.4, FULL - lw - 0.4
    card(s, rx, 1.5, rw, 5.4)
    text(s, rx + 0.35, 1.6, rw - 0.7, 1.4, f"{best} of {total}", size=76, bold=True, color=BLUE)
    text(s, rx + 0.35, 3.05, rw - 0.7, 0.4, "labelled US fields called right, with no US training", size=16, color=INK)
    rows = [("Small model", best, BLUE), ("Big model", N.LABELLED[1][2], GREY), ("Always say \"grain\"", N.LABELLED[-1][2], GREY)]
    scale = (rw - 3.6) / total
    for i, (name, value, color) in enumerate(rows):
        y = 4.0 + i * 0.85
        text(s, rx + 0.35, y + 0.08, 2.3, 0.4, name, size=15, bold=i == 0, color=INK)
        bar(s, rx + 2.6, y, value * scale, 0.5, color)
        text(s, rx + 2.6 + value * scale + 0.12, y + 0.06, 0.8, 0.4, str(value), size=17, bold=i == 0, color=INK)
    text(s, rx + 0.35, 6.42, rw - 0.7, 0.35, "This is the field test. Counties and the state are on the next slide.", size=13)
    notes(s, "Our model is small: six numbers and a simple formula, trained only on European fields. "
             "On the 44 labelled US fields it got 42 right. A much bigger model got 37.",
          "我们的模型很小：六个数加一个简单公式，只用欧洲的地块训练。在美国 44 个带标签的地块上，它对了 42 个。一个大得多的模型对了 37 个。")

    # every place against USDA ---------------------------------------------------------------------
    s = new_slide(prs, "Against USDA: right where silage is rare, a clear miss in the heaviest dairy county")
    body_picture(s, "s05_places.png")
    notes(s, "This is every place where we can compare our number with USDA. Where silage is rare we read right, and the Michigan "
             "test passed. In the heaviest dairy county, Marathon in Wisconsin, we read far too low, and Maryland 2025 reads low too.",
          "这是我们所有能和美国农业部数字对比的地方。青贮少的地方我们读得对，密歇根的检验通过了。"
          "在奶牛最多的威斯康星 Marathon 县，我们读得太低；马里兰 2025 也偏低。")

    # what went wrong and what we did ----------------------------------------------------------------
    s = new_slide(prs, "What went wrong, and what we did about it")
    cols = [(MX, 4.75, "What went wrong"), (MX + 4.95, 3.75, "What we did"), (MX + 8.9, FULL - 8.9, "What came out")]
    for x, w, head in cols:
        text(s, x + 0.2, 1.42, w - 0.2, 0.35, head, size=14, bold=True, color=BLUE)
    fixes = [
        ("The first county total read far too high: 22% for 1.3%",
         "Count only the fields seen clearly around the cut", "That county: 22% → 4%"),
        ("A correction with a neighbour county failed its own test: Kent 21% for 5.5%",
         "Dropped it", "Kent: 21% → 12.9%"),
        ("Cloud hid the harvest of over half the Eastern Shore's corn",
         "Add up by harvest window; report the corn not seen", "Worst miss 7.4 → 3.8 points, on counties already seen"),
        ("Grain cut early and green looks like silage",
         "Checked the crop history of those fields", "Shown, not solved"),
        ("Double-crop corn was not scored",
         "Scored it with the same model, afterwards", "Maryland 2025: 4.75% → 6.7% (USDA 8.9%)"),
        ("The big model did not travel to a new country",
         "Dropped dates in training; tried a sequence model", "37 → 39 of 44; 0.817 against 0.765 in Europe"),
        ("A northern dairy county reads far too low: Marathon 4% for 50%",
         "Half its fields could not be judged. Cause under investigation now", "Open: a clear miss"),
    ]
    rh = 0.66
    for i, cells in enumerate(fixes):
        y = 1.85 + i * (rh + 0.06)
        card(s, MX, y, FULL, rh)
        for (x, w, _), cell, strong in zip(cols, cells, (False, False, True)):
            text(s, x + 0.2, y, w - 0.3, rh, cell, size=14, bold=strong, color=INK, anchor=MSO_ANCHOR.MIDDLE, spacing=1.05)
    notes(s, "Here is the tuning in one table. Each row is a problem the data showed us, what we did, and what came out. "
             "Two are still open: early green grain, and the miss in Marathon County.",
          "这一页用一张表讲完调优过程：每一行是数据暴露出的一个问题、我们的做法和结果。"
          "有两个还没解决：趁绿早收的籽粒玉米，以及 Marathon 县读数过低。")

    # the first county total ------------------------------------------------------------------
    f = N.FIRST_COUNTY_TOTAL
    s = new_slide(prs, "Adding fields up to a county went wrong at first")
    card(s, MX, 1.6, half, 3.0)
    text(s, MX + 0.4, 1.8, half - 0.8, 0.5, "We said", size=18, bold=True, color=INK)
    text(s, MX + 0.4, 2.35, half - 0.8, 1.6, f"{f['ours']:.0f}%", size=96, bold=True, color=BLUE)
    text(s, MX + 0.4, 3.95, half - 0.8, 0.5, "of the corn is silage", size=15)
    card(s, MX + half + 0.4, 1.6, half, 3.0)
    text(s, MX + half + 0.8, 1.8, half - 0.8, 0.5, "The Census of Agriculture says", size=18, bold=True, color=INK)
    text(s, MX + half + 0.8, 2.35, half - 0.8, 1.6, f"{f['census']}%", size=96, bold=True, color=INK)
    text(s, MX + half + 0.8, 3.95, half - 0.8, 0.5, "of the corn is silage", size=15)
    text(s, MX, 4.85, FULL, 0.4, "Queen Anne's County, Maryland, 2022. Our first try, checked against a rule we had written down before.", size=14)
    text(s, MX, 5.5, FULL, 1.5, "Where silage is rare, a few false alarms on a lot of grain add up.\n"
         "So we went back to the data. The next three slides are what it taught us.", size=20, color=INK, spacing=1.2)
    notes(s, "Then we added the fields up to a county, and it went wrong. We said 22 percent silage where the Census says 1.3. "
             "So we went back to the data.",
          "接着我们把地块加总到县，结果错了。我们算出青贮占 22%，而普查是 1.3%。于是我们回到数据里找原因。")

    # lesson 1 --------------------------------------------------------------------------------
    s = new_slide(prs, "Lesson 1: count only the fields we saw clearly around the cut")
    body_picture(s, "s09_gap_evidence.png")
    notes(s, "First lesson. When the two pictures around the harvest are far apart, the model says silage far too often. "
             "So we count only the fields we saw clearly.",
          "第一个教训：收割前后两张影像隔得远的时候，模型判成青贮的次数多得离谱。所以我们只数那些看清了的地块。")

    # lesson 2 --------------------------------------------------------------------------------
    worst_before, worst_after = N.WINDOW_FIX_WORST
    s = new_slide(prs, "Lesson 2: clouds. Half the corn was cut in one cloudy fortnight")
    pw = 9.3
    picture(s, "pic_kent_cloudy.png", MX, 1.42, pw)
    text(s, MX + pw + 0.3, 1.6, FULL - pw - 0.3, 1.8, "A Kent County field, 2022.\nLast clear look on 2 September, the next on "
         "17 September. The cut is somewhere in between.", size=14, spacing=1.15)
    tile(s, MX, 3.8, half, 3.1, "More than half", "of the corn in two Eastern Shore counties was cut in one cloudy fortnight. "
         "No field there was seen clearly in those two weeks.", big_size=40)
    tile(s, MX + half + 0.4, 3.8, half, 3.1, f"{worst_before} → {worst_after} points",
         "The worst county miss, once we add up by harvest window and say how much corn we could not see.\n"
         "Found on counties we had already seen, so not proof yet.", big_size=40)
    notes(s, "Second lesson: clouds. On the Eastern Shore, more than half of the corn was cut in one cloudy fortnight. "
             "So we add up by harvest window, and we say how much we could not see.",
          "第二个教训：云。在东岸，一半以上的玉米是在同一个多云的两周里收的。所以我们按收割窗口来加总，并且说明有多少没看到。")

    # what we could not fix --------------------------------------------------------------------
    soy, hay = N.KENT_CALLS_HISTORY[0][1], N.KENT_CALLS_HISTORY[2][1]
    s = new_slide(prs, "What we could not fix: grain cut early and green looks like silage")
    picture(s, "pic_kent_early.png", MX, 1.42, pw)
    text(s, MX + pw + 0.3, 1.6, FULL - pw - 0.3, 1.8, "A Kent County field the model calls silage.\nGreen in late August, "
         "bare by 2 September.", size=14, spacing=1.15)
    tile(s, MX, 3.8, third, 3.1, f"{soy}%", "of the ground under such fields also grows soybeans. That is grain ground.",
         big_color=INK)
    tile(s, MX + third + 0.3, 3.8, third, 3.1, f"{hay}%", "of it grows hay. Around real silage fields in a dairy county, "
         f"hay is {N.FREDERICK_CALLS_HAY[0]} to {N.FREDERICK_CALLS_HAY[1]}%.", big_color=INK)
    tile(s, MX + 2 * (third + 0.3), 3.8, third, 3.1, "We say so", "Probably grain cut early while still green. "
         "Crop history shows it. The picture cannot.", fill=BLUE, big_size=40)
    notes(s, "One thing we could not fix. Some grain is cut early while still green, and in the pictures it looks like silage. "
             "Crop history shows the difference; pictures do not.",
          "有一件事我们没能解决：有些籽粒玉米趁绿早收，在图片上看起来和青贮一样。轮作历史能看出区别，图片看不出。")

    # the whole team system, and where the model sits -------------------------------------------------
    s = new_slide(prs, "Where the model sits in the team's system")
    chain = [
        ("In", 1.7, "Satellite images and field outlines", False),
        ("Our part: the model layer", 2.8, "A toolbox the agent calls: a call for every field, county and state totals, "
         "and the evidence behind them.", True),
        ("Bridge", 2.2, "Loads the toolbox. Adds a prior from earlier USDA years and the per-field evidence.", False),
        ("Supervisor agent", 2.2, "Runs the county steps in a fixed order. A critic searches USDA reports.", False),
        ("Reporter and analyst screen", 2.0, "Explain the result.", False),
    ]
    gap = (FULL - sum(c[1] for c in chain)) / (len(chain) - 1)
    x = MX
    for i, (head, w, line, ours) in enumerate(chain):
        card(s, x, 1.5, w, 2.35, fill=BLUE if ours else WHITE, line=None if ours else GREY)
        text(s, x + 0.2, 1.65, w - 0.4, 0.75, head, size=16, bold=True, color=WHITE if ours else INK, spacing=1.05)
        text(s, x + 0.2, 2.45, w - 0.4, 1.35, line, size=13, color=WHITE if ours else INK2, spacing=1.12)
        if i < len(chain) - 1:
            arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + w + gap * 0.15), Inches(2.5), Inches(gap * 0.7), Inches(0.3))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = rgb(INK2)
            arrow.line.fill.background()
            arrow.shadow.inherit = False
        x += w + gap
    text(s, MX, 4.1, FULL, 0.35, "Around it, built by our teammates", size=14, bold=True, color=INK)
    around = [
        ("Planner", "Which fields to visit."),
        ("Recommender", "Grain fields near dairies that are short of silage."),
        ("Picture check", "Looks at the before-and-after pictures of a field."),
        ("Price module", "What the silage and grain split does to prices."),
        ("Four-state check", "Checks 2025 in four states (PA, WI, IA, NE), about 300 fields each."),
    ]
    aw = (FULL - 4 * 0.25) / 5
    for i, (head, line) in enumerate(around):
        x = MX + i * (aw + 0.25)
        card(s, x, 4.55, aw, 1.65)
        text(s, x + 0.2, 4.68, aw - 0.4, 0.4, head, size=15, bold=True, color=INK)
        text(s, x + 0.2, 5.12, aw - 0.4, 1.0, line, size=12.5, spacing=1.1)
    text(s, MX, 6.45, FULL, 0.7, "Our model is the blue box. The agent reaches it through the bridge, which loads the module with the small "
         "model behind it.", size=16, color=INK, spacing=1.15)
    notes(s, "This is the team's whole system. Our model is the blue box, and the agent calls it through a bridge. "
             "The other boxes are our teammates' parts.",
          "这是团队的整个系统。我们的模型是蓝色那一块，智能体通过一个桥接层来调用它。其余的方框是队友们做的部分。")

    # what the team gets, what comes next -----------------------------------------------------
    s = new_slide(prs, "What the team gets, and what comes next")
    card(s, MX, 1.5, half, 4.2)
    text(s, MX + 0.4, 1.7, half - 0.8, 0.5, "What the team gets", size=22, bold=True, color=BLUE)
    text(s, MX + 0.4, 2.4, half - 0.8, 3.2, "A toolbox the agent already calls: classify a county's fields, add them up, "
         "say what could not be seen.\n\nA call for every field, with the evidence behind it.\n\n"
         "A demo map, and notebooks that read new counties.", size=16, color=INK, spacing=1.15)
    card(s, MX + half + 0.4, 1.5, half, 4.2)
    text(s, MX + half + 0.8, 1.7, half - 0.8, 0.5, "What comes next", size=22, bold=True, color=BLUE)
    text(s, MX + half + 0.8, 2.4, half - 0.8, 3.2, "Radar for cloudy weeks: it sees through cloud, and the reader is built.\n\n"
         "Score the double-crop corn that today's state number leaves out.\n\n"
         "A land-use hint (hay and pasture mean dairy), shown beside the satellite number.", size=16, color=INK, spacing=1.15)
    card(s, MX, 5.95, FULL, 1.0, fill=BLUE, line=None)
    text(s, MX + 0.4, 5.95, FULL - 0.8, 1.0, "This is a preliminary baseline, checked in Maryland, two Michigan counties and two Wisconsin counties; "
         "it missed the heaviest dairy county. Predictions anywhere else are examples, not verified results.", size=17, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "The team gets a toolbox, a demo map and notebooks. Next come radar for cloudy weeks, double-crop corn, and a land-use hint. "
             "This is a first baseline, checked in Maryland, Michigan and Wisconsin only, and it missed one county badly.",
          "团队拿到的是：工具箱、演示地图和笔记本。接下来要做的是：多云那几周用雷达、给双季玉米打分、加上土地利用的提示。"
          "这只是一个初步的基线，只在马里兰、密歇根和威斯康星检验过，其中一个县错得很多。")

    # backup 1 ------------------------------------------------------------------------------------
    s = new_slide(prs, "Backup: why the simple model won")
    body_picture(s, "s07_why_simple_won.png")
    notes(s, "Backup. In a new country the small model loses almost nothing; the big one loses more. "
             "Dropping dates in training lifts the big model from 37 to 39 of 44. "
             "A sequence model does better in Europe, 0.817 against 0.765, but it has not been tested on the US fields.",
          "备用页。换一个国家，小模型几乎不掉精度，大模型掉得多。训练时丢掉一些日期，大模型从 37 升到 39（共 44）。"
          "序列模型在欧洲更好（0.817 对 0.765），但还没有在美国的地块上检验过。")

    # backup 2 ------------------------------------------------------------------------------------
    s = new_slide(prs, "Backup: each test had its rule written down first")
    ph = 3.25
    picture(s, "s10_maryland_test.png", MX, 1.4, ph * 1920 / 972)
    text(s, MX + ph * 1920 / 972 + 0.3, 1.6, FULL - ph * 1920 / 972 - 0.3, 3.0, "Every test rule went into a log, with the time, before the "
         "test ran.\n\nThe 44 labelled fields were scored three times, each time logged, and never used for training.\n\n"
         "Failures are shown as failures.", size=15, color=INK, spacing=1.15)
    text(s, MX, 4.8, FULL, 0.35, "From the log: the rule first, then the test", size=14, bold=True, color=INK)
    logged = [("16:12", "16:40", "Queen Anne's", "failed"), ("17:30", "18:11", "Kent", "failed"),
              ("18:25", "19:11", "Michigan", "passed"), ("18:25", "19:14", "Maryland 2025", "reads low"),
              ("22:55", "22:58", "Wisconsin", "failed")]
    sw = (FULL - 4 * 0.2) / 5
    for i, (rule_at, test_at, what, result) in enumerate(logged):
        x = MX + i * (sw + 0.2)
        good = result == "passed"
        card(s, x, 5.25, sw, 1.65, fill=BLUE if good else WHITE, line=None if good else GREY)
        text(s, x + 0.25, 5.4, sw - 0.5, 0.35, f"{rule_at}  rule written", size=13, color=WHITE if good else INK2)
        text(s, x + 0.25, 5.75, sw - 0.5, 0.35, f"{test_at}  test run once", size=13, color=WHITE if good else INK2)
        text(s, x + 0.2, 6.2, sw - 0.4, 0.6, f"{what}: {result}", size=15, bold=True, color=WHITE if good else INK)
    notes(s, "Backup. Every test rule was written in a log before the test ran. "
             "The 44 labelled fields were scored three times, each time logged.",
          "备用页。每条检验规则都在检验之前写进了日志。44 个带标签的地块一共打了三次分，每次都有记录。")

    # backup 3 ------------------------------------------------------------------------------------
    s = new_slide(prs, "Backup: Maryland 2025 in detail")
    body_picture(s, "state_2025.png")
    notes(s, "Backup. For Maryland in 2025 we get 22,700 acres of silage; USDA reports 40,000. "
             "With double-crop corn scored afterwards it is 33,135, still low.",
          "备用页。2025 年的马里兰，我们算出青贮 22,700 英亩，美国农业部是 40,000。事后把双季玉米也打分后是 33,135，仍然偏低。")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"{out.name}: {len(prs.slides)} slides, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path of the .pptx to write")
    build(ap.parse_args().out)
