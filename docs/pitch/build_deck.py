"""Builds the slide deck (PowerPoint, 16:9) from the figure bodies in docs/pitch/deck/.

    python docs/pitch/deck_figs.py --pictures <folder with the field picture strips>
    python docs/pitch/build_deck.py --out <folder>/FieldVision_model_story.pptx

Every slide has an editable title (the message of the slide), a picture or editable boxes, and
speaker notes: short English sentences, then the same in Chinese. Needs python-pptx.
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


def rgb(hex6):
    return RGBColor.from_string(hex6)


def new_slide(prs, title, size=28, top=0.32, height=1.0):
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # "Title Only"
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(MX), Inches(top), Inches(W - 2 * MX), Inches(height)
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
    full = W - 2 * MX

    # 1 ------------------------------------------------------------------------------------------
    s = new_slide(prs, "Chopped for silage or combined for grain?\nWe read it for every corn field, from space",
                  size=36, top=0.75, height=1.9)
    text(s, MX, 2.8, full, 0.5, "FieldVision: the model module. Hack the Harvest.", size=20)
    picture(s, "pic_frederick_silage.png", MX, 3.75, full)
    text(s, MX, 6.3, full, 0.6, "One corn field in Frederick County, Maryland, through the 2022 season (Sentinel-2 true colour, 10 m pixels). "
         "The model calls it silage: cut while still green, around 2 September.", size=13)
    notes(s, "This is the model part of FieldVision. The question is simple. For each corn field, was it chopped for silage "
             "or combined for grain? We answer it from free satellite images.",
          "这是 FieldVision 的模型部分。问题很简单：每一块玉米地，是切碎做了青贮，还是用收割机收了籽粒？我们用免费的卫星影像来回答。")

    # 2 ------------------------------------------------------------------------------------------
    s = new_slide(prs, "The task is full of traps, and the US has only 44 labelled fields")
    traps = [
        ("Clouds", "The harvest can fall between two clear images."),
        ("Early is not silage", "Some grain is cut early too. The date alone misleads."),
        ("Few silage fields", "About 6 to 8% of US corn is silage. Always saying grain looks right."),
        ("Double crops", "Corn sown after a small grain follows another calendar."),
    ]
    cw, gap = (full - 3 * 0.3) / 4, 0.3
    for i, (head, line) in enumerate(traps):
        x = MX + i * (cw + gap)
        card(s, x, 1.6, cw, 2.4)
        text(s, x + 0.22, 1.82, cw - 0.44, 0.45, head, size=18, bold=True, color=INK)
        text(s, x + 0.22, 2.45, cw - 0.44, 1.4, line, size=15)
    text(s, MX, 4.15, full, 0.35, "The traps the organisers warned about", size=13)
    card(s, MX, 4.75, full, 2.1, fill=BLUE, line=None)
    text(s, MX + 0.4, 4.95, 2.4, 1.7, "44", size=80, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(s, MX + 3.0, 4.95, full - 3.5, 1.7, "The catch: the only labelled US fields are 44 field-years at one research farm in "
         "Beltsville, Maryland.\nToo few to train on. We kept them for testing only.", size=20, color=WHITE,
         anchor=MSO_ANCHOR.MIDDLE, spacing=1.15)
    notes(s, "The organisers warned about four traps: clouds, early harvest that is not silage, few silage fields, and double crops. "
             "And there is a catch. The only labelled US fields are 44, all at one research farm. That is too few to train on.",
          "主办方提醒了四个陷阱：云、早收不等于青贮、青贮地块很少、双季作物。还有一个难点：美国带标签的地块只有 44 个，全在一个试验农场。"
          "这点数据不够训练。")

    # 3 ------------------------------------------------------------------------------------------
    s = new_slide(prs, f"So we built our own data: {N.FIELDS_LABELLED_EUROPE:,} labelled fields in Europe, "
                  f"about {round(N.FIELDS_US_TOTAL, -3):,} unlabelled in the US")
    tiles = [
        (f"{N.FIELDS_LABELLED_EUROPE:,}", "labelled European fields",
         "From farm registries: Austria 2022-2024, France 2022, Netherlands 2025.\nSilage or grain, as the farmer declared it."),
        (f"{round(N.FIELDS_US_TOTAL, -3):,}", "unlabelled US corn fields",
         f"{N.FIELDS_US_MARYLAND_2022:,} in four Maryland counties and {N.FIELDS_US_MICHIGAN_2022:,} in two Michigan counties (2022); "
         f"{N.FIELDS_US_MARYLAND_2025:,} in a random Maryland 2025 sample."),
        ("44", "labelled US field-years", "Beltsville, 2021-2023.\nUsed for testing only, never for training."),
    ]
    tw = (full - 2 * 0.3) / 3
    if (FIG / "map_kent.png").exists():
        # numbers down the left, a real county on the right
        lw = 6.0
        for i, (num, head, line) in enumerate(tiles):
            y = 1.55 + i * 1.7
            text(s, MX, y, 2.15, 0.85, num, size=44, bold=True, color=BLUE if i < 2 else INK, anchor=MSO_ANCHOR.MIDDLE)
            text(s, MX + 2.25, y + 0.02, lw - 2.25, 0.4, head, size=17, bold=True, color=INK)
            text(s, MX + 2.25, y + 0.45, lw - 2.25, 1.15, line, size=12.5)
        text(s, MX, 6.7, lw, 0.5, "Per field: only the pixels inside its outline, on every clear Sentinel-2 date, April to November.",
             size=12.5, color=INK)
        mw = 5.75
        picture(s, "map_kent.png", W - MX - mw, 1.5, mw)
        text(s, W - MX - 3.2, 5.45, 3.2, 1.3, "Kent County, Maryland, 2022:\nall 1,363 corn fields.\nGreen: called silage. Yellow: called grain. "
             "Grey: not judged.", size=12.5, spacing=1.15)
    else:
        for i, (num, head, line) in enumerate(tiles):
            x = MX + i * (tw + 0.3)
            card(s, x, 1.6, tw, 3.55)
            text(s, x + 0.3, 1.8, tw - 0.6, 1.1, num, size=60, bold=True, color=BLUE if i < 2 else INK)
            text(s, x + 0.3, 3.0, tw - 0.6, 0.5, head, size=18, bold=True, color=INK)
            text(s, x + 0.3, 3.55, tw - 0.6, 1.5, line, size=14)
        text(s, MX, 5.5, full, 1.2, "For every field: only the pixels inside its outline, on every clear Sentinel-2 date from April to "
             "November.\nUS field outlines are USDA Crop Sequence Boundaries. The reading ran on Colab.", size=17, color=INK, spacing=1.2)
    notes(s, "So we built our own data. We took 8,000 labelled fields from farm registries in Austria, France and the Netherlands. "
             "We also read about 18,000 unlabelled corn fields in Maryland and Michigan. For each field we keep only the pixels "
             "inside its outline.",
          "所以我们自己建了数据。从奥地利、法国、荷兰的农场登记里取了 8,000 个带标签的地块；又读取了马里兰和密歇根约 18,000 个没有标签的玉米地块。"
          "每块地只取边界以内的像素。")

    # 4 ------------------------------------------------------------------------------------------
    s = new_slide(prs, "Silage is cut green, within days. Grain is left to dry first")
    bw = 11.2
    bx = (W - bw) / 2
    text(s, bx, 1.42, bw, 0.35, "Called silage: still green on 31 August, cut around 2 September, green again within weeks (a cover crop)",
         size=16, bold=True, color=INK)
    picture(s, "pic_frederick_silage.png", bx, 1.8, bw)
    text(s, bx, 4.12, bw, 0.35, "Called grain: brown by late September, cut by 7 October", size=16, bold=True, color=INK)
    picture(s, "pic_frederick_grain.png", bx, 4.5, bw)
    text(s, bx, 6.85, bw, 0.4, "Two corn fields in Frederick County, Maryland, 2022. Sentinel-2 true colour, 10 m pixels. "
         "The calls are the model's: these fields have no label.", size=12)
    notes(s, "These are real satellite pictures of two Maryland fields. The top field was still green on 31 August and was cut around "
             "2 September, so the model calls it silage. The bottom field turned brown first and was cut in October, so the model "
             "calls it grain.",
          "这是马里兰两块地的真实卫星图片。上面这块 8 月 31 日还是绿的，9 月 2 日前后就割了，模型判为青贮。下面这块先变黄，10 月才收，模型判为籽粒。")

    # 5 ------------------------------------------------------------------------------------------
    s = new_slide(prs, "The model: check the evidence, then six numbers and a logistic regression")
    body_picture(s, "module_diagram.png")
    notes(s, "The model has three steps. First it checks whether the satellite saw the harvest. Then a logistic regression on six "
             "numbers says silage or grain. Last, the fields are added up to county and state acres.",
          "模型有三步。第一步检查卫星有没有看到收割。第二步用六个数做逻辑回归，判断青贮还是籽粒。第三步把地块加起来，得到县和州的面积。")

    # 6 ------------------------------------------------------------------------------------------
    s = new_slide(prs, "Trained only in Europe, the small model gets 42 of the 44 US fields right")
    body_picture(s, "fig3_labelled_fields.png")
    notes(s, "We trained only on European fields. Then we scored the organisers' 44 labelled fields. The small model got 42 right. "
             "Boosted trees got 37, and always saying grain gets 32.",
          "我们只用欧洲的地块训练，然后在主办方 44 个带标签的地块上打分。小模型对了 42 个，提升树对了 37 个，全部猜籽粒能对 32 个。")

    # 7 ------------------------------------------------------------------------------------------
    s = new_slide(prs, "Simple won because it travels. Two ideas now help the bigger models catch up")
    body_picture(s, "s07_why_simple_won.png")
    notes(s, "Why did the simple model win? In a new country it loses almost nothing, and the big model loses a lot. "
             "Dropping dates in training helps the big model: 37 becomes 39 of 44. A sequence model looks good in Europe, "
             "but it has one seed and is still running.",
          "为什么简单的模型赢了？换一个国家，它几乎不掉精度，大模型掉得多。训练时随机丢掉一些观测日期能帮到大模型：44 个里从 37 升到 39。"
          "序列模型在欧洲看起来不错，但只跑了一个随机种子，还在进行中。")

    # 8 ------------------------------------------------------------------------------------------
    f = N.FIRST_COUNTY_TOTAL
    s = new_slide(prs, f"From fields to counties, the first attempt failed: {f['ours']:.0f}% silage where the Census says {f['census']}%")
    hw = (full - 0.4) / 2
    card(s, MX, 1.6, hw, 3.0)
    text(s, MX + 0.4, 1.8, hw - 0.8, 0.5, "Our first county total", size=18, bold=True, color=INK)
    text(s, MX + 0.4, 2.35, hw - 0.8, 1.6, f"{f['ours']}%", size=96, bold=True, color=BLUE)
    text(s, MX + 0.4, 3.95, hw - 0.8, 0.5, "of the corn area called silage", size=15)
    card(s, MX + hw + 0.4, 1.6, hw, 3.0)
    text(s, MX + hw + 0.8, 1.8, hw - 0.8, 0.5, "USDA Census of Agriculture 2022", size=18, bold=True, color=INK)
    text(s, MX + hw + 0.8, 2.35, hw - 0.8, 1.6, f"{f['census']}%", size=96, bold=True, color=INK)
    text(s, MX + hw + 0.8, 3.95, hw - 0.8, 0.5, "of the harvested corn area is silage", size=15)
    text(s, MX, 4.85, full, 0.4, "Queen Anne's County, Maryland, 2022. A check we had written down before opening the county.",
         size=14)
    text(s, MX, 5.45, full, 1.6, "Where silage is rare, a few false alarms on a lot of grain add up to a large wrong total.\n"
         "The tuning starts here: four steps, each driven by what the data showed.", size=20, color=INK, spacing=1.2)
    notes(s, "Then we added fields up to a county, and the first attempt failed. In Queen Anne's County we said 22 percent silage "
             "where the Census says 1.3. From here on, the data told us what to fix.",
          "接着我们把地块加总到县。第一次尝试失败了：Queen Anne's 县我们算出青贮占 22%，而普查是 1.3%。从这里开始，是数据告诉我们该改什么。")

    # 9 ------------------------------------------------------------------------------------------
    s = new_slide(prs, "Tuning 1: count only the fields whose harvest was clearly seen")
    body_picture(s, "s09_gap_evidence.png")
    notes(s, "We looked at the evidence. Fields with a long gap between images were called silage far more often. "
             "So we sort every field: clearly seen, partly seen, or not judged. Only clearly seen fields are counted.",
          "我们去看证据。前后两张影像间隔长的地块，被判成青贮的比例高得多。所以我们给每块地分级：看清了、部分看到、不判断。只有看清了的才计数。")

    # 10 -----------------------------------------------------------------------------------------
    s = new_slide(prs, "Tuning 2: an honest test, and our correction step failed it")
    body_picture(s, "s10_maryland_test.png")
    notes(s, "Next we tested honestly: we wrote the pass and fail lines before opening two new counties. "
             "Kent read 21 percent against 5.5, which is a fail. So we dropped the correction step.",
          "接下来是老老实实的检验。在打开两个新的县之前，我们先写下了及格线和不及格线。Kent 县读数 21%，普查是 5.5%，不及格。所以我们放弃了这个校正步骤。")

    # 11 -----------------------------------------------------------------------------------------
    s = new_slide(prs, "Tuning 3: on the Eastern Shore, half the corn was cut in one cloudy fortnight")
    pw = 9.3
    picture(s, "pic_kent_cloudy.png", MX, 1.42, pw)
    text(s, MX + pw + 0.3, 1.6, full - pw - 0.3, 1.8, "A Kent County field, 2022.\nLast clear look on 2 September, the next on "
         "17 September: 15 days apart. The cut is somewhere in between.", size=14, spacing=1.15)
    picture(s, "s11_window_fix.png", MX, 3.6, full)
    notes(s, "On the Eastern Shore more than half of the corn was cut in one cloudy fortnight. No field was seen clearly "
             "in those two weeks. Adding up by harvest window cut the worst miss from 7.4 to 3.8 points. "
             "But we found this on counties we had already seen, so it is not proof.",
          "为什么是 Kent？在东岸，一半以上的玉米是在同一个多云的两周里收的，那两周没有一块地被看清。按收割窗口加总，最大偏差从 7.4 降到 3.8 个百分点。"
          "但这是在已经看过的县上找到的，所以不算证明。")

    # 12 -----------------------------------------------------------------------------------------
    s = new_slide(prs, "Tuning 4: Kent's early green fields grow on grain ground, and pictures cannot tell")
    picture(s, "pic_kent_early.png", MX, 1.42, pw)
    text(s, MX + pw + 0.3, 1.6, full - pw - 0.3, 1.8, "A Kent County field the model calls silage.\nGreen on 23 August, bare by "
         "2 September, green again in October.", size=14, spacing=1.15)
    picture(s, "s12_crop_history.png", MX, 3.75, full)
    notes(s, "What are Kent's early green fields? Their crop history is soybeans and wheat, with almost no hay, "
             "so they are probably grain cut early while still green. In the pictures they look just like silage. "
             "We cannot tell them apart, and we say so.",
          "Kent 县那些很早就割掉的绿色地块是什么？它们的轮作历史是大豆和小麦，几乎没有牧草。所以很可能是趁绿早收的籽粒玉米。"
          "在图片上它们和青贮一模一样，我们分不开，也如实这么说。")

    # 13 -----------------------------------------------------------------------------------------
    s = new_slide(prs, "The clean test: two Michigan counties we had never looked at. It passed")
    body_picture(s, "s13_michigan.png")
    notes(s, "Then the clean test: two Michigan counties that we had never looked at, with the rule written first. "
             "We read 2.1 percent against 1.8, and 30.0 against 24.2. It passed.",
          "然后是干净的检验：密歇根两个我们从没看过的县，规则事先写好。我们读出 2.1%（普查 1.8%）和 30.0%（普查 24.2%）。通过了。")

    # 14 -----------------------------------------------------------------------------------------
    s = new_slide(prs, "The state the organisers score: Maryland 2025 reads low, and we say so")
    body_picture(s, "state_2025.png")
    notes(s, "The organisers score state acres. For Maryland in 2025 we read 4.75 percent silage where USDA says 8.9, so we are low. "
             "Double-crop corn that we did not score explains part of the gap, not all of it. This is open, and we say it is open.",
          "主办方按州级面积打分。2025 年的马里兰，我们读出青贮占 4.75%，美国农业部是 8.9%，我们偏低。没有打分的双季玉米能解释一部分差距，但不是全部。"
          "这个问题还没解决，我们也明说它没解决。")

    # 15 -----------------------------------------------------------------------------------------
    s = new_slide(prs, "What the team gets, and how we kept ourselves honest")
    card(s, MX, 1.5, hw, 3.05)
    text(s, MX + 0.4, 1.7, hw - 0.8, 0.5, "What the team gets", size=22, bold=True, color=BLUE)
    text(s, MX + 0.4, 2.35, hw - 0.8, 2.1, "A toolbox the agent already calls: classify a county's fields, add them up, "
         "say what could not be seen.\nA call for every field, with its evidence: harvest dates, greenness, moisture, residue.\n"
         "A demo map of the labelled fields, and notebooks that read new counties on Colab.", size=15, color=INK, spacing=1.2)
    card(s, MX + hw + 0.4, 1.5, hw, 3.05)
    text(s, MX + hw + 0.8, 1.7, hw - 0.8, 0.5, "How we kept ourselves honest", size=22, bold=True, color=BLUE)
    text(s, MX + hw + 0.8, 2.35, hw - 0.8, 2.1, "Every test rule went into a log, with the time, before the test was run.\n"
         "The 44 labelled fields were scored three times, each time logged, and never used for training.\n"
         "Failures are shown as failures.", size=15, color=INK, spacing=1.2)
    # the log as a timeline: rule written, then the test, then what came out (times from the access log, 14 Nov)
    text(s, MX, 4.8, full, 0.35, "From the log: the rule first, then the test", size=14, bold=True, color=INK)
    steps = [("16:12", "16:40", "Queen Anne's", "failed"), ("17:30", "18:11", "Kent", "failed"),
             ("18:25", "19:11", "Michigan", "passed"), ("18:25", "19:14", "Maryland 2025", "reads low")]
    sw = (full - 3 * 0.25) / 4
    for i, (rule_at, test_at, what, result) in enumerate(steps):
        x = MX + i * (sw + 0.25)
        good = result == "passed"
        card(s, x, 5.25, sw, 1.65, fill=BLUE if good else WHITE, line=None if good else GREY)
        text(s, x + 0.25, 5.4, sw - 0.5, 0.35, f"{rule_at}  rule written", size=13, color=WHITE if good else INK2)
        text(s, x + 0.25, 5.75, sw - 0.5, 0.35, f"{test_at}  test run once", size=13, color=WHITE if good else INK2)
        text(s, x + 0.25, 6.2, sw - 0.5, 0.5, f"{what}: {result}", size=17, bold=True, color=WHITE if good else INK)
    notes(s, "The team gets a toolbox that the agent already calls, a demo map, and notebooks. And we kept ourselves honest. "
             "Every test rule was logged before the test. The labelled set was scored three times, each time logged.",
          "团队拿到的是：智能体已经在调用的工具箱、一个演示地图，还有笔记本。我们也让自己保持诚实：每条检验规则都在检验之前写进日志；"
          "带标签的那组地块一共打了三次分，每次都有记录。")

    # 16 -----------------------------------------------------------------------------------------
    s = new_slide(prs, "Next: radar for cloudy weeks, double-crop corn, a land-use prior")
    nxt = [
        ("1", "Radar for cloudy weeks", "Radar sees through cloud. A Sentinel-1 reader for the same fields is built; "
         "it is not used in any number yet."),
        ("2", "Score double-crop corn", "Corn after a small grain is outside today's state figure. A first look calls 61% of it "
         "silage; no labelled double-crop field exists to check that."),
        ("3", "A land-use prior, shown beside", "Hay and pasture around a field point to dairy. Shown next to the satellite figure, "
         "not mixed into it."),
    ]
    for i, (num, head, line) in enumerate(nxt):
        x = MX + i * (tw + 0.3)
        card(s, x, 1.6, tw, 4.2)
        text(s, x + 0.3, 1.8, tw - 0.6, 1.0, num, size=54, bold=True, color=BLUE)
        text(s, x + 0.3, 2.95, tw - 0.6, 0.9, head, size=20, bold=True, color=INK)
        text(s, x + 0.3, 3.85, tw - 0.6, 1.9, line, size=15)
    text(s, MX, 6.15, full, 0.8, "The field calls travel. The adding-up is the open problem, and these three attack it.",
         size=20, color=INK)
    notes(s, "Three next steps. Radar can see through clouds, and the reader is built. Double-crop corn should be scored. "
             "And a land-use prior can be shown beside the satellite figure.",
          "接下来三件事：雷达能穿透云层，读取程序已经做好；双季玉米应该纳入打分；还可以把土地利用的先验信息放在卫星数字旁边一起展示。")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"{out.name}: {len(prs.slides)} slides, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path of the .pptx to write")
    build(ap.parse_args().out)
