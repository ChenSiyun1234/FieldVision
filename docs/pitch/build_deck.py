"""Builds the slide deck (PowerPoint, 16:9): a nine-slide story for the judges, then slides that go deeper.

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


ONE_MINUTE = """

=== ONE MINUTE: the whole model story, to say as it is ===
1. USDA asked: from space, was a cornfield chopped for silage, or combined for grain?
2. The US has almost no labelled fields. So we trained on 8,000 fields from farm registries in Europe.
3. Our model had never seen an American field. At Beltsville it got 42 of USDA's 44 labelled fields right.
4. Where clouds hid the harvest, a second satellite, Landsat, brought most of it back into view.
5. We then tested five places we had never looked at, each once, with the rule written down first.
6. Michigan: 14.4 percent, USDA says 14.7. New York, where more than half the corn is silage: 48.7, USDA says 53.2.
7. Pennsylvania's largest dairy county: within a point. Wisconsin: 5 points low. Pennsylvania as a whole we missed.
8. The next step is American labelled fields.

=== 一分钟版本（上面八句的中文） ===
1. 美国农业部问：从太空看，一块玉米地是切碎做了青贮，还是收了籽粒？
2. 美国几乎没有带标签的地块，所以我们用欧洲农场登记里的 8,000 个地块来训练。
3. 我们的模型从没见过美国的地。在 Beltsville，美国农业部 44 个带标签的地块它对了 42 个。
4. 云挡住收割期的地方，第二颗卫星 Landsat 把其中大部分又看回来了。
5. 然后我们测了五个从没看过的地方，每个只跑一次，规则都事先写好。
6. 密歇根：14.4%，美国农业部 14.7%。纽约州，那里一半以上的玉米是青贮：48.7%，美国农业部 53.2%。
7. 宾夕法尼亚最大的奶牛县：只差不到一个点。威斯康星：低 5 个点。整个宾夕法尼亚我们读错了。
8. 下一步是美国本土带标签的地块。"""


def build(out):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    half = (FULL - 0.4) / 2
    third = (FULL - 2 * 0.3) / 3

    # ===== the story, ten slides ======================================================================

    # 1  the question ----------------------------------------------------------------------------
    s = new_slide(prs, "Can you tell what happened in a cornfield from space?", size=38, top=0.7, height=1.3)
    text(s, MX, 2.15, FULL, 1.1, "Chopped for silage, or combined for grain? That is USDA's question.\n"
         "Our answer: yes, for most fields. And we can show you where we cannot.", size=22, color=INK, spacing=1.2)
    picture(s, "pic_frederick_silage.png", MX, 3.75, FULL)
    text(s, MX, 6.3, FULL, 0.6, "One real corn field in Maryland through the 2022 season, seen by the Sentinel-2 satellite. "
         "Green in August, bare in early September, green again in October.", size=13)
    notes(s, "Here is the question USDA asked us. Can you tell from space what happened in a cornfield? "
             "Was it chopped for silage, or combined for grain?",
          "这是美国农业部给我们的问题：能不能从太空看出一块玉米地发生了什么？是切碎做了青贮，还是收了籽粒？"
          + ONE_MINUTE)

    # 2  the catch, and the idea -------------------------------------------------------------------
    s = new_slide(prs, "Nobody labels American cornfields. European farmers do. So we learned from them")
    card(s, MX, 1.6, half - 0.5, 3.6)
    text(s, MX + 0.4, 1.8, half - 1.3, 0.5, "United States", size=20, bold=True, color=INK)
    text(s, MX + 0.4, 2.35, half - 1.3, 1.6, "44", size=110, bold=True, color=INK)
    text(s, MX + 0.4, 4.2, half - 1.3, 0.9, "labelled corn fields, all on one research farm. Too few to teach a model.", size=16)
    arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(MX + half - 0.35), Inches(3.1), Inches(0.9), Inches(0.6))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = rgb(INK2)
    arrow.line.fill.background()
    arrow.shadow.inherit = False
    card(s, MX + half + 0.7, 1.6, half - 0.3, 3.6, fill=BLUE, line=None)
    text(s, MX + half + 1.1, 1.8, half - 1.1, 0.5, "Austria, France, the Netherlands", size=20, bold=True, color=WHITE)
    text(s, MX + half + 1.1, 2.35, half - 1.1, 1.6, f"{N.FIELDS_LABELLED_EUROPE:,}", size=110, bold=True, color=WHITE)
    text(s, MX + half + 1.1, 4.2, half - 1.1, 0.9, "fields where the farmer told the government: silage, or grain.", size=16, color=WHITE)
    text(s, MX, 5.6, FULL, 1.3, "Two new data sources: European farm registries to learn from, and a second satellite, Landsat, for cloudy weeks.\n"
         "The model learns in Europe. It is tested here, on fields it has never seen.",
         size=22, color=INK, spacing=1.2)
    notes(s, "To teach a computer you need examples, and the US has only 44 labelled fields. "
             "But in Europe, farmers tell their governments what each field is for. So we learned from 8,000 European fields.",
          "教电脑需要例子，而美国只有 44 个带标签的地块。但在欧洲，农民要向政府申报每块地的用途。所以我们用 8,000 个欧洲地块来学习。")

    # 3  the signal ----------------------------------------------------------------------------------
    s = new_slide(prs, "The signal: silage is cut green and wet. Grain is left to dry first")
    if (FIG / "pic_before_after.jpg").exists():
        ph = 5.05
        from PIL import Image
        with Image.open(FIG / "pic_before_after.jpg") as im:
            pw = ph * im.width / im.height
        px = W - MX - pw
        picture(s, "pic_before_after.jpg", px, 1.4, pw)
        lw = px - MX - 0.2
        for frac, label in ((0.125, "just before the cut"), (0.381, "just after"),
                            (0.638, "just before the cut"), (0.894, "just after")):
            text(s, MX, 1.4 + frac * ph - 0.05, lw, 0.4, label, size=14)
        text(s, MX, 1.4 + 0.125 * ph - 0.65, lw, 0.5, "Silage", size=24, bold=True, color=BLUE)
        text(s, MX, 1.4 + 0.638 * ph - 0.65, lw, 0.5, "Grain", size=24, bold=True, color=INK)
        text(s, MX, 6.6, FULL, 0.6, "Twelve real fields in Frederick County, Maryland, 2022, as our model calls them. Picked by rule, not by eye "
             "(two of the silage picks were cut late and look more like grain; the rule kept them).", size=12)
    else:
        picture(s, "pic_frederick_silage.png", MX, 1.8, FULL)
    notes(s, "This is what the satellite sees. Silage is cut while the plant is still green and wet. "
             "Grain is left to dry and turn brown first.",
          "这就是卫星看到的。青贮是趁植株还绿、还湿的时候割的；籽粒玉米要先晾干、变黄。")

    # 4  how it works ---------------------------------------------------------------------------------
    s = new_slide(prs, "How our model works: look, check, decide. If clouds hid the harvest, it does not guess")
    steps = [
        ("1", "Look", "One season of free satellite pictures of the field."),
        ("2", "Check", "Did we see the harvest? If clouds hid it, the field is \"not judged\"."),
        ("3", "Decide", "Six measurements of the crop at harvest. The key one: how wet it is just before the cut."),
    ]
    for i, (num, head, line) in enumerate(steps):
        x = MX + i * (third + 0.3)
        card(s, x, 1.5, third, 2.25, fill=BLUE if i == 1 else WHITE, line=None if i == 1 else GREY)
        text(s, x + 0.3, 1.55, 0.9, 1.0, num, size=54, bold=True, color=WHITE if i == 1 else BLUE)
        text(s, x + 1.25, 1.78, third - 1.5, 0.6, head, size=26, bold=True, color=WHITE if i == 1 else INK)
        text(s, x + 0.3, 2.65, third - 0.6, 1.0, line, size=15, color=WHITE if i == 1 else INK2, spacing=1.12)
    pw = 8.6
    picture(s, "pic_kent_cloudy.png", MX, 4.1, pw)
    text(s, MX + pw + 0.35, 4.3, FULL - pw - 0.35, 2.2, "A real field in Kent County.\n\nFifteen days with no clear picture, right when it was "
         "cut. We do not count this field.", size=16, color=INK, spacing=1.15)
    notes(s, "Our model has three steps: look, check, decide. And it has one rule. If clouds hid the harvest, we do not guess.",
          "我们的模型有三步：看、查、判断。还有一条规矩：如果云挡住了收割，我们就不猜。")

    # 5  the test on USDA's labelled fields --------------------------------------------------------------
    best, total = N.LABELLED[0][2], N.LABELLED_TOTAL
    c = N.CONFUSION
    silage, grain = c["silage_found"] + c["silage_missed"], c["grain_called_silage"] + c["grain_right"]
    calls = c["silage_found"] + c["grain_called_silage"]
    s = new_slide(prs, f"Our model had never seen an American field. On USDA's {total} labelled fields it got {best} right")
    lw = 4.9
    card(s, MX, 1.5, lw, 2.35, fill=BLUE, line=None)
    text(s, MX + 0.35, 1.55, lw - 0.7, 1.4, f"{best} of {total}", size=76, bold=True, color=WHITE)
    text(s, MX + 0.35, 3.0, lw - 0.7, 0.7, "called right, trained with zero US labels", size=16, color=WHITE)
    rx, rw = MX + lw + 0.35, FULL - lw - 0.35
    card(s, rx, 1.5, rw, 2.35)
    rows = [("Our small model", best, BLUE), ("A much bigger model", N.LABELLED[1][2], GREY),
            ("Always say \"grain\"", N.LABELLED[-1][2], GREY)]
    scale = (rw - 4.0) / total
    for i, (name, value, color) in enumerate(rows):
        y = 1.72 + i * 0.68
        text(s, rx + 0.3, y + 0.05, 2.6, 0.4, name, size=15, bold=i == 0, color=INK)
        bar(s, rx + 2.95, y, value * scale, 0.45, color)
        text(s, rx + 2.95 + value * scale + 0.12, y + 0.03, 0.8, 0.4, str(value), size=17, bold=i == 0, color=INK)
    tiles = [
        (f"{c['silage_found']} of {silage}", f"silage fields found\n(recall {100 * c['silage_found'] / silage:.0f}%)"),
        (f"{c['silage_found']} of {calls}", f"silage calls that were right\n(precision {100 * c['silage_found'] / calls:.0f}%)"),
        (f"{c['grain_called_silage']} of {grain}", f"grain fields called silage\n(false positive rate {100 * c['grain_called_silage'] / grain:.0f}%)"),
    ]
    for i, (big, label) in enumerate(tiles):
        tile(s, MX + i * (third + 0.3), 4.05, third, 2.05, big, label, big_size=44, label_size=15)
    rt = N.RESPONSE_TIME
    text(s, MX, 6.35, FULL, 0.6, f"Response time: about a minute per county on a laptop (measured: {rt['fields']:,} fields in "
         f"{rt['seconds']} seconds, {rt['ms_per_field']} ms per field), once the satellite pictures are read.", size=16, color=INK)
    notes(s, "Our model had never seen an American field. On USDA's 44 labelled fields it got 42 right. "
             "It found 11 of the 12 silage fields, with one false alarm. A whole county takes about a minute on a laptop (measured).",
          "我们的模型从没见过美国的地。在美国农业部 44 个带标签的地块上，它对了 42 个。12 块青贮地找到了 11 块，只有一次误报。在笔记本电脑上，一个县大约一分钟算完（实测）。")

    # 6  the scoreboard against USDA ---------------------------------------------------------------------
    s = new_slide(prs, "Within 5 points in Michigan, New York and Wisconsin; right in Lancaster; Pennsylvania missed")
    body_picture(s, "s06_scoreboard.png")
    notes(s, "Then we tried to break it. We compared fourteen places with USDA's own numbers. "
             "Five of them we had never looked at, and we ran each once, with the rule written down first. "
             "Michigan: 14.4 percent, USDA says 14.7. New York, where more than half the corn is silage: 48.7, USDA says 53.2. "
             "Lancaster, Pennsylvania's largest dairy county: within a point. Wisconsin: 5 points low. Pennsylvania as a whole we missed. "
             "In Wisconsin a warning check we had set up went off. By the rule we wrote down first, that marks the Wisconsin figure as unreliable.",
          "然后我们想办法把它考倒：拿十四个地方和美国农业部自己的数字对比。"
          "其中五个地方我们从没看过，每个只跑一次，规则都事先写好。"
          "密歇根：14.4%，美国农业部 14.7%。纽约州，那里一半以上的玉米是青贮：48.7%，美国农业部 53.2%。"
          "宾夕法尼亚最大的奶牛县 Lancaster：只差不到一个点。威斯康星：低 5 个点。整个宾夕法尼亚我们读错了。"
          "在威斯康星，我们事先设的一个预警触发了；按我们事先写下的规则，威斯康星这个数字要标为“不可靠”。")

    # 7  a second satellite for the cloudy counties --------------------------------------------------------------
    s = new_slide(prs, "A second satellite sees past the clouds: in the three cloudy counties our figure moved towards USDA's")
    body_picture(s, "s_landsat.png")
    notes(s, "In three counties, clouds hid the harvest. So we added a second satellite, Landsat, and in all three our figure "
             "moved towards USDA's. Kent is almost right now; Marathon is only partly fixed. "
             "In a fourth county, which clouds did not hit, it changed little. "
             "In the 2025 state tests the two satellites together saw 96 percent of the corn in Wisconsin and 98 percent in Pennsylvania.",
          "有三个县的收割期被云挡住了，所以我们加了第二颗卫星 Landsat，这三个县的读数都向美国农业部的数字靠拢。"
          "Kent 县现在几乎对了，Marathon 县只修好了一部分。第四个县没有被云影响，读数几乎没变。"
          "在 2025 年的两个州级测试里，两颗卫星合起来在威斯康星看到了 96% 的玉米，在宾夕法尼亚看到了 98%。")

    # 8  what the data taught us ---------------------------------------------------------------------------
    s = new_slide(prs, "What went wrong while we built it, and what it taught us")
    f = N.FIRST_COUNTY_TOTAL
    pa = N.PENNSYLVANIA_2025
    lessons = [
        ("Do not count what you cannot see",
         f"Our first county total said {f['ours']:.0f}% silage. USDA says {f['census']}%. Counting only fields whose harvest we saw: 4%.", False),
        ("Test your own fix",
         "A clever correction failed the test we had written for it (21% for 5.5%). We dropped it.", False),
        ("A second satellite sees past the clouds",
         "In three cloudy counties, Landsat brought the corn back into view and our figure moved towards USDA's.", False),
        ("Four of five clean tests within 5 points",
         "Including New York, where more than half the corn is silage. Pennsylvania as a whole reads less than half of USDA's figure, "
         f"with {pa['seen']}% of the corn in view, and we do not yet know why.", True),
    ]
    for i, (head, line, open_) in enumerate(lessons):
        x = MX + (i % 2) * (half + 0.4)
        y = 1.5 + (i // 2) * 2.75
        card(s, x, y, half, 2.5, fill=BLUE if open_ else WHITE, line=None if open_ else GREY)
        text(s, x + 0.35, y + 0.2, 0.9, 1.0, str(i + 1), size=48, bold=True, color=WHITE if open_ else BLUE)
        text(s, x + 1.2, y + 0.3, half - 1.5, 0.9, head, size=21, bold=True, color=WHITE if open_ else INK, spacing=1.05)
        text(s, x + 1.2, y + 1.2, half - 1.5, 1.2, line, size=15, color=WHITE if open_ else INK2, spacing=1.15)
    notes(s, "The data taught us four things on the way. Do not count what you cannot see. Test your own fix. "
             "A second satellite sees past clouds. And of five places we had never looked at, four are within 5 points of USDA, "
             "including New York, where more than half the corn is silage. Pennsylvania as a whole reads less than half, and we do not yet know why.",
          "一路上数据教了我们四件事：看不见的不要数；自己的修正也要检验；第二颗卫星能看穿云层；"
          "还有，五个从没看过的地方里，有四个和美国农业部相差在 5 个点以内，包括一半以上玉米是青贮的纽约州。"
          "整个宾夕法尼亚读出的还不到一半，原因我们还不知道。")

    # 8  the whole team system -------------------------------------------------------------------------------
    s = new_slide(prs, "Where our model sits in the team's system")
    chain = [
        ("In", 1.7, "Satellite pictures and field outlines", False),
        ("Our part: the model", 2.8, "A call for every field, county and state totals, and the evidence behind them.", True),
        ("Bridge", 2.2, "Loads the model. Adds what USDA reported in earlier years.", False),
        ("Supervisor agent", 2.2, "Runs the steps in a fixed order. A critic checks against USDA reports.", False),
        ("Reporter and analyst screen", 2.0, "Explain the result to a person.", False),
    ]
    gap = (FULL - sum(c_[1] for c_ in chain)) / (len(chain) - 1)
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
    text(s, MX, 6.45, FULL, 0.7, "Our model is the blue box. The agent calls it, then explains the answer to a person.", size=16,
         color=INK, spacing=1.15)
    notes(s, "This is the team's whole system. Our model is the blue box. The agent calls it, and then explains the answer.",
          "这是团队的整个系统。我们的模型是蓝色那一块。智能体调用它，然后把答案解释给人听。")

    # 9  what USDA gets, and what comes next -------------------------------------------------------------------
    s = new_slide(prs, "What USDA gets today, and what comes next")
    card(s, MX, 1.5, half, 3.55)
    text(s, MX + 0.4, 1.7, half - 0.8, 0.5, "Today", size=24, bold=True, color=BLUE)
    text(s, MX + 0.4, 2.4, half - 0.8, 2.6, "A call for every corn field, with the evidence behind it: 42 of 44 right where we could check.\n\n"
         "State and county totals with a plain record: four of five clean tests within 5 points, one missed.\n\n"
         "Free public data, no US labels, about a minute per county on a laptop.", size=16, color=INK, spacing=1.08)
    card(s, MX + half + 0.4, 1.5, half, 3.55)
    text(s, MX + half + 0.8, 1.7, half - 0.8, 0.5, "Next", size=24, bold=True, color=BLUE)
    text(s, MX + half + 0.8, 2.4, half - 0.8, 2.6, "American labelled fields, or training against USDA's county totals, for the places that read low.\n\n"
         "Landsat, the second satellite, everywhere.\n\n"
         "Labelled double-crop corn fields: those calls are untested.", size=16, color=INK, spacing=1.08)
    card(s, MX, 5.3, FULL, 1.6, fill=BLUE, line=None)
    text(s, MX + 0.4, 5.42, FULL - 0.8, 0.7, "We do not ask you to trust the model. We show where it can be trusted, and where not yet.",
         size=22, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(s, MX + 0.4, 6.15, FULL - 0.8, 0.65, "A preliminary baseline, checked in five states. "
         "Predictions anywhere else are examples, not verified results.", size=14, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "So what does USDA get? A call for every field, with its evidence, from free data. "
             "State and county totals, with a plain record of where they held and where they did not. "
             "Next we need American labelled fields.",
          "那么美国农业部能得到什么？每块地一个判断，附上依据，用的是免费数据。"
          "还有州和县的总数，哪里对、哪里不对，都有清楚的记录。下一步需要美国本土带标签的地块。")

    # ===== more detail: each slide stands on its own ====================================================================

    s = new_slide(prs, "The slides after this one go deeper. Each stands on its own", size=34, top=2.6, height=1.4)
    text(s, MX, 4.2, FULL, 1.2, "Slides 1 to 10 tell the story in order. Take any of the slides that follow into your own deck: "
         "each has its point in the title and one picture.", size=20, spacing=1.2)
    notes(s, "This slide only separates the story from the detail. Do not show it.",
          "这一页只是把主线和细节分开，演讲时不要放。")

    s = new_slide(prs, "Clouds hid Marathon County's silage harvest for 25 days. A second satellite found most of it")
    g = N.MARATHON_GAP
    cw = 7.9
    picture(s, "marathon_gap.png", MX, 1.45, cw)
    tx, tw = MX + cw + 0.35, FULL - cw - 0.35
    text(s, tx, 1.55, tw, 0.5, "What happened", size=18, bold=True, color=BLUE)
    text(s, tx, 2.0, tw, 1.9, f"In September, Marathon's fields had one clear picture each. Racine had four.\n\n"
         f"{g['fields_cut_inside']}% of Marathon's fields were cut inside a {g['days']}-day cloudy gap ({g['start']} to {g['end']}). "
         "The fields we did see were the late harvest, which is grain.", size=13.5, color=INK, spacing=1.1)
    ls = N.LANDSAT_MARATHON
    text(s, tx, 3.9, tw, 0.4, "What a second satellite did", size=18, bold=True, color=BLUE)
    text(s, tx, 4.35, tw, 1.0, f"Adding Landsat brought the corn we could see from {ls['seen_before']}% to {ls['seen_after']}%. "
         f"Our figure rose from {ls['silage_before']}% to {ls['silage_after']}%. USDA says {ls['usda']}%.", size=13.5, color=INK, spacing=1.1)
    text(s, tx, 5.4, tw, 0.4, "What is still missing", size=18, bold=True, color=BLUE)
    text(s, tx, 5.85, tw, 1.1, "Silage in the north is cut after the crop has yellowed. A model trained in Europe reads that as grain. "
         "It needs labelled northern fields.", size=13.5, color=INK, spacing=1.1)
    notes(s, "Clouds hid the silage harvest for 25 days, so we only saw the late fields, which are grain. "
             "A second satellite, Landsat, brought most of the corn back into view, and our figure rose from 4 to 14 percent. "
             "The rest needs labelled fields from the north.",
          "云把青贮收割期挡住了 25 天，我们只看到了晚收的地，那些是籽粒。第二颗卫星 Landsat 把大部分玉米又看回来了，读数从 4% 升到 14%。"
          "剩下的差距需要北方带标签的地块来解决。")

    s = new_slide(prs, "Every place checked against USDA, with both ways of adding up")
    body_picture(s, "s05_places.png")
    notes(s, "Every place we compared with USDA, with both ways of adding up the fields. "
             "The grey bands are the ranges USDA's own figures allow. "
             "The five newest rows were opened last, once each. Michigan 2025: 14.4 percent for USDA's 14.7. New York 2025: 48.7 for 53.2. "
             "Lancaster County 2023: 26.6 for 27.4. Wisconsin 2025: 16.6 for 21.6, right on our pass mark of five points. "
             "Pennsylvania 2025: 15.2 for 32.8, a clear miss.",
          "所有和美国农业部对比过的地方，两种加总方式都列出来。灰色带是美国农业部自己的数字允许的范围。"
          "最新的五行是最后打开的五个地方，每个只跑一次。2025 年密歇根：我们 14.4%，美国农业部 14.7%。2025 年纽约州：48.7% 对 53.2%。"
          "2023 年 Lancaster 县：26.6% 对 27.4%。2025 年威斯康星：16.6% 对 21.6%，正好压在我们 5 个点的及格线上。"
          "2025 年宾夕法尼亚：15.2% 对 32.8%，明显读低了。")

    s = new_slide(prs, "Starting from USDA's last figure: closer in six of eight counties we tuned on, worse in four of five states")
    body_picture(s, "s_update.png")
    notes(s, "USDA already has an older number for every place. We tried starting from it and letting the pictures update it, field by field. "
             "In six of the eight counties we tuned on, it moved closer to the truth. In four of five states it moved away; in New York it moved closer, but not close enough to pass. "
             "So we do not rely on it.",
          "美国农业部对每个地方都已经有一个旧数字。我们试过从它出发，让卫星图片逐块更新它。"
          "在调优用过的八个县里，有六个更接近真值；但五个州里有四个离真值更远了，纽约州虽然更近，也没有达标。所以我们不依赖它。")

    s = new_slide(prs, "What went wrong, what we did, and what came out")
    cols = [(MX, 4.75, "What went wrong"), (MX + 4.95, 3.75, "What we did"), (MX + 8.9, FULL - 8.9, "What came out")]
    for x, w, head in cols:
        text(s, x + 0.2, 1.36, w - 0.2, 0.35, head, size=14, bold=True, color=BLUE)
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
         "Dropped dates in training; tried a sequence model", "37 → 39 of 44; the sequence model reaches 42, a tie"),
        ("A northern dairy county reads far too low: Marathon 4% for 50%. A 25-day cloudy spell lay over the silage harvest, "
         "and our \"not seen\" measure did not flag it",
         "Added Landsat, a second satellite: corn seen 29% → 83%. Repairing the measure so such a county is reported as a range",
         "4.3% → 13.8% (USDA 49.9%). The rest needs northern training fields"),
        ("Wisconsin and Pennsylvania read low with the corn in view; Michigan reads right",
         "Tried three replacement timing measures, rule for choosing written first",
         "None helped; module unchanged. Why is still open"),
    ]
    y = 1.76
    for i, cells in enumerate(fixes):
        rh = 1.0 if len(cells[0]) > 120 else 0.52
        card(s, MX, y, FULL, rh)
        for (x, w, _), cell, strong in zip(cols, cells, (False, False, True)):
            text(s, x + 0.2, y, w - 0.3, rh, cell, size=13.5, bold=strong, color=INK, anchor=MSO_ANCHOR.MIDDLE, spacing=1.05)
        y += rh + 0.06
    notes(s, "The tuning in one table: each row is a problem the data showed us, what we did, and what came out. "
             "Early green grain is not solved, the cloudy dairy county is only partly fixed, and Pennsylvania as a whole still reads low. "
             "A teammate's idea was also tried on thirteen places: count a field we could not judge as silage if it is green again in November. "
             "It helped a little in some places and pushed Lancaster from 26.6 to 40.1 percent. It is not in the model.",
          "一张表讲完调优过程：每一行是数据暴露出的一个问题、我们的做法和结果。"
          "趁绿早收的籽粒玉米还没解决；多云的那个奶牛县只修好了一部分。"
          "整个宾夕法尼亚仍然读得偏低。"
          "队友的一个想法也在十三个地方试过：没法判断、但十一月又变绿的地块算作青贮。"
          "有些地方略有改善，但 Lancaster 从 26.6% 被推到了 40.1%。它没有进入模型。")

    s = new_slide(prs, "Why we count only the fields we saw clearly")
    body_picture(s, "s09_gap_evidence.png")
    notes(s, "When the two pictures around the harvest are far apart, the model says silage far too often. "
             "So we count only the fields we saw clearly.",
          "收割前后两张影像隔得远的时候，模型判成青贮的次数多得离谱。所以我们只数那些看清了的地块。")

    worst_before, worst_after = N.WINDOW_FIX_WORST
    s = new_slide(prs, "On Maryland's Eastern Shore, half the corn was cut in one cloudy fortnight")
    pw = 9.3
    picture(s, "pic_kent_cloudy.png", MX, 1.42, pw)
    text(s, MX + pw + 0.3, 1.6, FULL - pw - 0.3, 1.8, "A Kent County field, 2022.\nLast clear look on 2 September, the next on "
         "17 September. The cut is somewhere in between.", size=14, spacing=1.15)
    tile(s, MX, 3.8, half, 3.1, "More than half", "of the corn in two Eastern Shore counties was cut in one cloudy fortnight. "
         "No field there was seen clearly in those two weeks.", big_size=40)
    tile(s, MX + half + 0.4, 3.8, half, 3.1, f"{worst_before} → {worst_after} points",
         "The worst county miss, once we add up by harvest window and say how much corn we could not see.\n"
         "Found on counties we had already seen, so not proof yet.", big_size=40)
    notes(s, "On the Eastern Shore, more than half of the corn was cut in one cloudy fortnight. "
             "So we add up by harvest window, and we say how much we could not see.",
          "在东岸，一半以上的玉米是在同一个多云的两周里收的。所以我们按收割窗口来加总，并且说明有多少没看到。")

    soy, hay = N.KENT_CALLS_HISTORY[0][1], N.KENT_CALLS_HISTORY[2][1]
    s = new_slide(prs, "Grain cut early and green looks like silage from space")
    picture(s, "pic_kent_early.png", MX, 1.42, pw)
    text(s, MX + pw + 0.3, 1.6, FULL - pw - 0.3, 1.8, "A Kent County field the model calls silage.\nGreen in late August, "
         "bare by 2 September.", size=14, spacing=1.15)
    tile(s, MX, 3.8, third, 3.1, f"{soy}%", "of the ground under such fields also grows soybeans. That is grain ground.",
         big_color=INK)
    tile(s, MX + third + 0.3, 3.8, third, 3.1, f"{hay}%", "of it grows hay. Around real silage fields in a dairy county, "
         f"hay is {N.FREDERICK_CALLS_HAY[0]} to {N.FREDERICK_CALLS_HAY[1]}%.", big_color=INK)
    tile(s, MX + 2 * (third + 0.3), 3.8, third, 3.1, "We say so", "Probably grain cut early while still green. "
         "Crop history shows it. The picture cannot.", fill=BLUE, big_size=40)
    notes(s, "Some grain is cut early while still green, and in the pictures it looks like silage. "
             "Crop history shows the difference; pictures do not.",
          "有些籽粒玉米趁绿早收，在图片上看起来和青贮一样。轮作历史能看出区别，图片看不出。")

    s = new_slide(prs, "Why we kept the simple model: a bigger one only ties it, and the simple one can be explained")
    body_picture(s, "s07_why_simple_won.png")
    notes(s, "In a new country the small model loses almost nothing; the big one loses more. "
             "A sequence model ties the small one on USDA's labelled fields, 42 of 44, and is no better on the counties. "
             "We kept the small one because it can be explained in one sentence; the sequence model is steadier when pictures are missing, so it is next.",
          "换一个国家，小模型几乎不掉精度，大模型掉得多。序列模型在美国农业部带标签的地块上和小模型打平（44 对 42），在各县上也不更好。"
          "我们留下小模型，因为一句话就能讲清它；序列模型在影像缺失时更稳，所以它是下一个要试的。")

    s = new_slide(prs, "Each test had its rule written down before it was run")
    ph = 2.65
    pw = ph * 1920 / 972
    picture(s, "s10_maryland_test.png", MX, 1.35, pw)
    text(s, MX + pw + 0.3, 1.45, FULL - pw - 0.3, 2.6, "Every test rule went into a log, with the time, before the test ran.\n\n"
         "The 44 labelled fields were scored four times, each time logged, and never used for training.\n\n"
         "Failures are shown as failures. In Wisconsin 2025 a warning check we had written first went off, so by our own rule "
         "that figure is marked unreliable. Our own predictions are logged too: we wrote that Michigan 2025 would read low "
         "and that New York 2025 would fail. Both passed.", size=15, color=INK, spacing=1.12)
    text(s, MX, 4.05, FULL, 0.35, "From the log: the rule first, then the test", size=14, bold=True, color=INK)
    wi, pa, la, mi, ny = N.WISCONSIN_2025, N.PENNSYLVANIA_2025, N.LANCASTER_2023, N.MICHIGAN_2025, N.NEW_YORK_2025
    logged = [
        ("16:12", "16:40", "Queen Anne's", "failed: 22% for USDA's 1.3%"),
        ("18:25", "19:11", "Michigan, two counties", "passed: 2.1% and 30.0% for USDA's 1.8% and 24.2%"),
        ("18:25", "19:14", "Maryland 2025", "reads low: 4.75% for USDA's 8.9%"),
        ("22:55", "22:58", "Wisconsin 2023, two counties", "failed: Marathon 4.3% for USDA's 32 to 50%"),
        ("00:18", "01:43", "Wisconsin 2025", f"on the pass mark, marked unreliable: {wi['ours']}% for USDA's {wi['usda']}%"),
        ("00:18", "01:53", "Pennsylvania 2025", f"failed: {pa['ours']}% for USDA's {pa['usda']}%"),
        ("00:18", "02:13", "Lancaster County 2023", f"passed: {la['ours']}% for USDA's {la['census_2022']}% (2022 Census)"),
        ("02:22", "04:42", "Michigan 2025", f"passed: {mi['ours']}% for USDA's {mi['usda']}% (we had predicted low: wrong)"),
        ("02:22", "04:55", "New York 2025", f"passed: {ny['ours']}% for USDA's {ny['usda']}% (we had predicted a fail: wrong)"),
    ]
    rh = 0.25
    for i, (rule_at, test_at, what, result) in enumerate(logged):
        y = 4.43 + i * (rh + 0.025)
        good = result.startswith("passed")
        card(s, MX, y, FULL, rh, fill=BLUE if good else WHITE, line=None if good else GREY)
        text(s, MX + 0.2, y, 3.6, rh, f"{rule_at} rule written    {test_at} test run once", size=12, color=WHITE if good else INK2,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, MX + 3.6, y, 2.8, rh, what, size=12.5, bold=True, color=WHITE if good else INK, anchor=MSO_ANCHOR.MIDDLE)
        text(s, MX + 6.4, y, FULL - 6.6, rh, result, size=12.5, color=WHITE if good else INK, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Every test rule was written in a log before the test ran. "
             "The 44 labelled fields were scored four times, each time logged. "
             "In Wisconsin 2025 our own warning check went off, so by our rule that figure is marked unreliable. "
             "We log our own predictions too: we wrote that Michigan 2025 would read low and that New York 2025 would fail. Both passed.",
          "每条检验规则都在检验之前写进了日志。44 个带标签的地块一共打了四次分，每次都有记录。"
          "2025 年威斯康星的测试里，我们自己设的预警触发了，所以按我们的规则，那个数字标为“不可靠”。"
          "我们自己的预测也有记录：我们写过 2025 年密歇根会读低、纽约州会不及格，结果两个都通过了。")

    s = new_slide(prs, "Maryland 2025, the state USDA scores: we read low")
    body_picture(s, "state_2025.png")
    notes(s, "For Maryland in 2025 we get 22,700 acres of silage; USDA reports 40,000. "
             "With double-crop corn scored afterwards it is 33,135, still low. "
             "With a second satellite added afterwards we see 99 percent of the corn, and it still reads low. So it is not clouds.",
          "2025 年的马里兰，我们算出青贮 22,700 英亩，美国农业部是 40,000。事后把双季玉米也打分后是 33,135，仍然偏低。"
          "事后再加上第二颗卫星，能看到 99% 的玉米，读数还是偏低。所以原因不是云。")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"{out.name}: {len(prs.slides)} slides, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path of the .pptx to write")
    build(ap.parse_args().out)
