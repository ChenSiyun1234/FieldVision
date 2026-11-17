"""Draw the README header art and tidy the figure corners.

Run from the repository root, after scripts/make_readme_figures.py:

    python scripts/make_readme_art.py

Writes hero.svg, stats.svg and model_stats.svg into docs/readme/, rounds the
corners of the figure PNGs there and makes the framed prize-check picture.
"""

from pathlib import Path
import random

OUT = Path(__file__).resolve().parents[1] / "docs" / "readme"
FONT = "'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"

GOLD = "#f2d16b"
INK = "#0b2a1e"


def hero() -> str:
    w, h = 1200, 300
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="FieldVision: Grand Prize, Hack the Harvest 2026, USDA NASS challenge">
  <defs>
    <style>
      text {{ font-family: {FONT}; }}
      .draw {{ fill: none; stroke-width: 3.5; stroke-linecap: round; stroke-linejoin: round;
               stroke-dasharray: 640; stroke-dashoffset: 0; animation: draw 9s ease-out infinite; }}
      .late {{ animation-name: drawlate; }}
      .dot {{ opacity: 1; animation: pop 9s ease-out infinite; }}
      @keyframes drawlate {{ 0%, 8% {{ stroke-dashoffset: 640; }} 40% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: 0; }} }}
      @keyframes draw {{ 0% {{ stroke-dashoffset: 640; }} 32% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: 0; }} }}
      @keyframes pop {{ 0%, 30% {{ opacity: 0; }} 38%, 100% {{ opacity: 1; }} }}
    </style>
  </defs>

  <rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="20" fill="#ffffff" stroke="#dde7e1" stroke-width="1.5"/>

  <text x="64" y="112" font-size="62" font-weight="800" fill="#13261c" letter-spacing="-0.5">FieldVision</text>
  <text x="66" y="150" font-size="20" fill="#4f6358">Silage or grain corn, read from free satellite data</text>
  <rect x="66" y="172" width="64" height="3" rx="1.5" fill="#1d6b47"/>
  <text x="66" y="212" font-size="17" fill="#13261c">Grand Prize, Hack the Harvest 2026 · USDA NASS challenge</text>
  <text x="66" y="240" font-size="14.5" fill="#4f6358">Judged by USDA, NVIDIA, Niantic Spatial and UC Davis</text>

  <rect x="830" y="34" width="322" height="232" rx="14" fill="#f3f8f5" stroke="#dde7e1"/>
  <text x="850" y="62" font-size="13" fill="#4f6358">Greenness through the season</text>
  <line x1="850" y1="222" x2="1132" y2="222" stroke="#13261c" stroke-opacity="0.15"/>
  <path class="draw" stroke="#1d6b47" d="M850,216 C900,216 930,96 980,84 C1000,80 1012,82 1022,86 L1030,210 L1132,214"/>
  <path class="draw late" stroke="#c98f0c" d="M850,216 C900,216 930,106 980,94 C1010,88 1042,116 1078,162 L1088,170 L1097,212 L1132,215"/>
  <circle class="dot" cx="1026" cy="86" r="5" fill="#1d6b47"/>
  <circle class="dot" cx="1088" cy="170" r="5" fill="#c98f0c"/>
  <circle cx="856" cy="245" r="5" fill="#1d6b47"/>
  <text x="868" y="250" font-size="13" fill="#13261c">silage, cut green</text>
  <circle cx="1000" cy="245" r="5" fill="#c98f0c"/>
  <text x="1012" y="250" font-size="13" fill="#13261c">grain, left to dry</text>
</svg>
"""


def strip(cards, label) -> str:
    w, h, cw, gap = 1200, 132, 282, 24
    parts = []
    for i, (big, line1, line2) in enumerate(cards):
        x = i * (cw + gap)
        parts.append(
            f'<rect x="{x + 1}" y="1" width="{cw - 2}" height="{h - 2}" rx="14" fill="url(#card)" stroke="#2f6b4f"/>'
            f'<text x="{x + 22}" y="58" font-size="40" font-weight="800" fill="{GOLD}">{big}</text>'
            f'<text x="{x + 22}" y="86" font-size="14" fill="#e8f3ed">{line1}</text>'
            f'<text x="{x + 22}" y="108" font-size="13" fill="#a9c5b6">{line2}</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{label}">
  <defs>
    <linearGradient id="card" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{INK}"/>
      <stop offset="1" stop-color="#17493a"/>
    </linearGradient>
    <style>text {{ font-family: {FONT}; }}</style>
  </defs>
  {''.join(parts)}
</svg>
"""


def round_png(path: Path, radius: int) -> None:
    from PIL import Image, ImageDraw

    im = Image.open(path).convert("RGBA")
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, im.width - 1, im.height - 1), radius, fill=255
    )
    im.putalpha(mask)
    im.save(path)


def framed_check(src: Path, dst: Path) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    photo = Image.open(src).convert("RGBA")
    pad = 14
    card = Image.new("RGBA", (photo.width + 2 * pad, photo.height + 2 * pad), "#ffffff")
    card.paste(photo, (pad, pad))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, card.width - 1, card.height - 1), 18, fill=255
    )
    card.putalpha(mask)
    card = card.rotate(-2, expand=True, resample=Image.BICUBIC)

    margin = 40
    canvas = Image.new(
        "RGBA", (card.width + 2 * margin, card.height + 2 * margin), (0, 0, 0, 0)
    )
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 110), (margin, margin + 10), card.split()[3])
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(14)))
    canvas.alpha_composite(card, (margin, margin))
    canvas.save(dst)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hero.svg").write_text(hero(), encoding="utf-8", newline="\n")
    (OUT / "stats.svg").write_text(
        strip(
            [
                (
                    "42 / 44",
                    "labelled US fields called right",
                    "trained with zero US labels",
                ),
                (
                    "4 of 5",
                    "checks within 5 points of USDA",
                    "rules written first, each run once",
                ),
                (
                    "13.3 → 1.4",
                    "point gap with USDA history added",
                    "Texas, California, Colorado",
                ),
                (
                    "0",
                    "unsupported numbers from agents",
                    "5/5 routing, 4/4 Q&amp;A checks",
                ),
            ],
            "42 of 44 fields right; 4 of 5 checks within 5 points of USDA; "
            "gap 13.3 to 1.4 points with USDA history; 0 unsupported numbers from the agents",
        ),
        encoding="utf-8",
        newline="\n",
    )
    (OUT / "model_stats.svg").write_text(
        strip(
            [
                (
                    "8,000",
                    "European training fields",
                    "Austria, France, the Netherlands",
                ),
                ("0", "US labels used in training", "all 44 kept for the test"),
                ("6", "crop-timing features", "one logistic regression"),
                ("266", "commits inside the 22.5-hour build", "of my 271 on main"),
            ],
            "8,000 European training fields; 0 US labels used in training; "
            "6 crop-timing features; 266 commits inside the 22.5-hour build",
        ),
        encoding="utf-8",
        newline="\n",
    )
    for png in sorted(OUT.glob("0[1-8]_*.png")):
        round_png(png, 26)
    framed_check(OUT / "grand_prize_check.jpg", OUT / "grand_prize_check_card.png")


if __name__ == "__main__":
    main()
