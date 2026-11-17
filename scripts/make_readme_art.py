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
    w, h = 1200, 380
    rng = random.Random(7)
    tiles = []
    palette = ["#3f9a6a", "#58b782", "#2f7d55", "#d9b84a", "#8fcf9f", "#c9a53a"]
    for row in range(10):
        for col in range(34):
            x, y = 6 + col * 35.3, 6 + row * 37
            tiles.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="31" height="33" rx="5" '
                f'fill="{rng.choice(palette)}" opacity="{rng.uniform(0.10, 0.34):.2f}"/>'
            )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="FieldVision: Grand Prize, 1st place, Hack the Harvest 2026, USDA NASS challenge">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0a2419"/>
      <stop offset="0.55" stop-color="#12402d"/>
      <stop offset="1" stop-color="#1c5a40"/>
    </linearGradient>
    <linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0.30" stop-color="#fff" stop-opacity="0"/>
      <stop offset="0.75" stop-color="#fff" stop-opacity="1"/>
    </linearGradient>
    <mask id="tilemask"><rect width="{w}" height="{h}" fill="url(#fade)"/></mask>
    <clipPath id="frame"><rect width="{w}" height="{h}" rx="20"/></clipPath>
    <radialGradient id="gold" cx="0.38" cy="0.32" r="0.8">
      <stop offset="0" stop-color="#fdeaa5"/>
      <stop offset="0.55" stop-color="#d9ac2f"/>
      <stop offset="1" stop-color="#a47516"/>
    </radialGradient>
    <clipPath id="coin"><circle cx="150" cy="196" r="74"/></clipPath>
    <style>
      text {{ font-family: {FONT}; }}
      .draw {{ fill: none; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round;
               stroke-dasharray: 640; stroke-dashoffset: 0; animation: draw 9s ease-out infinite; }}
      .late {{ animation-name: drawlate; }}
      .shine {{ animation: sweep 5s ease-in-out infinite; }}
      .dot {{ opacity: 1; animation: pop 9s ease-out infinite; }}
      @keyframes drawlate {{ 0%, 8% {{ stroke-dashoffset: 640; }} 40% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: 0; }} }}
      @keyframes draw {{ 0% {{ stroke-dashoffset: 640; }} 32% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: 0; }} }}
      @keyframes sweep {{ 0%, 55% {{ transform: translateX(-170px); }} 85%, 100% {{ transform: translateX(170px); }} }}
      @keyframes pop {{ 0%, 30% {{ opacity: 0; }} 38%, 100% {{ opacity: 1; }} }}
    </style>
  </defs>

  <g clip-path="url(#frame)">
    <rect width="{w}" height="{h}" fill="url(#bg)"/>
    <g mask="url(#tilemask)">{''.join(tiles)}</g>

    <polygon points="92,70 138,70 152,150 108,170" fill="#2b78d6"/>
    <polygon points="162,70 208,70 192,170 148,150" fill="#4a97f0"/>
    <circle cx="150" cy="202" r="76" fill="#7e5a12"/>
    <circle cx="150" cy="196" r="74" fill="url(#gold)"/>
    <circle cx="150" cy="196" r="59" fill="none" stroke="#8a6516" stroke-width="2.5"/>
    <text x="150" y="216" text-anchor="middle" font-size="74" font-weight="800" fill="#6b4a0e">1</text>
    <text x="150" y="242" text-anchor="middle" font-size="10.5" font-weight="700" letter-spacing="1.6" fill="#6b4a0e">GRAND PRIZE</text>
    <g clip-path="url(#coin)">
      <rect class="shine" x="130" y="100" width="34" height="200" fill="#fff" opacity="0.38" transform="skewX(-18)"/>
    </g>

    <text x="262" y="150" font-size="66" font-weight="800" fill="#ffffff" letter-spacing="-0.5">FieldVision</text>
    <text x="264" y="186" font-size="19" fill="#d7e9df">Silage or grain corn, read from free satellite data</text>
    <rect x="264" y="204" width="500" height="2" fill="#d4a72c"/>
    <text x="264" y="238" font-size="19" fill="#ffffff">Hack the Harvest 2026 · USDA NASS challenge</text>
    <text x="264" y="275" font-size="27" font-weight="800" fill="{GOLD}">Grand Prize · 1st place · $5,000</text>
    <text x="264" y="305" font-size="15" fill="#cfe3d8">Judges from USDA, NVIDIA, Niantic Spatial and UC Davis</text>

    <rect x="830" y="66" width="322" height="248" rx="14" fill="#06180f" fill-opacity="0.78" stroke="#ffffff" stroke-opacity="0.16"/>
    <text x="850" y="96" font-size="13" fill="#cfe3d8">Greenness through the season</text>
    <line x1="850" y1="268" x2="1132" y2="268" stroke="#ffffff" stroke-opacity="0.25"/>
    <path class="draw" stroke="#7fe0a6" d="M850,262 C900,262 930,132 980,120 C1000,116 1012,118 1022,122 L1030,256 L1132,260"/>
    <path class="draw late" stroke="{GOLD}" d="M850,262 C900,262 930,142 980,130 C1010,124 1042,152 1078,198 L1088,206 L1097,258 L1132,261"/>
    <circle class="dot" cx="1026" cy="122" r="5" fill="#7fe0a6"/>
    <circle class="dot" cx="1088" cy="206" r="5" fill="{GOLD}"/>
    <circle cx="856" cy="291" r="5" fill="#7fe0a6"/>
    <text x="868" y="296" font-size="13" fill="#e8f3ed">silage, cut green</text>
    <circle cx="1000" cy="291" r="5" fill="{GOLD}"/>
    <text x="1012" y="296" font-size="13" fill="#e8f3ed">grain, left to dry</text>
  </g>
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
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius, fill=255)
    im.putalpha(mask)
    im.save(path)


def framed_check(src: Path, dst: Path) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    photo = Image.open(src).convert("RGBA")
    pad = 14
    card = Image.new("RGBA", (photo.width + 2 * pad, photo.height + 2 * pad), "#ffffff")
    card.paste(photo, (pad, pad))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card.width - 1, card.height - 1), 18, fill=255)
    card.putalpha(mask)
    card = card.rotate(-2, expand=True, resample=Image.BICUBIC)

    margin = 40
    canvas = Image.new("RGBA", (card.width + 2 * margin, card.height + 2 * margin), (0, 0, 0, 0))
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
                ("42 / 44", "labelled US fields called right", "trained with zero US labels"),
                ("4 of 5", "checks within 5 points of USDA", "rules written first, each run once"),
                ("13.3 → 1.4", "point gap with USDA history added", "Texas, California, Colorado"),
                ("0", "unsupported numbers from agents", "5/5 routing, 4/4 Q&amp;A checks"),
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
                ("8,000", "European training fields", "Austria, France, the Netherlands"),
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
