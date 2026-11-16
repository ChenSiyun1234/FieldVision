"""Draws docs/readme/00_banner.png: the README header.

Facts on the banner: Hack the Harvest 2026 is a USDA NASS challenge; the Grand Prize was
$5,000 in cash for 1 winning team (event page, https://hack-the-harvest.devpost.com/); judges
came from USDA, NVIDIA, Niantic Spatial and UC Davis (same page).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "readme" / "00_banner.png"
W, H = 2400, 700
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# deep green gradient with a soft diagonal
x = np.linspace(0, 1, 600); y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)
g = 0.55 * X + 0.45 * Y
c0 = np.array([0x14, 0x3A, 0x2B]) / 255; c1 = np.array([0x2F, 0x6B, 0x4F]) / 255
img = c0[None, None, :] * (1 - g[..., None]) + c1[None, None, :] * g[..., None]
ax.imshow(img, extent=[0, W, 0, H], aspect="auto", zorder=0)

# faint field rows
for i in range(-6, 30):
    ax.plot([1180 + i * 60, 1500 + i * 60 + 520], [-20, H + 20], color="#ffffff", alpha=0.035, lw=14, zorder=1)

GOLD = "#D4AF37"; GOLD_D = "#A8842A"; GOLD_L = "#F1D77A"; CREAM = "#FCFCFA"
cx, cy = 360, 340
# ribbon
ax.add_patch(Polygon([[cx - 150, cy + 250], [cx - 40, cy + 250], [cx + 20, cy - 40], [cx - 70, cy - 120], [cx - 170, cy - 40]],
                     closed=True, fc="#2878D6", ec="#1b4f8c", lw=3, zorder=3))
ax.add_patch(Polygon([[cx + 150, cy + 250], [cx + 40, cy + 250], [cx - 20, cy - 40], [cx + 70, cy - 120], [cx + 170, cy - 40]],
                     closed=True, fc="#3C8CE6", ec="#1b4f8c", lw=3, zorder=3))
# medal
ax.add_patch(Circle((cx, cy - 10), 190, fc=GOLD_D, ec="none", zorder=4))
ax.add_patch(Circle((cx, cy), 185, fc=GOLD, ec=GOLD_L, lw=6, zorder=5))
ax.add_patch(Circle((cx, cy), 150, fc="none", ec=GOLD_D, lw=4, zorder=6))
for k in range(48):
    a = 2 * np.pi * k / 48
    ax.plot([cx + 160 * np.cos(a), cx + 172 * np.cos(a)], [cy + 160 * np.sin(a), cy + 172 * np.sin(a)], color=GOLD_D, lw=3, zorder=6)
ax.text(cx, cy + 22, "1", ha="center", va="center", fontsize=118, fontweight="bold", color="#6E5516", zorder=7)
ax.text(cx, cy - 92, "GRAND PRIZE", ha="center", va="center", fontsize=19, fontweight="bold", color="#6E5516", zorder=7)

# text block
tx = 700
ax.text(tx, 530, "FieldVision", fontsize=86, fontweight="bold", color=CREAM, va="center", zorder=7)
ax.text(tx, 440, "Silage vs. grain corn from satellite time series", fontsize=34, color="#DCE9E1", va="center", zorder=7)
ax.plot([tx, tx + 1420], [388, 388], color=GOLD, lw=3, zorder=7)
ax.text(tx, 330, "Hack the Harvest 2026  ·  USDA NASS challenge", fontsize=30, color=CREAM, va="center", zorder=7)
ax.text(tx, 262, "Grand Prize  ·  1st place  ·  $5,000", fontsize=36, fontweight="bold", color=GOLD_L, va="center", zorder=7)
ax.text(tx, 190, "Judges from USDA, NVIDIA, Niantic Spatial and UC Davis", fontsize=26, color="#DCE9E1", va="center", zorder=7)
ax.text(tx, 120, "Sentinel-2 and Landsat 8/9   ·   Python   ·   scikit-learn   ·   Google Colab", fontsize=22, color="#9FC3B0", va="center", zorder=7)

fig.savefig(OUT, dpi=100, facecolor="#143A2B")
im = Image.open(OUT).convert("RGB")
im.thumbnail((2400, 700))
im.save(OUT, optimize=True)
print(OUT, im.size, OUT.stat().st_size)
