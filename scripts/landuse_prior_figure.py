"""Figure for docs/landuse-prior.md: county forage share (2019-2023 mean) against the Census 2022 silage share.

    python scripts/landuse_prior_figure.py --data C:/t/fv/data/us/landuse --out docs/img/landuse_prior.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES1, SERIES2 = '#2a78d6', '#eb6834'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    d = pd.read_csv(a.data / 'census2022_with_landuse.csv', dtype={'fips': str}).dropna(subset=['forage_share_mean1923'])
    c0, c1 = json.loads((a.data / 'coefficients.json').read_text())['forage_share_mean1923|linear']
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    size = 4 + 60 * d.total / d.total.max()
    ax.scatter(100 * d.forage_share_mean1923, 100 * d.silage_share, s=size, color=SERIES1, alpha=0.35,
               linewidths=0, label=f'County, Census 2022 ({len(d):,}; dot area = corn acres)')
    x = np.linspace(0, d.forage_share_mean1923.quantile(0.995), 50)
    ax.plot(100 * x, 100 * np.clip(c0 + c1 * x, 0, 1), color=SERIES2, linewidth=2,
            label=f'Weighted fit: share = {c0:.3f} + {c1:.2f} x forage share')
    ax.set_xlabel('Forage share of CDL cropland, 2019-2023 mean (%)', color=INK2)
    ax.set_ylabel('Census 2022 silage share of corn (%)', color=INK2)
    ax.set_title('Silage share rises with forage land use (a prior, not a field label)', color=INK, fontsize=11, loc='left')
    ax.set_xlim(0, 100 * d.forage_share_mean1923.quantile(0.995))
    ax.set_ylim(0, 100)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)
    leg = ax.legend(loc='upper left', frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE)
    print(a.out, a.out.stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
