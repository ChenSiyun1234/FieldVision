"""Small networks that read a field through the season.

All four share the temporal part and differ only in what they see at one date:

  SeqNet       the ten band medians of the field (the control: no spatial information)
  PixelSetNet  the set of pixels inside the field, in any order (distribution, not layout)
  ChipSeqNet   the image chip, read by a 2D convolution (layout too)
  Conv3dNet    the chips stacked on an even time grid and read by 3D convolutions,
               space and time in the same filters (the "video" form)

Sizes are deliberately small (81,000 to 435,000 weights): the training
sets are a few thousand fields.
"""
from __future__ import annotations

import math

import torch
from torch import nn

PERIODS = (365.0, 180.0, 90.0, 45.0, 22.0, 11.0)      # days


class TimeEncoding(nn.Module):
    """Sines and cosines of the day count, mixed by a linear layer."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("freq", torch.tensor([2 * math.pi / p for p in PERIODS]))
        self.proj = nn.Linear(2 * len(PERIODS), dim)

    def forward(self, days):
        a = days.unsqueeze(-1) * self.freq
        return self.proj(torch.cat([a.sin(), a.cos()], dim=-1))


class Temporal(nn.Module):
    """Self-attention over the dates of one field, then mean and max over real dates."""

    def __init__(self, dim=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.time = TimeEncoding(dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(dim, 1))

    def forward(self, tokens, days, valid):
        if not bool(valid.any(1).all()):          # a field with no date at all: keep one empty step
            valid = valid.clone()
            valid[~valid.any(1), 0] = True
        h = self.encoder(tokens + self.time(days), src_key_padding_mask=~valid)
        h = self.norm(h)
        m = valid.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        peak = h.masked_fill(~m, -1e4).max(1).values
        return self.head(torch.cat([mean, peak], dim=-1)).squeeze(-1)


class SeqNet(nn.Module):
    def __init__(self, n_in=10, dim=64, **kw):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_in, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.temporal = Temporal(dim, **kw)

    def forward(self, x, days, valid):
        return self.temporal(self.embed(x), days, valid)


class PixelSetNet(nn.Module):
    """Pixel-set encoder: a shared network per pixel, mean and spread over the set,
    then the temporal part. The order of the pixels does not matter."""

    def __init__(self, n_in=10, dim=64, hidden=64, **kw):
        super().__init__()
        self.pixel = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, hidden), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(2 * hidden + 1, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.temporal = Temporal(dim, **kw)

    def forward(self, px, use, days, valid):
        """px (B, L, S, C) pixels; use (B, L, S) True for a clear pixel inside the field."""
        h = self.pixel(px)
        m = use.unsqueeze(-1).float()
        n = m.sum(2).clamp(min=1)
        mean = (h * m).sum(2) / n
        var = ((h - mean.unsqueeze(2)) ** 2 * m).sum(2) / n
        count = torch.log1p(use.float().sum(2, keepdim=True)) / 5.0
        tokens = self.merge(torch.cat([mean, (var + 1e-6).sqrt(), count], dim=-1))
        return self.temporal(tokens, days, valid)


def _norm(channels):
    return nn.GroupNorm(8, channels)


class ChipSeqNet(nn.Module):
    """A 2D convolution reads each date's chip; attention reads the dates."""

    def __init__(self, n_in=12, dim=64, **kw):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, 32, 3, padding=1), _norm(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), _norm(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), _norm(64), nn.ReLU())
        self.merge = nn.Linear(128, dim)
        self.temporal = Temporal(dim, **kw)

    def forward(self, chips, days, valid):
        """chips (B, L, C, H, W)."""
        b, length = chips.shape[:2]
        h = self.conv(chips.flatten(0, 1))
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=-1)
        return self.temporal(self.merge(h).view(b, length, -1), days, valid)


class Conv3dNet(nn.Module):
    """3D convolutions over (time, height, width) on evenly spaced frames."""

    def __init__(self, n_in=13, width=32, dropout=0.1):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv3d(a, b, 3, padding=1), _norm(b), nn.ReLU(),
                                 nn.Conv3d(b, b, 3, padding=1), _norm(b), nn.ReLU())

        self.body = nn.Sequential(block(n_in, width), nn.MaxPool3d((1, 2, 2)),
                                  block(width, 2 * width), nn.MaxPool3d((2, 2, 2)),
                                  block(2 * width, 2 * width))
        self.head = nn.Sequential(nn.Linear(4 * width, 2 * width), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(2 * width, 1))

    def forward(self, frames):
        """frames (B, C, T, H, W); the last input channel says whether a frame exists."""
        h = self.body(frames)
        h = torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=-1)
        return self.head(h).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
