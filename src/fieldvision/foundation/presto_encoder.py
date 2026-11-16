"""The Presto encoder, trimmed to what we need to compute embeddings.

Copied from https://github.com/nasaharvest/presto (presto/presto.py,
commit 11e207a), MIT License, Copyright (c) 2024 Presto Authors. The
original package imports Earth Engine and Google Cloud at import time, so
only the encoder and its building blocks are kept here. Changes from the
original:

- band constants are written out instead of imported from dataops
- tensors follow the input's device instead of a global `device`
- the decoder, fine-tuning wrappers and training utilities are dropped

The layer names match the original, so the released weights
(data/default_model.pt, keys starting with "encoder.") load unchanged.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright
notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
WITHOUT WARRANTY OF ANY KIND.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F

from .presto_bands import BAND_GROUPS, DW_MISSING, EMBED_DIM, PRESTO_BANDS  # noqa: F401


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if attn_mask is not None:
            # True = this key takes part in attention
            attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, N, 1))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.ls1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Identity()

    def forward(self, x, attn_mask=None):
        x = x + self.ls1(self.attn(self.norm1(x), attn_mask))
        return x + self.ls2(self.mlp(self.norm2(x)))


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    if isinstance(positions, int):
        positions = list(range(positions))
    table = np.array([[p / np.power(T, 2 * (j // 2) / d_hid) for j in range(d_hid)]
                      for p in positions])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.FloatTensor(table)


def get_month_encoding_table(d_hid):
    """Sinusoid month encoding for months 0-11 (0 = January)."""
    assert d_hid % 2 == 0
    angles = np.arange(0, 13) / (12 / (2 * np.pi))
    sin_table = np.sin(np.stack([angles for _ in range(d_hid // 2)], axis=-1))
    cos_table = np.cos(np.stack([angles for _ in range(d_hid // 2)], axis=-1))
    return torch.FloatTensor(np.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1))


def month_to_tensor(month, batch_size, seq_len, device):
    """Month index of every time step, starting from `month` (int or one per sample)."""
    if isinstance(month, int):
        assert 0 <= month < 12
        return torch.fmod(torch.arange(month, month + seq_len, dtype=torch.long),
                          12).expand(batch_size, seq_len).to(device)
    month = torch.as_tensor(month)
    assert int(month.max()) < 12
    if month.ndim == 1:
        return torch.stack([torch.fmod(torch.arange(int(m), int(m) + seq_len, dtype=torch.long), 12)
                            for m in month]).to(device)
    return month.to(device)


class Encoder(nn.Module):
    def __init__(self, embedding_size=EMBED_DIM, channel_embed_ratio=0.25, month_embed_ratio=0.25,
                 depth=2, mlp_ratio=4, num_heads=8, max_sequence_length=24):
        super().__init__()
        self.band_groups = BAND_GROUPS
        self.embedding_size = embedding_size
        self.band_group_to_idx = {g: i for i, g in enumerate(self.band_groups)}
        self.band_group_to_idx["dynamic_world"] = max(self.band_group_to_idx.values()) + 1
        self.eo_patch_embed = nn.ModuleDict(
            {g: nn.Linear(len(idx), embedding_size) for g, idx in self.band_groups.items()})
        self.dw_embed = nn.Embedding(DW_MISSING + 1, embedding_size)
        self.latlon_embed = nn.Linear(3, embedding_size)
        self.blocks = nn.ModuleList([Block(embedding_size, num_heads, mlp_ratio)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embedding_size)
        self.max_sequence_length = max_sequence_length
        pos_size = int(embedding_size * (1 - (channel_embed_ratio + month_embed_ratio)))
        ch_size = int(embedding_size * channel_embed_ratio)
        month_size = int(embedding_size * month_embed_ratio)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_sequence_length, pos_size),
                                      requires_grad=False)
        self.pos_embed.data.copy_(get_sinusoid_encoding_table(max_sequence_length, pos_size))
        self.month_embed = nn.Embedding.from_pretrained(get_month_encoding_table(month_size),
                                                        freeze=True)
        self.channel_embed = nn.Embedding(len(self.band_groups) + 1, ch_size)

    @staticmethod
    def cartesian(latlons):
        with torch.no_grad():
            r = latlons * math.pi / 180
            lats, lons = r[:, 0], r[:, 1]
            return torch.stack([torch.cos(lats) * torch.cos(lons),
                                torch.cos(lats) * torch.sin(lons), torch.sin(lats)], dim=-1)

    @staticmethod
    def mask_tokens(x, mask):
        mask = mask.bool()
        sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
        x = x.gather(1, indices[:, :, None].expand_as(x))
        x = x * sorted_mask.unsqueeze(-1)
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        updated_mask = 1 - sorted_mask[:, :max_length]
        return x, indices, updated_mask

    def tokens(self, x, dynamic_world, latlons, mask=None, month=0):
        """Run the transformer; returns (tokens, token_mask, token_group, token_time).

        tokens       [B, N, D] output of the last block (before the final LayerNorm)
        token_mask   [B, N] 1 where the token is padding (masked input)
        token_group  [B, N] channel-group index of each token (-1 = lat/lon token)
        token_time   [B, N] time step of each token (-1 for lat/lon and SRTM)
        """
        dev = x.device
        if mask is None:
            mask = torch.zeros_like(x)
        B, T = x.shape[0], x.shape[1]
        months = month_to_tensor(month, B, T, dev)
        month_embedding = self.month_embed(months)
        positional = repeat(self.pos_embed[:, :T, :], "b t d -> (repeat b) t d", repeat=B)
        all_tokens, all_masks, all_groups, all_times = [], [], [], []
        steps = torch.arange(T, device=dev).expand(B, T)
        for group, idx in self.band_groups.items():
            tok = self.eo_patch_embed[group](x[:, :, idx])
            ch = self.channel_embed(torch.tensor(self.band_group_to_idx[group], device=dev))
            ch = repeat(ch, "d -> b t d", b=B, t=T)
            if group == "SRTM":
                emb = torch.cat((torch.zeros_like(month_embedding[:, 0:1]), ch[:, 0:1],
                                 torch.zeros_like(positional[:, 0:1])), dim=-1)
                sl = slice(0, 1)
            else:
                emb = torch.cat((month_embedding, ch, positional), dim=-1)
                sl = slice(None)
            tok = tok[:, sl] + emb
            all_tokens.append(tok)
            all_masks.append(torch.max(mask[:, sl][:, :, idx], dim=-1)[0])
            n = tok.shape[1]
            all_groups.append(torch.full((B, n), self.band_group_to_idx[group], device=dev))
            all_times.append(steps[:, sl] if group != "SRTM" else torch.full((B, 1), -1, device=dev))
        tok = self.dw_embed(dynamic_world)
        ch = self.channel_embed(torch.tensor(self.band_group_to_idx["dynamic_world"], device=dev))
        ch = repeat(ch, "d -> b t d", b=B, t=T)
        tok = tok + torch.cat((month_embedding, ch, positional), dim=-1)
        all_tokens.append(tok)
        all_masks.append((dynamic_world == DW_MISSING).to(x.dtype))
        all_groups.append(torch.full((B, T), self.band_group_to_idx["dynamic_world"], device=dev))
        all_times.append(steps)

        x = torch.cat(all_tokens, dim=1)
        m = torch.cat(all_masks, dim=1)
        groups = torch.cat(all_groups, dim=1)
        times = torch.cat(all_times, dim=1)
        x, idx, upd_mask = self.mask_tokens(x, m)
        groups = groups.gather(1, idx)[:, :x.shape[1]]
        times = times.gather(1, idx)[:, :x.shape[1]]

        latlon_tokens = self.latlon_embed(self.cartesian(latlons)).unsqueeze(1)
        x = torch.cat((latlon_tokens, x), dim=1)
        upd_mask = torch.cat((torch.zeros(B, 1, device=dev), upd_mask.to(dev).float()), dim=1)
        groups = torch.cat((torch.full((B, 1), -1, device=dev), groups), dim=1)
        times = torch.cat((torch.full((B, 1), -1, device=dev), times), dim=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=~upd_mask.bool())
        return x, upd_mask, groups, times

    def forward(self, x, dynamic_world, latlons, mask=None, month=0):
        """The pooled embedding Presto's authors use downstream: mean of the
        non-masked tokens, then the final LayerNorm. Shape [B, 128]."""
        tok, upd_mask, _, _ = self.tokens(x, dynamic_world, latlons, mask, month)
        keep = 1 - upd_mask.unsqueeze(-1)
        x_mean = (tok * keep).sum(dim=1) / keep.sum(dim=1)
        return self.norm(x_mean)


def load_encoder(weights_path, map_location="cpu"):
    """Encoder with the released Presto weights (data/default_model.pt)."""
    state = torch.load(weights_path, map_location=map_location, weights_only=True)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    enc = Encoder()
    enc.load_state_dict(enc_state, strict=True)
    enc.eval()
    return enc
