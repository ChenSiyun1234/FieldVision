"""Frozen Presto embeddings, one 128-number vector per field-season.

Pooling: Presto's own downstream recipe. The encoder produces one token per
(channel group, month) that is not masked plus one location token; the
field vector is the mean of all non-masked tokens followed by the encoder's
final LayerNorm. Masked months and the modalities we lack contribute
nothing. This is what the Presto authors feed to their linear and
random-forest heads, so it is the fairest "off the shelf" use.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .presto_bands import EMBED_DIM

DEFAULT_WEIGHTS = Path(r"C:\t\fv\data\foundation\vendor\presto\data\default_model.pt")
EMB_COLS = [f"presto_{i:03d}" for i in range(EMBED_DIM)]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def embed_batch(batch, weights=DEFAULT_WEIGHTS, batch_size=512, device="cpu", encoder=None):
    """PrestoBatch -> DataFrame(field_id, presto_000 ... presto_127, n_months_observed).

    Fields with no observed month get NaN embeddings (Presto would only see
    the location token, which says nothing about the crop).
    """
    import torch

    from .presto_encoder import load_encoder
    enc = encoder if encoder is not None else load_encoder(weights)
    enc = enc.to(device).eval()
    N = len(batch.field_id)
    out = np.full((N, EMBED_DIM), np.nan, dtype=np.float32)
    seen = batch.n_months_observed > 0
    idx = np.flatnonzero(seen)
    with torch.no_grad():
        for a in range(0, len(idx), batch_size):
            sel = idx[a:a + batch_size]
            e = enc(torch.from_numpy(batch.x[sel]).to(device),
                    torch.from_numpy(batch.dynamic_world[sel]).to(device),
                    torch.from_numpy(batch.latlons[sel]).to(device),
                    mask=torch.from_numpy(batch.mask[sel]).to(device),
                    month=int(batch.month))
            out[sel] = e.cpu().numpy()
    df = pd.DataFrame(out, columns=EMB_COLS)
    df.insert(0, "field_id", batch.field_id)
    df["n_months_observed"] = batch.n_months_observed
    return df
