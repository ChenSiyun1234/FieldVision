"""Train the sequence and stream models on fixed folds and write held-out predictions.

    python -m fieldvision.stream.train --folds folds.parquet --timeseries <parquet ...>
        [--chips <chips.npz ...>] --models seq pixelset chipseq conv3d --out <dir>

`folds.parquet` has one row per labelled field: field_id, block_id, country,
year, y (1 silage, 0 grain) and fold (the spatial fold of the reference
run). Two schemes are trained: `spatial` (each fold held out) and
`loco:<country>` (each country held out). The output has the columns of the
main line's prediction tables, so the same summaries and paired
comparisons apply.

Every setting below was fixed before any result was seen and is the same
for every model and fold; nothing is tuned on a held-out fold, there is no
early stopping, and the last epoch is the model. Band scaling comes from
the training fields of the fold only.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.indices import BANDS
from .series import (GRID_START, REL_START, attach_chips, build_sequences, greenup_doy,
                     regular_grid, time_axis)

MISSING = 65535
CLEAR_SCL = (4, 5, 6, 7)
KINDS = ("seq", "pixelset", "chipseq", "conv3d")
TIMES = {"calendar": "cal", "greenup": "rel"}


@dataclass(frozen=True)
class Settings:
    epochs: int = 30
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2
    drop_dates: float = 0.2       # share of dates hidden from a field in each training step
    min_dates: int = 4
    set_size: int = 32            # pixels drawn per date in training (all of them at test time)
    dim: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    seeds: int = 3


def model_name(kind, time_mode):
    return f"{kind}-{TIMES[time_mode]}"


def load_chip_files(paths):
    """Concatenate chip stores (see fieldvision.stream.pixels) from several runs."""
    parts = [np.load(p, allow_pickle=False) for p in paths]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("field_id", "date", "bands", "scl", "inside")}
    out["field_id"] = out["field_id"].astype(str)
    out["date"] = out["date"].astype(str)
    return out


class StreamData:
    """Everything the models read, for one list of fields."""

    def __init__(self, ts, field_ids, chips=None):
        self.seq = build_sequences(ts, field_ids)
        self.chips = chips
        self.obs, self.dropped = (attach_chips(self.seq, chips) if chips is not None else (None, 0))
        s = self.seq
        self.anchor = greenup_doy(s["x"], s["doy"], s["valid"])
        self.days = {"calendar": time_axis(s["doy"], s["valid"]),
                     "greenup": time_axis(s["doy"], s["valid"], self.anchor)}
        self.grid = {"calendar": regular_grid(s["doy"], s["valid"], GRID_START),
                     "greenup": regular_grid(self.days["greenup"], s["valid"], REL_START)}

    def band_stats(self, rows):
        v = self.seq["x"][rows][self.seq["valid"][rows]]
        return v.mean(0).astype(np.float32), np.maximum(v.std(0), 1e-3).astype(np.float32)


def _hide_dates(valid, rate, min_dates, rng):
    """Training only: hide a random share of each field's dates, keeping at least `min_dates`."""
    drop = valid & (rng.random(valid.shape) < rate)
    kept = (valid & ~drop).sum(1)
    drop[kept < min_dates] = False
    return valid & ~drop


def _chip_channels(data, obs, stats):
    """(…, 12, K, K): ten scaled bands (0 where unusable), clear mask, inside mask."""
    c = data.chips
    safe = np.maximum(obs, 0)
    raw = c["bands"][safe]
    clear = np.isin(c["scl"][safe], CLEAR_SCL) & (raw != MISSING).all(-3)
    inside = c["inside"][safe]
    mean, std = stats
    x = (raw.astype(np.float32) / 10000.0 - mean[:, None, None]) / std[:, None, None]
    x = np.where(clear[..., None, :, :], x, 0.0)
    exists = (obs >= 0)[..., None, None]
    clear, inside = clear & exists, inside & exists
    x = np.where(exists[..., None, :, :], x, 0.0)
    return np.concatenate([x, clear[..., None, :, :].astype(np.float32),
                           inside[..., None, :, :].astype(np.float32)], axis=-3).astype(np.float32)


def _turn(a, k, flip):
    a = np.rot90(a, k, axes=(-2, -1))
    return a[..., ::-1] if flip else a


def make_batch(data, rows, kind, time_mode, stats, cfg, rng=None):
    """Numpy inputs of one batch for a model kind; `rng` switches the training augmentations on."""
    s = data.seq
    valid = s["valid"][rows]
    if rng is not None and kind != "conv3d":
        valid = _hide_dates(valid, cfg.drop_dates, cfg.min_dates, rng)
    # real dates come first in every row, so trimming the padding is a slice
    used = np.flatnonzero(valid.any(0))
    width = int(used.max()) + 1 if len(used) else 1
    valid = valid[:, :width]
    days = data.days[time_mode][rows][:, :width] / 100.0
    mean, std = stats
    if kind == "seq":
        x = (s["x"][rows][:, :width] - mean) / std
        return {"x": x * valid[..., None], "days": days, "valid": valid}
    if kind == "conv3d":
        grid = data.grid[time_mode][rows]
        pos = np.take_along_axis(data.obs[rows], np.maximum(grid, 0), axis=1)
        pos = np.where(grid >= 0, pos, -1)
        if rng is not None:
            pos = np.where(rng.random(pos.shape) < cfg.drop_dates, -1, pos)
        ch = _chip_channels(data, pos, stats)                              # (B, T, 12, K, K)
        exists = np.broadcast_to((pos >= 0)[..., None, None, None], ch[:, :, :1].shape)
        frames = np.concatenate([ch, exists.astype(np.float32)], axis=2)
        if rng is not None:
            frames = np.stack([_turn(f, int(rng.integers(4)), bool(rng.integers(2))) for f in frames])
        return {"frames": np.ascontiguousarray(frames.transpose(0, 2, 1, 3, 4))}
    obs = np.where(valid, data.obs[rows][:, :width], -1)
    ch = _chip_channels(data, obs, stats)                                  # (B, L, 12, K, K)
    if kind == "chipseq":
        if rng is not None:
            ch = np.stack([_turn(f, int(rng.integers(4)), bool(rng.integers(2))) for f in ch])
        return {"chips": np.ascontiguousarray(ch), "days": days, "valid": valid}
    b, length = ch.shape[:2]
    px = ch[:, :, :10].reshape(b, length, 10, -1).transpose(0, 1, 3, 2)   # (B, L, S, 10)
    use = (ch[:, :, 10] * ch[:, :, 11]).reshape(b, length, -1) > 0
    if rng is not None and px.shape[2] > cfg.set_size:
        score = np.where(use, rng.random(use.shape), 2.0)
        pick = np.argsort(score, axis=2)[:, :, :cfg.set_size]
        px = np.take_along_axis(px, pick[..., None], axis=2)
        use = np.take_along_axis(use, pick, axis=2)
    # a date with no clear pixel inside the field tells nothing: switch it off
    valid = valid & use.any(2)
    return {"px": np.ascontiguousarray(px), "use": use, "days": days, "valid": valid}


def build_model(kind, cfg):
    from . import nets
    kw = dict(layers=cfg.layers, heads=cfg.heads, dropout=cfg.dropout)
    if kind == "seq":
        return nets.SeqNet(len(BANDS), cfg.dim, **kw)
    if kind == "pixelset":
        return nets.PixelSetNet(len(BANDS), cfg.dim, **kw)
    if kind == "chipseq":
        return nets.ChipSeqNet(len(BANDS) + 2, cfg.dim, **kw)
    if kind == "conv3d":
        return nets.Conv3dNet(len(BANDS) + 3, dropout=cfg.dropout)
    raise ValueError(f"unknown model kind {kind!r}")


def fit_predict(data, y, kind, time_mode, train, test, cfg=Settings(), seed=0, device="cpu"):
    """Train one model on `train` rows and return silage probabilities for `test` rows."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    stats = data.band_stats(train)
    model = build_model(kind, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(train) / cfg.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    target = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def forward(rows, augment):
        batch = make_batch(data, rows, kind, time_mode, stats, cfg, rng if augment else None)
        return model(**{k: torch.from_numpy(v).to(device) for k, v in batch.items()})

    losses = []
    for _ in range(cfg.epochs):
        model.train()
        order, total = rng.permutation(train), 0.0
        for a in range(0, len(order), cfg.batch):
            rows = order[a:a + cfg.batch]
            opt.zero_grad()
            loss = loss_fn(forward(rows, True), target[rows].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(rows)
        losses.append(total / len(train))
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(test), 2 * cfg.batch):
            out.append(torch.sigmoid(forward(test[a:a + 2 * cfg.batch], False)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32), losses


def make_splits(folds, schemes):
    """(scheme, fold name, train rows, test rows) for the spatial folds and each held-out country."""
    out = []
    if "spatial" in schemes:
        for name in sorted(folds["fold"].unique()):
            out.append(("spatial", str(name), np.flatnonzero(folds["fold"] != name),
                        np.flatnonzero(folds["fold"] == name)))
    if "loco" in schemes:
        for c in sorted(folds["country"].unique()):
            out.append((f"loco:{c}", str(c), np.flatnonzero(folds["country"] != c),
                        np.flatnonzero(folds["country"] == c)))
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", required=True)
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--chips", nargs="*", default=[])
    ap.add_argument("--models", nargs="+", default=["seq"], choices=KINDS)
    ap.add_argument("--time", nargs="+", default=["greenup", "calendar"], choices=list(TIMES))
    ap.add_argument("--schemes", nargs="+", default=["loco", "spatial"], choices=["loco", "spatial"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=Settings.seeds)
    ap.add_argument("--epochs", type=int, default=Settings.epochs)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args(argv)


def main(argv=None):
    import torch

    from ..features.loading import load_timeseries
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cfg = Settings(epochs=args.epochs, seeds=args.seeds)
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)

    folds = pd.read_parquet(args.folds).reset_index(drop=True)
    folds["field_id"] = folds["field_id"].astype(str)
    ts, _ = load_timeseries(args.timeseries, field_ids=folds["field_id"])
    chips = load_chip_files(args.chips) if args.chips else None
    data = StreamData(ts, folds["field_id"].to_numpy(), chips)
    has = data.seq["valid"].any(1)
    if not has.all():
        print(f"{int((~has).sum())} fields without a usable date are left out of every model", flush=True)
        folds = folds[has].reset_index(drop=True)
        data = StreamData(ts, folds["field_id"].to_numpy(), chips)
    y = folds["y"].to_numpy(int)
    print(f"{len(folds)} fields on {device}; dates per field median "
          f"{int(np.median(data.seq['valid'].sum(1)))}; steps without a chip dropped: {data.dropped}",
          flush=True)
    (out / "settings.json").write_text(json.dumps(
        {"settings": asdict(cfg), "args": vars(args), "device": device, "fields": int(len(folds)),
         "steps_without_chip": int(data.dropped)}, indent=1), encoding="utf-8")

    meta = folds[["field_id", "block_id", "country", "year"]].copy()
    meta["low_evidence"] = folds["low_evidence"] if "low_evidence" in folds.columns else False
    for scheme, fold, train, test in make_splits(folds, args.schemes):
        for kind in args.models:
            if kind != "seq" and chips is None:
                raise SystemExit(f"model {kind} needs --chips")
            for mode in args.time:
                name = model_name(kind, mode)
                part = out / "parts" / f"{scheme.replace(':', '_')}_{fold}_{name}.parquet"
                if part.exists():
                    continue
                t0 = time.perf_counter()
                runs = [fit_predict(data, y, kind, mode, train, test, cfg, seed, device)
                        for seed in range(cfg.seeds)]
                frame = meta.iloc[test].copy()
                frame["scheme"], frame["variant"], frame["fold"], frame["model"] = scheme, "relative", fold, name
                frame["y"] = y[test]
                for k, (p, _) in enumerate(runs):
                    frame[f"p_seed{k}"] = p
                frame["p"] = np.mean([p for p, _ in runs], axis=0)
                frame["threshold"] = 0.5
                frame["pred"] = (frame["p"] >= 0.5).astype(int)
                frame["fit_seconds"] = time.perf_counter() - t0
                frame.to_parquet(part, index=False)
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(frame["y"], frame["p"]) if frame["y"].nunique() == 2 else float("nan")
                print(f"{scheme} {fold} {name}: {len(train)} train / {len(test)} test, loss "
                      f"{runs[0][1][0]:.3f} -> {runs[0][1][-1]:.3f}, ROC-AUC {auc:.3f}, "
                      f"{frame['fit_seconds'].iloc[0]:.0f}s", flush=True)
    parts = sorted((out / "parts").glob("*.parquet"))
    preds = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    preds.to_parquet(out / "predictions.parquet", index=False)
    print(f"wrote {out / 'predictions.parquet'}: {len(preds)} rows, models {sorted(preds['model'].unique())}")
    return preds


if __name__ == "__main__":
    main()
