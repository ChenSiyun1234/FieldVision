"""Train one stream model on every labelled field and score fields it has never seen.

    python -m fieldvision.stream.apply --folds folds.parquet --timeseries <training parquet ...>
        --target-timeseries <parquet ...> --model seq --time greenup --out <dir>
        [--chips <npz ...> --target-chips <npz ...>]

The target fields need no labels: their ids are taken from the target
time series, nothing else about them is read. Output: one row per target
field with the silage probability of every seed and their mean. Settings
are the fixed ones of `fieldvision.stream.train`; band scaling comes from
the training fields only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .train import KINDS, TIMES, Settings, StreamData, fit_predict, load_chip_files, model_name


def fit_and_score(ts, labelled, target_ids, kind, time_mode, chips=None, cfg=Settings(), device="cpu"):
    """labelled: frame with field_id and y. Returns a frame for `target_ids`."""
    train_ids = labelled["field_id"].astype(str).to_numpy()
    target_ids = np.asarray(target_ids).astype(str)
    if set(train_ids) & set(target_ids):
        raise ValueError("target fields must not be among the training fields")
    data = StreamData(ts, np.concatenate([train_ids, target_ids]), chips)
    has = data.seq["valid"].any(1)
    train = np.flatnonzero(has[:len(train_ids)])
    test = len(train_ids) + np.flatnonzero(has[len(train_ids):])
    y = np.concatenate([labelled["y"].to_numpy(int), np.zeros(len(target_ids), int)])
    out = pd.DataFrame({"field_id": target_ids, "model": model_name(kind, time_mode),
                        "n_dates": data.seq["valid"].sum(1)[len(train_ids):]})
    for seed in range(cfg.seeds):
        p = np.full(len(target_ids), np.nan)
        p[test - len(train_ids)] = fit_predict(data, y, kind, time_mode, train, test, cfg, seed, device)[0]
        out[f"p_seed{seed}"] = p
    out["p"] = out[[f"p_seed{k}" for k in range(cfg.seeds)]].mean(axis=1)
    out["n_train"] = len(train)
    return out


def main(argv=None):
    import torch

    from ..features.loading import load_timeseries
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", required=True)
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--target-timeseries", nargs="+", required=True)
    ap.add_argument("--chips", nargs="*", default=[])
    ap.add_argument("--target-chips", nargs="*", default=[])
    ap.add_argument("--model", default="seq", choices=KINDS)
    ap.add_argument("--time", default="greenup", choices=list(TIMES))
    ap.add_argument("--seeds", type=int, default=Settings.seeds)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labelled = pd.read_parquet(args.folds)
    train_ts, _ = load_timeseries(args.timeseries, field_ids=labelled["field_id"])
    target_ts, _ = load_timeseries(args.target_timeseries)
    target_ids = sorted(target_ts["field_id"].unique())
    chips = load_chip_files([*args.chips, *args.target_chips]) if args.chips else None
    cfg = Settings(seeds=args.seeds)
    scores = fit_and_score(pd.concat([train_ts, target_ts], ignore_index=True), labelled, target_ids,
                           args.model, args.time, chips, cfg, device)
    scores.to_parquet(out / "scores.parquet", index=False)
    (out / "settings.json").write_text(json.dumps({"settings": asdict(cfg), "args": vars(args),
                                                   "device": device, "n_train": int(scores["n_train"].iloc[0]),
                                                   "n_target": int(len(scores))}, indent=1), encoding="utf-8")
    print(f"{len(scores)} target fields scored by {scores['model'].iloc[0]} trained on "
          f"{scores['n_train'].iloc[0]} fields; wrote {out / 'scores.parquet'}")
    return scores


if __name__ == "__main__":
    main()
