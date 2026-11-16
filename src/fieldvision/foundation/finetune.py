"""One light fine-tuning run of Presto, evaluated on the comparison's folds.

    python -m fieldvision.foundation.finetune --run <compare output dir> --timeseries ... --fields ...

For each spatial fold of an earlier `compare` run (same fields, same test
blocks, read back from its predictions.parquet), a fresh copy of the
pretrained encoder has its last transformer block and final LayerNorm
unfrozen, a linear head is added, and both are trained on the training
fields of that fold only, for a fixed number of epochs chosen before
looking at any result (no early stopping on the test fold). The test fold
is then scored once. Output rows have the same columns as compare's
predictions, with model "ft-presto", so the same summaries apply.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig
from ..model.harness import META_COLS, compare_models, summarize_predictions
from .adapter import build_presto_inputs
from .compare import fold_table, markdown_table, paired_fold_diffs, verdict
from .embed import DEFAULT_WEIGHTS


def train_fold(batch, tr, te, y, weights, epochs=15, seed=0, lr_head=1e-3, lr_enc=1e-4,
               batch_size=64):
    import torch
    from torch import nn

    from .presto_encoder import load_encoder
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = load_encoder(weights)
    for p in enc.parameters():
        p.requires_grad_(False)
    trainable = list(enc.blocks[-1].parameters()) + list(enc.norm.parameters())
    for p in trainable:
        p.requires_grad_(True)
    head = nn.Linear(enc.embedding_size, 1)
    opt = torch.optim.AdamW([{"params": trainable, "lr": lr_enc},
                             {"params": head.parameters(), "lr": lr_head}], weight_decay=1e-4)
    t = {k: torch.from_numpy(getattr(batch, k)) for k in ("x", "mask", "dynamic_world", "latlons")}
    yt = torch.from_numpy(y.astype(np.float32))
    pos = float(y[tr].mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))

    def forward(idx):
        idx = torch.from_numpy(idx)
        e = enc(t["x"][idx], t["dynamic_world"][idx], t["latlons"][idx], mask=t["mask"][idx],
                month=int(batch.month))
        return head(e).squeeze(-1)

    losses = []
    for _ in range(epochs):
        enc.train()
        order = rng.permutation(tr)
        tot = 0.0
        for a in range(0, len(order), batch_size):
            idx = order[a:a + batch_size]
            opt.zero_grad()
            loss = loss_fn(forward(idx), yt[torch.from_numpy(idx)])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        losses.append(tot / len(tr))
    enc.eval()
    with torch.no_grad():
        p = torch.sigmoid(forward(np.asarray(te))).numpy()
    return p, losses


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="output dir of fieldvision.foundation.compare")
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    run = Path(args.run)
    prev = pd.read_parquet(run / "predictions.parquet")
    sp = prev[prev["scheme"] == "spatial"]
    ref = sp[sp["model"] == "phys-gbm"][META_COLS + ["fold", "y"]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = load_fields(args.fields)
        ts, _ = load_timeseries(args.timeseries, field_ids=ref["field_id"])
    fields = fields.set_index("field_id").loc[ref["field_id"]].reset_index()
    batch = build_presto_inputs(ts, fields, cfg=FeatureConfig())
    y = ref["y"].to_numpy(int)
    folds = ref["fold"].to_numpy()

    rows, log = [], {}
    for fold in sorted(np.unique(folds)):
        te = np.flatnonzero(folds == fold)
        tr = np.flatnonzero(folds != fold)
        t0 = time.perf_counter()
        p, losses = train_fold(batch, tr, te, y, args.weights, args.epochs, args.seed)
        log[str(fold)] = {"train_loss": losses, "seconds": time.perf_counter() - t0,
                          "n_train": int(len(tr)), "n_test": int(len(te))}
        part = ref.iloc[te].copy()
        part["scheme"], part["variant"], part["model"] = "spatial", "relative", "ft-presto"
        part["p"], part["threshold"] = p, 0.5
        part["pred"] = (p >= 0.5).astype(int)
        rows.append(part)
        print(f"{fold}: {len(tr)} train / {len(te)} test, loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
              f"{log[str(fold)]['seconds']:.0f}s", flush=True)
    ft = pd.concat(rows, ignore_index=True)
    ft.to_parquet(run / "predictions_finetune.parquet", index=False)
    allp = pd.concat([sp, ft], ignore_index=True)
    summary = summarize_predictions(allp, n_boot=args.n_boot, seed=args.seed)
    tab = fold_table(summary, "spatial")
    keep = ["ft-presto", "emb-logistic", "emb-gbm", "raw-gbm", "phys-gbm", "both-gbm", "physraw-gbm"]
    tab = tab[tab["model"].isin(keep)].set_index("model").loc[[m for m in keep if m in set(tab["model"])]].reset_index()
    comps = {}
    for b in ("emb-logistic", "raw-gbm", "phys-gbm"):
        boot = compare_models(allp, "ft-presto", b, "spatial", "relative", n_boot=args.n_boot,
                              seed=args.seed)["roc_auc"]
        diffs = paired_fold_diffs(allp, "ft-presto", b)
        comps[f"ft-presto vs {b}"] = {"auc_diff_pooled": boot["diff"], "auc_diff_ci95": boot["ci"],
                                      "auc_diff_per_fold": diffs, "verdict": verdict(diffs, boot)}
    (run / "finetune_summary.json").write_text(json.dumps(
        {"args": vars(args), "folds": log, "table": tab.to_dict(orient="records"),
         "comparisons": comps}, indent=1, default=str), encoding="utf-8")
    md = ["# Light fine-tuning (last block + LayerNorm + linear head), same folds", "",
          markdown_table(tab), ""]
    for k, c in comps.items():
        md.append(f"- {k}: pooled {c['auc_diff_pooled']:+.3f}, CI [{c['auc_diff_ci95'][0]:+.3f}, "
                  f"{c['auc_diff_ci95'][1]:+.3f}], per fold "
                  + ", ".join(f"{v:+.3f}" for v in c["auc_diff_per_fold"].values()) + f" -> {c['verdict']}")
    (run / "FINETUNE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
