"""LIGHTLY ADAPTED evaluation: recalibrate only the decision threshold.

    python -m fieldvision.model.evaluate_adapted --model <run dir> --timeseries <...> \
        --fields <...> --out <dir> --n-calib 10 --seed 0

This is not the zero-shot number. It answers a different question: if a
user labelled a handful of fields in the new region, how much would moving
the threshold help?

 - `--n-calib` fields are drawn from the evaluation set with a fixed seed,
   stratified by label only (never by features or scores);
 - on those fields only the threshold of the primary model is chosen
   (maximum balanced accuracy, ties broken towards the original threshold);
   the model itself, the features and the low-evidence rule are unchanged;
 - metrics are reported on the remaining fields only, next to the zero-shot
   metrics on exactly the same remaining fields and the majority baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import fingerprint, ledger_check_and_record, ledger_path, write_json
from .evaluate import check_frozen, choose_groups, evaluate_frame, prepare, write_report, misclassified
from .metrics import metrics_from_counts
from .predict import add_input_args
from .report import fmt


def pick_calibration(labels, n_calib, seed):
    """Indices of calibration fields: stratified by label, fixed seed, at least one per class."""
    labels = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    idx = []
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("threshold recalibration needs both classes in the evaluation set")
    k_pos = int(np.clip(round(n_calib * n_pos / len(labels)), 1, n_pos - 1 if n_pos > 1 else 1))
    k_neg = int(np.clip(n_calib - k_pos, 1, n_neg - 1 if n_neg > 1 else 1))
    for val, k in ((1, k_pos), (0, k_neg)):
        pool = np.flatnonzero(labels == val)
        idx.extend(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())
    return np.sort(np.array(idx, dtype=int))


def fit_threshold(y, p, original):
    """Threshold with the highest balanced accuracy on (y, p); ties go to the one nearest `original`."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    u = np.unique(p)
    cands = np.r_[u[0] - 1e-9, (u[1:] + u[:-1]) / 2, u[-1] + 1e-9] if len(u) > 1 else np.r_[u, u + 1e-9]
    pred = p[None, :] >= cands[:, None]
    tp = (pred & (y == 1)).sum(1)
    fp = (pred & (y == 0)).sum(1)
    fn = ((~pred) & (y == 1)).sum(1)
    tn = ((~pred) & (y == 0)).sum(1)
    bal = metrics_from_counts(tp, fp, fn, tn)["balanced_accuracy"]
    best = np.nanmax(bal)
    ties = cands[np.isclose(bal, best)]
    return float(ties[np.argmin(np.abs(ties - original))]), float(best)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Threshold-recalibrated evaluation (NOT zero-shot).")
    add_input_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=10, help="fields used to move the threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--allow-repeat", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    st = prepare(args)
    preds, meta = st["preds"], st["meta"]
    prim = preds[preds["model"] == st["name"]].reset_index(drop=True)
    if args.n_calib >= len(prim) - 2:
        raise SystemExit(f"--n-calib {args.n_calib} leaves too few fields to report on "
                         f"({len(prim)} labelled)")
    fp = fingerprint(prim["field_id"])
    frozen = check_frozen(args.model, st["name"], meta)
    ledger = (ledger_path(out), f"external_adapted_n{args.n_calib}_s{args.seed}", fp,
              frozen["id"], out, args.allow_repeat)
    earlier = ledger_check_and_record(*ledger, record=False)
    cal_idx = pick_calibration(prim["y"], args.n_calib, args.seed)
    cal_ids = set(prim.loc[cal_idx, "field_id"])
    cal = prim[prim["field_id"].isin(cal_ids)]
    thr, cal_bal = fit_threshold(cal["y"], cal["p"], st["model"].threshold)

    rest = preds[~preds["field_id"].isin(cal_ids)].copy()
    adapted = rest[rest["model"] == st["name"]].copy()
    adapted["model"] = f"{st['name']} (threshold {thr:.3f}, adapted)"
    adapted["threshold"] = thr
    adapted["pred"] = (adapted["p"] >= thr).astype(int)
    frame = pd.concat([rest, adapted], ignore_index=True)
    groups_col, groups_note = choose_groups(st["fields"][st["fields"]["field_id"].isin(rest["field_id"])])
    results = {m: evaluate_frame(g, args.n_boot, args.seed, groups_col)
               for m, g in frame.groupby("model")}

    cal[["field_id", "year", "label", "p"]].to_csv(out / "calibration_fields.csv", index=False)
    miss = misclassified(adapted, st["X"], st["fields"])
    miss.to_csv(out / "misclassified.csv", index=False)
    metrics = {
        "mode": "threshold_recalibrated", "NOT_ZERO_SHOT": True, "frozen_model": frozen,
        "fit_on_evaluation_set": f"decision threshold only, on {len(cal)} fields",
        "run_id": meta["run_id"], "model": st["name"],
        "original_threshold": st["model"].threshold, "adapted_threshold": thr,
        "calibration_balanced_accuracy": cal_bal, "n_calibration": int(len(cal)),
        "calibration_seed": args.seed, "n_reported": int(adapted.shape[0]),
        "evaluation_set_fingerprint": fp, "earlier_evaluations_of_this_set": earlier,
        "n_fields": int(len(st["fields"])), "n_labelled": int(adapted.shape[0]),
        "n_unlabelled_ignored": st["n_unlabelled"],
        "bootstrap_unit": groups_col, "bootstrap_note": groups_note, "results": results,
        "response_time": {"features_seconds_per_1000": st["info"]["seconds"] / max(len(st["X"]), 1) * 1000},
        "low_evidence_reasons": st["X"]["low_evidence_reason"].replace("", "ok").value_counts().to_dict(),
    }
    write_json(out / "metrics.json", metrics)
    adapted_name = adapted["model"].iloc[0]
    extra = (f"Threshold moved from {fmt(st['model'].threshold)} to {fmt(thr)} using {len(cal)} "
             f"fields (seed {args.seed}; list in `calibration_fields.csv`). All numbers below are "
             f"on the other {adapted.shape[0]} fields. The row `{st['name']}` is the zero-shot "
             f"model on those same fields; `{adapted_name}` is the adapted one.")
    shown = dict(st)
    shown["name"] = adapted_name
    write_report(out, metrics, results, shown, miss, None,
                 title=f"External evaluation: LIGHTLY ADAPTED (threshold recalibrated on "
                       f"{len(cal)} fields) - NOT the zero-shot number",
                 intro="Only the decision threshold was refit, on a small labelled sample of this "
                       "set. Report it next to, never instead of, the zero-shot result.",
                 extra=extra, main_rows=["majority", st["name"], adapted_name])
    ledger_check_and_record(*ledger, record=True)
    print(f"adapted threshold {thr:.3f} (was {st['model'].threshold:.3f}) from {len(cal)} fields; "
          f"reported on {adapted.shape[0]} -> {out}")
    return metrics


if __name__ == "__main__":
    main()
