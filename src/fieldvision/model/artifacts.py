"""Saving and loading a trained run, the per-field predictions table, and the
single-use ledger for the holdout and external evaluation sets."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pickle
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.season import FeatureConfig

RUN_META = "run.json"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return o


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2, ensure_ascii=False)


FORBIDDEN_SPLITS = {"external_eval"}
FORBIDDEN_COUNTRIES = {"US"}


def refuse_evaluation_fields(fields):
    """Stop if any field is reserved for external evaluation.

    US fields (the Beltsville set and anything else American) and any field
    marked split=external_eval must never be trained, cross-validated or
    tuned on; they are scored only by `evaluate`, once.
    """
    split = fields["split"].astype(str).str.strip().str.lower() if "split" in fields else None
    bad = fields["country"].astype(str).str.upper().isin(FORBIDDEN_COUNTRIES)
    if split is not None:
        bad |= split.isin(FORBIDDEN_SPLITS)
    if bad.any():
        files = sorted(fields.loc[bad, "fields_file"].astype(str).unique()) \
            if "fields_file" in fields else []
        raise SystemExit(
            f"refusing to train on {int(bad.sum())} fields reserved for external evaluation "
            f"(country US or split=external_eval), from {files}. Evaluate them with "
            "`python -m fieldvision.model.evaluate` after the model is frozen.")


def frozen_model_id(run_dir, name):
    """Hash of the saved model file and the training configuration it came from."""
    run_dir = Path(run_dir)
    h = hashlib.sha256()
    h.update((run_dir / "models" / f"{name}.pkl").read_bytes())
    with open(run_dir / RUN_META, encoding="utf-8") as fh:
        meta = json.load(fh)
    cfg = {k: meta.get(k) for k in ("feature_config", "variant", "threshold", "feature_names",
                                    "training", "holdout_patterns")}
    h.update(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


def save_run(run_dir, models, meta):
    """models: name -> fitted model. meta: dict with at least primary, variant, feature_config.

    Also records a frozen identifier per model (see frozen_model_id).
    """
    run_dir = Path(run_dir)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        with open(run_dir / "models" / f"{name}.pkl", "wb") as fh:
            pickle.dump(model, fh)
    write_json(run_dir / RUN_META, meta)
    meta = dict(meta)
    meta["frozen_model_ids"] = {name: frozen_model_id(run_dir, name) for name in models}
    write_json(run_dir / RUN_META, meta)


def load_run(run_dir, which=None):
    """(model, meta, feature_config, majority_model) for a trained run.

    which: model name; default the run's primary model.
    """
    run_dir = Path(run_dir)
    with open(run_dir / RUN_META, encoding="utf-8") as fh:
        meta = json.load(fh)
    name = which or meta["primary"]
    path = run_dir / "models" / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"no model {name!r} in {run_dir} (have "
                                f"{sorted(p.stem for p in (run_dir / 'models').glob('*.pkl'))})")
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    majority = None
    if (run_dir / "models" / "majority.pkl").exists():
        with open(run_dir / "models" / "majority.pkl", "rb") as fh:
            majority = pickle.load(fh)
    known = {f.name for f in dc_fields(FeatureConfig)}
    raw = {k: (tuple(v) if isinstance(v, list) else v)
           for k, v in meta.get("feature_config", {}).items() if k in known}
    return model, meta, FeatureConfig(**raw), majority


# --------------------------------------------------------------------------
# per-field predictions (what the demo reads)
# --------------------------------------------------------------------------

PREDICTION_COLUMNS = [
    "field_id", "p_silage", "predicted_label", "low_evidence", "low_evidence_reason",
    "harvest_start", "harvest_end", "harvest_interval_days", "pre_harvest_ndvi",
    "pre_harvest_ndmi", "n_usable_dates", "max_gap_days",
]


def _doy_to_date(year, doy):
    year = pd.to_numeric(pd.Series(year), errors="coerce").to_numpy(dtype=float)
    doy = np.asarray(doy, dtype=float)
    out = pd.Series(pd.NaT, index=range(len(doy)), dtype="datetime64[ns]")
    ok = np.isfinite(year) & np.isfinite(doy)
    if ok.any():
        jan1 = pd.to_datetime(year[ok].astype(int).astype(str) + "-01-01")
        out[ok] = jan1 + pd.to_timedelta(doy[ok] - 1, unit="D")
    return out.to_numpy()


def predictions_frame(features, fields, p, threshold, model_name, keep_columns=(), model=None,
                      reference=None, prevalence=None, train_prevalence=None):
    """One row per field for the demo and for county aggregation.

    harvest_start / harvest_end bracket the harvest: the last usable
    observation still standing and the first one clearly harvested. The
    true date is somewhere in between.

    p_silage is calibrated for the class share of the training data
    (balanced). With `prevalence` (e.g. 0.07 for the US average) the table
    also gets p_silage_at_prevalence, the same probability moved to that
    share by Bayes' rule, and the label it implies at 0.5.
    With `model` and `reference` (from run.json) it gets the readable
    evidence and a reason sentence per field (see explain.py).
    """
    out = _base_predictions(features, fields, p, threshold, model_name, keep_columns)
    if prevalence is not None and train_prevalence:
        from .metrics import prior_shift
        ps = prior_shift(p, prevalence, train_prevalence)
        out["p_silage_at_prevalence"] = ps
        out["predicted_label_at_prevalence"] = np.where(ps >= 0.5, "silage", "grain")
        out["assumed_prevalence"] = float(prevalence)
    if model is not None and reference:
        from .explain import explanation_columns
        ex = explanation_columns(out, features, model, reference)
        out = pd.concat([out.reset_index(drop=True), ex], axis=1)
    return out


def _base_predictions(features, fields, p, threshold, model_name, keep_columns=()):
    f = fields.set_index("field_id").reindex(features["field_id"])
    p = np.asarray(p, dtype=float)
    reason = features["low_evidence_reason"].astype(str).to_numpy().copy()
    no_ts = features["q_n_rows"].to_numpy(dtype=float) == 0
    reason[no_ts] = "no_timeseries"
    out = pd.DataFrame({
        "field_id": features["field_id"].to_numpy(),
        "p_silage": p,
        "predicted_label": np.where(p >= threshold, "silage", "grain"),
        "low_evidence": features["low_evidence"].to_numpy(dtype=bool),
        "low_evidence_reason": reason,
        "harvest_start": _doy_to_date(f["year"].to_numpy(), features["harv_lo_doy"]),
        "harvest_end": _doy_to_date(f["year"].to_numpy(), features["harv_hi_doy"]),
        "harvest_interval_days": features["q_harv_width_days"].to_numpy(dtype=float),
        "pre_harvest_ndvi": features["pre_ndvi"].to_numpy(dtype=float),
        "pre_harvest_ndmi": features["pre_ndmi"].to_numpy(dtype=float),
        "n_usable_dates": features["q_n_usable_season"].to_numpy(dtype=float).astype(int),
        "max_gap_days": features["q_max_gap_days"].to_numpy(dtype=float),
    })
    for c in ("year", "country", "block_id", "area_ha", "centroid_lon", "centroid_lat", "label"):
        if c in f.columns:
            out[c] = f[c].to_numpy()
    for c in keep_columns:
        if c in f.columns and c not in out.columns:
            out[c] = f[c].to_numpy()
    out["model"] = model_name
    out["threshold"] = float(threshold)
    return out


# --------------------------------------------------------------------------
# single-use ledger
# --------------------------------------------------------------------------

def fingerprint(field_ids):
    h = hashlib.sha256()
    for fid in sorted(map(str, field_ids)):
        h.update(fid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def ledger_path(out_dir):
    env = os.environ.get("FIELDVISION_LEDGER")
    if env:
        return Path(env)
    return Path(out_dir).resolve().parent / "single_use_ledger.jsonl"


def ledger_check_and_record(path, kind, set_fingerprint, model_id, out_dir, allow_repeat=False,
                            record=True):
    """Refuse a second use of an evaluation set unless allowed, and record this one.

    kind is e.g. 'holdout', 'external_zero_shot', 'external_adapted_...'.
    Call once with record=False before computing anything and once with
    record=True after the results are written, so a crash does not burn the set.
    Returns the list of earlier uses of the same set (any model).
    """
    path = Path(path)
    earlier = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == kind and rec.get("set") == set_fingerprint:
                    earlier.append(rec)
    same_model = [r for r in earlier if r.get("model") == model_id]
    if same_model and not allow_repeat:
        raise RuntimeError(
            f"this {kind} set ({set_fingerprint}) was already evaluated with this model on "
            f"{same_model[0].get('time')} -> {same_model[0].get('out')}. It is single-use. "
            "Pass --allow-repeat only if you know why, and say so in the report.")
    if not record:
        return earlier
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"),
                             "kind": kind, "set": set_fingerprint, "model": model_id,
                             "out": str(out_dir), "repeat": bool(same_model)}) + "\n")
    return earlier
