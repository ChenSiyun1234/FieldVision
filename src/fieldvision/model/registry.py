"""Frozen model folders for the team's registry.

    python -m fieldvision.model.registry export --run <run dir> --which gbm --name ts_model_v1 \
        --registry C:\\t\\fv\\data\\registry --training-files <files...> [--eval <eval dir>]

A frozen model is a folder, not a weights file:

    <registry>/<name>/
      model.json            what the model is; thresholds in full for rules
      model.joblib          the fitted object (ML models; also rules, for exactness)
      feature_schema.json   feature names in order, units, how gaps are handled, feature settings
      applicability.json    what data the model needs, when it abstains, what it was validated on
      model_card.md         metrics, confusion matrix, calibration, known limits
      provenance.json       training data hashes, git commit, dates, run and frozen ids

Folders are written once. Exporting to an existing folder is refused; make a new version.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

from ..features.season import FeatureConfig, feature_kind
from .artifacts import frozen_model_id, load_run, write_json


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def feature_unit(name):
    n = name.lower()
    if n.endswith("_doy"):
        return "day of year"
    if "days" in n:
        return "days"
    if n.startswith("q_n_") or n.endswith("_flag") or n.endswith("_above_summer"):
        return "count or 0/1 flag"
    if any(n.endswith(s) for s in ("_over_peak", "_frac", "_ratio", "_rel", "_over_dndvi")):
        return "ratio (field relative to itself)"
    if "slope" in n:
        return "index units per day"
    if n.startswith("gi_"):
        return "index x days (integral)"
    if any(f"_{b.lower()}" in n for b in ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                                          "B11", "B12")) or "bright" in n:
        return "surface reflectance 0-1"
    return "normalised index, unitless (-1..1)"


def _gap_rule(model):
    base = getattr(model, "base_", model)
    kind = type(base).__name__
    if kind == "GBMModel":
        return ("missing values stay NaN; LightGBM sends them down the branch learned for missing "
                "values in training")
    if kind == "LogisticModel":
        return "missing values replaced by the training median, then standardised"
    if kind == "RuleModel":
        return ("a field with no detected harvest gets a fixed score learned in training; a field "
                "with timing but no moisture value is judged on timing alone")
    return "no features used"


def _model_kind(model):
    base = getattr(model, "base_", model)
    return type(base).__name__


def model_json(model, name, description):
    base = getattr(model, "base_", model)
    out = {"name": name, "description": description, "class": _model_kind(model),
           "positive_class": "silage", "decision": "silage if p_silage >= threshold",
           "threshold": float(model.threshold), "features": list(model.feature_names),
           "probability": "Platt-calibrated for a 50/50 training class balance"
           if hasattr(model, "platt_") else "rule score (not a calibrated probability)"}
    if hasattr(model, "platt_") and model.platt_ is not None:
        out["platt"] = {"a": float(model.platt_.coef_[0][0]), "b": float(model.platt_.intercept_[0])}
    if type(base).__name__ == "RuleModel":
        out["rule"] = base.to_dict()
        out["rule_text"] = base.describe()
        if hasattr(model, "platt_") and model.platt_ is not None and base.mode_ != "constant":
            # where the calibrated score crosses the threshold: logit(raw) = -b/a, raw = sigmoid(2*margin)
            a, b = out["platt"]["a"], out["platt"]["b"]
            margin = -b / a / 2.0
            out["effective_rule"] = (f"silage if {base.timing} <= "
                                     f"{base.t_time_ - margin * base.scale_time_:.1f}"
                                     + ("" if base.mode_ == "timing_only" else
                                        f" ({base.mode_.upper()} {base.moisture} test as in rule)"))
    return out


def _metrics_line(m):
    def f(k):
        v = m.get(k)
        return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"
    ci = m.get("exact95") or m.get("ci95") or {}
    def c(k):
        x = ci.get(k)
        return f" [{x[0]:.2f}, {x[1]:.2f}]" if x and None not in x else ""
    return (f"silage precision {f('precision')}{c('precision')}, silage recall {f('recall')}{c('recall')}, "
            f"grain precision {f('grain_precision')}{c('grain_precision')}, grain recall "
            f"{f('grain_recall')}{c('grain_recall')}, FPR {f('fpr')}{c('fpr')}, ROC-AUC {f('roc_auc')}"
            f"{c('roc_auc')}, n {m['n']} ({m['n_silage']} silage)")


def _confusion(m):
    c = m["confusion"]
    return ("| declared \\ predicted | silage | grain |\n|---|---|---|\n"
            f"| silage | {c['tp']} | {c['fn']} |\n| grain | {c['fp']} | {c['tn']} |")


def model_card(name, which, meta, run_metrics, eval_metrics, description, note=""):
    v = meta["variant"]
    L = [f"# {name}\n", description + "\n"]
    if note:
        L.append(f"**{note}**\n")
    t = meta["training"]
    L.append(f"Trained on European parcel registry labels only: {t['n_fields']} fields "
             f"({t['n_silage']} silage), countries {t['countries']}, years {t['years']}, imagery "
             f"collections {t['collections']}. Training run `{meta['run_id']}`, created "
             f"{meta['created']}. No US data was used for training, tuning or feature design.\n")
    L.append(f"Model: {meta['models'][which]}\n")
    sp = run_metrics["summary"].get("spatial", {}).get(v, {})
    if which in sp:
        L.append("## European spatial cross-validation (5 folds grouped by 10 km block)\n")
        for subset, label in (("all", "All fields"), ("confident", "Confident subset")):
            m = sp[which][subset]
            extra = f", coverage {m['coverage']:.2f}" if "coverage" in m else ""
            L.append(f"- {label}: {_metrics_line(m)}{extra}")
        L.append("\nConfusion (all fields):\n")
        L.append(_confusion(sp[which]["all"]))
        base = sp.get("always_grain", {}).get("all")
        if base:
            L.append(f"\nAlways-grain baseline on the same fields: accuracy {base['accuracy']:.3f}.\n")
        cal = (run_metrics.get("calibration") or {}).get(which)
        if cal:
            L.append(f"Calibration on held-out folds: Brier {cal['brier']:.3f}, ECE {cal['ece']:.3f}.\n")
        prev = [r for r in run_metrics.get("prevalence_table", []) if r["model"] == which
                and abs(r["prevalence"] - 0.07) < 1e-9]
        def _f(v):
            return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"
        for r in prev:
            L.append(f"At a 7 % silage share ({r['operating_point']}): precision {_f(r['precision'])}, "
                     f"recall {_f(r['recall'])}.")
        L.append("")
    if eval_metrics and which in eval_metrics.get("results", {}):
        r = eval_metrics["results"][which]
        L.append("## Zero-shot on the organisers' Beltsville set (Evaluation 1, run once)\n")
        L.append(f"Frozen model scored on {eval_metrics['n_labelled']} labelled field-years without "
                 "any fitting on them; exact (Clopper-Pearson) 95 % intervals.\n")
        L.append(f"- All: {_metrics_line(r['all'])}, accuracy {r['all']['accuracy']:.3f}")
        L.append(f"- Confident subset: {_metrics_line(r['confident'])}, coverage "
                 f"{r['confident']['coverage']:.2f}")
        for yr, m in r.get("per_year", {}).items():
            L.append(f"- {yr}: {_metrics_line(m)}")
        L.append("\n" + _confusion(r["all"]))
        ag = eval_metrics["results"].get("always_grain", {}).get("all")
        if ag:
            L.append(f"\nAlways grain on the same field-years: accuracy {ag['accuracy']:.3f}.\n")
    L.append("## Known limits\n")
    L.append("- Trained on one or a few European regions; the US harvest calendar and management "
             "differ. Leave-one-country-out results, when available, are the best estimate of "
             "transfer.\n- Needs a usable Sentinel-2 observation before and after the harvest drop; "
             "cloud gaps widen the harvest bracket and push fields into abstention.\n"
             "- Probabilities assume a 50/50 class balance; US corn is about 7 % silage, so silage "
             "precision in practice is far lower than the balanced numbers (see the 7 % line).\n"
             "- European labels are farmer declarations; some are wrong.\n")
    return "\n".join(L)


def export(run_dir, which, name, registry_dir, training_files, git_commit, eval_dir=None,
           description="", confidence_tier="medium", note=""):
    run_dir, registry_dir = Path(run_dir), Path(registry_dir)
    folder = registry_dir / name
    if folder.exists():
        raise SystemExit(f"{folder} exists; frozen folders are written once. Use a new version.")
    model, meta, cfg, _ = load_run(run_dir, which)
    fid = frozen_model_id(run_dir, which)
    with open(run_dir / "metrics.json", encoding="utf-8") as fh:
        run_metrics = json.load(fh)
    eval_metrics = None
    if eval_dir:
        with open(Path(eval_dir) / "metrics.json", encoding="utf-8") as fh:
            eval_metrics = json.load(fh)
    # write into a temporary folder and rename at the end, so a failure never leaves a
    # half-written "frozen" folder behind
    final = folder
    folder = final.with_name(final.name + ".partial")
    if folder.exists():
        import shutil
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    import joblib
    joblib.dump(model, folder / "model.joblib")
    write_json(folder / "model.json", model_json(model, name, description))
    write_json(folder / "feature_schema.json", {
        "computed_by": "fieldvision.features.build.build_features (time series -> one row per field)",
        "input_contract": "long Sentinel-2 table: field_id, date, clear_fraction, B02..B12 "
                          "reflectance 0-1 (NaN = looked, nothing clear)",
        "feature_config": cfg.to_dict(),
        "features": [{"name": c, "kind": feature_kind(c), "unit": feature_unit(c)}
                     for c in model.feature_names],
        "gaps": _gap_rule(model),
    })
    write_json(folder / "applicability.json", {
        "model": name,
        "requires": {"sensors": ["Sentinel-2 L2A surface reflectance, bands B02-B12"],
                     "optical_clear_obs_in_harvest_window": 1,
                     "clear_obs_before_and_after_drop": True,
                     "min_usable_dates_jul_nov_for_confident": cfg.low_evidence_min_usable,
                     "max_harvest_bracket_days_for_confident": cfg.low_evidence_max_interval_days,
                     "corn_peak_window_doy": list(cfg.peak_window)},
        "abstains_when": ["fewer usable dates than required", "no harvest drop found",
                          "harvest bracket wider than the limit", "drop seen only once"],
        "confidence_tier": confidence_tier,
        "validated_on": ["European spatial CV: " + ", ".join(
            f"{c}" for c in meta["training"]["countries"]) + " " + ", ".join(meta["training"]["years"])]
        + (["Beltsville 2021-2023 zero-shot, Evaluation 1"] if eval_metrics else []),
    })
    prov = {
        "name": name, "exported": dt.datetime.now().isoformat(timespec="seconds"),
        "training_run": str(run_dir), "run_id": meta["run_id"], "trained": meta["created"],
        "frozen_model_id": fid, "git_commit_of_training_code": git_commit,
        "training_data": [{"path": str(p), "sha256": file_sha256(p)} for p in training_files],
        "training_command": meta.get("command"),
    }
    if eval_metrics:
        prov["evaluated"] = {"dir": str(eval_dir), "frozen_model_id": eval_metrics["frozen_model"]["id"]}
    write_json(folder / "provenance.json", prov)
    (folder / "model_card.md").write_text(model_card(name, which, meta, run_metrics, eval_metrics,
                                                     description, note), encoding="utf-8")
    folder.rename(final)
    return final


def load(registry_dir, name):
    """(model, model.json, feature config) from a registry folder. Read-only."""
    folder = Path(registry_dir) / name
    with open(folder / "model.json", encoding="utf-8") as fh:
        mj = json.load(fh)
    with open(folder / "feature_schema.json", encoding="utf-8") as fh:
        fs = json.load(fh)
    known = {f.name for f in dc_fields(FeatureConfig)}
    cfg = FeatureConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                           for k, v in fs["feature_config"].items() if k in known})
    if (folder / "model.joblib").exists():
        import joblib
        model = joblib.load(folder / "model.joblib")
    elif "rule" in mj:
        from .models import RuleModel
        model = RuleModel.from_dict(mj["rule"])
    else:
        raise FileNotFoundError(f"{folder} has neither model.joblib nor rule thresholds")
    return model, mj, cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export a trained model to the registry.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--which", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--registry", required=True)
    e.add_argument("--training-files", nargs="+", required=True)
    e.add_argument("--git-commit", required=True)
    e.add_argument("--eval", default=None)
    e.add_argument("--description", default="")
    e.add_argument("--confidence-tier", default="medium")
    e.add_argument("--note", default="", help="statement printed in bold at the top of the card")
    a = ap.parse_args(argv)
    folder = export(a.run, a.which, a.name, a.registry, a.training_files, a.git_commit, a.eval,
                    a.description, a.confidence_tier, a.note)
    print(f"exported {a.which} -> {folder}")


if __name__ == "__main__":
    main()
