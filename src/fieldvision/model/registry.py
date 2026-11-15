"""Frozen model folders for the team's registry.

    python -m fieldvision.model.registry export --run <run dir> --which gbm --name ts_model_v1 \
        --training-files <files...> [--eval <eval dir>]    (--registry defaults to registry/)

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

# the repo's registry/ folder (src/fieldvision/model/registry.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPO_ROOT / "registry"

# --------------------------------------------------------------------------
# nothing written into the repo may carry a machine-specific path
# --------------------------------------------------------------------------
import re as _re

_ABS_PATH = _re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|/(?:home|Users|tmp|workspace|mnt)/)[^\s\"'`|,;)]*")


def portable_path(p):
    """An absolute path made safe to publish.

    .../src/fieldvision/<pkg>/<module>.py -> python -m fieldvision.<pkg>.<module>
    anything inside the repo                -> repo-relative
    anything under a 'data' folder          -> data/<rest>
    anything else                           -> <external>/<file name>
    """
    s = str(p).replace("\\", "/")
    m = _re.search(r"/src/fieldvision/([\w/]+)\.py$", s)
    if m:
        return "python -m fieldvision." + m.group(1).replace("/", ".")
    root = str(REPO_ROOT).replace("\\", "/").rstrip("/")
    if s.lower().startswith(root.lower() + "/"):
        return s[len(root) + 1:]
    i = s.lower().find("/data/")
    if i >= 0:
        return "data/" + s[i + len("/data/"):]
    return "<external>/" + s.rstrip("/").split("/")[-1]


def portable_text(text):
    return _ABS_PATH.sub(lambda m: portable_path(m.group(0)), str(text))


def portable(obj):
    """Apply portable_text to every string in a JSON-like object."""
    if isinstance(obj, str):
        return portable_text(obj)
    if isinstance(obj, dict):
        return {k: portable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [portable(v) for v in obj]
    return obj


def portable_command(argv):
    """`python -m <module> <args>` with every path argument made portable."""
    if not argv:
        return None
    head = portable_path(argv[0]) if str(argv[0]).endswith(".py") else str(argv[0])
    if not head.startswith("python -m"):
        head = "python " + head
    return " ".join([head] + [portable_text(a) for a in argv[1:]])


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
    mj = model_json(model, name, description)
    # class medians of the evidence features, so explanations work without the training run
    mj["evidence_reference"] = meta.get("evidence_reference")
    base = getattr(model, "base_", model)
    if type(base).__name__ == "RuleModel" and (run_dir / "features.parquet").exists():
        import pandas as pd
        tr = pd.read_parquet(run_dir / "features.parquet", columns=[base.timing, "split"])
        # the regional-threshold policy shifts the rule by (region median - this median)
        mj["training_median_timing"] = float(tr.loc[tr["split"] == "development", base.timing].median())
    write_json(folder / "model.json", portable(mj))
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
    write_json(folder / "provenance.json", portable(prov))
    (folder / "model_card.md").write_text(portable_text(model_card(
        name, which, meta, run_metrics, eval_metrics, description, note)), encoding="utf-8")
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
    e.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    e.add_argument("--training-files", nargs="+", required=True)
    e.add_argument("--git-commit", required=True)
    e.add_argument("--eval", default=None)
    e.add_argument("--description", default="")
    e.add_argument("--confidence-tier", default="medium")
    e.add_argument("--note", default="", help="statement printed in bold at the top of the card")
    c = sub.add_parser("composite")
    c.add_argument("--run", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    c.add_argument("--training-files", nargs="+", required=True)
    c.add_argument("--git-commit", required=True)
    c.add_argument("--family-runs", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "composite":
        folder = export_composite(a.run, a.name, a.registry, a.training_files, a.git_commit,
                                  a.family_runs)
        print(f"exported composite -> {folder}")
        return
    folder = export(a.run, a.which, a.name, a.registry, a.training_files, a.git_commit, a.eval,
                    a.description, a.confidence_tier, a.note)
    print(f"exported {a.which} -> {folder}")




COMPOSITE_MEMBERS = ("timing_rule", "rule", "logistic")


def composite_oof_metrics(run_dir, members=COMPOSITE_MEMBERS, spread_limit=0.5):
    """Held-out metrics of the member-average, rebuilt from a run's out-of-fold predictions.

    Each member was fit inside the training part of every split, so the average of their
    out-of-fold probabilities is an out-of-fold prediction of the composite.
    """
    from .metrics import classification_metrics, exact_intervals
    import pandas as pd
    cv = pd.read_parquet(Path(run_dir) / "cv_predictions.parquet")
    with open(Path(run_dir) / "run.json", encoding="utf-8") as fh:
        variant = json.load(fh)["variant"]
    out = {}
    for scheme, g in cv[(cv["variant"] == variant) & cv["model"].isin(members)].groupby("scheme"):
        if scheme == "holdout":
            continue
        P = g.pivot(index="field_id", columns="model", values="p")
        meta = g.drop_duplicates("field_id").set_index("field_id").loc[P.index]
        y = meta["y"].to_numpy(int)
        p = P.mean(axis=1).to_numpy()
        votes = (P.to_numpy() >= 0.5).sum(axis=1)
        spread = (P.max(axis=1) - P.min(axis=1)).to_numpy()
        abstain = meta["low_evidence"].to_numpy(bool) | ((votes > 0) & (votes < P.shape[1])
                                                         & (spread >= spread_limit))
        m = classification_metrics(y, (p >= 0.5).astype(int), p)
        mc = classification_metrics(y[~abstain], (p[~abstain] >= 0.5).astype(int), p[~abstain])
        mc["exact95"] = exact_intervals(mc["confusion"])
        m["exact95"] = exact_intervals(m["confusion"])
        out[scheme] = {"all": m, "confident": mc, "abstain_rate": float(abstain.mean())}
    return out


def export_composite(run_dir, name, registry_dir, training_files, git_commit, family_runs=(),
                     spread_limit=0.5, description=""):
    """composite_v1: the frozen members of `run_dir`, averaged; no refit."""
    import joblib
    import pickle
    from .models import CompositeModel
    run_dir, registry_dir = Path(run_dir), Path(registry_dir)
    final = registry_dir / name
    if final.exists():
        raise SystemExit(f"{final} exists; frozen folders are written once. Use a new version.")
    members = {}
    for m in COMPOSITE_MEMBERS:
        with open(run_dir / "models" / f"{m}.pkl", "rb") as fh:
            members[m] = pickle.load(fh)
    comp = CompositeModel.from_members(members, spread_limit=spread_limit)
    _, meta, cfg, _ = load_run(run_dir, "logistic")
    folder = final.with_name(final.name + ".partial")
    if folder.exists():
        import shutil
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    joblib.dump(comp, folder / "model.joblib")
    write_json(folder / "model.json", {
        "name": name, "class": "CompositeModel", "positive_class": "silage",
        "decision": "silage if the mean of the members' calibrated probabilities >= 0.5",
        "threshold": 0.5, "members": {k: model_json(v, k, "") for k, v in members.items()},
        "member_frozen_ids": {k: frozen_model_id(run_dir, k) for k in members},
        "abstain": f"low evidence, or members split their votes with probabilities at least "
                   f"{spread_limit} apart", "spread_limit": spread_limit,
        "why_mean_not_vote": "Austria 2024 spatial CV: ROC-AUC 0.766 for the mean vs 0.739 for "
                             "a majority vote"})
    write_json(folder / "feature_schema.json", {
        "computed_by": "fieldvision.features.build.build_features",
        "feature_config": cfg.to_dict(),
        "features": [{"name": c, "kind": feature_kind(c), "unit": feature_unit(c)}
                     for c in comp.feature_names],
        "gaps": "each member handles gaps its own way (rules: fixed score without a harvest; "
                "logistic: training median)"})
    write_json(folder / "applicability.json", {
        "model": name,
        "requires": {"sensors": ["Sentinel-2 L2A surface reflectance, bands B02-B12"],
                     "optical_clear_obs_in_harvest_window": 1,
                     "clear_obs_before_and_after_drop": True,
                     "min_usable_dates_jul_nov_for_confident": cfg.low_evidence_min_usable,
                     "max_harvest_bracket_days_for_confident": cfg.low_evidence_max_interval_days},
        "abstains_when": ["low evidence (see the members)", "members disagree strongly"],
        "confidence_tier": "medium",
        "validated_on": ["European data only; not scored on Beltsville before Evaluation 2"]})
    write_json(folder / "provenance.json", portable({
        "name": name, "exported": dt.datetime.now().isoformat(timespec="seconds"),
        "members_from_run": str(run_dir), "run_id": meta["run_id"],
        "git_commit_of_code": git_commit,
        "training_data": [{"path": str(p), "sha256": file_sha256(p)} for p in training_files],
        "note": "added after Evaluation 1 as a fifth family for Evaluation 2 (see ACCESS_LOG.md)"}))
    L = [f"# {name}\n", (description or "Average of the calibrated probabilities of three "
                         "low-capacity models trained on European registry labels: the timing "
                         "rule, the timing+moisture rule and the six-feature logistic.") + "\n",
         "**Not scored on the Beltsville evaluation set. It was added after Evaluation 1 and joins "
         "the Evaluation 2 candidates under the rule written in ACCESS_LOG.md.**\n",
         f"Members (frozen, from run `{meta['run_id']}`, trained on {meta['training']['n_fields']} "
         f"fields of {meta['training']['countries']} {meta['training']['years']}):\n"]
    L += [f"- {k}: {v.describe()}" for k, v in members.items()]
    L.append("\nAbstains when a field is low evidence or the members split their votes with "
             f"probabilities at least {spread_limit} apart.\n")
    for label, rd in [("this model's training run", run_dir)] + [("family, other run", r) for r in family_runs]:
        res = composite_oof_metrics(rd, spread_limit=spread_limit)
        L.append(f"## Held-out results of the member average ({label}: `{Path(rd).name}`)\n")
        L.append("Rebuilt from the members' out-of-fold predictions; exact 95 % intervals.\n")
        for scheme, r in res.items():
            L.append(f"- {scheme}: {_metrics_line(r['all'])}, accuracy {r['all']['accuracy']:.3f}; "
                     f"abstain rate {r['abstain_rate']:.2f}; not abstained: "
                     f"{_metrics_line(r['confident'])}")
        L.append("")
    L.append("## Known limits\n")
    L.append("- On Austria 2024 spatial CV the average is slightly below the logistic alone "
             "(ROC-AUC 0.766 vs 0.783); its case is robustness across places and years, which "
             "leave-one-country-out has to show.\n- The members' strong-disagreement flag rarely "
             "fires on Austrian data (about 0.4 % of fields); most abstentions come from low "
             "evidence.\n- Probabilities assume a 50/50 class balance.\n")
    (folder / "model_card.md").write_text(portable_text("\n".join(L)), encoding="utf-8")
    folder.rename(final)
    return final




# --------------------------------------------------------------------------
# error rates for adjusting counted totals (European cross-validation only)
# --------------------------------------------------------------------------

RATE_MEMBERS = {"composite": COMPOSITE_MEMBERS}


def rates_from_run(run_dir, model, scheme_prefix=("spatial", "loyo:", "loco:"), spread_limit=0.5,
                   max_bracket=None, alpha=0.10):
    """True/false positive counts on held-out fields that the model did not abstain on.

    Uses a run's out-of-fold predictions. For 'composite' in a run that predates it, the
    member average is rebuilt from the members' out-of-fold probabilities. Returns
    {scheme: {tp, fp, fn, tn, tpr, fpr, n}}.
    """
    import pandas as pd
    cv = pd.read_parquet(Path(run_dir) / "cv_predictions.parquet")
    variant = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))["variant"]
    cv = cv[cv["variant"] == variant]
    feats = None
    if max_bracket is not None:
        feats = pd.read_parquet(Path(run_dir) / "features.parquet",
                                columns=["field_id", "q_harv_width_days"]).set_index("field_id")
    out = {}
    for scheme, g in cv.groupby("scheme"):
        if not scheme.startswith(scheme_prefix):
            continue
        if model in set(g["model"]):
            m = g[g["model"] == model].set_index("field_id")
            p, y, ab = m["p"].to_numpy(), m["y"].to_numpy(int), m["low_evidence"].to_numpy(bool)
            ids = m.index
            if model == "composite":
                P = g[g["model"].isin(COMPOSITE_MEMBERS)].pivot(index="field_id", columns="model",
                                                                values="p").loc[m.index]
                votes = (P.to_numpy() >= 0.5).sum(axis=1)
                spread = (P.max(axis=1) - P.min(axis=1)).to_numpy()
                ab = ab | ((votes > 0) & (votes < P.shape[1]) & (spread >= spread_limit))
        elif model == "composite" and set(COMPOSITE_MEMBERS) <= set(g["model"]):
            P = g[g["model"].isin(COMPOSITE_MEMBERS)].pivot(index="field_id", columns="model", values="p")
            meta = g.drop_duplicates("field_id").set_index("field_id").loc[P.index]
            p = P.mean(axis=1).to_numpy()
            y = meta["y"].to_numpy(int)
            ids = P.index
            votes = (P.to_numpy() >= 0.5).sum(axis=1)
            spread = (P.max(axis=1) - P.min(axis=1)).to_numpy()
            ab = meta["low_evidence"].to_numpy(bool) | ((votes > 0) & (votes < P.shape[1])
                                                        & (spread >= spread_limit))
        else:
            continue
        if max_bracket is not None:      # the clear evidence tier only
            br = feats["q_harv_width_days"].reindex(ids).to_numpy(float)
            ab = ab | ~(br <= max_bracket)
        keep = ~ab
        pred = p[keep] >= 0.5
        yy = y[keep]
        tp, fp = int((pred & (yy == 1)).sum()), int((pred & (yy == 0)).sum())
        fn, tn = int((~pred & (yy == 1)).sum()), int((~pred & (yy == 0)).sum())
        out[scheme] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(keep.sum()),
                       "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1)}
        pk, yk = p[keep], yy
        if (yk == 0).any() and (yk == 1).any():
            # recall at the score that flags `alpha` of grain (for the local-threshold step)
            t_s = float(np.quantile(pk[yk == 0], 1 - alpha))
            out[scheme]["matched_tpr_alpha_0.10"] = float((pk[yk == 1] >= t_s).mean())
            out[scheme]["n_silage"] = int((yk == 1).sum())
    return out


def local_rates(run_dir, model, alpha=0.10, clear_max=12.0, wide_max=20.0):
    """Matched recall for the local-threshold county step, per evidence tier.

    From a run's spatial out-of-fold scores of non-abstained fields, measured inside each
    source domain (country-year) and averaged by scored fields
    (fieldvision.anomaly.local.matched_tpr). Tiers: clear (bracket <= clear_max), wide
    (clear_max < bracket <= wide_max) and both together."""
    import pandas as pd
    from ..anomaly.local import matched_tpr
    cv = pd.read_parquet(Path(run_dir) / "cv_predictions.parquet")
    variant = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))["variant"]
    m = cv[(cv["variant"] == variant) & (cv["scheme"] == "spatial")
           & (cv["model"] == model)].set_index("field_id")
    br = pd.read_parquet(Path(run_dir) / "features.parquet", columns=["field_id", "q_harv_width_days"]
                         ).set_index("field_id")["q_harv_width_days"].reindex(m.index).to_numpy(float)
    ok = ~m["low_evidence"].to_numpy(bool)
    groups = (m["country"].astype(str) + "-" + m["year"].astype(str)).to_numpy()
    tiers = {"clear": ok & (br <= clear_max), "wide": ok & (br > clear_max) & (br <= wide_max),
             "clear+wide": ok & (br <= wide_max)}
    out = {"alpha": alpha, "source": "spatial out-of-fold, non-abstained, per country-year",
           "domains": sorted(set(groups))}
    for name, keep in tiers.items():
        _, tpr = matched_tpr(m["p"].to_numpy()[keep], m["y"].to_numpy(int)[keep], alpha,
                             groups[keep])
        out[name] = {"matched_tpr": tpr, "n": int(keep.sum())}
    return out


def write_rates(folder, model, runs, primary, max_bracket=None, primary_clear=None, local_run=None):
    """rates.json beside a frozen model: European held-out rates, used by the toolbox to
    correct counted silage area. runs: {label: (run_dir, model name in that run)}."""
    sets = {}
    for label, (run_dir, name) in runs.items():
        for scheme, r in rates_from_run(run_dir, name).items():
            sets[f"{label}:{scheme}"] = r
        if max_bracket is not None:
            for scheme, r in rates_from_run(run_dir, name, max_bracket=max_bracket).items():
                sets[f"clear:{label}:{scheme}"] = r
    if primary not in sets:
        raise ValueError(f"primary rate set {primary!r} not among {sorted(sets)}")
    doc = {"note": "true/false positive rates of this model family on held-out European fields it "
                   "did not abstain on (no US labels); added after export, not read by the model",
           "primary": primary, "sets": sets}
    if primary_clear:
        doc["primary_clear"] = primary_clear
        doc["clear_tier_max_bracket_days"] = max_bracket
    if local_run is not None:          # (run_dir, model name in that run)
        doc["local"] = local_rates(local_run[0], local_run[1], clear_max=max_bracket or 12.0)
    write_json(Path(folder) / "rates.json", portable(doc))
    return doc


if __name__ == "__main__":
    main()
