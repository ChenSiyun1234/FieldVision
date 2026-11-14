"""In-site models: fitted on the Beltsville field-years themselves, validated on them.

    python -m fieldvision.beltsville.in_site --timeseries <barc parquet x3> \
        --fields <barc geojson x3> --austria-features <at2024 features.parquet> --out <dir>

Four low-capacity candidates (with 8 silage field-years per fold, nothing with more capacity):

  logistic_barc     six-feature logistic fitted on the Beltsville training folds only
  rule_barc         timing+moisture rule fitted on the Beltsville training folds only
  eu_logistic_adj   the European logistic (registry eu_logistic_v1) with only its intercept
                    and decision threshold refitted on the training folds
  logistic_at_barc  six-feature logistic on Austria 2024 plus the training folds, Beltsville
                    rows up-weighted so both sources carry the same total weight

plus the reference `timing_rule_v1` procedure. Two validation designs:

  leave-one-year-out    train on two seasons, test on the third. The same polygons appear in
                        training in other seasons, so this is "a new season on known fields".
  leave-one-field-out   all seasons of one polygon held out together: "a new field".

Nothing here feeds the European models or the second evaluation.
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig
from ..model.artifacts import write_json
from ..model.metrics import classification_metrics, exact_intervals
from ..model.models import LOGISTIC_FEATURES, LogisticModel, RuleModel, _Base, _matrix
from ..model.registry import load as load_registry

FEATURES = LOGISTIC_FEATURES["relative"]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _balanced_threshold(y, p, default=0.5):
    from ..model.evaluate_adapted import fit_threshold
    if len(np.unique(y)) < 2:
        return default
    return fit_threshold(y, p, default)[0]


class EUIntercept(_Base):
    """The frozen European logistic; refit only an intercept shift and the threshold."""
    name = "eu_logistic_adj"

    def __init__(self, eu_model):
        self.eu = eu_model
        self.feature_names = list(eu_model.feature_names)

    def fit(self, X, y):
        y = np.asarray(y, int)
        z = _logit(self.eu.predict_proba(X))

        def loss(b):
            p = _sigmoid(z + b)
            return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        self.shift_ = float(minimize_scalar(loss, bounds=(-10, 10), method="bounded").x)
        self.threshold = float(_balanced_threshold(y, self.predict_proba(X)))
        return self

    def predict_proba(self, X):
        return _sigmoid(_logit(self.eu.predict_proba(X)) + self.shift_)

    def describe(self):
        return (f"European logistic, logit shifted by {self.shift_:+.2f}, threshold "
                f"{self.threshold:.3f}")


class PooledLogistic(_Base):
    """Logistic on Austria + Beltsville training rows, both sources weighted equally."""
    name = "logistic_at_barc"

    def __init__(self, X_at, y_at, features=FEATURES):
        self.X_at, self.y_at = X_at, np.asarray(y_at, int)
        self.feature_names = list(features)

    def fit(self, X, y):
        y = np.asarray(y, int)
        A = np.vstack([_matrix(self.X_at, self.feature_names), _matrix(X, self.feature_names)])
        yy = np.r_[self.y_at, y]
        w = np.r_[np.ones(len(self.y_at)), np.full(len(y), len(self.y_at) / max(len(y), 1))]
        self.est_ = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler(),
                                  LogisticRegression(class_weight="balanced", max_iter=2000))
        self.est_.fit(A, yy, logisticregression__sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(_matrix(X, self.feature_names))[:, 1]

    def describe(self):
        coef = self.est_[-1].coef_[0]
        return "logit = " + " ".join(f"{c:+.2f}*z({n})" for n, c in zip(self.feature_names, coef))


def candidates(eu_model, X_at, y_at):
    return {
        "timing_rule_v1 procedure": partial(RuleModel, timing="harv_mid_doy", moisture="pre_ndvi",
                                            min_gain=0.01, name="timing_rule_v1"),
        "logistic_barc": partial(LogisticModel, FEATURES),
        "rule_barc": partial(RuleModel, timing="harv_days_since_peak", moisture="pre_ndmi",
                             min_gain=0.01, name="rule_barc"),
        "eu_logistic_adj": partial(EUIntercept, eu_model),
        "logistic_at_barc": partial(PooledLogistic, X_at, y_at),
    }


def cross_validate(X, y, groups, makers):
    """Out-of-fold p and call for every candidate, one fold per distinct group value."""
    rows = []
    for g in sorted(np.unique(groups)):
        te = groups == g
        if len(np.unique(y[~te])) < 2:
            continue
        for name, mk in makers.items():
            m = mk().fit(X[~te], y[~te])
            p = m.predict_proba(X[te])
            rows.append(pd.DataFrame({"field_id": X["field_id"][te].to_numpy(), "model": name,
                                      "fold": str(g), "y": y[te], "p": p,
                                      "pred": (p >= m.threshold).astype(int),
                                      "low_evidence": X["low_evidence"][te].to_numpy(bool)}))
    return pd.concat(rows, ignore_index=True)


def summary_table(oof, design):
    out = []
    for name, g in oof.groupby("model", sort=False):
        y, pred = g["y"].to_numpy(), g["pred"].to_numpy()
        c = classification_metrics(y, pred)["confusion"]
        ex = exact_intervals(c)
        n = len(g)
        conf = ~g["low_evidence"].to_numpy()
        cc = classification_metrics(y[conf], pred[conf])["confusion"]
        out.append({"design": design, "model": name, "correct": c["tp"] + c["tn"], "n": n,
                    "accuracy": (c["tp"] + c["tn"]) / n, "accuracy_ci": ex["accuracy"],
                    "silage_found": c["tp"], "n_silage": c["tp"] + c["fn"],
                    "grain_called_silage": c["fp"], "n_grain": c["fp"] + c["tn"],
                    "abstain_rate": float((~conf).mean()),
                    "correct_not_abstained": cc["tp"] + cc["tn"], "n_not_abstained": int(conf.sum())})
    return pd.DataFrame(out)


def load_inputs(fields_paths, ts_paths):
    fields = load_fields(fields_paths, keep_columns=["OBJECTID"])
    fields = fields[np.isfinite(fields["y"].to_numpy(float))].reset_index(drop=True)
    ts, _ = load_timeseries(ts_paths, field_ids=fields["field_id"])
    X, _ = build_features(ts, fields, FeatureConfig())
    y = fields["y"].to_numpy(int)
    years = pd.to_numeric(fields["year"]).to_numpy(int)
    unit = fields["field_id"].str.replace(r"-(19|20)\d\d$", "", regex=True).to_numpy()
    return fields, X, y, years, unit


CARD_SENTENCE = ("44 field-years cannot separate models that differ by one or two fields: each "
                 "field-year is about 2.3 points of accuracy and each silage field-year about 8.3 "
                 "points of silage recall.")

# what gets frozen: registry name -> (candidate in the comparison, description)
EXPORTS = {
    "barc_logistic_v1": ("logistic_barc",
                         "Six-feature logistic regression fitted on the 44 Beltsville field-years "
                         "only (the team's plan: train on Beltsville, test on held-out Beltsville)."),
    "eu_barc_logistic_v1": ("logistic_at_barc",
                            "Six-feature logistic regression fitted on the Austria 2024 registry "
                            "fields together with the 44 Beltsville field-years, the two sources "
                            "weighted equally. This is what the toolbox's in_site policy uses."),
}


def table_markdown(tab, references):
    head = ("| validation | model | correct of 44 [95% CI] | silage found of 12 | grain called "
            "silage of 32 | abstain rate | correct where not abstained |\n|---|---|---|---|---|---|---|")
    rows = [head]
    for r in tab.to_dict("records"):
        lo, hi = r["accuracy_ci"]
        rows.append(f"| {r['design']} | {r['model']} | {r['correct']} [{lo:.2f}, {hi:.2f}] | "
                    f"{r['silage_found']} | {r['grain_called_silage']} | {r['abstain_rate']:.2f} | "
                    f"{r['correct_not_abstained']} of {r['n_not_abstained']} |")
    for name, text in references:
        rows.append(f"| reference | {name} | {text} |  |  |  |  |")
    return "\n".join(rows)


REFERENCES = [("always grain", "32"), ("timing_rule_v1 (leave-one-year-out)", "40"),
              ("European logistic, zero-shot (Evaluation 1, secondary result)", "42")]


def permutation_test(makers, X, y, groups, names, n=200, seed=20261114):
    """Correct-of-n when the training labels are shuffled within each fold."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in names:
        mk = makers[name]
        scores = []
        for _ in range(n):
            correct = 0
            for g in np.unique(groups):
                te = groups == g
                ytr = rng.permutation(y[~te])
                if len(np.unique(ytr)) < 2:
                    continue
                m = mk().fit(X[~te], ytr)
                correct += int(((m.predict_proba(X[te]) >= m.threshold).astype(int) == y[te]).sum())
            scores.append(correct)
        s = np.array(scores)
        out[name] = {"n_shuffles": n, "median": float(np.median(s)),
                     "p95": float(np.percentile(s, 95)), "max": int(s.max()),
                     "reaching_43": int((s >= 43).sum()), "reaching_40": int((s >= 40).sum())}
    return out


def permutation_markdown(perm):
    rows = ["| model | median correct of 44 | 95th percentile | max | shuffles reaching 43 | "
            "shuffles reaching 40 |", "|---|---|---|---|---|---|"]
    for name, r in perm.items():
        rows.append(f"| {name} | {r['median']:.0f} | {r['p95']:.0f} | {r['max']} | "
                    f"{r['reaching_43']} of {r['n_shuffles']} | {r['reaching_40']} of {r['n_shuffles']} |")
    return "\n".join(rows)


def export(name, X, y, at, registry, fields_paths, ts_paths, git_commit, tab, perm):
    """Fit one candidate on all 44 field-years (plus Austria where it uses it) and freeze it."""
    import datetime as dt
    import joblib
    from ..model.registry import feature_unit, file_sha256, portable, portable_text
    candidate, description = EXPORTS[name]
    folder = Path(registry) / name
    if folder.exists():
        raise SystemExit(f"{folder} exists; frozen folders are written once")
    # build classes through the package path, so the frozen object does not refer to
    # __main__ when this module is run with python -m
    from . import in_site as _module
    from ..model.models import LogisticModel as _Logistic
    if candidate == "logistic_at_barc":
        model = _module.PooledLogistic(at, at["y"].to_numpy(int)).fit(X, y)
        model.X_at = model.y_at = None      # keep the fitted pipeline, not the training rows
        uses_austria = True
    else:
        model = _Logistic(FEATURES).fit(X, y)
        uses_austria = False
    tmp = folder.with_name(folder.name + ".partial")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    joblib.dump(model, tmp / "model.joblib")
    write_json(tmp / "model.json", {
        "name": name, "class": type(model).__name__, "positive_class": "silage",
        "decision": "silage if p_silage >= 0.5", "threshold": 0.5, "features": FEATURES,
        "description": model.describe(), "training": description,
        "comparison_name": candidate})
    write_json(tmp / "feature_schema.json", {
        "computed_by": "fieldvision.features.build.build_features",
        "feature_config": FeatureConfig().to_dict(),
        "features": [{"name": c, "unit": feature_unit(c)} for c in FEATURES],
        "gaps": "missing values replaced by the training median, then standardised"})
    write_json(tmp / "applicability.json", {
        "model": name,
        "requires": {"sensors": ["Sentinel-2 L2A surface reflectance, bands B02-B12"],
                     "clear_obs_before_and_after_drop": True},
        "use_only_for": "Beltsville (FIPS 24033); fitted on these fields, untested elsewhere",
        "abstains_when": ["low evidence: harvest bracket over 20 days, fewer than 6 usable dates, "
                          "no or unconfirmed drop"],
        "confidence_tier": "in-site",
        "validated_on": "Beltsville 2021-2023, leave-one-year-out and leave-one-field-out"})
    data = list(fields_paths) + list(ts_paths)
    write_json(tmp / "provenance.json", portable({
        "name": name, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit_of_code": git_commit,
        "training_data": [{"path": str(p), "sha256": file_sha256(p)} for p in data],
        "austria_training_rows": int(len(at)) if uses_austria else 0,
        "note": "fitted on Beltsville labels (in-site)" + (" plus Austria 2024" if uses_austria else "")
                + "; kept separate from the European models and from the second evaluation"}))
    perm_text = permutation_markdown(perm)
    if uses_austria:
        reading = ("With the Beltsville training labels shuffled, this model still reaches up to 42 "
                   "of 44 (median 34), because the Austrian rows carry the signal: the Beltsville "
                   "labels add about one field over the European logistic's zero-shot 42. Present "
                   "its 43 as the European model nudged by Beltsville, not as a model learned at "
                   "Beltsville.")
    else:
        reading = ("With the training labels shuffled it reaches at most 40 of 44 (median 23); the "
                   "real 43 is far in the tail of that distribution.")
    card = [f"# {name}\n", description + "\n",
            "**In-site model: it is fitted on the Beltsville labels, so it applies to Beltsville "
            "only and its numbers come from cross-validation on the same 44 field-years.**\n",
            f"Fitted model: {model.describe()}\n",
            f"In the comparison below it is `{candidate}`.\n",
            "## Cross-validation on Beltsville (exact Clopper-Pearson intervals)\n",
            table_markdown(tab, REFERENCES),
            "\nLeave-one-field-out (all seasons of a polygon held out together) is the honest "
            "number for a new field; leave-one-year-out is a new season on fields already seen. "
            "Three logistic variants tie at 43 of 44 under both designs; the single miss in every "
            "case is barc-327-2021 (2021 fold: 17 of 18; 2022: 7 of 7; 2023: 19 of 19), a declared "
            "silage field that was dry before the cut and looks like grain on every measure. The "
            "8 abstained field-years still get a call and are counted in the 43. Everything fitted "
            "(imputation, scaling, the logistic) is fitted inside the training fold; the threshold "
            "is a fixed 0.5 and the abstain flag uses only each field's own data quality.\n",
            CARD_SENTENCE + "\n",
            "## Label-permutation check (leave-one-year-out, 200 seeded shuffles of the training "
            "labels within each fold)\n", perm_text, "\n" + reading + "\n",
            "The out-of-fold rows in `results/MD_<year>.jsonl` come from leave-one-field-out.\n",
            "## Known limits\n",
            "- Fitted on one farm; do not apply it to other counties (the toolbox's in_site policy "
            "is limited to FIPS 24033).\n- 12 silage field-years in total; the intervals above are "
            "wide.\n- Probabilities are class-balanced, not calibrated to Beltsville's silage share.\n"]
    (tmp / "model_card.md").write_text(portable_text("\n".join(card)), encoding="utf-8")
    tmp.rename(folder)
    return folder


def main(argv=None):
    ap = argparse.ArgumentParser(description="In-site model comparison on Beltsville.")
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--austria-features", required=True,
                    help="features.parquet of the Austria 2024 training run")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", nargs="*", default=["leave-one-year-out", "leave-one-field-out"])
    ap.add_argument("--export-to", default=None,
                    help="registry folder to export barc_logistic_v1 and eu_barc_logistic_v1 to")
    ap.add_argument("--git-commit", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fields, X, y, years, unit = load_inputs(a.fields, a.timeseries)
    at = pd.read_parquet(a.austria_features)
    at = at[at["split"] == "development"].reset_index(drop=True)
    from ..model.registry import DEFAULT_REGISTRY
    eu, _, _ = load_registry(a.registry or DEFAULT_REGISTRY, "eu_logistic_v1")
    makers = candidates(eu, at, at["y"].to_numpy(int))
    tables, oofs = [], []
    for design in a.designs:
        groups = years if design == "leave-one-year-out" else unit
        oof = cross_validate(X, y, groups, makers)
        oof["design"] = design
        oofs.append(oof)
        tables.append(summary_table(oof, design))
    oof = pd.concat(oofs, ignore_index=True)
    oof.to_csv(out / "in_site_oof.csv", index=False)
    tab = pd.concat(tables, ignore_index=True)
    tab.to_csv(out / "in_site_table.csv", index=False)
    write_json(out / "in_site_table.json", tab.to_dict("records"))
    for r in tab.to_dict("records"):
        lo, hi = r["accuracy_ci"]
        print(f"{r['design']:20s} {r['model']:26s} {r['correct']}/{r['n']} [{lo:.2f}, {hi:.2f}] "
              f"silage {r['silage_found']}/{r['n_silage']} grain->silage "
              f"{r['grain_called_silage']}/{r['n_grain']} abstain {r['abstain_rate']:.2f} "
              f"(not abstained {r['correct_not_abstained']}/{r['n_not_abstained']})")
    (out / "in_site_table.md").write_text(table_markdown(tab, REFERENCES) + "\n\n" + CARD_SENTENCE + "\n",
                                          encoding="utf-8")
    perm = permutation_test(makers, X, y, years, ["logistic_barc", "logistic_at_barc"])
    write_json(out / "permutation_test.json", perm)
    print(permutation_markdown(perm))
    if a.export_to:
        for name in EXPORTS:
            folder = export(name, X, y, at, a.export_to, a.fields, a.timeseries, a.git_commit,
                            tab, perm)
            print(f"exported -> {folder}")
    return tab, oof


if __name__ == "__main__":
    main()
