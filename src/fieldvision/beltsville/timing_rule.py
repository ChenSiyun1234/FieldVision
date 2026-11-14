"""timing_rule_v1: thresholds on the drop date and the greenness just before it, fitted on
Beltsville and validated by leave-one-year-out (train on two years, test on the third).

    python -m fieldvision.beltsville.timing_rule --timeseries <barc parquet x3> \
        --fields <barc geojson x3> --registry registry --out <report dir> \
        --git-commit <sha>

The rule: silage if the harvest drop comes on or before day T (mid-point of the bracket
between the last standing and first harvested observation) [AND | OR] NDVI at the last
standing observation is at least G. Which of timing-only / AND / OR, and T and G, are chosen
on the training years only by balanced accuracy (fieldvision.model.models.RuleModel). The
final registry model is the same procedure fitted on all three years; its expected
performance is the leave-one-year-out result, not its fit on the data it saw.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from ..features.season import FeatureConfig
from ..model.artifacts import write_json
from ..model.metrics import classification_metrics, exact_intervals
from ..model.models import RuleModel
from ..model.registry import feature_unit, file_sha256

NAME = "timing_rule_v1"
TIMING = "harv_mid_doy"
GREEN = "pre_ndvi"


def make_rule():
    return RuleModel(timing=TIMING, moisture=GREEN, min_gain=0.01, name=NAME)


def loyo(X, y, years):
    """Out-of-fold predictions and the rule fitted in each fold."""
    oof = pd.DataFrame({"field_id": X["field_id"], "year": years, "y": y,
                        "low_evidence": X["low_evidence"].to_numpy(bool)})
    oof["p"] = np.nan
    folds = []
    for yr in sorted(np.unique(years)):
        te = years == yr
        rule = make_rule().fit(X[~te], y[~te])
        oof.loc[te, "p"] = rule.predict_proba(X[te])
        folds.append({"test_year": int(yr), "n_train": int((~te).sum()),
                      "n_train_silage": int(y[~te].sum()), "rule": rule.describe(),
                      "thresholds": rule.to_dict()})
    oof["pred"] = (oof["p"] >= 0.5).astype(int)
    return oof, folds


def _block(y, pred, p):
    m = classification_metrics(y, pred, p)
    m["exact95"] = exact_intervals(m["confusion"])
    return m


def _fmt(m, k):
    v, ci = m.get(k), m["exact95"].get(k)
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v:.3f}"


def _row(name, m):
    c = m["confusion"]
    return (f"| {name} | {m['n']} ({m['n_silage']}) | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} | "
            f"{_fmt(m, 'accuracy')} | {_fmt(m, 'precision')} | {_fmt(m, 'recall')} | "
            f"{_fmt(m, 'grain_precision')} | {_fmt(m, 'grain_recall')} |")


HEAD = ("| set | n (silage) | silage called silage | silage called grain | grain called silage | "
        "grain called grain | accuracy | silage precision | silage recall | grain precision | "
        "grain recall |\n|---|---|---|---|---|---|---|---|---|---|---|")


def card(oof, folds, final, counts, n_fields):
    L = [f"# {NAME}\n",
         "Thresholds on the harvest drop date and on the greenness just before it, fitted on the "
         "organisers' Beltsville field-years and validated by leave-one-year-out (train on two "
         "years, test on the third). This is not the European model: it uses Beltsville labels, "
         "so its numbers are an in-site estimate, and the folds share fields (the same polygon "
         "appears in several years).\n",
         f"Final rule (fitted on all {n_fields} field-years): {final.describe()}\n",
         "## Field-years by year and class\n",
         counts.to_markdown(index=False) if hasattr(counts, "to_markdown") and _tab() else
         "```\n" + counts.to_string(index=False) + "\n```",
         "\n## Leave-one-year-out results (exact Clopper-Pearson 95 % intervals)\n", HEAD]
    y, pred, p = oof["y"].to_numpy(), oof["pred"].to_numpy(), oof["p"].to_numpy()
    pooled = _block(y, pred, p)
    ag = _block(y, np.zeros_like(y), np.zeros(len(y)))
    for f in folds:
        g = oof[oof["year"] == f["test_year"]]
        L.append(_row(f"test {f['test_year']}", _block(g["y"], g["pred"], g["p"])))
    L.append(_row("pooled (all 3 folds)", pooled))
    L.append(_row("always grain, pooled", ag))
    conf = ~oof["low_evidence"].to_numpy()
    pc = _block(y[conf], pred[conf], p[conf])
    L.append(_row("pooled, not abstained", pc))
    L.append(f"\nAbstain rate (low evidence: harvest bracket too wide, too few dates, no or "
             f"unconfirmed drop): {1 - conf.mean():.3f} ({int((~conf).sum())} of {len(conf)}). The rule "
             "still makes a call on those; the row above leaves them out.\n")
    L.append("Each field-year is 1/44 of the pooled accuracy, about 2.3 points; each silage "
             "field-year is 1/12 of silage recall, about 8.3 points. With 12 silage field-years the "
             "intervals above are wide, and differences of one or two fields are not meaningful.\n")
    L.append(f"Baseline: always grain scores {ag['accuracy']:.3f} ({ag['confusion']['tn']} of "
             f"{ag['n']}) with silage recall 0.\n")
    L.append("## Rule fitted in each fold\n")
    for f in folds:
        L.append(f"- test {f['test_year']} (trained on {f['n_train']}, {f['n_train_silage']} silage): "
                 f"{f['rule']}")
    L.append("\n## Known limits\n")
    L.append("- Fitted and validated on one farm in Maryland; the thresholds encode its calendar "
             "and will not hold where corn is planted or cut at other times.\n"
             "- Leave-one-year-out reuses the same fields across folds, so it measures a new season "
             "on known fields, not a new place.\n- The score is a rule margin, not a calibrated "
             "probability.\n- Needs a clear observation before and after the drop.\n")
    return "\n".join(L), pooled, pc, ag


def _tab():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit and validate timing_rule_v1 on Beltsville.")
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--git-commit", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    folder = Path(a.registry) / NAME
    if folder.exists():
        raise SystemExit(f"{folder} exists; frozen folders are written once")
    cfg = FeatureConfig()
    fields = load_fields(a.fields, keep_columns=["OBJECTID"])
    fields = fields[np.isfinite(fields["y"].to_numpy(float))].reset_index(drop=True)
    ts, _ = load_timeseries(a.timeseries, field_ids=fields["field_id"])
    X, _ = build_features(ts, fields, cfg)
    y = fields["y"].to_numpy(int)
    years = pd.to_numeric(fields["year"]).to_numpy(int)
    oof, folds = loyo(X, y, years)
    oof.to_csv(out / "loyo_predictions.csv", index=False)
    final = make_rule().fit(X, y)
    counts = pd.crosstab(fields["year"], fields["label"]).reset_index()
    text, pooled, pc, ag = card(oof, folds, final, counts, len(fields))
    (out / "REPORT.md").write_text(text, encoding="utf-8")

    folder.mkdir(parents=True)
    write_json(folder / "model.json", {
        "name": NAME, "class": "RuleModel", "positive_class": "silage",
        "decision": "silage if the rule holds (score >= 0.5)", "threshold": 0.5,
        "features": [TIMING, GREEN], "rule": final.to_dict(), "rule_text": final.describe(),
        "probability": "rule margin squashed to 0-1; not a calibrated probability"})
    write_json(folder / "feature_schema.json", {
        "computed_by": "fieldvision.features.build.build_features",
        "feature_config": cfg.to_dict(),
        "features": [{"name": TIMING, "unit": feature_unit(TIMING),
                      "meaning": "mid-point between the last standing and first harvested clear observation"},
                     {"name": GREEN, "unit": feature_unit(GREEN),
                      "meaning": "NDVI at the last standing clear observation"}],
        "gaps": "a field with no detected drop gets a fixed score learned in training; a field "
                "with a drop date but no greenness value is judged on the date alone"})
    write_json(folder / "applicability.json", {
        "model": NAME,
        "requires": {"sensors": ["Sentinel-2 L2A surface reflectance (B04, B08 and the bands the "
                                 "low-evidence rule uses)"],
                     "optical_clear_obs_in_harvest_window": 1,
                     "clear_obs_before_and_after_drop": True},
        "abstains_when": ["harvest bracket wider than %g days" % cfg.low_evidence_max_interval_days,
                          "fewer than %d usable dates Jul-Nov" % cfg.low_evidence_min_usable,
                          "no drop found", "drop seen only once"],
        "confidence_tier": "high (simple, explainable) within Beltsville; untested elsewhere",
        "validated_on": "Beltsville 2021-2023, leave-one-year-out (3 folds), %d field-years" % len(fields)})
    write_json(folder / "provenance.json", {
        "name": NAME, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit_of_code": a.git_commit,
        "training_data": [{"path": str(p), "sha256": file_sha256(p)} for p in a.fields + a.timeseries],
        "note": "fitted on Beltsville labels; kept separate from the European models"})
    (folder / "model_card.md").write_text(text, encoding="utf-8")
    write_json(out / "metrics.json", {"pooled": pooled, "pooled_not_abstained": pc,
                                      "always_grain": ag, "folds": folds,
                                      "abstain_rate": float(oof["low_evidence"].mean())})
    print(f"LOYO pooled accuracy {pooled['accuracy']:.3f}, silage recall {pooled['recall']:.3f}, "
          f"FPR {pooled['fpr']:.3f}; always grain {ag['accuracy']:.3f}; final: {final.describe()}")
    return pooled


if __name__ == "__main__":
    main()
