"""Per-field predictions in the format of `fieldvision.app.export --predictions`, from one
frozen registry model, one file per region (e.g. barc-2021, barc-2022, barc-2023).

    python -m fieldvision.model.app_predictions --model eu_logistic_v1 \
        --timeseries <...> --fields <...> --region-prefix barc --out <predictions dir>

The app requires out-of-fold predictions for labelled regions. A model trained only on
European data and applied to US fields satisfies that; a model fitted on the same fields
does not, and this command refuses registry models whose provenance says so unless
--allow-in-sample is given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_features
from ..features.loading import load_fields, load_timeseries
from .artifacts import predictions_frame, write_json
from .registry import DEFAULT_REGISTRY, load


def flip_reasons(model, X, p, threshold):
    """The strongest factor pulling against the call, in the model's own terms."""
    try:
        contrib = model.explain(X)
    except Exception:
        contrib = None
    out = np.full(len(X), "", dtype=object)
    if contrib is None or contrib.empty:
        return out
    a = np.nan_to_num(contrib.to_numpy(float))
    names = np.array(contrib.columns)
    for i in range(len(X)):
        call_silage = p[i] >= threshold
        against = -a[i] if call_silage else a[i]       # positive = pushes towards the other class
        j = int(np.argmax(against))
        if against[j] > 0:
            out[i] = (f"{names[j]} pulled towards {'grain' if call_silage else 'silage'} "
                      f"({a[i, j]:+.2f} log-odds); p = {p[i]:.2f} vs threshold {threshold:.2f}")
        else:
            out[i] = f"every factor agreed; p = {p[i]:.2f} vs threshold {threshold:.2f}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predictions for the app export from a registry model.")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--model", required=True)
    ap.add_argument("--timeseries", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--region-prefix", required=True, help="region id = <prefix>-<year>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-in-sample", action="store_true")
    a = ap.parse_args(argv)
    prov = json.loads((Path(a.registry) / a.model / "provenance.json").read_text(encoding="utf-8"))
    in_sample = "fitted on Beltsville" in str(prov.get("note", ""))
    if in_sample and not a.allow_in_sample:
        raise SystemExit(f"{a.model} was fitted on these fields; the app needs out-of-fold predictions")
    model, mj, cfg = load(a.registry, a.model)
    fields = load_fields(a.fields)
    ts, _ = load_timeseries(a.timeseries, field_ids=fields["field_id"])
    X, _ = build_features(ts, fields, cfg)
    p = model.predict_proba(X)
    pf = predictions_frame(X, fields.drop(columns=["label", "y"], errors="ignore"), p,
                           mj["threshold"], a.model, model=model,
                           reference=_reference(a.registry, a.model))
    app = pd.DataFrame({
        "field_id": pf["field_id"], "p_silage": pf["p_silage"],
        "harvest_start": pd.to_datetime(pf["harvest_start"]).dt.date,
        "harvest_end": pd.to_datetime(pf["harvest_end"]).dt.date,
        "pre_ndvi": pf["pre_harvest_ndvi"], "pre_ndmi": pf["pre_harvest_ndmi"],
        "low_evidence": pf["low_evidence"].astype(bool),
        "low_evidence_reason": pf["low_evidence_reason"].replace("", None),
        "explanation": pf.get("reason"),
        "flip_reason": flip_reasons(model, X, p, mj["threshold"]),
        "post_ndvi": X["post_ndvi"].to_numpy(float), "post_ndti": X["post_ndti"].to_numpy(float),
        "pred_label": pf["predicted_label"], "model": a.model,
    })
    app["year"] = pd.to_numeric(fields.set_index("field_id").reindex(app["field_id"])["year"]).to_numpy()
    out = Path(a.out)
    written = {}
    for yr, g in app.groupby("year"):
        d = out / f"{a.region_prefix}-{int(yr)}"
        d.mkdir(parents=True, exist_ok=True)
        g.drop(columns="year").to_parquet(d / "predictions.parquet", index=False)
        written[str(d / "predictions.parquet")] = int(len(g))
    write_json(out / f"{a.region_prefix}_{a.model}_written.json",
               {"model": a.model, "files": written, "out_of_fold": not in_sample})
    for k, n in written.items():
        print(f"{n} fields -> {k}")
    return app


def _reference(registry, name):
    """Class medians for the reason sentence: from the training run named in provenance."""
    prov = json.loads((Path(registry) / name / "provenance.json").read_text(encoding="utf-8"))
    run = prov.get("training_run")
    if run and (Path(run) / "run.json").exists():
        return json.loads((Path(run) / "run.json").read_text(encoding="utf-8")).get("evidence_reference")
    return None


if __name__ == "__main__":
    main()
