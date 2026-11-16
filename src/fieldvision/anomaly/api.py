"""Importable anomaly view: fit on a mostly-grain reference table, score a target table.

    from fieldvision.anomaly.api import fit_reference, score_target, confidence_tier
    ref = fit_reference(X_reference)            # rows of features.build.build_features output
    out = score_target(ref, X_target)           # field_id, anomaly_score, anomaly_threshold, anomaly_flag
    tier = confidence_tier(classifier_pred, out["anomaly_flag"], low_evidence)

The reference table carries no labels and none are read: every row is taken
as "normal". It should be mostly grain corn; see docs/anomaly-view.md for how
much silage it can contain before the score degrades. The flag threshold is a
high quantile (default 98th percentile) of the reference fields' own scores,
each scored by a model fit on the other reference fields (cross-fitting), so
it is set from the reference set alone, before any target field is seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..model.models import LOGISTIC_FEATURES
from .scorers import NormalModel, cross_fit_scores

# the recipe chosen in docs/anomaly-view.md
DEFAULT_FEATURES = list(LOGISTIC_FEATURES["relative"])
DEFAULT_SCORER = "knn"
DEFAULT_QUANTILE = 0.98
DEFAULT_K = 5
DEFAULT_CALIB_FOLDS = 5
MIN_REFERENCE = 50


class ReferenceModel:
    """What fit_reference returns; plain attributes so it pickles and can be described."""

    def __init__(self, model, features, scorer, quantile, threshold, n_reference, reference_scores):
        self.model, self.features, self.scorer = model, list(features), scorer
        self.quantile, self.threshold, self.n_reference = quantile, float(threshold), int(n_reference)
        self.reference_scores = np.asarray(reference_scores, dtype=float)

    def describe(self):
        return (f"anomaly view: {self.scorer} on {len(self.features)} features, reference of "
                f"{self.n_reference} fields, flag above their {self.quantile:.0%} quantile "
                f"({self.threshold:.3f})")


def _feature_matrix(table, features):
    missing = [c for c in features if c not in table.columns]
    if missing:
        raise KeyError(f"feature table lacks {missing}")
    return table[features].to_numpy(dtype=float)


def fit_reference(reference, features=None, scorer=DEFAULT_SCORER, quantile=DEFAULT_QUANTILE,
                  k=DEFAULT_K, calib_folds=DEFAULT_CALIB_FOLDS, seed=0, drop_low_evidence=True):
    """Learn "normal" from a reference feature table (no labels used).

    drop_low_evidence  leave out reference rows the feature code marks low evidence
                       (too few usable dates), when the column is present
    """
    features = list(features or DEFAULT_FEATURES)
    ref = reference
    if drop_low_evidence and "low_evidence" in ref.columns:
        ref = ref[~ref["low_evidence"].astype(bool)]
    A = _feature_matrix(ref, features)
    A = A[np.isfinite(A).any(axis=1)]
    if len(A) < MIN_REFERENCE:
        raise ValueError(f"reference has {len(A)} usable fields; need at least {MIN_REFERENCE}")
    model = NormalModel(scorer, max_dim=10, seed=seed, k=k).fit(A)
    s = cross_fit_scores(A, scorer, calib_folds, max_dim=10, seed=seed, k=k)
    return ReferenceModel(model, features, scorer, quantile, np.quantile(s, quantile), len(A), s)


def score_target(ref, target):
    """Score every target row. anomaly_ratio = score / threshold (>= 1 means flagged)."""
    s = ref.model.score(_feature_matrix(target, ref.features))
    out = pd.DataFrame({"anomaly_score": s, "anomaly_threshold": ref.threshold,
                        "anomaly_ratio": s / ref.threshold, "anomaly_flag": (s >= ref.threshold).astype(int)},
                       index=target.index)
    if "field_id" in target.columns:
        out.insert(0, "field_id", target["field_id"].to_numpy())
    return out


def confidence_tier(classifier_pred, anomaly_flag, low_evidence=None):
    """'agree' when the classifier's silage call matches the anomaly flag, 'disagree' when
    not, 'abstain' where the field is low evidence (or either input is missing)."""
    c = pd.Series(np.asarray(classifier_pred, dtype=float))
    a = pd.Series(np.asarray(anomaly_flag, dtype=float))
    tier = np.where(c == a, "agree", "disagree").astype(object)
    bad = c.isna().to_numpy() | a.isna().to_numpy()
    if low_evidence is not None:
        bad |= np.asarray(low_evidence, dtype=bool)
    tier[bad] = "abstain"
    return tier
