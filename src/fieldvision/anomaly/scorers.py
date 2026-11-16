"""One-class scorers fit on "normal" fields only. Higher score = less like the normal set.

Three standard recipes, kept small on purpose:
  mahalanobis  robust (minimum covariance determinant) distance to the normal cloud
  iforest      isolation forest
  knn          mean distance to the k nearest fields in a memory bank of normal fields

Every learned step (median imputation, median/IQR scaling, the optional PCA,
the scorer itself and the flagging threshold) is fit on the normal set handed
to `fit` and nothing else.

`OneClassModel` wraps a scorer so it plugs into model.harness.run_splits like
the supervised models: `fit(X, y)` keeps the grain rows (y == 0) as the normal
set, optionally mixes in a share of silage rows to imitate an impure normal
set, and ignores the labels from then on. `predict_proba` returns the anomaly
score (not a probability; ROC-AUC only needs the order), and `threshold` is a
high quantile of the normal set's own scores, computed out-of-sample by
cross-fitting inside the normal set, so it is fixed before any test field is
seen.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

SCORERS = ("mahalanobis", "iforest", "knn")


class Preprocess:
    """Median imputation, median/IQR scaling and, above `max_dim` columns, PCA."""

    def __init__(self, max_dim=10, seed=0):
        self.max_dim = max_dim
        self.seed = seed

    def fit(self, A):
        A = np.asarray(A, dtype=float)
        med = np.nanmedian(A, axis=0)
        self.fill_ = np.where(np.isfinite(med), med, 0.0)
        B = self._fill(A)
        q1, q3 = np.percentile(B, [25, 75], axis=0)
        scale = q3 - q1
        sd = B.std(axis=0)
        scale = np.where(scale > 1e-9, scale, np.where(sd > 1e-9, sd, 1.0))
        self.center_, self.scale_ = np.median(B, axis=0), scale
        Z = (B - self.center_) / self.scale_
        self.pca_ = None
        if self.max_dim and Z.shape[1] > self.max_dim:
            self.pca_ = PCA(n_components=min(self.max_dim, Z.shape[0] - 1), random_state=self.seed).fit(Z)
        return self

    def _fill(self, A):
        return np.where(np.isfinite(A), A, self.fill_[None, :])

    def transform(self, A):
        Z = (self._fill(np.asarray(A, dtype=float)) - self.center_) / self.scale_
        return self.pca_.transform(Z) if self.pca_ is not None else Z


class MahalanobisScorer:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, Z):
        try:
            self.cov_ = MinCovDet(random_state=self.seed).fit(Z)
        except (ValueError, np.linalg.LinAlgError):
            self.cov_ = EmpiricalCovariance().fit(Z)
        return self

    def score(self, Z):
        return np.sqrt(np.maximum(self.cov_.mahalanobis(Z), 0.0))


class IForestScorer:
    def __init__(self, seed=0, n_estimators=200):
        self.seed, self.n_estimators = seed, n_estimators

    def fit(self, Z):
        self.est_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed).fit(Z)
        return self

    def score(self, Z):
        return -self.est_.score_samples(Z)


class KNNScorer:
    """Mean distance to the k nearest members of the memory bank."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, Z):
        self.k_ = min(self.k, len(Z))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_).fit(Z)
        return self

    def score(self, Z):
        d, _ = self.nn_.kneighbors(Z, n_neighbors=self.k_)
        return d.mean(axis=1)


def make_scorer(name, seed=0, k=5):
    if name == "mahalanobis":
        return MahalanobisScorer(seed)
    if name == "iforest":
        return IForestScorer(seed)
    if name == "knn":
        return KNNScorer(k)
    raise ValueError(f"unknown scorer {name!r}; choose from {SCORERS}")


class NormalModel:
    """Preprocessing + scorer, fit on a normal set (a plain array, no labels)."""

    def __init__(self, scorer="knn", max_dim=10, seed=0, k=5):
        self.scorer, self.max_dim, self.seed, self.k = scorer, max_dim, seed, k

    def fit(self, A):
        self.pre_ = Preprocess(self.max_dim, self.seed).fit(A)
        self.est_ = make_scorer(self.scorer, self.seed, self.k).fit(self.pre_.transform(A))
        return self

    def score(self, A):
        return self.est_.score(self.pre_.transform(A))


def cross_fit_scores(A, scorer="knn", n_folds=3, max_dim=10, seed=0, k=5):
    """Out-of-sample scores of every row of the normal set (each scored by a model
    fit on the other parts), so a quantile of them is an honest threshold."""
    A = np.asarray(A, dtype=float)
    rng = np.random.default_rng(seed)
    part = rng.permutation(len(A)) % n_folds
    out = np.empty(len(A))
    for f in range(n_folds):
        te = part == f
        out[te] = NormalModel(scorer, max_dim, seed, k).fit(A[~te]).score(A[te])
    return out


class OneClassModel:
    """Harness-compatible wrapper: fit on grain rows (plus optional silage contamination)."""

    def __init__(self, features, scorer="knn", contamination=0.0, quantile=0.98, calib_folds=3,
                 max_dim=10, seed=0, k=5):
        self.feature_names = list(features)
        self.scorer, self.contamination, self.quantile = scorer, float(contamination), quantile
        self.calib_folds, self.max_dim, self.seed, self.k = calib_folds, max_dim, seed, k
        self.name = f"oneclass-{scorer}"
        self.threshold = np.inf

    def normal_rows(self, y):
        """Indices of the training rows used as 'normal': all grain, plus silage so that
        silage makes up `contamination` of the normal set."""
        y = np.asarray(y).astype(int)
        grain = np.flatnonzero(y == 0)
        if self.contamination <= 0:
            return grain
        sil = np.flatnonzero(y == 1)
        n = int(round(self.contamination / (1 - self.contamination) * len(grain)))
        if n > len(sil):
            raise ValueError(f"need {n} silage rows for {self.contamination:.0%} contamination, have {len(sil)}")
        rng = np.random.default_rng(self.seed)
        return np.sort(np.concatenate([grain, rng.choice(sil, n, replace=False)]))

    def fit(self, X, y):
        rows = self.normal_rows(y)
        A = X[self.feature_names].to_numpy(float)[rows]
        self.n_normal_, self.n_silage_in_normal_ = len(rows), int(np.asarray(y)[rows].sum())
        self.model_ = NormalModel(self.scorer, self.max_dim, self.seed, self.k).fit(A)
        if self.calib_folds and self.calib_folds > 1:
            s = cross_fit_scores(A, self.scorer, self.calib_folds, self.max_dim, self.seed, self.k)
        else:
            s = self.model_.score(A)
        self.threshold = float(np.quantile(s, self.quantile))
        return self

    def predict_proba(self, X):
        return self.model_.score(X[self.feature_names].to_numpy(float))

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def explain(self, X):
        return None

    def describe(self):
        return (f"one-class {self.scorer} on {len(self.feature_names)} features, "
                f"{self.n_normal_} normal fields ({self.n_silage_in_normal_} silage mixed in), "
                f"flag above the normal set's {self.quantile:.0%} quantile")
