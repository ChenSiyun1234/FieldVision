import numpy as np
import pandas as pd
import pytest

from fieldvision.anomaly.scorers import (SCORERS, NormalModel, OneClassModel, Preprocess,
                                         cross_fit_scores)


def _normal(n=400, d=4, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


@pytest.mark.parametrize("scorer", SCORERS)
def test_outliers_score_higher_than_inliers(scorer):
    A = _normal()
    m = NormalModel(scorer, seed=0).fit(A)
    inl = _normal(200, seed=1)
    out = _normal(200, seed=2) + 4.0
    s_in, s_out = m.score(inl), m.score(out)
    # nearly every shifted field is scored above nearly every normal one
    assert (s_out[:, None] > s_in[None, :]).mean() > 0.97


def test_preprocess_is_fit_on_the_normal_set_only_and_fills_missing():
    A = _normal(100, 3)
    A[::7, 1] = np.nan
    pre = Preprocess(max_dim=10).fit(A)
    med = np.nanmedian(A[:, 1])
    B = np.array([[0.0, np.nan, 0.0]])
    # a missing value becomes the normal set's median, i.e. 0 after centring on the median
    z = pre.transform(B)
    assert np.isfinite(z).all()
    assert abs(z[0, 1] - (med - pre.center_[1]) / pre.scale_[1]) < 1e-12
    # transform of new data does not change the fitted statistics
    c = pre.center_.copy()
    pre.transform(_normal(50, 3, seed=5) + 100)
    assert np.array_equal(c, pre.center_)


def test_pca_only_above_max_dim():
    assert Preprocess(max_dim=10).fit(_normal(100, 6)).pca_ is None
    pre = Preprocess(max_dim=10).fit(_normal(100, 30))
    assert pre.transform(_normal(5, 30)).shape == (5, 10)


def _table(n_grain=300, n_sil=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(np.vstack([rng.normal(size=(n_grain, 3)), rng.normal(size=(n_sil, 3)) + 3]),
                     columns=["a", "b", "c"])
    y = np.r_[np.zeros(n_grain, int), np.ones(n_sil, int)]
    return X, y


def test_normal_set_is_grain_only_without_contamination():
    X, y = _table()
    m = OneClassModel(["a", "b", "c"], "knn").fit(X, y)
    assert m.n_normal_ == 300 and m.n_silage_in_normal_ == 0


@pytest.mark.parametrize("c", [0.01, 0.05, 0.20])
def test_contamination_share(c):
    X, y = _table()
    m = OneClassModel(["a", "b", "c"], "mahalanobis", contamination=c, seed=3)
    rows = m.normal_rows(y)
    share = y[rows].mean()
    assert abs(share - c) < 0.005
    assert (y[rows] == 0).sum() == 300            # every grain field is kept
    assert len(np.unique(rows)) == len(rows)


def test_contamination_needs_enough_silage():
    X, y = _table(n_grain=300, n_sil=10)
    with pytest.raises(ValueError):
        OneClassModel(["a"], "knn", contamination=0.2).normal_rows(y)


@pytest.mark.parametrize("scorer", SCORERS)
def test_threshold_flags_about_two_percent_of_new_normal_fields(scorer):
    X, y = _table(n_grain=1500, n_sil=100)
    m = OneClassModel(["a", "b", "c"], scorer, quantile=0.98, seed=0).fit(X, y)
    fresh = pd.DataFrame(_normal(4000, 3, seed=9), columns=["a", "b", "c"])
    flagged = m.predict(fresh).mean()
    assert 0.005 < flagged < 0.05
    assert m.predict(X[y == 1]).mean() > 0.8


def test_cross_fit_scores_are_out_of_sample():
    A = _normal(300, 3)
    # with k=1, in-sample knn distance is 0 for every row; cross-fitted ones are not
    assert (NormalModel("knn", k=1).fit(A).score(A) == 0).all()
    assert (cross_fit_scores(A, "knn", 3, k=1) > 0).all()


def test_labels_beyond_grain_versus_silage_are_not_used():
    X, y = _table()
    a = OneClassModel(["a", "b", "c"], "iforest", seed=1).fit(X, y).predict_proba(X)
    # shuffling which silage rows are which changes nothing: silage rows are never read
    X2 = X.copy()
    sil = np.flatnonzero(y == 1)
    X2.iloc[sil] = X.iloc[np.random.default_rng(0).permutation(sil)].to_numpy()
    b = OneClassModel(["a", "b", "c"], "iforest", seed=1).fit(X2, y).predict_proba(X)
    assert np.allclose(a, b)
