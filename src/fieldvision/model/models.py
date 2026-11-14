"""Three classifiers behind one interface, plus the majority-class baseline.

    model.fit(X, y)            X: feature DataFrame, y: 1 = silage, 0 = grain
    model.predict_proba(X)     score for silage in [0, 1]
    model.predict(X)           1 where the score reaches model.threshold
    model.feature_names        the columns the model actually reads
    model.describe()           something a person can read

Each model picks its own columns out of the full feature table, so the
evaluation code can hand the same table to all of them. Everything that
is learned (thresholds, imputation medians, scaling, trees) is learned in
fit() and nowhere else.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.season import signal_features

RULE_TIMING = {"relative": "harv_days_since_peak", "calendar": "harv_mid_doy",
               "all": "harv_days_since_peak"}
RULE_MOISTURE = "pre_ndmi"

LOGISTIC_FEATURES = {
    "relative": ["harv_days_since_peak", "harv_decline_days", "pre_ndvi_frac", "pre_ndmi",
                 "sen_ndvi_slope_rel", "post_ndti"],
    "calendar": ["harv_mid_doy", "gi_ndvi_aug_oct", "pre_ndvi", "pre_ndmi",
                 "sen_ndvi_slope", "post_ndti"],
}
LOGISTIC_FEATURES["all"] = list(dict.fromkeys(LOGISTIC_FEATURES["relative"]
                                              + LOGISTIC_FEATURES["calendar"]))

GBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                  class_weight="balanced", n_jobs=4, verbose=-1)


def _matrix(X, names):
    a = X[names].to_numpy(dtype=float)
    a[~np.isfinite(a)] = np.nan
    return a


class _Base:
    name = "base"
    threshold = 0.5
    feature_names: list = []

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def describe(self):
        return self.name


class MajorityModel(_Base):
    """Predicts the most common class of the training data for every field."""
    name = "majority"

    def __init__(self):
        self.feature_names = []

    def fit(self, X, y):
        self.prevalence_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.prevalence_)

    def describe(self):
        cls = "silage" if self.prevalence_ >= 0.5 else "grain"
        return f"always {cls} (training share of silage {self.prevalence_:.3f})"


def _robust_scale(v):
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return 1.0
    q1, q3 = np.percentile(v, [25, 75])
    s = (q3 - q1) / 1.349
    if s <= 0:
        s = float(np.std(v))
    return s if s > 0 else 1.0


def _candidates(v, n=41):
    v = v[np.isfinite(v)]
    return np.unique(np.quantile(v, np.linspace(0.02, 0.98, n)))


class RuleModel(_Base):
    """Silage if harvested early (and/or still wet just before harvest).

        timing <= T_time   [AND | OR]   moisture >= T_moist

    Both thresholds and the choice between timing only, AND, OR are fit on
    the training data by maximising balanced accuracy. The moisture test is
    only added if it improves training balanced accuracy by more than
    `min_gain`. A field with no detected harvest gets the silage share seen
    among such fields in training; a field with timing but no moisture
    value is judged on timing alone.

    The score is a squashed distance to the thresholds. It ranks fields but
    is not a calibrated probability.
    """
    name = "rule"

    def __init__(self, timing="harv_days_since_peak", moisture=RULE_MOISTURE, min_gain=0.01):
        self.timing = timing
        self.moisture = moisture
        self.min_gain = min_gain
        self.feature_names = [timing, moisture]

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        d = _matrix(X, [self.timing])[:, 0]
        m = _matrix(X, [self.moisture])[:, 0]
        self.prevalence_ = float(y.mean())
        has_d = np.isfinite(d)
        n_missing = int((~has_d).sum())
        # shrink towards the overall share so a handful of fields cannot decide it
        self.p_missing_ = float((y[~has_d].sum() + 5 * self.prevalence_) / (n_missing + 5))
        self.mode_ = "constant"
        self.t_time_ = self.t_moist_ = float("nan")
        self.scale_time_ = _robust_scale(d)
        self.scale_moist_ = _robust_scale(m)

        d, m, yy = d[has_d], m[has_d], y[has_d]
        n_pos, n_neg = int(yy.sum()), int((1 - yy).sum())
        if n_pos == 0 or n_neg == 0:
            return self
        td = _candidates(d)
        A = d[:, None] <= td[None, :]
        A1, A0 = A[yy == 1].astype(float), A[yy == 0].astype(float)
        bal_time = 0.5 * (A1.sum(0) / n_pos + 1 - A0.sum(0) / n_neg)
        i = int(np.argmax(bal_time))
        best = (float(bal_time[i]), "timing_only", float(td[i]), float("nan"))

        has_m = np.isfinite(m)
        if has_m.sum() >= 10:
            tm = _candidates(m)
            wet = m[:, None] >= tm[None, :]
            for mode in ("and", "or"):
                # fields without a moisture value fall back to the timing test
                B = np.where(has_m[:, None], wet, mode == "and")
                B1, B0 = B[yy == 1].astype(float), B[yy == 0].astype(float)
                if mode == "and":
                    tp, fp = A1.T @ B1, A0.T @ B0
                else:
                    tp = n_pos - (1 - A1).T @ (1 - B1)
                    fp = n_neg - (1 - A0).T @ (1 - B0)
                bal = 0.5 * (tp / n_pos + 1 - fp / n_neg)
                a, b = np.unravel_index(int(np.argmax(bal)), bal.shape)
                if bal[a, b] > bal_time[i] + self.min_gain and bal[a, b] > best[0]:
                    best = (float(bal[a, b]), mode, float(td[a]), float(tm[b]))
        self.train_balanced_accuracy_, self.mode_, self.t_time_, self.t_moist_ = best
        return self

    def predict_proba(self, X):
        d = _matrix(X, [self.timing])[:, 0]
        m = _matrix(X, [self.moisture])[:, 0]
        if self.mode_ == "constant":
            return np.full(len(d), self.prevalence_)
        margin = (self.t_time_ - d) / self.scale_time_
        if self.mode_ in ("and", "or"):
            mm = (m - self.t_moist_) / self.scale_moist_
            both = np.minimum(margin, mm) if self.mode_ == "and" else np.maximum(margin, mm)
            margin = np.where(np.isfinite(mm), both, margin)
        with np.errstate(over="ignore", invalid="ignore"):
            p = 1.0 / (1.0 + np.exp(-2.0 * margin))
        return np.where(np.isfinite(d), p, self.p_missing_)

    def describe(self):
        if self.mode_ == "constant":
            return "no usable timing in training data; constant prediction"
        text = f"silage if {self.timing} <= {self.t_time_:.1f}"
        if self.mode_ in ("and", "or"):
            text += f" {self.mode_.upper()} {self.moisture} >= {self.t_moist_:.3f}"
        text += f"; no harvest detected -> score {self.p_missing_:.2f}"
        return text


class _Constant:
    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, a):
        return np.column_stack([np.full(len(a), 1 - self.p), np.full(len(a), self.p)])


class LogisticModel(_Base):
    """Logistic regression on a handful of physical features.

    Missing values are replaced by the training median, then the features
    are standardised; both steps are fit on the training data only.
    """
    name = "logistic"

    def __init__(self, features, C=1.0):
        self.feature_names = list(features)
        self.C = C

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2 or not self.feature_names:
            self.est_ = _Constant(y.mean())
            return self
        self.est_ = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000),
        )
        self.est_.fit(_matrix(X, self.feature_names), y)
        return self

    def predict_proba(self, X):
        if isinstance(self.est_, _Constant):
            return self.est_.predict_proba(X)[:, 1]
        return self.est_.predict_proba(_matrix(X, self.feature_names))[:, 1]

    def describe(self):
        if isinstance(self.est_, _Constant):
            return "single class in training data; constant prediction"
        coef = self.est_[-1].coef_[0]
        parts = [f"{c:+.2f}*z({n})" for n, c in zip(self.feature_names, coef)]
        return "logit = " + " ".join(parts) + f" {self.est_[-1].intercept_[0]:+.2f}"


class GBMModel(_Base):
    """LightGBM on the full feature set. NaN is handled by the trees themselves.

    Hyper-parameters are fixed up front and not tuned.
    """
    name = "gbm"

    def __init__(self, features, seed=0, **params):
        self.feature_names = list(features)
        self.seed = seed
        self.params = {**GBM_PARAMS, **params}

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2 or not self.feature_names:
            self.est_ = _Constant(y.mean())
            return self
        params = dict(self.params)
        params.setdefault("min_child_samples", int(np.clip(len(y) // 50, 5, 20)))
        self.est_ = LGBMClassifier(random_state=self.seed, **params)
        self.est_.fit(_matrix(X, self.feature_names), y)
        return self

    def predict_proba(self, X):
        if isinstance(self.est_, _Constant):
            return self.est_.predict_proba(X)[:, 1]
        return self.est_.predict_proba(_matrix(X, self.feature_names))[:, 1]

    def describe(self):
        return f"LightGBM, {len(self.feature_names)} features, {self.params['n_estimators']} trees"


MODEL_ORDER = ["majority", "rule", "logistic", "gbm"]


def make_factories(columns, variant="relative", exclude=(), use_quality=False, seed=0):
    """name -> zero-argument constructor, for the given feature variant.

    variant      'relative', 'calendar' or 'all' (see features.season.feature_kind)
    exclude      column names to withhold from the logistic and GBM models
    use_quality  also give the GBM the q_* data-quality columns (off by
                 default: cloudiness says something about the region, not
                 about how the field was harvested)
    """
    columns = list(columns)
    exclude = set(exclude)
    gbm_cols = [c for c in signal_features(columns, variant) if c not in exclude]
    if use_quality:
        gbm_cols += [c for c in columns if c.startswith("q_")]
    logit_cols = [c for c in LOGISTIC_FEATURES[variant] if c in columns and c not in exclude]
    timing = RULE_TIMING[variant]
    return {
        "majority": MajorityModel,
        "rule": lambda: RuleModel(timing=timing),
        "logistic": lambda: LogisticModel(logit_cols),
        "gbm": lambda: GBMModel(gbm_cols, seed=seed),
    }
