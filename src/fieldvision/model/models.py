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

from functools import partial

import numpy as np
import pandas as pd
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
    a = np.array(X[names].to_numpy(dtype=float), dtype=float, copy=True)
    a[~np.isfinite(a)] = np.nan
    return a


class _Base:
    name = "base"
    threshold = 0.5
    feature_names: list = []

    def explain(self, X):
        return None

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


class AlwaysGrainModel(_Base):
    """Calls every field grain. Right about 93 % of the time on US corn and useless."""
    name = "always_grain"

    def __init__(self):
        self.feature_names = []

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.zeros(len(X))

    def describe(self):
        return "always grain"


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

    def __init__(self, timing="harv_days_since_peak", moisture=RULE_MOISTURE, min_gain=0.01,
                 name="rule"):
        self.name = name
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

    def explain(self, X):
        """Signed margins of the two tests (positive = towards silage), in robust-scale units."""
        if self.mode_ == "constant":
            return None
        d = _matrix(X, [self.timing])[:, 0]
        m = _matrix(X, [self.moisture])[:, 0]
        out = pd.DataFrame({self.timing: (self.t_time_ - d) / self.scale_time_}, index=X.index)
        if self.mode_ in ("and", "or"):
            out[self.moisture] = (m - self.t_moist_) / self.scale_moist_
        return out

    def describe(self):
        if self.mode_ == "constant":
            return "no usable timing in training data; constant prediction"
        text = f"silage if {self.timing} <= {self.t_time_:.1f}"
        if self.mode_ in ("and", "or"):
            text += f" {self.mode_.upper()} {self.moisture} >= {self.t_moist_:.3f}"
        text += f"; no harvest detected -> score {self.p_missing_:.2f}"
        return text

    _STATE = ("timing", "moisture", "min_gain", "name", "mode_", "t_time_", "t_moist_",
              "scale_time_", "scale_moist_", "p_missing_", "prevalence_", "threshold")

    def to_dict(self):
        """Every learned number, as plain JSON (the rule can be rebuilt from it alone)."""
        out = {}
        for k in self._STATE:
            v = getattr(self, k, None)
            out[k] = None if isinstance(v, float) and not np.isfinite(v) else v
        return out

    @classmethod
    def from_dict(cls, d):
        gain = d.get("min_gain", 0.01)
        m = cls(timing=d["timing"], moisture=d["moisture"],
                min_gain=float("inf") if gain is None else gain, name=d.get("name", "rule"))
        for k in cls._STATE:
            if k in d and k not in ("timing", "moisture", "min_gain", "name"):
                v = d[k]
                setattr(m, k, float("nan") if v is None and k.endswith("_") and k != "mode_" else v)
        return m


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

    def explain(self, X):
        """Per-field contribution of each feature to the log-odds (coef * standardised value)."""
        if isinstance(self.est_, _Constant):
            return None
        a = self.est_[:-1].transform(_matrix(X, self.feature_names))
        return pd.DataFrame(a * self.est_[-1].coef_[0][None, :], columns=self.feature_names,
                            index=X.index)

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

    def explain(self, X):
        """Per-field SHAP-style contributions to the log-odds from LightGBM itself."""
        if isinstance(self.est_, _Constant):
            return None
        c = self.est_.predict(_matrix(X, self.feature_names), pred_contrib=True)
        return pd.DataFrame(np.asarray(c)[:, :-1], columns=self.feature_names, index=X.index)

    def describe(self):
        return f"LightGBM, {len(self.feature_names)} features, {self.params['n_estimators']} trees"


class CalibratedModel(_Base):
    """Platt scaling of another model, learned inside the training data only.

    fit(): an inner cross-validation over the training fields (grouped by
    `block_id` when that column is present, so neighbours stay together)
    gives out-of-fold scores; a one-feature logistic regression on their
    logit maps scores to probabilities; then the base model is refit on all
    training fields. Probabilities are calibrated for the class share of
    the training data (balanced by design); see metrics.prior_shift for
    other shares.
    """

    def __init__(self, make_base, n_inner=3, seed=0):
        self.make_base = make_base
        self.n_inner = n_inner
        self.seed = seed
        self.base_ = make_base()
        self.name = self.base_.name
        self.feature_names = self.base_.feature_names

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
        y = np.asarray(y).astype(int)
        self.train_prevalence_ = float(y.mean())
        self.platt_ = None
        n_min = int(min(y.sum(), len(y) - y.sum()))
        if n_min >= self.n_inner * 2:
            oof = np.full(len(y), np.nan)
            groups = X["block_id"].astype(str).to_numpy() if "block_id" in X.columns else None
            if groups is not None and len(np.unique(groups)) >= self.n_inner:
                cv = StratifiedGroupKFold(self.n_inner, shuffle=True, random_state=self.seed)
                splits = cv.split(np.zeros(len(y)), y, groups)
            else:
                cv = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed)
                splits = cv.split(np.zeros(len(y)), y)
            for tr, te in splits:
                if len(np.unique(y[tr])) < 2:
                    continue
                oof[te] = self.make_base().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
            ok = np.isfinite(oof)
            if ok.sum() > 10 and len(np.unique(y[ok])) == 2:
                self.platt_ = LogisticRegression(C=1e4, max_iter=1000)
                self.platt_.fit(self._logit(oof[ok])[:, None], y[ok])
        self.base_ = self.make_base().fit(X, y)
        return self

    def raw_proba(self, X):
        return self.base_.predict_proba(X)

    def predict_proba(self, X):
        raw = self.base_.predict_proba(X)
        if self.platt_ is None:
            return raw
        return self.platt_.predict_proba(self._logit(raw)[:, None])[:, 1]

    def explain(self, X):
        return self.base_.explain(X)

    def describe(self):
        text = self.base_.describe()
        if self.platt_ is None:
            return text + " (calibration skipped: too few training fields)"
        a, b = float(self.platt_.coef_[0][0]), float(self.platt_.intercept_[0])
        return text + f"; Platt calibration p = sigmoid({a:.2f}*logit(score) {b:+.2f})"


class CompositeModel(_Base):
    """Average of the calibrated probabilities of the low-capacity members
    (timing rule, timing+moisture rule, six-feature logistic).

    Chosen over a majority vote on Austrian spatial CV (ROC-AUC 0.766 vs 0.739). Members are
    fit inside fit() like any other model. `disagreement(X)` flags fields where the members
    split their votes and their probabilities are at least `spread_limit` apart; together
    with the low-evidence flag that is when the composite should abstain.
    """
    name = "composite"

    def __init__(self, makers, spread_limit=0.5, seed=0):
        self.makers = dict(makers)
        self.spread_limit = spread_limit
        self.seed = seed
        names = []
        for mk in self.makers.values():
            names += list(mk().feature_names)
        self.feature_names = list(dict.fromkeys(names))

    @classmethod
    def from_members(cls, members, spread_limit=0.5):
        """Build from already fitted members (e.g. the frozen models of a run), no refit."""
        obj = cls.__new__(cls)
        obj.makers = {}
        obj.spread_limit = spread_limit
        obj.seed = 0
        obj.members_ = dict(members)
        names = []
        for m in obj.members_.values():
            names += list(m.feature_names)
        obj.feature_names = list(dict.fromkeys(names))
        return obj

    def fit(self, X, y):
        self.members_ = {n: CalibratedModel(mk, seed=self.seed).fit(X, y)
                         for n, mk in self.makers.items()}
        return self

    def member_proba(self, X):
        return pd.DataFrame({n: m.predict_proba(X) for n, m in self.members_.items()}, index=X.index)

    def predict_proba(self, X):
        return self.member_proba(X).mean(axis=1).to_numpy()

    def disagreement(self, X):
        P = self.member_proba(X)
        votes = (P >= 0.5).sum(axis=1).to_numpy()
        spread = (P.max(axis=1) - P.min(axis=1)).to_numpy()
        return (votes > 0) & (votes < P.shape[1]) & (spread >= self.spread_limit)

    def explain(self, X):
        parts = []
        for n, m in self.members_.items():
            e = m.explain(X)
            if e is not None:
                parts.append(e.add_prefix(f"{n}:") / len(self.members_))
        return pd.concat(parts, axis=1) if parts else None

    def describe(self):
        return "mean of calibrated probabilities of: " + "; ".join(
            f"{n} [{m.describe()}]" for n, m in self.members_.items())


MODEL_ORDER = ["majority", "always_grain", "timing_rule", "rule", "logistic", "gbm", "composite"]


def make_factories(columns, variant="relative", exclude=(), use_quality=False, seed=0,
                   calibrate=True):
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
    # partials, not lambdas: the calibrated wrapper keeps its base factory and must pickle
    base = {
        # the organisers' suggested baseline: calendar date of the sustained NDVI
        # decline, one threshold, nothing else
        "timing_rule": partial(RuleModel, timing="harv_mid_doy", min_gain=float("inf"),
                               name="timing_rule"),
        "rule": partial(RuleModel, timing=timing),
        "logistic": partial(LogisticModel, logit_cols),
        "gbm": partial(GBMModel, gbm_cols, seed=seed),
    }
    out = {"majority": MajorityModel, "always_grain": AlwaysGrainModel}
    for name, make in base.items():
        out[name] = (lambda m=make: CalibratedModel(m, seed=seed)) if calibrate else make
    members = {k: base[k] for k in ("timing_rule", "rule", "logistic")}
    out["composite"] = partial(CompositeModel, members, seed=seed)
    return out
