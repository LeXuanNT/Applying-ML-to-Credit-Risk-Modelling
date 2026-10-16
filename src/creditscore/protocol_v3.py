"""Experimental protocol exactly as written in thesis draft v3 (Chapters 1-3).

Ch.3 3.3  stratified split 70/15/15; missing indicators kept when a logistic test of
          BAD on the indicator is significant (Ch.2 2.3, p < 0.05); joint (iterative) imputation
          for CLAGE-CLNO-DELINQ-NINQ; median / mode for the rest; one-hot (drop first);
          StandardScaler; SMOTE / ADASYN.
Ch.2 2.3  Bayesian Optimisation chooses imputer + oversampler + model hyperparameters together.
Revised   imputer options optimized / simple / simple_ind / native (NaN kept for XGBoost, LightGBM,
          with class weighting instead of SMOTE/ADASYN in that sub-space); see scripts/missing_study.py.
Ch.3 3.4  YOJ binned (Unemployed, 0-5, 5-10, 10-20, 20-40 years); degree-2 polynomial
          features of LOAN, VALUE, CLAGE replace the originals; WoE (optimal binning, Ch.2 2.5)
          + IV for logistic regression.
Ch.1/2/3  models: Logistic (baseline, WoE), Decision Tree, Random Forest, Gradient Boosting
          (XGBoost, LightGBM), SVM, Neural Network (MLP), Gaussian Naive Bayes, KNN.
Ch.1 1.2  impact of each imbalance technique (SMOTE, ROS, RUS; Ch.2 2.4 class_weight;
          Ch.3 ADASYN).
Ch.3 3.7  BayesSearchCV, 50 iterations, 10-fold stratified CV, objective F1.
Ch.2 2.2  Accuracy, Precision, Recall, F1, MCC, ROC/AUC, confusion matrix.
Ch.3 3.5  selection (creditscore.selection: DeLong + average rank) -> final model, re-trained on train + validation, evaluated on test.
Ch.3 3.6  feature importance.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

from .data import CAT_FEATURES, NUM_FEATURES, RAW_FEATURES, TARGET
from .features import WoEBinner

warnings.filterwarnings("ignore")

JOINT = ["CLAGE", "CLNO", "DELINQ", "NINQ"]          # Ch.3: structured-missing group
POLY = ["LOAN", "VALUE", "CLAGE"]                    # Ch.3 3.4
YOJ_EDGES = [-np.inf, 0, 5, 10, 20, np.inf]          # Ch.3 3.4 (last bin "20-40": max YOJ is 41)
YOJ_LABELS = ["Unemployed", "0-5", "5-10", "10-20", "20-40"]

LABELS = {"LR": "Logistic Regression (WoE)", "DT": "Decision Tree", "RF": "Random Forest",
          "XGB": "XGBoost", "LGBM": "LightGBM", "SVM": "SVM", "MLP": "Neural Network (MLP)",
          "GNB": "Gaussian Naive Bayes", "KNN": "K-Nearest Neighbors"}
MODELS = list(LABELS)


# ------------------------------------------------------------------ Ch.2 2.3 test
def missing_indicator_tests(train: pd.DataFrame) -> pd.DataFrame:
    """Logistic regression P(BAD=1) = sigma(b0 + b1*M_i) for each feature with missing values."""
    rows = []
    for c in RAW_FEATURES:
        m = train[c].isna().astype(float)
        if m.sum() == 0:
            continue
        res = sm.Logit(train[TARGET].astype(float), sm.add_constant(m)).fit(disp=0)
        rows.append({"feature": c, "n_missing": int(m.sum()), "beta1": res.params.iloc[1],
                     "odds_ratio": float(np.exp(res.params.iloc[1])), "p_value": res.pvalues.iloc[1],
                     "keep_indicator": res.pvalues.iloc[1] < 0.05})
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# ------------------------------------------------------------------ preprocessing (Ch.3 3.3-3.4)
def _poly2(Z: pd.DataFrame) -> pd.DataFrame:
    """Degree-2 polynomial terms of POLY (same names/order as sklearn PolynomialFeatures); NaN propagates."""
    out = {c: Z[c] for c in POLY}
    for i, a in enumerate(POLY):
        for b in POLY[i:]:
            out[f"{a}^2" if a == b else f"{a} {b}"] = Z[a] * Z[b]
    return pd.DataFrame(out, index=Z.index)


IMPUTERS = ["optimized", "simple", "simple_ind", "native"]


class V3Preprocessor(BaseEstimator, TransformerMixin):
    """Missing-data strategy (Ch.2 2.3 / Ch.3 3.3, revised):
    'optimized'  significant missing indicators + joint iterative imputation (CLAGE-CLNO-DELINQ-NINQ) + median/mode
    'simple'     median / mode only
    'simple_ind' median / mode + significant missing indicators
    'native'     no imputation: numeric NaN left for the learner (XGBoost / LightGBM), categorical NaN -> 'Missing'
    Then YOJ bins, polynomial terms, one-hot (drop first), scaling."""

    def __init__(self, imputer: str = "optimized", seed: int = 42):
        self.imputer = imputer
        self.seed = seed

    def fit(self, X, y=None):
        X = X[RAW_FEATURES]
        self.indicators_ = []
        if self.imputer in ("optimized", "simple_ind") and y is not None:
            t = missing_indicator_tests(X.assign(**{TARGET: np.asarray(y)}))
            self.indicators_ = t.loc[t.keep_indicator, "feature"].tolist()
        if self.imputer == "optimized":
            self.joint_ = IterativeImputer(random_state=self.seed, max_iter=15,
                                           initial_strategy="median").fit(X[JOINT])
        self.medians_ = X[NUM_FEATURES].median()
        self.modes_ = {c: X[c].mode().iloc[0] for c in CAT_FEATURES}
        Z = self._numeric(X)
        D = self._assemble(X, Z, fit=True)
        self.scaler_ = StandardScaler().fit(D)
        self.columns_ = list(D.columns)
        return self

    def _numeric(self, X):
        Z = X[NUM_FEATURES].copy()
        if self.imputer == "native":
            return Z
        if self.imputer == "optimized":
            Z[JOINT] = np.clip(self.joint_.transform(X[JOINT]), 0, None)
        return Z.fillna(self.medians_)

    def _assemble(self, X, Z, fit=False):
        parts = [_poly2(Z[POLY]), Z[[c for c in NUM_FEATURES if c not in POLY + ["YOJ"]]]]
        for c in self.indicators_:
            parts.append(X[c].isna().astype(int).rename(f"{c}_MISSING"))
        yoj = pd.cut(Z["YOJ"], YOJ_EDGES, labels=YOJ_LABELS).astype(object)
        native = self.imputer == "native"
        cats = {c: X[c].fillna("Missing" if native else self.modes_[c]) for c in CAT_FEATURES}
        cats["YOJ_BIN"] = yoj.where(Z["YOJ"].notna(), "Missing").astype(str)
        if fit:
            self.levels_ = {c: sorted(v.unique(), key=str) for c, v in cats.items()}
            self.levels_["YOJ_BIN"] = [v for v in YOJ_LABELS + ["Missing"] if v in set(cats["YOJ_BIN"])]
        for c, v in cats.items():
            for lev in self.levels_[c][1:]:
                parts.append((v == lev).astype(int).rename(f"{c}_{lev}"))
        return pd.concat(parts, axis=1)

    def transform(self, X):
        X = X[RAW_FEATURES]
        D = self._assemble(X, self._numeric(X)).reindex(columns=self.columns_, fill_value=0)
        return pd.DataFrame(self.scaler_.transform(D), columns=self.columns_, index=X.index)


class WoEFeatures(BaseEstimator, TransformerMixin):
    """WoE transformation + IV screening (IV >= 0.02) for logistic regression (Ch.3 3.4)."""

    def __init__(self, min_iv: float = 0.02):
        self.min_iv = min_iv

    def fit(self, X, y):
        self.binner_ = WoEBinner().fit(X[RAW_FEATURES], pd.Series(np.asarray(y), index=X.index))
        iv = self.binner_.iv_table()
        self.features_ = iv.loc[iv.iv >= self.min_iv, "feature"].tolist()
        return self

    def transform(self, X):
        return self.binner_.transform(X[RAW_FEATURES])[self.features_]


# ------------------------------------------------------------------ models
def _estimator(name, seed, class_weight=None):
    return {
        "LR": LogisticRegression(max_iter=5000, class_weight=class_weight),
        "DT": DecisionTreeClassifier(random_state=seed, class_weight=class_weight),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight=class_weight),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=seed, n_jobs=2,
                             eval_metric="logloss", tree_method="hist",
                             scale_pos_weight=4.0 if class_weight else 1.0),
        "LGBM": LGBMClassifier(n_estimators=300, random_state=seed, verbose=-1, n_jobs=2, class_weight=class_weight),
        "SVM": SVC(random_state=seed, class_weight=class_weight),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, early_stopping=True, random_state=seed),
        "GNB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
    }[name]


class Resampler(BaseEstimator):
    """Oversampler chosen by name, so Bayesian optimisation can search over it."""

    def __init__(self, kind: str = "adasyn", seed: int = 42):
        self.kind = kind
        self.seed = seed

    def fit_resample(self, X, y):
        cls = {"smote": SMOTE, "adasyn": ADASYN, "ros": RandomOverSampler, "rus": RandomUnderSampler,
               "smote_tomek": SMOTETomek, "smote_enn": SMOTEENN}
        if self.kind == "none":
            return X, y
        return cls[self.kind](random_state=self.seed).fit_resample(X, y)


HYBRIDS = ["smote_tomek", "smote_enn"]                # hybrid over+under sampling (Batista et al. 2004)
SAMPLERS = {k: (lambda s, k=k: Resampler(k, s)) for k in ["smote", "adasyn", "ros", "rus", "none"] + HYBRIDS}


def build(name: str, sampler: str = "adasyn", imputer: str = "optimized", seed: int = 42,
          class_weight=None) -> ImbPipeline:
    prep = WoEFeatures() if name == "LR" else V3Preprocessor(imputer=imputer, seed=seed)
    return ImbPipeline([("prep", prep), ("sampler", SAMPLERS[sampler](seed)),
                        ("clf", _estimator(name, seed, class_weight))])


def space(name: str) -> dict:
    common = {"sampler__kind": Categorical(["smote", "adasyn"] + HYBRIDS)}
    # plain median imputation ("simple") is excluded: tree models read the imputed median as a hidden missing flag,
    # so a genuine applicant near the median is scored as if the field were missing (found when building the app).
    imp = {"prep__imputer": Categorical(["optimized", "simple_ind"])}
    s = {
        "LR": {"clf__C": Real(1e-3, 1e2, prior="log-uniform")},
        "DT": {"clf__max_depth": Integer(2, 15), "clf__min_samples_leaf": Integer(1, 100),
               "clf__criterion": Categorical(["gini", "entropy"])},
        "RF": {"clf__n_estimators": Integer(100, 500), "clf__max_depth": Integer(3, 30),
               "clf__min_samples_leaf": Integer(1, 20), "clf__max_features": Real(0.1, 0.8)},
        "XGB": {"clf__n_estimators": Integer(100, 600), "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "clf__max_depth": Integer(2, 10), "clf__subsample": Real(0.6, 1.0),
                "clf__colsample_bytree": Real(0.5, 1.0), "clf__min_child_weight": Integer(1, 10)},
        "LGBM": {"clf__n_estimators": Integer(100, 600), "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                 "clf__num_leaves": Integer(8, 128), "clf__min_child_samples": Integer(5, 60),
                 "clf__colsample_bytree": Real(0.5, 1.0)},
        "SVM": {"clf__C": Real(1e-2, 1e3, prior="log-uniform"), "clf__gamma": Real(1e-4, 1e1, prior="log-uniform")},
        "MLP": {"clf__hidden_layer_sizes": Integer(8, 256), "clf__alpha": Real(1e-6, 1e-1, prior="log-uniform"),
                "clf__learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform")},
        "GNB": {"clf__var_smoothing": Real(1e-12, 1e-1, prior="log-uniform")},
        "KNN": {"clf__n_neighbors": Integer(1, 60), "clf__weights": Categorical(["uniform", "distance"]),
                "clf__p": Integer(1, 2)},
    }[name]
    return {**common, **({} if name == "LR" else imp), **s}


NATIVE_MODELS = ("XGB", "LGBM")          # learners with built-in missing-value handling
NATIVE_SHARE = 0.3                       # share of the BO budget spent on the native-NaN sub-space


def spaces(name: str, n_iter: int):
    """BO search spaces. For XGBoost / LightGBM a second sub-space keeps NaN (no imputation); because
    SMOTE/ADASYN cannot interpolate NaN, that sub-space handles imbalance with class weighting instead."""
    base = space(name)
    if name not in NATIVE_MODELS:
        return [(base, n_iter)]
    nat = {k: v for k, v in base.items() if k not in ("sampler__kind", "prep__imputer")}
    nat["prep__imputer"] = Categorical(["native"])
    nat["sampler__kind"] = Categorical(["none"])
    nat.update({"clf__scale_pos_weight": Categorical([4.0])} if name == "XGB"
               else {"clf__class_weight": Categorical(["balanced"])})
    k = max(1, round(n_iter * NATIVE_SHARE))
    return [(base, n_iter - k), (nat, k)]


# ------------------------------------------------------------------ metrics (Ch.2 2.2)
def metrics(y, yhat, score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "mcc": matthews_corrcoef(y, yhat),
            "auc": roc_auc_score(y, score), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def score_of(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


# ------------------------------------------------------------------ Ch.1: imbalance techniques
def imbalance_comparison(X, y, models=MODELS, seed=42, n_splits=5) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    rows = []
    for m in models:
        for st in ["none", "class_weight", "ros", "rus", "smote", "adasyn"] + HYBRIDS:
            if st == "class_weight" and m in ("MLP", "GNB", "KNN"):
                continue
            pipe = build(m, "none" if st == "class_weight" else st, seed=seed,
                         class_weight="balanced" if st == "class_weight" else None)
            t0 = time.time()
            r = cross_validate(pipe, X, y, cv=cv, scoring=["f1", "recall", "precision", "roc_auc"])
            rows.append({"model": m, "strategy": st, "f1": r["test_f1"].mean(), "recall": r["test_recall"].mean(),
                         "precision": r["test_precision"].mean(), "auc": r["test_roc_auc"].mean(),
                         "seconds": time.time() - t0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Ch.3 3.7 tuning
def tune(name, X, y, n_iter=50, n_splits=10, seed=42):
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    opt = BayesSearchCV(build(name, seed=seed), spaces(name, n_iter), n_iter=n_iter, cv=cv, scoring="f1",
                        random_state=seed, n_jobs=1, refit=False, return_train_score=False)
    t0 = time.time()
    opt.fit(X, y)
    secs = time.time() - t0
    cvr = pd.DataFrame(opt.cv_results_)
    best = opt.best_index_
    fold_cols = [c for c in cvr.columns if c.startswith("split") and c.endswith("_test_score")]
    params = dict(opt.best_params_)
    info = {"model": name, "cv_f1": opt.best_score_, "cv_f1_std": float(cvr.loc[best, fold_cols].std()),
            "best_params": params, "tuning_seconds": secs, "n_iter": n_iter, "cv_folds": n_splits}
    # best CV F1 reached by each imputer / sampler option (Ch.2 2.3: BO chooses imputer and oversampler)
    opts = []
    for k in ("param_prep__imputer", "param_sampler__kind"):
        if k in cvr:
            g = cvr.assign(opt=cvr[k].astype(str))
            for o, s in g.groupby("opt")["mean_test_score"].agg(["max", "size"]).iterrows():
                opts.append({"model": name, "parameter": k.replace("param_", ""), "option": o,
                             "best_cv_f1": s["max"], "n_trials": int(s["size"])})
    return opt.best_params_, info, pd.DataFrame(opts)


def fit_with(name, params, X, y, seed=42):
    est = build(name, seed=seed).set_params(**params)
    t0 = time.time()
    est.fit(X, y)
    return est, time.time() - t0


# ------------------------------------------------------------------ Ch.3 3.5 model selection
# Model selection (Ch.3 3.5) is in creditscore.selection: DeLong eligibility + equal-weight average rank
# (Lessmann et al., 2015). The earlier author-set weighted ranking was removed because its weights had no source.


# ------------------------------------------------------------------ Ch.3 3.6 feature importance
def feature_importance(model, name: str) -> pd.DataFrame:
    clf, prep = model.named_steps["clf"], model.named_steps["prep"]
    cols = prep.features_ if name == "LR" else prep.columns_
    if name == "LR":
        v = clf.coef_[0]
        return pd.DataFrame({"feature": cols, "coefficient": v, "importance": np.abs(v)}) \
            .sort_values("importance", ascending=False).reset_index(drop=True)
    if name == "LGBM":
        v = clf.booster_.feature_importance(importance_type="gain")
    elif name == "XGB":
        g = clf.get_booster().get_score(importance_type="total_gain")  # total, comparable with LightGBM gain
        v = np.array([g.get(f"f{k}", g.get(c, 0.0)) for k, c in enumerate(cols)])
    elif hasattr(clf, "feature_importances_"):
        v = clf.feature_importances_
    else:
        return pd.DataFrame()
    v = np.asarray(v, float)
    return pd.DataFrame({"feature": cols, "importance": v / v.sum()}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)
