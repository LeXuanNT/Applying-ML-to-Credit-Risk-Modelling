"""Stage 3 - PD models.

Champion baseline: logistic regression on WoE features (the bank scorecard).
Challengers: logistic regression on raw features (+ degree-2 terms, Chapter 3),
Decision Tree, Random Forest, XGBoost, LightGBM, MLP, SVM.

All challengers are full pipelines (preprocessing -> [scaling] -> [resampling] -> model),
so imputation and resampling are re-fitted inside every CV fold and never see
validation / test rows. Tuning: Bayesian optimisation (scikit-optimize BayesSearchCV)
with stratified K-fold CV on the training set, objective = ROC-AUC.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

from .features import HMEQPreprocessor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_LABELS = {
    "LR_WoE": "Logistic (WoE scorecard)",
    "LR_raw": "Logistic (raw + poly)",
    "DT": "Decision Tree",
    "RF": "Random Forest",
    "XGB": "XGBoost",
    "LGBM": "LightGBM",
    "MLP": "MLP",
    "SVM": "SVM (RBF)",
}
CHALLENGERS = ["LR_raw", "DT", "RF", "XGB", "LGBM", "MLP", "SVM"]
STRATEGIES = ["none", "class_weight", "smote", "adasyn"]
SCALED = {"LR_raw", "MLP", "SVM"}


class PolyTerms(BaseEstimator, TransformerMixin):
    """Squares and pairwise products of selected columns (Chapter 3: LOAN, VALUE, CLAGE)."""

    def __init__(self, cols=("LOAN", "VALUE", "CLAGE")):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        c = list(self.cols)
        for i, a in enumerate(c):
            for b in c[i:]:
                X[f"{a}*{b}"] = X[a] * X[b]
        return X


def _base_estimator(name: str, seed: int, class_weight: bool, pos_weight: float):
    cw = "balanced" if class_weight else None
    if name == "LR_raw":
        return LogisticRegression(max_iter=5000, class_weight=cw)
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed, class_weight=cw, min_samples_leaf=20)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                      class_weight=cw, min_samples_leaf=2)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.9,
                             colsample_bytree=0.8, random_state=seed, n_jobs=2,
                             eval_metric="logloss", tree_method="hist",
                             scale_pos_weight=pos_weight if class_weight else 1.0)
    if name == "LGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              random_state=seed, class_weight=cw, verbose=-1, n_jobs=2)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=64, alpha=1e-3, max_iter=600,
                             early_stopping=True, random_state=seed)
    if name == "SVM":
        return SVC(C=1.0, gamma="scale", class_weight=cw, random_state=seed)
    raise KeyError(name)


def supports_class_weight(name: str) -> bool:
    return name != "MLP"


def build_pipeline(name: str, strategy: str = "class_weight", seed: int = 42,
                   pos_weight: float = 4.0) -> ImbPipeline:
    steps = [("prep", HMEQPreprocessor(seed=seed))]
    if name == "LR_raw":
        steps.append(("poly", PolyTerms()))
    if name in SCALED:
        steps.append(("scale", StandardScaler()))
    if strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=seed)))
    elif strategy == "adasyn":
        steps.append(("sampler", ADASYN(random_state=seed)))
    steps.append(("clf", _base_estimator(name, seed, strategy == "class_weight", pos_weight)))
    return ImbPipeline(steps)


def search_space(name: str) -> dict:
    s = {
        "LR_raw": {"clf__C": Real(1e-3, 1e2, prior="log-uniform")},
        "DT": {"clf__max_depth": Integer(2, 12), "clf__min_samples_leaf": Integer(5, 100),
               "clf__criterion": Categorical(["gini", "entropy"])},
        "RF": {"clf__n_estimators": Integer(200, 600), "clf__max_depth": Integer(4, 24),
               "clf__min_samples_leaf": Integer(1, 20), "clf__max_features": Real(0.2, 0.8)},
        "XGB": {"clf__n_estimators": Integer(100, 600),
                "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "clf__max_depth": Integer(2, 8), "clf__subsample": Real(0.6, 1.0),
                "clf__colsample_bytree": Real(0.5, 1.0), "clf__min_child_weight": Integer(1, 10),
                "clf__reg_lambda": Real(1e-3, 10, prior="log-uniform")},
        "LGBM": {"clf__n_estimators": Integer(100, 600),
                 "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                 "clf__num_leaves": Integer(8, 64), "clf__min_child_samples": Integer(5, 60),
                 "clf__subsample": Real(0.6, 1.0), "clf__subsample_freq": Categorical([1]),
                 "clf__colsample_bytree": Real(0.5, 1.0),
                 "clf__reg_lambda": Real(1e-3, 10, prior="log-uniform")},
        "MLP": {"clf__hidden_layer_sizes": Integer(8, 128),
                "clf__alpha": Real(1e-5, 1e-1, prior="log-uniform"),
                "clf__learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform")},
        "SVM": {"clf__C": Real(1e-2, 1e2, prior="log-uniform"),
                "clf__gamma": Real(1e-4, 1.0, prior="log-uniform")},
    }
    return s[name]


def scores_of(model, X) -> np.ndarray:
    """PD-like score: predict_proba if available, else decision_function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


# ------------------------------------------------------------ imbalance ablation
def make_cv(n_splits: int, seed: int, groups=None):
    """Stratified K-fold, or StratifiedGroupKFold when near-duplicate groups are given.
    Returns (cv, fit_kwargs)."""
    if groups is None:
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed), {}
    return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed), {"groups": np.asarray(groups)}


def imbalance_ablation(X: pd.DataFrame, y: pd.Series, models=CHALLENGERS, seed: int = 42,
                       n_splits: int = 5, groups=None) -> pd.DataFrame:
    """CV on train only: AUC, PR-AUC and F1@0.5 for each model x imbalance strategy."""
    cv, kw = make_cv(n_splits, seed, groups)
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    rows = []
    for m in models:
        for st in STRATEGIES:
            if st == "class_weight" and not supports_class_weight(m):
                continue
            pipe = build_pipeline(m, st, seed, pos_weight)
            method = "decision_function" if m == "SVM" else "predict_proba"
            t0 = time.time()
            s = cross_val_predict(pipe, X, y, cv=cv, method=method, **kw)
            s = s[:, 1] if s.ndim == 2 else s
            thr = 0.0 if m == "SVM" else 0.5
            rows.append({"model": m, "strategy": st, "cv_auc": roc_auc_score(y, s),
                         "cv_pr_auc": average_precision_score(y, s),
                         "cv_f1_at_default_threshold": f1_score(y, (s > thr).astype(int)),
                         "seconds": time.time() - t0})
    return pd.DataFrame(rows)


def choose_strategy(ablation: pd.DataFrame, tol: float = 0.003) -> dict[str, str]:
    """Rule fixed before seeing test data: best CV AUC; if within `tol` of the best,
    prefer the simpler option (none < class_weight < smote < adasyn)."""
    order = {s: i for i, s in enumerate(STRATEGIES)}
    out = {}
    for m, g in ablation.groupby("model"):
        best = g["cv_auc"].max()
        ok = g[g["cv_auc"] >= best - tol].copy()
        ok["rank"] = ok["strategy"].map(order)
        out[m] = ok.sort_values("rank").iloc[0]["strategy"]
    return out


# ------------------------------------------------------------ tuning
def tune(name: str, X: pd.DataFrame, y: pd.Series, strategy: str, n_iter: int = 30,
         n_splits: int = 5, seed: int = 42, groups=None):
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    pipe = build_pipeline(name, strategy, seed, pos_weight)
    cv, kw = make_cv(n_splits, seed, groups)
    opt = BayesSearchCV(pipe, search_space(name), n_iter=n_iter, cv=cv, scoring="roc_auc",
                        random_state=seed, n_jobs=1, refit=False)
    t0 = time.time()
    opt.fit(X, y, **kw)
    best = clone(pipe).set_params(**opt.best_params_)
    if name == "SVM":
        best.set_params(clf__probability=True)
    best.fit(X, y)
    info = {"model": name, "strategy": strategy, "cv_auc": opt.best_score_,
            "best_params": {k.replace("clf__", ""): (round(v, 5) if isinstance(v, float) else v)
                            for k, v in opt.best_params_.items()},
            "tuning_seconds": time.time() - t0, "n_iter": n_iter, "cv_folds": n_splits}
    return best, info


# ------------------------------------------------------------ WoE scorecard
def fit_scorecard(W_train: pd.DataFrame, y: pd.Series, seed: int = 42, n_splits: int = 5, groups=None):
    """Logistic on WoE. C chosen by CV AUC on a small grid (no class weights, so the
    intercept stays on the real default rate and PDs remain calibrated)."""
    cv, kw = make_cv(n_splits, seed, groups)
    res = []
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 10.0]:
        s = cross_val_predict(LogisticRegression(C=C, max_iter=5000), W_train, y, cv=cv,
                              method="predict_proba", **kw)[:, 1]
        res.append((roc_auc_score(y, s), C))
    best_auc, best_C = max(res)
    lr = LogisticRegression(C=best_C, max_iter=5000).fit(W_train, y)
    return lr, {"model": "LR_WoE", "strategy": "none", "cv_auc": best_auc,
                "best_params": {"C": best_C}, "n_features": W_train.shape[1]}
