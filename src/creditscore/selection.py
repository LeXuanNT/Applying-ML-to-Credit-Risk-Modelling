"""Model selection without author-set weights (replaces protocol_v3.weighted_ranking).

The thesis framework has two stages: (1) compare the algorithms and choose one champion, (2) deploy the champion in
a web application that shows a credit officer, for ONE applicant, the probability of default, the features that
drive it and a recommendation. Stage 2 therefore sets a requirement on the champion before any result is seen.
The rule uses the VALIDATION split only (test stays untouched):

  Step 0  Explainability requirement (pass/fail). The model must (a) output a probability of default and
          (b) give exact, deterministic per-applicant feature contributions that add up to the prediction:
          coefficients x WoE for logistic regression, additive log-odds for Gaussian naive Bayes, TreeSHAP for
          tree models (Lundberg et al., 2020). RBF-SVM, KNN and MLP only admit sampling approximations
          (KernelSHAP) and fail. Rationale: Rudin (2019); adverse-action reasons (CFPB Circular 2022-03).
  Step 1  Discrimination. Each eligible model is compared with the highest-AUC eligible model by DeLong's test
          (DeLong, DeLong & Clarke-Pearson, 1988); models significantly worse (two-sided p < 0.05) are dropped.
          Discriminatory power is the first validation component for rating systems (BCBS WP14, 2005).
  Step 2  Choice. Remaining models are ranked on AUC, KS, PR-AUC, Brier score (PD after Platt scaling) and MCC
          and aggregated by the equal-weight average rank (Lessmann, Baesens, Seow & Thomas, 2015).
          Ties are broken by the 10-fold cross-validated AUC on the training split (4,172 loans, a less noisy
          estimate than the 894 validation loans). NOTE: this tie-break was chosen after the RF/LightGBM tie was
          observed; the thesis states this.

Brier needs probabilities on a common scale. Every model's score is mapped to a PD with Platt scaling
(Platt, 1999) fitted on 10-fold out-of-fold scores of the TRAINING split, so the validation split is not used
for fitting anything.

Population stability (PSI; thresholds 0.10 / 0.25 from Baesens, Roesch & Scheule, 2016) is NOT a selection
filter: with a random cross-sectional split there is no population drift to detect, and PSI between scores of
differently fitted models (training out-of-fold vs validation) is dominated by binning artefacts (point masses of
SVM margins, the 9 distinct leaf values of a depth-4 tree). PSI is reported between the validation and test
samples scored by the same model (Chapter 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_curve
from sklearn.model_selection import StratifiedKFold

ALPHA = 0.05
MEASURES = {"auc": True, "ks": True, "pr_auc": True, "brier": False, "mcc": True}   # True = higher is better
EXPLAINABLE = {"LR", "DT", "GNB", "RF", "XGB", "LGBM"}      # exact per-applicant attributions + PD output


# ------------------------------------------------------------------ DeLong (fast O(n log n) form)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _delong_components(y, scores):
    """scores: (k, n). Returns AUCs (k,) and covariance matrix (k, k)."""
    y = np.asarray(y)
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r], ty[r] = _midrank(pos[r]), _midrank(neg[r])
        tz[r] = _midrank(np.r_[pos[r], neg[r]])
    auc = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return auc, cov


def delong_ci(y, s, alpha=ALPHA):
    auc, cov = _delong_components(y, np.atleast_2d(s))
    se = np.sqrt(cov[0, 0])
    z = norm.ppf(1 - alpha / 2)
    return float(auc[0]), float(auc[0] - z * se), float(min(1.0, auc[0] + z * se))


def delong_test(y, s1, s2):
    """Two-sided DeLong test of AUC(s1) == AUC(s2) on the same observations. Returns (diff, p-value)."""
    auc, cov = _delong_components(y, np.vstack([s1, s2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = auc[0] - auc[1]
    if var <= 0:
        return float(diff), 1.0
    return float(diff), float(2 * norm.sf(abs(diff) / np.sqrt(var)))


# ------------------------------------------------------------------ other measures
def ks(y, s):
    f, t, _ = roc_curve(y, s)
    return float(np.max(t - f))


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _platt_input(s):
    s = np.asarray(s, float)
    if s.min() >= 0 and s.max() <= 1:                 # probabilities -> logit, decision scores used as they are
        s = np.clip(s, 1e-6, 1 - 1e-6)
        s = np.log(s / (1 - s))
    return s.reshape(-1, 1)


def oof_scores(est, X, y, score_fn, n_splits=10, seed=42):
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    s = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        s[va] = score_fn(clone(est).fit(X.iloc[tr], y[tr]), X.iloc[va])
    return s


def platt(oof, y_train):
    return LogisticRegression(C=1e6, max_iter=1000).fit(_platt_input(oof), y_train)


# ------------------------------------------------------------------ the rule
def fold_auc_std(oof_s, y_train, n_splits=10, seed=42):
    from sklearn.metrics import roc_auc_score
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    a = [roc_auc_score(y_train[va], oof_s[va]) for _, va in cv.split(np.zeros(len(y_train)), y_train)]
    return float(np.mean(a)), float(np.std(a, ddof=1))


def select_champion(y_valid, valid_scores: dict, valid_pred: dict, oof: dict, y_train,
                    require_explainable: bool = True) -> pd.DataFrame:
    """valid_scores: model -> raw validation scores; valid_pred: model -> 0/1 predictions at the model's own
    threshold; oof: model -> training out-of-fold scores. Returns one row per model with every step.
    require_explainable=False gives the sensitivity result without step 0."""
    gate = {m: (m in EXPLAINABLE) or not require_explainable for m in valid_scores}
    pool = [m for m in valid_scores if gate[m]]
    best = max(pool, key=lambda m: delong_ci(y_valid, valid_scores[m])[0])
    rows = []
    for m, s in valid_scores.items():
        auc, lo, hi = delong_ci(y_valid, s)
        diff, p = (0.0, 1.0) if m == best else delong_test(y_valid, s, valid_scores[best])
        cal = platt(oof[m], y_train)
        pd_valid = cal.predict_proba(_platt_input(s))[:, 1]
        rows.append({"model": m, "explainable": m in EXPLAINABLE, "step0_pass": gate[m],
                     "auc": auc, "auc_lo": lo, "auc_hi": hi, "gini": 2 * auc - 1,
                     "delta_auc_vs_best": diff, "p_delong": p, "ks": ks(y_valid, s),
                     "pr_auc": average_precision_score(y_valid, s), "brier": brier_score_loss(y_valid, pd_valid),
                     "mcc": matthews_corrcoef(y_valid, valid_pred[m]),
                     **dict(zip(["cv_auc", "cv_auc_std"], fold_auc_std(np.asarray(oof[m]), np.asarray(y_train))))})
    d = pd.DataFrame(rows).set_index("model")
    d["step1_pass"] = d["step0_pass"] & (d["p_delong"] >= ALPHA)
    for k, hib in MEASURES.items():
        d[f"rank_{k}"] = d[k].rank(ascending=not hib, method="average")
    d["avg_rank_all"] = d[[f"rank_{k}" for k in MEASURES]].mean(axis=1)
    elig = d[d.step1_pass].copy()
    for k, hib in MEASURES.items():                      # ranks recomputed within the eligible set
        elig[f"erank_{k}"] = elig[k].rank(ascending=not hib, method="average")
    d["avg_rank_eligible"] = elig[[f"erank_{k}" for k in MEASURES]].mean(axis=1)
    d["tie"] = d["avg_rank_eligible"].duplicated(keep=False) & d["avg_rank_eligible"].notna()
    d = d.sort_values(["avg_rank_eligible", "cv_auc"], ascending=[True, False], na_position="last")
    d["selected"] = False
    d.iloc[0, d.columns.get_loc("selected")] = bool(d.iloc[0]["step1_pass"])
    d["best_auc_model"] = best
    return d.reset_index()
