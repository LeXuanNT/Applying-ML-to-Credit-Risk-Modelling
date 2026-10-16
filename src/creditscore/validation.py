"""Validation: discrimination (AUC/Gini/KS/PR-AUC) with bootstrap CIs, paired bootstrap
tests vs the champion, calibration (Brier, Platt recalibration), stability (PSI/CSI),
and threshold metrics at a cut-off chosen on the validation set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from .credit_metrics import (bootstrap_auc, calibration_table, decile_table, gini, ks,  # noqa: F401
                             psi, psi_label)


def discrimination(y, p, n_boot: int = 1000, seed: int = 0) -> dict:
    b = bootstrap_auc(y, p, n_boot=n_boot, seed=seed)
    return {"auc": b["auc"], "auc_lo": b["auc_lo"], "auc_hi": b["auc_hi"],
            "gini": b["gini"], "gini_lo": b["gini_lo"], "gini_hi": b["gini_hi"],
            "ks": ks(y, p), "pr_auc": average_precision_score(y, p)}


def paired_auc_test(y, p_a, p_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap of AUC(a) - AUC(b) on the same resampled rows; two-sided p-value."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    diffs = np.asarray(diffs)
    d = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    pval = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"auc_diff": d, "diff_lo": np.percentile(diffs, 2.5), "diff_hi": np.percentile(diffs, 97.5),
            "gini_diff": 2 * d, "p_value": min(1.0, max(pval, 1 / len(diffs)))}


def best_f1_threshold(y, p) -> float:
    """Cut-off maximising F1 - chosen on the VALIDATION set, then frozen for test."""
    qs = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y, (p >= t).astype(int)) for t in qs]
    return float(qs[int(np.argmax(f1s))])


def threshold_metrics(y, p, thr: float) -> dict:
    yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"threshold": thr, "accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def brier(y, p) -> float:
    return float(brier_score_loss(y, np.clip(p, 0, 1)))


class PlattCalibrator:
    """Logistic recalibration of a score: PD = sigmoid(a + b * logit(score)).
    Fit on the validation set; the test set stays untouched."""

    def fit(self, s, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(s), y)
        return self

    @staticmethod
    def _z(s):
        s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s)).reshape(-1, 1)

    def predict(self, s):
        return self.lr_.predict_proba(self._z(s))[:, 1]


def hosmer_lemeshow(y, p, n: int = 10) -> dict:
    from scipy.stats import chi2
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["g"] = pd.qcut(d["p"].rank(method="first"), n, labels=False)
    g = d.groupby("g").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    stat = (((g["obs"] - g["exp"]) ** 2) / (g["exp"] * (1 - g["exp"] / g["n"]))).sum()
    return {"hl_stat": float(stat), "hl_p": float(1 - chi2.cdf(stat, n - 2))}


def csi_table(train_df: pd.DataFrame, other_df: pd.DataFrame, features: list[str],
              bins: int = 10) -> pd.DataFrame:
    rows = []
    for f in features:
        if not pd.api.types.is_numeric_dtype(train_df[f]):
            a = train_df[f].fillna("Missing").value_counts(normalize=True)
            b = other_df[f].fillna("Missing").value_counts(normalize=True).reindex(a.index, fill_value=0)
            a, b = np.clip(a.to_numpy(), 1e-6, None), np.clip(b.to_numpy(), 1e-6, None)
            v = float(np.sum((b - a) * np.log(b / a)))
        else:
            a, b = train_df[f].astype(float), other_df[f].astype(float)
            v = psi(a.fillna(-1e9), b.fillna(-1e9), bins) if a.nunique() > 1 else 0.0
        rows.append({"feature": f, "csi": v, "status": psi_label(v)})
    return pd.DataFrame(rows)
