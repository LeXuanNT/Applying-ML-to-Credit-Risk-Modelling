"""Credit-scoring metrics: WoE/IV, KS, Gini, PSI, bootstrap AUC, decile/calibration tables, PDO scaling.

Dependencies: numpy, pandas, scikit-learn. Convention: y = 1 means bad (default).

    python credit_metrics.py   # runs the self-test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 0.5  # added to empty good/bad cells before taking logs


# ---------------------------------------------------------------- WoE / IV
def woe_iv(x: pd.Series, y: pd.Series, bins: int | list = 10) -> tuple[pd.DataFrame, float]:
    """WoE table and IV for one feature. WoE = ln(%good / %bad).

    Numeric x is quantile-binned (or cut at the given edges); missing values get
    their own bin. Categorical x uses its categories. Fit on TRAINING data only;
    re-use the returned bin edges (table.attrs["edges"]) to transform test data.
    """
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    edges = None
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > (bins if isinstance(bins, int) else len(bins)):
        if isinstance(bins, int):
            edges = np.unique(np.nanquantile(x, np.linspace(0, 1, bins + 1)))
        else:
            edges = np.asarray(bins, float)
        edges[0], edges[-1] = -np.inf, np.inf
        grp = pd.cut(x, edges, include_lowest=True).astype(str)
    else:
        grp = x.astype(str)
    grp = grp.where(x.notna(), "Missing")
    t = pd.crosstab(grp, y).reindex(columns=[0, 1], fill_value=0)
    t.columns = ["good", "bad"]
    pg = (t["good"] + EPS) / (t["good"].sum() + EPS * len(t))
    pb = (t["bad"] + EPS) / (t["bad"].sum() + EPS * len(t))
    t["woe"] = np.log(pg / pb)
    t["iv_part"] = (pg - pb) * t["woe"]
    t["bad_rate"] = t["bad"] / (t["good"] + t["bad"])
    t.attrs["edges"] = edges
    return t, float(t["iv_part"].sum())


def iv_strength(iv: float) -> str:
    for cut, label in [(0.02, "not useful"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong")]:
        if iv < cut:
            return label
    return "suspicious (check leakage)"


def iv_table(X: pd.DataFrame, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    rows = [{"feature": c, "iv": woe_iv(X[c], y, bins)[1]} for c in X.columns]
    out = pd.DataFrame(rows).sort_values("iv", ascending=False)
    out["strength"] = out["iv"].map(iv_strength)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- discrimination
def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def bootstrap_auc(y, p, n_boot: int = 1000, seed: int = 0, p_other=None) -> dict:
    """AUC with a 95% bootstrap CI. With p_other, also the CI of AUC(p) - AUC(p_other)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs, diffs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a = roc_auc_score(y[i], p[i])
        aucs.append(a)
        if p_other is not None:
            diffs.append(a - roc_auc_score(y[i], np.asarray(p_other)[i]))
    out = {"auc": float(roc_auc_score(y, p)), "auc_lo": float(np.percentile(aucs, 2.5)),
           "auc_hi": float(np.percentile(aucs, 97.5))}
    out.update({"gini": 2 * out["auc"] - 1, "gini_lo": 2 * out["auc_lo"] - 1, "gini_hi": 2 * out["auc_hi"] - 1})
    if p_other is not None:
        out.update({"diff": float(np.mean(diffs)), "diff_lo": float(np.percentile(diffs, 2.5)),
                    "diff_hi": float(np.percentile(diffs, 97.5))})
    return out


def decile_table(y, p, n: int = 10) -> pd.DataFrame:
    """Gains table, riskiest decile first: bad rate, cumulative share of bads captured."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), n, labels=range(1, n + 1))
    g = d.groupby("decile", observed=True).agg(count=("y", "size"), bads=("y", "sum"),
                                               mean_pd=("p", "mean"))
    g["bad_rate"] = g["bads"] / g["count"]
    g["cum_bad_capture"] = g["bads"].cumsum() / g["bads"].sum()
    return g


def calibration_table(y, p, n: int = 10) -> pd.DataFrame:
    """Mean predicted PD vs observed bad rate per PD decile, plus the Brier score in attrs."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n, labels=False)
    g = d.groupby("bin").agg(mean_pd=("p", "mean"), observed=("y", "mean"), count=("y", "size"))
    g.attrs["brier"] = float(np.mean((d["p"] - d["y"]) ** 2))
    g.attrs["brier_base"] = float(np.mean((d["y"].mean() - d["y"]) ** 2))
    return g


# ---------------------------------------------------------------- stability
def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of `actual` (e.g. OOT scores) vs `expected` (train scores)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(v: float) -> str:
    return "stable" if v < 0.1 else ("monitor" if v < 0.25 else "significant shift")


# ---------------------------------------------------------------- scorecard scaling
def scaling(target_score: float = 600, target_odds: float = 50, pdo: float = 20) -> tuple[float, float]:
    factor = pdo / np.log(2)
    return factor, target_score - factor * np.log(target_odds)


def scale_score(pd_, target_score=600, target_odds=50, pdo=20):
    """Score = Offset + Factor * ln(odds_good). Higher score = lower risk."""
    factor, offset = scaling(target_score, target_odds, pdo)
    pd_ = np.clip(np.asarray(pd_, float), 1e-9, 1 - 1e-9)
    return offset + factor * np.log((1 - pd_) / pd_)


def points_table(woe_tables: dict, coefs: dict, intercept: float,
                 target_score=600, target_odds=50, pdo=20) -> pd.DataFrame:
    """Scorecard points per bin for a logistic model of ln(PD/(1-PD)) on WoE features.

    woe_tables: {feature: table from woe_iv}; coefs: {feature: beta}.
    """
    factor, offset = scaling(target_score, target_odds, pdo)
    n = len(coefs)
    rows = []
    for f, beta in coefs.items():
        for b, w in woe_tables[f]["woe"].items():
            rows.append({"feature": f, "bin": b, "woe": w,
                         "points": -(beta * w + intercept / n) * factor + offset / n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- decision layer
def cutoff_table(y, pd_, scores, cutoffs, lgd: float = 0.45, ead=1.0) -> pd.DataFrame:
    """Approval rate, bad rate and expected loss among approved applicants (score >= cutoff)."""
    y, pd_, s = np.asarray(y), np.asarray(pd_), np.asarray(scores)
    ead = np.broadcast_to(np.asarray(ead, float), y.shape)
    rows = []
    for c in cutoffs:
        ok = s >= c
        rows.append({"cutoff": c, "approval_rate": ok.mean(),
                     "bad_rate_approved": y[ok].mean() if ok.any() else np.nan,
                     "expected_loss": float(np.sum(pd_[ok] * lgd * ead[ok]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    n = 20_000
    X = pd.DataFrame({"util": rng.uniform(0, 1, n), "age": rng.integers(20, 70, n).astype(float),
                      "noise": rng.normal(size=n)})
    X.loc[rng.random(n) < 0.05, "age"] = np.nan
    logit = -3 + 3 * X["util"] - 0.03 * (X["age"].fillna(40) - 40)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))

    ivs = iv_table(X, y)
    print(ivs)
    assert ivs.iloc[0]["feature"] == "util" and ivs.set_index("feature").loc["noise", "iv"] < 0.02

    tabs, W = {}, pd.DataFrame(index=X.index)
    for c in ["util", "age"]:
        t, _ = woe_iv(X[c], y)
        tabs[c] = t
        g = pd.cut(X[c], t.attrs["edges"], include_lowest=True).astype(str).where(X[c].notna(), "Missing")
        W[c] = g.map(t["woe"])
    lr = LogisticRegression().fit(W, y)
    p = lr.predict_proba(W)[:, 1]
    assert all(b < 0 for b in lr.coef_[0]), "betas on WoE(good/bad) should be negative"

    pts = points_table(tabs, dict(zip(W.columns, lr.coef_[0])), lr.intercept_[0])
    # Sum of each applicant's bin points must equal the scaled score.
    total = sum(W[c].map(pts[pts.feature == c].set_index("woe")["points"]) for c in W.columns)
    assert np.allclose(total, scale_score(p), atol=1e-6), "points do not add up to the scaled score"
    assert abs(scale_score(1 / 51) - 600) < 1e-9 and abs(scale_score(1 / 101) - (600 + 20 * np.log2(100 / 50))) < 1e-9

    b = bootstrap_auc(y, p, n_boot=200, p_other=X["noise"])
    print({k: round(v, 3) for k, v in b.items()})
    assert b["auc_lo"] < b["auc"] < b["auc_hi"] and b["diff_lo"] > 0
    assert abs(gini(y, p) - (2 * b["auc"] - 1)) < 1e-12 and 0 < ks(y, p) < 1

    s = scale_score(p)
    assert psi(s, s) < 1e-9 and psi(s, s + 40) > 0.25
    print(decile_table(y, p).head(3))
    cal = calibration_table(y, p)
    assert cal.attrs["brier"] < cal.attrs["brier_base"]
    print(cutoff_table(y, p, s, np.quantile(s, [0.2, 0.4, 0.6]).round()))
    print("self-test passed")
