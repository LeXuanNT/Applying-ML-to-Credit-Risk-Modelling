"""Exploratory data analysis tables for Chapter 3 (descriptive, missingness, association, VIF).

Every number quoted in Chapter 3 comes from these functions (scripts/make_chapter3.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import CAT_FEATURES, NUM_FEATURES, RAW_FEATURES, TARGET


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    v = df[TARGET].value_counts().sort_index()
    return pd.DataFrame({"BAD": v.index, "n": v.values, "share": v.values / len(df)})


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count / share overall and by class, bad rate when missing, and phi with BAD."""
    y = df[TARGET]
    rows = []
    for c in RAW_FEATURES:
        m = df[c].isna().astype(int)
        if m.sum() == 0:
            phi, p = np.nan, np.nan
        else:
            ct = pd.crosstab(m, y)
            chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
            phi = np.sign(np.corrcoef(m, y)[0, 1]) * np.sqrt(chi2 / len(df))
        rows.append({"feature": c, "n_missing": int(m.sum()), "pct_missing": m.mean(),
                     "pct_missing_good": m[y == 0].mean(), "pct_missing_bad": m[y == 1].mean(),
                     "bad_rate_missing": y[m == 1].mean() if m.any() else np.nan,
                     "bad_rate_present": y[m == 0].mean(), "phi": phi, "p_value": p})
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def categorical_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CAT_FEATURES:
        g = df.groupby(df[c].fillna("Missing"))[TARGET].agg(["size", "mean"])
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[c].fillna("Missing"), df[TARGET]))
        for lev, r in g.sort_values("mean", ascending=False).iterrows():
            rows.append({"feature": c, "level": lev, "n": int(r["size"]), "share": r["size"] / len(df),
                         "bad_rate": r["mean"], "chi2_p": p})
    return pd.DataFrame(rows)


def numeric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics, skewness, medians by class, point-biserial r and Mann-Whitney p."""
    rows = []
    for c in NUM_FEATURES:
        x = df[[c, TARGET]].dropna()
        r, p = stats.pointbiserialr(x[TARGET], x[c])
        mw = stats.mannwhitneyu(x.loc[x[TARGET] == 0, c], x.loc[x[TARGET] == 1, c]).pvalue
        sw = stats.shapiro(x[c].sample(min(len(x), 5000), random_state=0)).statistic
        q = x[c].quantile([0.25, 0.5, 0.75])
        rows.append({"feature": c, "n": len(x), "mean": x[c].mean(), "std": x[c].std(), "min": x[c].min(),
                     "q1": q[0.25], "median": q[0.5], "q3": q[0.75], "max": x[c].max(), "skew": x[c].skew(),
                     "median_good": x.loc[x[TARGET] == 0, c].median(),
                     "median_bad": x.loc[x[TARGET] == 1, c].median(),
                     "mean_good": x.loc[x[TARGET] == 0, c].mean(), "mean_bad": x.loc[x[TARGET] == 1, c].mean(),
                     "r_pb": r, "p_pb": p, "p_mannwhitney": mw, "shapiro_w": sw})
    return pd.DataFrame(rows)


def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    """VIF on complete cases, WITH an intercept (centred R^2), and the uncentred version
    (no intercept) for comparison - the latter inflates VIF for variables with large means."""
    X = df[NUM_FEATURES].dropna().to_numpy(float)
    rows = []
    for i, c in enumerate(NUM_FEATURES):
        y, Z = X[:, i], np.delete(X, i, axis=1)
        Zc = np.c_[np.ones(len(Z)), Z]
        b, *_ = np.linalg.lstsq(Zc, y, rcond=None)
        r2 = 1 - ((y - Zc @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        b0, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2u = 1 - ((y - Z @ b0) ** 2).sum() / (y ** 2).sum()
        rows.append({"feature": c, "vif": 1 / (1 - r2), "vif_no_intercept": 1 / (1 - r2u)})
    out = pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
    out.attrs["n_complete"] = len(X)
    return out


def top_correlations(df: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    c = df[NUM_FEATURES].corr()
    iu = np.triu_indices_from(c, 1)
    pairs = pd.DataFrame({"a": c.index[iu[0]], "b": c.columns[iu[1]], "r": c.values[iu]})
    return pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)


def missing_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Jaccard similarity between missingness indicators (pairs of features missing together)."""
    M = df[RAW_FEATURES].isna()
    M = M.loc[:, M.any()]
    rows = []
    cols = list(M.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            inter = (M[a] & M[b]).sum()
            union = (M[a] | M[b]).sum()
            rows.append({"a": a, "b": b, "jaccard": inter / union if union else 0.0, "both_missing": int(inter)})
    return pd.DataFrame(rows).sort_values("jaccard", ascending=False).reset_index(drop=True)


def run_all_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"eda_target": target_summary(df), "eda_missing": missing_table(df),
            "eda_categorical": categorical_table(df), "eda_numeric": numeric_table(df),
            "eda_vif": vif_table(df), "eda_top_corr": top_correlations(df),
            "eda_missing_cooccurrence": missing_cooccurrence(df)}
