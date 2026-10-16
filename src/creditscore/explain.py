"""Explainability: SHAP for tree challengers, compared with IV / scorecard drivers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def shap_values(pipe, X: pd.DataFrame, max_rows: int = 2000, seed: int = 0):
    """SHAP values (log-odds scale) for a fitted tree pipeline: returns (values, Z)."""
    import shap

    steps = pipe.named_steps
    Z = steps["prep"].transform(X)
    for name in ("poly", "scale"):
        if name in steps:
            Z = steps[name].transform(Z)
    if len(Z) > max_rows:
        Z = Z.sample(max_rows, random_state=seed)
    clf = steps["clf"]
    ex = shap.TreeExplainer(clf)
    v = ex.shap_values(Z)
    if isinstance(v, list):
        v = v[1]
    v = np.asarray(v)
    if v.ndim == 3:
        v = v[:, :, 1]
    return v, Z, ex


def shap_importance(values: np.ndarray, Z: pd.DataFrame) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": Z.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


# engineered/one-hot columns -> source feature, to compare with IV ranking
def source_feature(col: str) -> str:
    for p in ("REASON_", "JOB_"):
        if col.startswith(p):
            return p[:-1]
    if col in ("DEBTINC_MISSING", "VALUE_MISSING"):
        return col.replace("_MISSING", "")
    return col


def grouped_shap(values: np.ndarray, Z: pd.DataFrame) -> pd.DataFrame:
    """Row-level SHAP summed over the columns derived from one source feature
    (one-hot dummies, missing flags) - valid because SHAP values are additive."""
    src = [source_feature(c) for c in Z.columns]
    return pd.DataFrame(values, columns=Z.columns, index=Z.index).T.groupby(src).sum().T


def shap_by_source(values: np.ndarray, Z: pd.DataFrame) -> pd.DataFrame:
    g = grouped_shap(values, Z).abs().mean().sort_values(ascending=False)
    out = g.rename("mean_abs_shap").reset_index().rename(columns={"index": "source"})
    out["shap_rank"] = np.arange(1, len(out) + 1)
    return out


def local_explanation(values_row: np.ndarray, Z_row: pd.Series, k: int = 3) -> pd.DataFrame:
    d = pd.DataFrame({"feature": Z_row.index, "value": Z_row.values, "shap": values_row})
    return d.sort_values("shap", ascending=False).head(k)
