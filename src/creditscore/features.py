"""Stage 1b - cleaning, feature engineering, WoE binning and IV.

Everything here is FIT ON TRAIN ONLY and then applied to validation / test.
(The original notebook fitted imputers and scalers on the full data or per split,
which leaks test information into training.)

Two feature views are produced:
* `HMEQPreprocessor`  -> imputed numeric matrix for ML models
  (missing indicators, joint iterative imputation, ratios, one-hot categories).
* `WoEBinner`         -> WoE-coded matrix for the logistic scorecard
  (missing values kept as their own bin, monotonic bins for numeric features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .credit_metrics import iv_strength
from .data import CAT_FEATURES, NUM_FEATURES

INFORMATIVE_MISSING = ["DEBTINC", "VALUE"]          # Chapter 3: informative missingness
JOINT_IMPUTE = ["CLAGE", "CLNO", "DELINQ", "NINQ"]  # Chapter 3: structured missing group
SIMPLE_IMPUTE = [c for c in NUM_FEATURES if c not in JOINT_IMPUTE]
EPS = 0.5


def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    """Loan-to-value and mortgage-to-value. NaN stays NaN (handled downstream)."""
    X = X.copy()
    X["LOAN_TO_VALUE"] = X["LOAN"] / X["VALUE"]
    X["MORTDUE_TO_VALUE"] = X["MORTDUE"] / X["VALUE"]
    return X


class HMEQPreprocessor(BaseEstimator, TransformerMixin):
    """Imputation + engineered features for the ML challengers. Fit on train only."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X: pd.DataFrame, y=None):
        X = X[NUM_FEATURES + CAT_FEATURES]
        self.joint_ = IterativeImputer(random_state=self.seed, max_iter=15,
                                       initial_strategy="median").fit(X[JOINT_IMPUTE])
        self.medians_ = X[SIMPLE_IMPUTE].median()
        self.modes_ = {c: X[c].mode().iloc[0] for c in CAT_FEATURES}
        self.categories_ = {c: sorted(X[c].fillna(self.modes_[c]).unique()) for c in CAT_FEATURES}
        Z = self._build(X)
        self.feature_names_out_ = list(Z.columns)
        return self

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = pd.DataFrame(index=X.index)
        for c in INFORMATIVE_MISSING:
            Z[f"{c}_MISSING"] = X[c].isna().astype(int)
        Z["N_MISSING"] = X[NUM_FEATURES + CAT_FEATURES].isna().sum(axis=1)
        num = X[NUM_FEATURES].copy()
        num[JOINT_IMPUTE] = np.clip(self.joint_.transform(X[JOINT_IMPUTE]), 0, None)
        num[SIMPLE_IMPUTE] = num[SIMPLE_IMPUTE].fillna(self.medians_)
        num = add_ratio_features(num)
        Z = pd.concat([num, Z], axis=1)
        for c in CAT_FEATURES:
            v = X[c].fillna(self.modes_[c])
            for cat in self.categories_[c][1:]:          # drop first level (reference)
                Z[f"{c}_{cat}"] = (v == cat).astype(int)
        return Z

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self._build(X[NUM_FEATURES + CAT_FEATURES])
        return Z[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


# ------------------------------------------------------------------ WoE binning
def _monotone_edges(x: pd.Series, y: pd.Series, max_bins: int, min_share: float) -> np.ndarray:
    """Quantile edges, merged until each bin has >= min_share of rows and the bad
    rate is monotonic (direction taken from the Spearman sign)."""
    xs = x.dropna()
    uniq = np.unique(xs)
    if len(uniq) <= 3 * max_bins:          # discrete counts (DELINQ, DEROG, NINQ): start per value
        edges = np.r_[-np.inf, (uniq[:-1] + uniq[1:]) / 2, np.inf]
    else:
        edges = np.unique(np.quantile(xs, np.linspace(0, 1, max_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) <= 2:
        return np.array([-np.inf, np.inf])
    ys = y[xs.index]
    sign = np.sign(pd.Series(xs).corr(ys, method="spearman")) or 1.0

    def stats(e):
        g = pd.cut(xs, e, include_lowest=True)
        t = ys.groupby(g, observed=False).agg(["size", "mean"])
        return t["size"].to_numpy(), t["mean"].to_numpy()

    while len(edges) > 2:
        n, br = stats(edges)
        small = np.where(n < min_share * len(x))[0]
        if len(small):
            i = small[0]
            # merge with the neighbour whose bad rate is closer
            if i == 0:
                j = 1
            elif i == len(n) - 1:
                j = i - 1
            else:
                j = i - 1 if abs(br[i] - br[i - 1]) <= abs(br[i] - br[i + 1]) else i + 1
            edges = np.delete(edges, max(i, j))
            continue
        d = np.diff(br) * sign
        bad = np.where(d < 0)[0]
        if not len(bad):
            break
        i = bad[np.argmin(np.abs(d[bad]))]
        edges = np.delete(edges, i + 1)
    return edges


class WoEBinner(BaseEstimator, TransformerMixin):
    """Fine-classing / coarse-classing with WoE = ln(%good / %bad). Missing = own bin.

    Higher WoE = safer bin, so logistic coefficients on WoE are expected to be negative.
    """

    def __init__(self, max_bins: int = 8, min_share: float = 0.05):
        self.max_bins = max_bins
        self.min_share = min_share

    def _bin(self, f: str, x: pd.Series) -> pd.Series:
        spec = self.spec_[f]
        if spec["kind"] == "num":
            lab = pd.cut(x, spec["edges"], include_lowest=True).astype(str)
        else:
            lab = x.astype(str).where(x.astype(str).isin(spec["levels"]), "RARE")
        return lab.where(x.notna(), "Missing")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = pd.Series(np.asarray(y), index=X.index).astype(int)
        self.spec_, self.tables_ = {}, {}
        for f in X.columns:
            x = X[f]
            if pd.api.types.is_numeric_dtype(x):
                self.spec_[f] = {"kind": "num",
                                 "edges": _monotone_edges(x, y, self.max_bins, self.min_share)}
            else:
                share = x.value_counts(normalize=True)
                self.spec_[f] = {"kind": "cat", "levels": list(share[share >= self.min_share].index)}
            lab = self._bin(f, x)
            t = pd.crosstab(lab, y).reindex(columns=[0, 1], fill_value=0)
            t.columns = ["good", "bad"]
            pg = (t["good"] + EPS) / (t["good"].sum() + EPS * len(t))
            pb = (t["bad"] + EPS) / (t["bad"].sum() + EPS * len(t))
            t["n"] = t["good"] + t["bad"]
            t["share"] = t["n"] / t["n"].sum()
            t["bad_rate"] = t["bad"] / t["n"]
            t["woe"] = np.log(pg / pb)
            t["iv_part"] = (pg - pb) * t["woe"]
            t = self._order(f, t)
            self.tables_[f] = t
        return self

    def _order(self, f, t):
        if self.spec_[f]["kind"] != "num":
            return t.sort_values("bad_rate", ascending=False)
        e = self.spec_[f]["edges"]
        order = [str(iv) for iv in pd.IntervalIndex.from_breaks(e, closed="right")]
        # pd.cut(include_lowest) labels the first interval with a tweaked left edge
        idx = [i for i in t.index if i != "Missing"]
        key = {lab: k for k, lab in enumerate(sorted(idx, key=lambda s: float(s.split(",")[0][1:])))}
        rows = sorted(idx, key=key.get) + (["Missing"] if "Missing" in t.index else [])
        return t.loc[rows]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for f in self.tables_:
            out[f] = self._bin(f, X[f]).map(self.tables_[f]["woe"]).fillna(0.0).astype(float)
        return out

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": f, "n_bins": len(t), "iv": float(t["iv_part"].sum())}
                for f, t in self.tables_.items()]
        out = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
        out["strength"] = out["iv"].map(iv_strength)
        return out

    def woe_table(self, f: str) -> pd.DataFrame:
        t = self.tables_[f].copy()
        t.index.name = "bin"
        return t.reset_index()[["bin", "n", "share", "bad", "bad_rate", "woe", "iv_part"]]


def scorecard_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Raw features (with NaN kept) + ratio features + missing count, for WoE binning."""
    Z = add_ratio_features(X[NUM_FEATURES + CAT_FEATURES])
    Z["N_MISSING"] = X[NUM_FEATURES + CAT_FEATURES].isna().sum(axis=1)
    return Z


def select_woe_features(woe_train: pd.DataFrame, iv: pd.DataFrame, min_iv: float = 0.02,
                        max_corr: float = 0.6) -> tuple[list[str], pd.DataFrame]:
    """Keep IV >= min_iv, then drop the weaker of any pair with |corr(WoE)| > max_corr."""
    cand = iv.loc[iv["iv"] >= min_iv, "feature"].tolist()
    keep, log = [], []
    for f in cand:                                   # cand is sorted by IV, strongest first
        clash = [k for k in keep if abs(woe_train[f].corr(woe_train[k])) > max_corr]
        if clash:
            log.append({"feature": f, "decision": "drop", "reason": f"corr>{max_corr} with {clash[0]}"})
        else:
            keep.append(f)
            log.append({"feature": f, "decision": "keep", "reason": ""})
    for f in iv.loc[iv["iv"] < min_iv, "feature"]:
        log.append({"feature": f, "decision": "drop", "reason": f"IV<{min_iv}"})
    return keep, pd.DataFrame(log)
