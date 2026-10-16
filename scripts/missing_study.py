"""Missing-data study for HMEQ: (A) mechanism tests, (B) imputation accuracy by amputation,
(C) predictive comparison of strategies that KEEP incomplete applications vs complete-case deletion.
Writes results/missing_study/*.csv.   python scripts/missing_study.py"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from creditscore.data import CAT_FEATURES, NUM_FEATURES, RAW_FEATURES, TARGET, load_hmeq, split_data  # noqa
from creditscore.features import WoEBinner  # noqa

OUT = ROOT / "results" / "missing_study"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
df = load_hmeq()
S = split_data(df, SEED)
dev = pd.concat([S["train"], S["valid"]], ignore_index=True)       # development = train + valid
test = S["test"].reset_index(drop=True)
MISS = [c for c in RAW_FEATURES if df[c].isna().any()]

# =============================== A. mechanism ===============================
# A1. Complete-case deletion: what would be lost
cc = df.dropna(subset=RAW_FEATURES)
nmiss = df[RAW_FEATURES].isna().sum(axis=1)
A1 = pd.DataFrame([
    {"subset": "Toàn bộ", "rows": len(df), "bads": int(df[TARGET].sum()), "bad_rate": df[TARGET].mean()},
    {"subset": "Hồ sơ đầy đủ (complete cases)", "rows": len(cc), "bads": int(cc[TARGET].sum()), "bad_rate": cc[TARGET].mean()},
    {"subset": "Hồ sơ có ít nhất 1 giá trị khuyết", "rows": int((nmiss > 0).sum()),
     "bads": int(df.loc[nmiss > 0, TARGET].sum()), "bad_rate": df.loc[nmiss > 0, TARGET].mean()}])
A1["share_rows"] = A1["rows"] / len(df)
A1["share_bads"] = A1["bads"] / df[TARGET].sum()
A1b = df.groupby(nmiss.clip(upper=4))[TARGET].agg(["size", "mean"]).reset_index()
A1b.columns = ["n_missing_fields(4=4+)", "rows", "bad_rate"]


# A2. Little's MCAR test (numeric variables, EM estimates of mean/covariance)
def em_mvn(X, iters=200, tol=1e-6):
    X = np.asarray(X, float)
    mu = np.nanmean(X, 0)
    S_ = np.diag(np.nanvar(X, 0))
    for _ in range(iters):
        Xh, C = X.copy(), np.zeros_like(S_)
        for i, row in enumerate(X):
            m = np.isnan(row)
            if not m.any():
                continue
            o = ~m
            Soo_inv = np.linalg.pinv(S_[np.ix_(o, o)])
            B = S_[np.ix_(m, o)] @ Soo_inv
            Xh[i, m] = mu[m] + B @ (row[o] - mu[o])
            C[np.ix_(m, m)] += S_[np.ix_(m, m)] - B @ S_[np.ix_(o, m)]
        mu_new = Xh.mean(0)
        D = Xh - mu_new
        S_new = (D.T @ D + C) / len(X)
        if np.max(np.abs(mu_new - mu)) < tol and np.max(np.abs(S_new - S_)) < tol:
            mu, S_ = mu_new, S_new
            break
        mu, S_ = mu_new, S_new
    return mu, S_


def little_mcar(Xdf):
    X = Xdf.to_numpy(float)
    Z = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)               # scale for numerical stability
    mu, Sg = em_mvn(Z)
    pat = pd.Series([tuple(r) for r in np.isnan(Z)])
    d2, df_ = 0.0, 0
    for p, idx in pat.groupby(pat).groups.items():
        o = ~np.array(p)
        if not o.any():
            continue
        sub = Z[np.asarray(idx)][:, o]
        diff = sub.mean(0) - mu[o]
        d2 += len(idx) * diff @ np.linalg.pinv(Sg[np.ix_(o, o)]) @ diff
        df_ += o.sum()
    df_ -= Z.shape[1]
    return d2, df_, stats.chi2.sf(d2, df_), pat.nunique()


d2, dof, p_mcar, npat = little_mcar(df[NUM_FEATURES])
A2 = pd.DataFrame([{"test": "Little's MCAR (10 biến định lượng)", "chi2": d2, "df": dof, "p_value": p_mcar,
                    "n_patterns": npat}])

# A3. Is missingness of X_j predictable from the OTHER observed variables? (MAR evidence) and from BAD?
rows = []
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for c in MISS:
    m = df[c].isna().astype(int)
    others = [v for v in RAW_FEATURES if v != c]
    Xo = pd.get_dummies(df[others].assign(**{v: df[v].fillna(df[v].median()) for v in others if v in NUM_FEATURES}),
                        columns=[v for v in others if v in CAT_FEATURES], dummy_na=True, dtype=float)
    Xo = Xo.join(df[others].isna().add_suffix("_na").astype(float))
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p_x = cross_val_predict(lr, Xo, m, cv=skf, method="predict_proba")[:, 1]
    p_xy = cross_val_predict(lr, Xo.assign(BAD=df[TARGET]), m, cv=skf, method="predict_proba")[:, 1]
    rows.append({"feature": c, "pct_missing": m.mean(), "auc_from_other_vars": roc_auc_score(m, p_x),
                 "auc_other_vars_plus_BAD": roc_auc_score(m, p_xy),
                 "bad_rate_missing": df.loc[m == 1, TARGET].mean(), "bad_rate_observed": df.loc[m == 0, TARGET].mean()})
A3 = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
print("A done", flush=True)

# ======================= B. imputation accuracy (amputation) =======================
# Mask complete cases using REAL missingness patterns drawn from incomplete rows; compare recovered values.
rng = np.random.default_rng(SEED)
num_cc = cc[NUM_FEATURES].reset_index(drop=True)
real_pat = df.loc[nmiss > 0, NUM_FEATURES].isna().to_numpy()
imps = {"Trung vị (median)": None,
        "KNN (k=5)": KNNImputer(n_neighbors=5),
        "MICE - BayesianRidge": IterativeImputer(max_iter=15, random_state=SEED),
        "MICE - ExtraTrees": IterativeImputer(ExtraTreesRegressor(n_estimators=50, min_samples_leaf=3, random_state=SEED,
                                                                  n_jobs=-1), max_iter=5, random_state=SEED)}
brows = []
for rep in range(3):
    mask = real_pat[rng.integers(0, len(real_pat), len(num_cc))]
    mask[:, ~mask.any(0)] = False
    Xm = num_cc.mask(mask)
    sd = num_cc.std()
    for name, imp in imps.items():
        if imp is None:
            Xi = Xm.fillna(Xm.median())
        else:
            mu_, s_ = Xm.mean(), Xm.std()
            Xi = pd.DataFrame(clone(imp).fit_transform((Xm - mu_) / s_), columns=NUM_FEATURES) * s_ + mu_
        for j, c in enumerate(NUM_FEATURES):
            if mask[:, j].sum() < 20:
                continue
            e = (Xi[c].to_numpy() - num_cc[c].to_numpy())[mask[:, j]]
            brows.append({"rep": rep, "imputer": name, "feature": c, "n_masked": int(mask[:, j].sum()),
                          "nrmse": np.sqrt(np.mean(e ** 2)) / sd[c]})
B = pd.DataFrame(brows)
B_sum = B.groupby(["imputer", "feature"]).nrmse.mean().unstack()
B_sum["mean"] = B_sum.mean(1)
B_sum = B_sum.sort_values("mean").reset_index()
print("B done", flush=True)


# ======================= C. predictive comparison =======================
class Prep(BaseEstimator, TransformerMixin):
    """Keep-all-rows strategies. num: 'median' | 'mice' | 'knn' | 'native'; ind: add missing indicators;
    cat: categorical NaN -> mode or its own 'Missing' level."""

    def __init__(self, num="median", ind=False, cat_missing=False):
        self.num, self.ind, self.cat_missing = num, ind, cat_missing

    def fit(self, X, y=None):
        Z = X[NUM_FEATURES]
        self.med_ = Z.median()
        self.mode_ = {c: X[c].mode().iloc[0] for c in CAT_FEATURES}
        self.mu_, self.sd_ = Z.mean(), Z.std()
        if self.num == "mice":
            self.imp_ = IterativeImputer(max_iter=15, random_state=SEED).fit((Z - self.mu_) / self.sd_)
        elif self.num == "knn":
            self.imp_ = KNNImputer(n_neighbors=5).fit((Z - self.mu_) / self.sd_)
        self.ind_cols_ = [c for c in RAW_FEATURES if X[c].isna().any()]
        self.cols_ = list(self._make(X).columns)
        return self

    def _make(self, X):
        Z = X[NUM_FEATURES]
        if self.num == "median":
            N = Z.fillna(self.med_)
        elif self.num == "native":
            N = Z.copy()
        else:
            N = pd.DataFrame(self.imp_.transform((Z - self.mu_) / self.sd_), columns=NUM_FEATURES,
                             index=X.index) * self.sd_ + self.mu_
        parts = [N]
        for c in CAT_FEATURES:
            v = X[c].fillna("Missing" if self.cat_missing else self.mode_[c])
            parts.append(pd.get_dummies(v, prefix=c, dtype=float))
        if self.ind:
            parts.append(X[self.ind_cols_].isna().add_suffix("_MISSING").astype(float))
        return pd.concat(parts, axis=1)

    def transform(self, X):
        return self._make(X).reindex(columns=self.cols_, fill_value=0.0)


class WoEPrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        self.b_ = WoEBinner().fit(X[RAW_FEATURES], y)
        return self

    def transform(self, X):
        return self.b_.transform(X[RAW_FEATURES])


MODELS = {
    "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    "RF": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                         n_jobs=-1, random_state=SEED),
    "LGBM": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                                   random_state=SEED, verbose=-1),
}
STRATS = {
    "S0 Xóa hồ sơ khuyết (complete-case)": ("cc", dict(num="median")),
    "S1 Median/mode": ("keep", dict(num="median")),
    "S2 Median + chỉ báo khuyết": ("keep", dict(num="median", ind=True, cat_missing=True)),
    "S3 MICE": ("keep", dict(num="mice")),
    "S4 MICE + chỉ báo khuyết": ("keep", dict(num="mice", ind=True, cat_missing=True)),
    "S5 KNN + chỉ báo khuyết": ("keep", dict(num="knn", ind=True, cat_missing=True)),
    "S6 Để nguyên NaN (native)": ("native", dict(num="native", cat_missing=True)),
    "S7 WoE với nhóm Missing": ("woe", {}),
}


def applicable(kind, model):
    return not ((kind == "native" and model != "LGBM") or (kind == "woe" and model != "LR"))


def fit_predict(kind, kw, model, Xtr, ytr, Xte):
    if kind == "cc":                                   # train only on complete rows; score everyone (median fill)
        keep = Xtr[RAW_FEATURES].notna().all(1).to_numpy()
        Xtr, ytr = Xtr[keep], ytr[keep]
    prep = WoEPrep() if kind == "woe" else Prep(**kw)
    pipe = make_pipeline(prep, MODELS[model]())
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


def scores(y, p, mask=None):
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "f1": f1_score(y, (p >= 0.5).astype(int)), "brier": brier_score_loss(y, p)}


Xd, yd = dev[RAW_FEATURES], dev[TARGET].to_numpy()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
crow = []
t0 = time.time()
for k, (tr, va) in enumerate(cv.split(Xd, yd)):
    inc = Xd.iloc[va].isna().any(axis=1).to_numpy()
    for s, (kind, kw) in STRATS.items():
        for m in MODELS:
            if not applicable(kind, m):
                continue
            p = fit_predict(kind, kw, m, Xd.iloc[tr], yd[tr], Xd.iloc[va])
            r = {"fold": k, "strategy": s, "model": m, **scores(yd[va], p)}
            r["auc_incomplete_rows"] = roc_auc_score(yd[va][inc], p[inc])
            r["auc_complete_rows"] = roc_auc_score(yd[va][~inc], p[~inc])
            crow.append(r)
    print(f"fold {k} {time.time() - t0:.0f}s", flush=True)
C = pd.DataFrame(crow)
C_sum = C.groupby(["model", "strategy"]).agg(
    auc=("auc", "mean"), auc_sd=("auc", "std"), pr_auc=("pr_auc", "mean"), f1=("f1", "mean"),
    brier=("brier", "mean"), auc_incomplete=("auc_incomplete_rows", "mean"),
    auc_complete=("auc_complete_rows", "mean")).reset_index()
# paired difference vs S1 (median) per model, across the 15 folds
base = C[C.strategy.str.startswith("S1")].set_index(["fold", "model"])["auc"]
D = C.join(base.rename("auc_s1"), on=["fold", "model"])
D["d"] = D["auc"] - D["auc_s1"]
pv = D.groupby(["model", "strategy"])["d"].agg(
    d_mean="mean", d_sd="std", wins=lambda x: int((x > 0).sum()),
    p_wilcoxon=lambda x: stats.wilcoxon(x).pvalue if (x != 0).any() else np.nan).reset_index()
C_sum = C_sum.merge(pv, on=["model", "strategy"])

# Final check on untouched test set (fit on dev = train+valid), bootstrap CI of AUC difference vs S1
trow, preds = [], {}
for s, (kind, kw) in STRATS.items():
    for m in MODELS:
        if applicable(kind, m):
            preds[(s, m)] = fit_predict(kind, kw, m, Xd, yd, test[RAW_FEATURES])
yt = test[TARGET].to_numpy()
B_ = np.random.default_rng(SEED).integers(0, len(yt), (1000, len(yt)))
for (s, m), p in preds.items():
    p1 = preds[(next(k for k in STRATS if k.startswith("S1")), m)]
    diffs = [roc_auc_score(yt[b], p[b]) - roc_auc_score(yt[b], p1[b]) for b in B_ if 0 < yt[b].sum() < len(b)]
    trow.append({"strategy": s, "model": m, **scores(yt, p), "d_auc_vs_S1": np.mean(diffs),
                 "ci_lo": np.percentile(diffs, 2.5), "ci_hi": np.percentile(diffs, 97.5)})
Tst = pd.DataFrame(trow)

for name, d in {"A1_complete_case": A1, "A1b_bad_by_nmissing": A1b, "A2_little_mcar": A2, "A3_mar_check": A3,
                "B_imputation_nrmse": B_sum, "C_cv_folds": C, "C_cv_summary": C_sum, "C_test": Tst}.items():
    d.to_csv(OUT / f"{name}.csv", index=False)
print(f"done {time.time() - t0:.0f}s")
