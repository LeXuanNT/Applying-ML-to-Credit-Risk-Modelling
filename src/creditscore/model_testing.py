"""Six-layer model testing framework (thesis §3.9, results in Ch.4).

Every check returns a TestResult with the layer, the criterion as stated in
Bảng 3.x, the measured result and pass/fail. THRESHOLDS holds every pass
criterion in one place so the code and the thesis text cite the same numbers.
Change a threshold only before running the tests, never after.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

# HMEQ schema (Bảng 3.1). Kept here so the module has no dependency on the data loader.
NUMERIC = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
CATEGORICAL = ["REASON", "JOB"]
FEATURES = NUMERIC + CATEGORICAL
REASONS = ["DebtCon", "HomeImp"]
JOBS = ["Other", "ProfExe", "Office", "Mgr", "Self", "Sales"]

THRESHOLDS = {
    "missing_drift_pp": 0.05,        # missing rate <= train rate + 5 pp
    "delong_alpha": 0.05,            # champion AUC > baseline, DeLong p < 0.05
    "calib_overall_pp": 0.02,        # |mean PD - default rate| <= 2 pp
    "calib_decile_pp": 0.10,         # each decile |mean PD - default rate| <= 10 pp
    "psi_max": 0.10,                 # PSI < 0.1 (Siddiqi 2006)
    "median_sensitivity_pp": 0.05,   # mean |dPD| around the train median <= 5 pp
    "integration_tol": 1e-9,         # PD app == PD pipeline
    "band_approve": 0.10,            # PD < 10% -> Approve
    "band_decline": 0.50,            # PD >= 50% -> Decline
    "shap_additivity_tol": 1e-6,     # base + sum(SHAP) == margin
    "regression_pd_tol": 1e-3,       # |dPD| <= 0.001, same model
    "regression_auc_drop": 0.005,    # retrained: AUC drop <= 0.005
    "regression_band_change": 0.01,  # retrained: <= 1% of applicants change band
}

# Expected sign of each variable's effect on PD, from the EDA in §3.2.
EXPECTED_SIGN = {"DEBTINC": +1, "DELINQ": +1, "DEROG": +1, "NINQ": +1, "CLAGE": -1}


@dataclass
class TestResult:
    layer: str
    test: str
    criterion: str
    result: str
    passed: bool


def _r(layer, test, criterion, result, passed):
    return TestResult(layer, test, criterion, result, bool(passed))


# ---------------------------------------------------------------- statistics
def delong(y, p1, p2):
    """DeLong et al. (1988) test for two correlated AUCs. Returns (auc1, auc2, p)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def components(p):
        p = np.asarray(p)
        x, z = p[pos], p[neg]
        # V10[i] = P(score of pos i > random neg), V01[j] = P(random pos > neg j)
        v10 = np.array([(np.sum(xi > z) + 0.5 * np.sum(xi == z)) / n for xi in x])
        v01 = np.array([(np.sum(x > zj) + 0.5 * np.sum(x == zj)) / m for zj in z])
        return v10.mean(), v10, v01

    a1, v10a, v01a = components(p1)
    a2, v10b, v01b = components(p2)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    if var <= 1e-15:                      # identical rankings: no difference to test
        return a1, a2, 1.0 if abs(a1 - a2) < 1e-12 else 0.0
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, float(2 * stats.norm.sf(abs(z)))


def bootstrap_auc_ci(y, p, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def psi(expected, actual, bins=10):
    """Population Stability Index on decile bins of the expected distribution."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (c - h, c + h)


def assign_band(pd_values):
    t = THRESHOLDS
    pd_values = np.asarray(pd_values)
    return np.where(pd_values < t["band_approve"], "Approve",
                    np.where(pd_values >= t["band_decline"], "Decline", "Review"))


def artifact_hash(model) -> str:
    return hashlib.sha256(pickle.dumps(model)).hexdigest()


# ------------------------------------------------------- 1. data quality
def test_data_quality(X_new: pd.DataFrame, X_train: pd.DataFrame) -> list[TestResult]:
    L, out = "Chất lượng dữ liệu", []
    missing_cols = [c for c in FEATURES if c not in X_new.columns]
    out.append(_r(L, "Cấu trúc (schema)", "Đủ 12 biến của Bảng 3.1",
                  "thiếu: " + (", ".join(missing_cols) or "không"), not missing_cols))
    if missing_cols:                      # later checks need the full schema
        return out

    problems = []
    if (X_new["LOAN"].dropna() <= 0).any():
        problems.append("LOAN <= 0")
    for c in ["MORTDUE", "VALUE", "DEBTINC", "CLAGE", "YOJ"]:
        if (X_new[c].dropna() < 0).any():
            problems.append(f"{c} < 0")
    for c in ["DEROG", "DELINQ", "NINQ", "CLNO"]:
        v = X_new[c].dropna()
        if (v < 0).any() or (v != np.round(v)).any():
            problems.append(f"{c} không phải số nguyên không âm")
    allowed = {"REASON": set(REASONS), "JOB": set(JOBS)}
    for c in CATEGORICAL:
        bad = set(X_new[c].dropna().unique()) - allowed[c]
        if bad:
            problems.append(f"{c} có nhóm lạ: {sorted(bad)}")
    out.append(_r(L, "Miền giá trị", "Không vi phạm miền giá trị",
                  "; ".join(problems) or "không vi phạm", not problems))

    tol = THRESHOLDS["missing_drift_pp"]
    drift = (X_new[FEATURES].isna().mean() - X_train[FEATURES].isna().mean())
    worst = drift.idxmax()
    out.append(_r(L, "Tỷ lệ khuyết", f"≤ tập huấn luyện + {tol*100:.0f} điểm %",
                  f"chênh lệch lớn nhất: {worst} {drift[worst]*100:+.1f} điểm %",
                  drift.max() <= tol))
    return out


# ------------------------------------------------------------- 2. model
def test_model(y, pd_champ, pd_base, pd_uncal, pd_reference_scores,
               champion=None, X_ref: pd.DataFrame | None = None,
               medians: pd.Series | None = None) -> list[TestResult]:
    L, t, out = "Mô hình", THRESHOLDS, []
    y = np.asarray(y)

    a1, a0, p = delong(y, pd_champ, pd_base)
    lo, hi = bootstrap_auc_ci(y, pd_champ)
    out.append(_r(L, "Khả năng phân biệt",
                  f"AUC vượt mô hình cơ sở (DeLong p < {t['delong_alpha']})",
                  f"AUC {a1:.3f} [{lo:.3f}; {hi:.3f}] so với {a0:.3f}; Gini {2*a1-1:.3f}; p = {p:.2g}",
                  a1 > a0 and p < t["delong_alpha"]))

    gap = pd_champ.mean() - y.mean()
    out.append(_r(L, "Hiệu chỉnh tổng thể", f"|PD TB − tỷ lệ vỡ nợ| ≤ {t['calib_overall_pp']*100:.0f} điểm %",
                  f"{pd_champ.mean()*100:.1f}% so với {y.mean()*100:.1f}% ({gap*100:+.1f} điểm %)",
                  abs(gap) <= t["calib_overall_pp"]))

    dec = pd.qcut(pd.Series(pd_champ).rank(method="first"), 10, labels=False)
    tab = pd.DataFrame({"d": dec, "p": pd_champ, "y": y}).groupby("d").mean()
    tab["gap"] = tab["p"] - tab["y"]
    worst = tab["gap"].abs().idxmax()
    out.append(_r(L, "Hiệu chỉnh theo decile", f"Mỗi decile ≤ {t['calib_decile_pp']*100:.0f} điểm %",
                  f"lệch lớn nhất: decile {worst+1}: {tab.loc[worst,'p']*100:.1f}% so với "
                  f"{tab.loc[worst,'y']*100:.1f}% ({tab.loc[worst,'gap']*100:+.1f} điểm %)",
                  tab["gap"].abs().max() <= t["calib_decile_pp"]))

    b1, b0 = brier_score_loss(y, pd_champ), brier_score_loss(y, pd_uncal)
    out.append(_r(L, "Brier score", "Sau hiệu chỉnh ≤ trước hiệu chỉnh",
                  f"{b1:.4f} so với {b0:.4f}", b1 <= b0))

    v = psi(pd_reference_scores, pd_champ)
    out.append(_r(L, "Ổn định (PSI)", f"PSI < {t['psi_max']}",
                  f"PSI = {v:.3f} (hai mẫu ngẫu nhiên cùng quần thể)", v < t["psi_max"]))

    if champion is not None and X_ref is not None and medians is not None:
        out.append(median_sensitivity(champion, X_ref, medians))
    return out


def median_sensitivity(champion, X_ref, medians, n=200, seed=42) -> TestResult:
    """Mean |ΔPD| when one variable is set to its exact train median vs. the
    median rounded to 2 significant figures or shifted by +1%.

    A smooth model barely moves. A model that has memorised the imputed median
    moves a lot on average (the r2 artefact: one applicant went 97.4% -> 5.8%).
    The mean over applicants is used, not the max, because a single applicant
    sitting on a genuine split point can move without any artefact.
    """
    rows = X_ref.sample(min(n, len(X_ref)), random_state=seed).copy()
    worst, worst_col = 0.0, None
    for c in NUMERIC:
        med = float(medians[c])
        if med == 0:
            continue
        pds = {}
        for k, val in {"exact": med, "rounded": float(f"{med:.2g}"), "shifted": med * 1.01}.items():
            r = rows.copy()
            r[c] = val
            pds[k] = champion.predict_pd(r)
        jump = float(np.mean(np.maximum(np.abs(pds["exact"] - pds["rounded"]),
                                        np.abs(pds["exact"] - pds["shifted"]))))
        if jump > worst:
            worst, worst_col = jump, c
    tol = THRESHOLDS["median_sensitivity_pp"]
    return _r("Mô hình", "Độ nhạy điền khuyết",
              f"|ΔPD| trung bình quanh trung vị ≤ {tol*100:.0f} điểm %",
              f"lớn nhất: {worst_col} {worst*100:.1f} điểm %", worst <= tol)


# ------------------------------------------------------- 3. integration
def test_integration(app_model, registry: dict, X_check: pd.DataFrame,
                     pipeline_pd: np.ndarray) -> list[TestResult]:
    L, out = "Tích hợp", []
    h = artifact_hash(app_model)
    out.append(_r(L, "Phiên bản mô hình", "Mã băm = phiên bản champion đã duyệt",
                  f"{h[:12]}… so với {registry['hash'][:12]}…", h == registry["hash"]))
    same = list(app_model.feature_names) == list(registry["features"])
    out.append(_r(L, "Danh sách biến", "Danh sách và thứ tự biến trùng khớp",
                  f"{len(app_model.feature_names)} biến, {'trùng' if same else 'lệch'}", same))
    diff = float(np.max(np.abs(app_model.predict_pd(X_check) - pipeline_pd)))
    out.append(_r(L, "PD ứng dụng = PD pipeline", f"Sai khác < {THRESHOLDS['integration_tol']:g}",
                  f"sai khác lớn nhất {diff:.1e}", diff < THRESHOLDS["integration_tol"]))
    return out


# --------------------------------------------------- 4. business rules
def test_business_rules(y, pd_champ) -> list[TestResult]:
    L, t, out = "Quy tắc nghiệp vụ", THRESHOLDS, []
    a, d = t["band_approve"], t["band_decline"]
    probe = np.array([a - 1e-9, a, a + 1e-9, d - 1e-9, d, d + 1e-9, 0.0, 1.0])
    expected = ["Approve", "Review", "Review", "Review", "Decline", "Decline", "Approve", "Decline"]
    got = list(assign_band(probe))
    out.append(_r(L, "Phân nhóm tại giá trị biên",
                  f"PD = {a:.0%} → Xem xét; PD = {d:.0%} → Từ chối",
                  "đúng" if got == expected else f"sai: {got}", got == expected))

    bands = assign_band(pd_champ)
    y = np.asarray(y)
    rows = []
    for b in ["Approve", "Review", "Decline"]:
        mask = bands == b
        k, n = int(y[mask].sum()), int(mask.sum())
        lo, hi = wilson_ci(k, n)
        rows.append((b, n, k / n if n else np.nan, lo, hi))
    rates = [r[2] for r in rows]
    mono = all(np.diff(rates) > 0)
    txt = " → ".join(f"{r[0]} {r[2]*100:.1f}% [{r[3]*100:.1f}; {r[4]*100:.1f}] (n={r[1]})" for r in rows)
    out.append(_r(L, "Tỷ lệ vỡ nợ theo nhóm", "Tăng đơn điệu Phê duyệt → Xem xét → Từ chối", txt, mono))
    return out


# --------------------------------------------------- 5. explainability
def test_explainability(champion, X: pd.DataFrame) -> list[TestResult]:
    L, out = "Khả năng giải thích", []
    base, contrib = champion.contributions(X)
    err = float(np.max(np.abs(base + contrib.sum(axis=1).values - champion.margin(X))))
    tol = THRESHOLDS["shap_additivity_tol"]
    out.append(_r(L, "Tính cộng SHAP (12 biến gốc)", f"|cơ sở + ΣSHAP − đầu ra| < {tol:g}",
                  f"sai khác lớn nhất {err:.1e}", err < tol))

    wrong = []
    signs = {}
    for var, sign in EXPECTED_SIGN.items():
        ok = X[var].notna()
        a, b = X.loc[ok, var], contrib.loc[ok, var]
        rho = 0.0 if b.nunique() < 2 else stats.spearmanr(a, b).correlation
        signs[var] = rho
        if np.sign(rho) != sign:
            wrong.append(var)
    out.append(_r(L, "Chiều tác động", "Cùng chiều với mục 3.2 (DEBTINC, DELINQ, DEROG, NINQ +; CLAGE −)",
                  ", ".join(f"{k} ρ={v:+.2f}" for k, v in signs.items()), not wrong))

    pdv = champion.predict_pd(X)
    declined = contrib[assign_band(pdv) == "Decline"]
    top3 = declined.apply(lambda r: list(r.nlargest(3).index), axis=1)
    n_ok = int(sum(len(v) == 3 and all(x in FEATURES for x in v) for v in top3))
    out.append(_r(L, "Lý do từ chối", "3 lý do từ 12 biến đầu vào cho mỗi hồ sơ bị từ chối",
                  f"{n_ok}/{len(declined)} hồ sơ", n_ok == len(declined) and len(declined) > 0))
    return out


# ------------------------------------------------------ 6. regression
def test_regression(golden_X, golden_pd, new_model, y=None, baseline_auc=None,
                    retrained=False) -> list[TestResult]:
    L, t, out = "Hồi quy", THRESHOLDS, []
    new_pd = new_model.predict_pd(golden_X)
    if not retrained:
        d = float(np.max(np.abs(new_pd - golden_pd)))
        out.append(_r(L, "PD trên golden dataset", f"|ΔPD| ≤ {t['regression_pd_tol']}",
                      f"|ΔPD| lớn nhất {d:.1e}", d <= t["regression_pd_tol"]))
    else:
        auc_new = roc_auc_score(y, new_pd)
        drop = baseline_auc - auc_new
        out.append(_r(L, "AUC sau huấn luyện lại", f"ΔAUC ≥ −{t['regression_auc_drop']}",
                      f"{auc_new:.3f} so với {baseline_auc:.3f} ({-drop:+.4f})",
                      drop <= t["regression_auc_drop"]))
        moved = float(np.mean(assign_band(new_pd) != assign_band(golden_pd)))
        out.append(_r(L, "Đổi nhóm quyết định", f"≤ {t['regression_band_change']:.0%} hồ sơ",
                      f"{moved*100:.2f}% hồ sơ", moved <= t["regression_band_change"]))
    return out


def to_frame(results: list[TestResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])
    df["Đánh giá"] = np.where(df["passed"], "Đạt", "Không đạt")
    return df.rename(columns={"layer": "Lớp kiểm thử", "test": "Kiểm thử",
                              "criterion": "Tiêu chí", "result": "Kết quả"})
