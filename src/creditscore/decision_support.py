"""Per-applicant explanation and decision support for the champion (stage 2 of the thesis framework).

* PD: the champion's raw score is mapped to a calibrated probability of default with Platt scaling
  (Platt, 1999) fitted on 10-fold out-of-fold scores of the data the champion was trained on.
* Reasons: exact TreeSHAP contributions (Lundberg et al., 2020) from LightGBM `pred_contrib` / XGBoost `pred_contribs`,
  on the log-odds scale. Platt scaling is linear in the log-odds, so the contributions remain exactly additive
  on the calibrated scale: logit(PD) = A * (phi_0 + sum_j phi_j) + B.
* Contributions of engineered columns are summed back to the 12 application variables: polynomial squares and
  missing indicators go to their source variable, an interaction LOAN x VALUE is split equally between LOAN and
  VALUE, one-hot levels and YOJ bins go to JOB / REASON / YOJ.
* Recommendation: three PD bands. The cut-offs are illustrative policy settings (risk appetite of the bank), not
  estimated quantities, and can be changed in the web application.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RAW_FEATURES

APPROVE_BELOW = 0.10          # PD < 10%            -> approve
DECLINE_FROM = 0.50           # PD >= 50%           -> decline (default more likely than not)
BANDS = ("Phê duyệt", "Xem xét thủ công", "Từ chối")


def source_variables(col: str) -> list[str]:
    """Raw application variables an engineered column is built from."""
    if col.endswith("_MISSING"):
        return [col[: -len("_MISSING")]]
    if col.startswith("YOJ_BIN"):
        return ["YOJ"]
    for cat in ("JOB", "REASON"):
        if col.startswith(cat + "_"):
            return [cat]
    if "^" in col:
        return [col.split("^")[0]]
    if " " in col:
        return col.split(" ")
    return [col]


def _clf_kind(clf) -> str:
    n = type(clf).__name__
    if n.startswith("LGBM"):
        return "lgbm"
    if n.startswith("XGB"):
        return "xgb"
    raise ValueError(f"exact log-odds TreeSHAP implemented for LightGBM / XGBoost only, got {n}")


def _margin(pipe, X: pd.DataFrame) -> np.ndarray:
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Z = prep.transform(X[RAW_FEATURES])
    if _clf_kind(clf) == "lgbm":
        return clf.booster_.predict(Z.values, raw_score=True)
    import xgboost as xgb
    return clf.get_booster().predict(xgb.DMatrix(Z.values, feature_names=list(Z.columns)), output_margin=True)


def _contribs(pipe, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Z = prep.transform(X[RAW_FEATURES])
    if _clf_kind(clf) == "lgbm":
        c = clf.booster_.predict(Z.values, pred_contrib=True)
    else:
        import xgboost as xgb
        c = clf.get_booster().predict(xgb.DMatrix(Z.values, feature_names=list(Z.columns)), pred_contribs=True)
    return pd.DataFrame(c[:, :-1], columns=Z.columns, index=X.index), c[:, -1]


def raw_contributions(pipe, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """TreeSHAP contributions (raw log-odds) summed to the 12 application variables, plus the base value."""
    phi, base = _contribs(pipe, X)
    out = pd.DataFrame(0.0, index=X.index, columns=RAW_FEATURES)
    for col in phi.columns:
        src = source_variables(col)
        for v in src:
            out[v] += phi[col] / len(src)
    return out, base


class Explainer:
    def __init__(self, pipe, platt_a: float, platt_b: float,
                 approve_below: float = APPROVE_BELOW, decline_from: float = DECLINE_FROM):
        self.pipe, self.a, self.b = pipe, float(platt_a), float(platt_b)
        self.approve_below, self.decline_from = approve_below, decline_from

    def margin(self, X):
        return _margin(self.pipe, X)

    def pd(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * self.margin(X) + self.b)))

    def band(self, p) -> np.ndarray:
        p = np.asarray(p)
        return np.where(p < self.approve_below, BANDS[0], np.where(p >= self.decline_from, BANDS[2], BANDS[1]))

    def explain(self, X) -> tuple[pd.DataFrame, np.ndarray]:
        """Contributions on the CALIBRATED log-odds scale, per application variable, and the calibrated base."""
        phi, base = raw_contributions(self.pipe, X)
        return phi * self.a, self.a * base + self.b

    def report(self, row: pd.DataFrame, top: int = 3) -> dict:
        """Everything the web application shows for one applicant."""
        p = float(self.pd(row)[0])
        phi, base = self.explain(row)
        s = phi.iloc[0].sort_values(key=np.abs, ascending=False)
        up = s[s > 0].head(top)
        down = s[s < 0].head(top)
        return {"pd": p, "band": str(self.band([p])[0]), "base_logodds": float(base[0]),
                "contributions": s, "risk_up": up, "risk_down": down}


def fit_platt(raw_margin_oof: np.ndarray, y) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(raw_margin_oof).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


LABELS_VI = {
    "LOAN": "Số tiền vay", "MORTDUE": "Dư nợ thế chấp", "VALUE": "Giá trị tài sản bảo đảm",
    "REASON": "Mục đích vay", "JOB": "Nghề nghiệp", "YOJ": "Số năm làm việc",
    "DEROG": "Số báo cáo xấu", "DELINQ": "Số hạn mức quá hạn", "CLAGE": "Tuổi hạn mức lâu nhất (tháng)",
    "NINQ": "Số lần truy vấn tín dụng gần đây", "CLNO": "Số hạn mức tín dụng", "DEBTINC": "Tỷ lệ nợ/thu nhập (%)",
}


def fmt_pd(p: float) -> str:
    return "> 99.9%" if p > 0.999 else ("< 0.1%" if p < 0.001 else f"{100 * p:.1f}%")


def _val(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "không khai báo"
    return f"{v:,.1f}" if isinstance(v, (float, np.floating)) else str(v)


def recommendation(rep: dict, row: pd.Series) -> list[str]:
    """Plain-language guidance for the credit officer (Vietnamese), built from the PD band and the SHAP reasons."""
    p, band = rep["pd"], rep["band"]
    out = []
    if band == BANDS[0]:
        out.append(f"Đề xuất PHÊ DUYỆT: xác suất vỡ nợ {fmt_pd(p)} thấp hơn ngưỡng phê duyệt.")
    elif band == BANDS[2]:
        out.append(f"Đề xuất TỪ CHỐI: xác suất vỡ nợ {fmt_pd(p)}, khả năng vỡ nợ cao hơn khả năng trả nợ.")
    else:
        out.append(f"Đề xuất XEM XÉT THỦ CÔNG: xác suất vỡ nợ {fmt_pd(p)} nằm giữa hai ngưỡng chính sách.")
    if len(rep["risk_up"]):
        reasons = "; ".join(f"{LABELS_VI[v]} = {_val(row[v])}" for v in rep["risk_up"].index)
        out.append(f"Yếu tố làm tăng rủi ro nhiều nhất: {reasons}.")
    if len(rep["risk_down"]):
        good = "; ".join(f"{LABELS_VI[v]} = {_val(row[v])}" for v in rep["risk_down"].index)
        out.append(f"Yếu tố làm giảm rủi ro: {good}.")
    missing = [v for v in RAW_FEATURES if pd.isna(row[v])]
    flagged = [v for v in missing if v in rep["risk_up"].index]
    if flagged:
        out.append("Hồ sơ thiếu thông tin " + ", ".join(LABELS_VI[v] for v in flagged)
                   + " và việc thiếu này làm tăng rủi ro theo mô hình: đề nghị bổ sung chứng từ trước khi quyết định.")
    if band == BANDS[1] and len(rep["risk_up"]):
        out.append("Khi thẩm định thủ công, ưu tiên xác minh các yếu tố làm tăng rủi ro ở trên "
                   "(ví dụ: lịch sử quá hạn, thu nhập để tính lại tỷ lệ nợ/thu nhập, định giá tài sản).")
    if band == BANDS[2]:
        out.append("Nêu rõ các yếu tố làm tăng rủi ro ở trên khi thông báo lý do từ chối cho khách hàng.")
    return out
