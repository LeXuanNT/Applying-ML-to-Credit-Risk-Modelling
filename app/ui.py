"""Dữ liệu và thành phần giao diện dùng chung cho ba trang của ứng dụng.

Mọi con số hiển thị đều đọc từ `results/v3/*.csv` (kết quả Chương 4) hoặc tính trực tiếp từ
`models/champion.joblib` trên tập kiểm tra đã lưu trong artefact. Không có số nào được gõ tay.
"""
from __future__ import annotations

import ast
import html
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "v3"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from creditscore.data import RAW_FEATURES, TARGET  # noqa: E402
from creditscore.decision_support import BANDS, Explainer  # noqa: E402
from creditscore.model_testing import bootstrap_auc_ci  # noqa: E402

# ------------------------------------------------------------------ palette (khớp .streamlit/config.toml)
INK, MUTED, LINE, TEAL = "#14213D", "#5B6475", "#D8DEE8", "#0F766E"
BAND_STYLE = {  # nền, chữ, màu trên thang PD
    BANDS[0]: ("#E6F4F1", "#0F5F58", "#5FB8AA"),
    BANDS[1]: ("#FEF3C7", "#854D0E", "#F2C75C"),
    BANDS[2]: ("#FDEEE6", "#9A3412", "#E8875A"),
}
MODEL_NAMES = {"LGBM": "LightGBM", "XGB": "XGBoost", "RF": "Random Forest", "LR": "Logistic Regression",
               "DT": "Decision Tree", "SVM": "SVM", "KNN": "KNN", "MLP": "MLP", "GNB": "Gaussian NB"}
CHAMPION = "LGBM"
BASELINE = "LR"


# ------------------------------------------------------------------ data
@st.cache_resource
def artifact() -> dict:
    return joblib.load(ROOT / "models" / "champion.joblib")


@st.cache_data
def results() -> dict[str, pd.DataFrame]:
    names = ["final_test", "evaluation", "gini_ks_test", "selection", "tuning", "stability", "split",
             "decision_bands_test", "calibration_test", "calibration_deciles", "shap_importance_test",
             "feature_importance", "example_applicants"]
    return {n: pd.read_csv(RES / f"{n}.csv") for n in names}


def explainer(approve: float | None = None, decline: float | None = None) -> Explainer:
    a = artifact()
    approve = a["approve_below"] if approve is None else approve
    decline = a["decline_from"] if decline is None else decline
    return Explainer(a["pipeline"], a["platt_a"], a["platt_b"], approve, max(decline, approve + 0.01))


@st.cache_data
def champion_test_scores() -> tuple[np.ndarray, np.ndarray]:
    """Nhãn thật và PD đã hiệu chỉnh của champion trên tập kiểm tra (n = 894)."""
    te = artifact()["test_applicants"]
    return te[TARGET].to_numpy(), explainer().pd(te[RAW_FEATURES])


@st.cache_data
def champion_auc_ci() -> dict:
    """Cùng thủ tục bootstrap (1,000 lần, seed 42) với results/v3/model_tests.csv."""
    y, p = champion_test_scores()
    lo, hi = bootstrap_auc_ci(y, p)
    return {"auc_lo": lo, "auc_hi": hi, "gini_lo": 2 * lo - 1, "gini_hi": 2 * hi - 1}


def model_test_metrics(code: str) -> dict:
    """Chỉ số trên tập kiểm tra. Champion: mô hình cuối (huấn luyện lại trên train + validation)."""
    r = results()
    gk = r["gini_ks_test"].set_index("model")
    if code == CHAMPION:
        row = r["final_test"].iloc[0]
        g = gk.loc["LGBM_final"]
        cal = r["calibration_test"].set_index("version")["brier"]
        brier = float(cal.iloc[-1])
    else:
        ev = r["evaluation"]
        row = ev[(ev.model == code) & (ev.split == "test")].iloc[0]
        g = gk.loc[code]
        brier = None
    return {k: float(row[k]) for k in ["auc", "f1", "precision", "recall", "mcc"]} | {
        "gini": float(g.gini), "ks": float(g.ks), "brier": brier,
        **{k: int(row[k]) for k in ["tn", "fp", "fn", "tp"]}}


def selection_table() -> pd.DataFrame:
    """Bảng chọn mô hình trên tập kiểm định, kèm bộ điền khuyết / xử lý mất cân bằng lớp đã chọn và PSI."""
    r = results()
    sel = r["selection"].copy()
    tun = r["tuning"].set_index("model")
    params = tun["best_params"].map(ast.literal_eval)
    sel["imputer"] = sel.model.map(lambda m: params[m].get("prep__imputer", "—"))
    sel["sampler"] = sel.model.map(lambda m: params[m].get("sampler__kind", "—"))
    sel["cv_f1"] = sel.model.map(tun["cv_f1"])
    sel["psi"] = sel.model.map(r["stability"].set_index("model")["psi_valid_test"])
    sel["name"] = sel.model.map(MODEL_NAMES)
    return sel.sort_values("avg_rank_all").reset_index(drop=True)


# ------------------------------------------------------------------ formatting
def f3(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def pct(v, d: int = 1) -> str:
    return f"{100 * v:.{d}f}%"


def esc(s) -> str:
    return html.escape(str(s))


# ------------------------------------------------------------------ html components
CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
.cs-eyebrow {font-size: 0.82rem; color: #5B6475; margin-bottom: 0.15rem;}
.cs-title {font-size: 1.9rem; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 0.4rem; color: #14213D;}
.cs-chips {display: flex; flex-wrap: wrap; gap: 8px; margin: 0.2rem 0 1rem;}
.cs-chip {display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px;
          background: #FFFFFF; border: 1px solid #D8DEE8; font-size: 0.82rem; color: #14213D;}
.cs-chip.ok {background: #E6F4F1; border-color: #B5DFD7; color: #0F5F58;}
.cs-kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 1rem;}
.cs-card {background: #FFFFFF; border: 1px solid #D8DEE8; border-radius: 14px; padding: 16px 18px;}
.cs-kpi-label {font-size: 0.82rem; color: #5B6475;}
.cs-kpi-value {font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 1.6rem; font-weight: 600;
               color: #14213D; line-height: 1.25; margin: 2px 0;}
.cs-kpi-sub {font-size: 0.75rem; color: #5B6475;}
.cs-cm {display: grid; grid-template-columns: 96px 1fr 1fr; gap: 8px; font-size: 0.82rem;}
.cs-cm .h {color: #5B6475; text-align: center; align-self: end;}
.cs-cm .r {color: #5B6475; display: flex; align-items: center;}
.cs-cell {border-radius: 10px; padding: 14px 6px; text-align: center;}
.cs-cell b {display: block; font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem;}
.cs-good {background: #E6F4F1; color: #0F5F58;} .cs-bad {background: #FDEEE6; color: #9A3412;}
.cs-step {font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600; color: #0F766E;}
.cs-step-t {font-weight: 600; margin: 4px 0;} .cs-step-d {font-size: 0.85rem; color: #5B6475;}
.cs-badge {display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: 1rem;}
.cs-pd {font-family: 'IBM Plex Mono', monospace; font-size: 3rem; font-weight: 600; line-height: 1.05; color: #14213D;}
.cs-scale {display: flex; height: 14px; border-radius: 7px; overflow: hidden; margin-top: 14px;}
.cs-marker {position: relative; height: 14px;}
.cs-ticks {position: relative; height: 16px; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #5B6475;}
.cs-rec {border-radius: 14px; padding: 16px 20px;}
.cs-rec h4 {margin: 0 0 6px; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase;}
.cs-rec ul {margin: 0; padding-left: 1.1rem;} .cs-rec li {margin: 4px 0; line-height: 1.5;}
.cs-side {border: 1px solid #2E3F68; border-radius: 12px; padding: 12px 14px; font-size: 0.82rem; margin-top: 8px;}
.cs-side .k {color: #A9B4C6;} .cs-side .v {font-family: 'IBM Plex Mono', monospace; color: #FFFFFF;}
.cs-side div {display: flex; justify-content: space-between; margin: 4px 0;}
.cs-side .t {font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A9B4C6; margin-bottom: 6px;}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def header(eyebrow: str, title: str) -> None:
    st.markdown(f"<div class='cs-eyebrow'>{esc(eyebrow)}</div><div class='cs-title'>{esc(title)}</div>",
                unsafe_allow_html=True)


def chips(items: list[tuple[str, bool]]) -> None:
    body = "".join(f"<span class='cs-chip{' ok' if ok else ''}'>{'✓ ' if ok else ''}{esc(t)}</span>"
                   for t, ok in items)
    st.markdown(f"<div class='cs-chips'>{body}</div>", unsafe_allow_html=True)


def kpis(cards: list[tuple[str, str, str]]) -> None:
    body = "".join(f"<div class='cs-card'><div class='cs-kpi-label'>{esc(l)}</div>"
                   f"<div class='cs-kpi-value'>{esc(v)}</div><div class='cs-kpi-sub'>{esc(s)}</div></div>"
                   for l, v, s in cards)
    st.markdown(f"<div class='cs-kpis'>{body}</div>", unsafe_allow_html=True)


def confusion(m: dict) -> None:
    st.markdown(
        "<div class='cs-cm'><span></span><span class='h'>Dự báo: Tốt</span><span class='h'>Dự báo: Xấu</span>"
        f"<span class='r'>Thực tế: Tốt</span><div class='cs-cell cs-good'><b>{m['tn']}</b>TN</div>"
        f"<div class='cs-cell cs-bad'><b>{m['fp']}</b>FP · từ chối nhầm</div>"
        f"<span class='r'>Thực tế: Xấu</span><div class='cs-cell cs-bad'><b>{m['fn']}</b>FN · bỏ sót vỡ nợ</div>"
        f"<div class='cs-cell cs-good'><b>{m['tp']}</b>TP</div></div>", unsafe_allow_html=True)


def sidebar_summary() -> None:
    sp = results()["split"].set_index("split")
    tot, te = sp.loc["total"], sp.loc["test"]
    st.sidebar.markdown(
        "<div class='cs-side'><div class='t'>Dữ liệu HMEQ</div>"
        f"<div><span class='k'>Số hồ sơ</span><span class='v'>{int(tot.rows):,}</span></div>"
        f"<div><span class='k'>Tỷ lệ vỡ nợ</span><span class='v'>{pct(tot.bad_rate)}</span></div>"
        f"<div><span class='k'>Biến đầu vào</span><span class='v'>{len(RAW_FEATURES)}</span></div>"
        f"<div><span class='k'>Tập kiểm tra</span><span class='v'>{int(te.rows):,}</span></div>"
        f"<div><span class='k'>Mô hình</span><span class='v'>{esc(artifact()['champion_label'])}</span></div></div>",
        unsafe_allow_html=True)


def hbar(df: pd.DataFrame, x: str, y: str, color: str = TEAL, fmt: str = ".1%", height: int | None = None,
         x_title: str | None = None):
    h = height or max(180, 30 * len(df))
    base = alt.Chart(df).encode(y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)))
    bars = base.mark_bar(color=color, cornerRadiusEnd=4, height=18).encode(
        x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format=fmt, grid=False)),
        tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)])
    text = base.mark_text(align="left", dx=4, font="IBM Plex Mono", fontSize=12, color=INK).encode(
        x=f"{x}:Q", text=alt.Text(f"{x}:Q", format=fmt))
    return (bars + text).properties(height=h)
