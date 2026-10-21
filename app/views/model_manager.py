"""Trang 1 - Quản lý mô hình (Model Manager)."""
import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

import ui
from creditscore.decision_support import LABELS_VI

r = ui.results()
sel = ui.selection_table()
order = sel.model.tolist()

head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
with head_l:
    ui.header("Giai đoạn 3 · Mô hình xác suất vỡ nợ (Probability of Default, PD)", "Quản lý mô hình")
with head_r:
    code = st.selectbox(
        "Mô hình", order, index=order.index(ui.CHAMPION),
        format_func=lambda m: ui.MODEL_NAMES[m] + (" · champion" if m == ui.CHAMPION else
                                                   " · baseline" if m == ui.BASELINE else ""))

is_champ = code == ui.CHAMPION
m = ui.model_test_metrics(code)
champ = ui.model_test_metrics(ui.CHAMPION)
sp = r["split"].set_index("split").loc["test"]
psi = float(r["stability"].set_index("model").loc[code, "psi_valid_test"])
ui.chips([
    (f"Tập kiểm tra (test) · n = {int(sp.rows)} · {int(sp.bads)} hồ sơ vỡ nợ", False),
    ("Chia ngẫu nhiên phân tầng 70/15/15 — HMEQ không có ngày nên không kiểm định ngoài thời gian (OOT)", False),
    (f"PSI validation → test = {psi:.3f}", psi < 0.1),
] + ([("Huấn luyện lại trên train + validation", False)] if is_champ else []))

if is_champ:
    ci = ui.champion_auc_ci()
    auc_sub = f"KTC 95%: [{ci['auc_lo']:.3f}; {ci['auc_hi']:.3f}]"
    gini_sub = f"KTC 95%: [{ci['gini_lo']:.3f}; {ci['gini_hi']:.3f}]"
    base = ui.model_test_metrics(ui.BASELINE)
    delta = ("So với baseline LR", f"{m['auc'] - base['auc']:+.3f}", "ΔAUC trên tập kiểm tra")
else:
    auc_sub, gini_sub = "KTC chỉ tính cho champion", "= 2 · AUC − 1"
    delta = ("So với champion", f"{m['auc'] - champ['auc']:+.3f}", "ΔAUC trên tập kiểm tra")
ui.kpis([
    ("AUC", ui.f3(m["auc"]), auc_sub),
    ("Gini", ui.f3(m["gini"]), gini_sub),
    ("KS", ui.f3(m["ks"]), "Kolmogorov–Smirnov"),
    ("F1", ui.f3(m["f1"]), "Tại ngưỡng của mô hình"),
    ("Brier", "—" if m["brier"] is None else f"{m['brier']:.4f}",
     "Sau hiệu chỉnh Platt" if is_champ else "Chỉ tính cho champion"),
    delta,
])

tab_metrics, tab_feat, tab_bands = st.tabs(["Chỉ số đánh giá", "Biến quan trọng", "Phân nhóm quyết định"])

with tab_metrics:
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        st.markdown(f"**Đường cong ROC** &nbsp; <span style='color:{ui.MUTED};font-size:.85rem'>AUC {m['auc']:.3f}</span>",
                    unsafe_allow_html=True)
        if is_champ:
            y, p = ui.champion_test_scores()
            fpr, tpr, _ = roc_curve(y, p)
            roc = pd.DataFrame({"FPR": fpr, "TPR": tpr})
            diag = pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})
            chart = (alt.Chart(diag).mark_line(strokeDash=[5, 5], color="#9AA4B5").encode(x="FPR:Q", y="TPR:Q")
                     + alt.Chart(roc).mark_line(color=ui.TEAL, strokeWidth=3, interpolate="step-after").encode(
                         x=alt.X("FPR:Q", title="Tỷ lệ dương tính giả (FPR)"),
                         y=alt.Y("TPR:Q", title="Tỷ lệ dương tính thật (TPR)"))).properties(height=320)
            st.altair_chart(chart, width="stretch")
        else:
            st.info("Repository chỉ lưu mô hình champion, nên đường ROC chỉ vẽ được cho LightGBM. "
                    "AUC của mô hình này lấy từ results/v3/gini_ks_test.csv.")
    with c2, st.container(border=True):
        st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
        ui.confusion(m)
        a, b, c = st.columns(3)
        a.metric("Precision", ui.f3(m["precision"]))
        b.metric("Recall", ui.f3(m["recall"]))
        c.metric("MCC", ui.f3(m["mcc"]))

with tab_feat, st.container(border=True):
    if is_champ:
        imp = r["shap_importance_test"].assign(Biến=lambda d: d.feature + " – " + d.feature.map(LABELS_VI))
        st.markdown("**Mức độ quan trọng của biến (SHAP)**")
        st.caption("Tỷ trọng |SHAP| trung bình trên tập kiểm tra; TreeSHAP của các cột kỹ thuật được quy về 12 biến gốc.")
        st.altair_chart(ui.hbar(imp, "share", "Biến"), width="stretch")
    else:
        fi = r["feature_importance"]
        fi = fi[fi.model == code].nsmallest(10, "rank")
        if fi.empty:
            st.info(f"{ui.MODEL_NAMES[code]} không có độ quan trọng nội tại chính xác "
                    "(chỉ giải thích được bằng SHAP lấy mẫu) — đây là lý do bị loại ở bước 0 nếu có.")
        else:
            fi = fi.assign(share=fi.importance / r["feature_importance"].query("model == @code").importance.sum())
            st.markdown("**Độ quan trọng nội tại của mô hình (10 cột đầu)**")
            st.caption("Theo cột đầu vào sau xử lý (kể cả biến chỉ báo thiếu, đa thức); "
                       + ("|hệ số| chuẩn hóa." if code == "LR" else "độ quan trọng của cây."))
            st.altair_chart(ui.hbar(fi, "share", "feature", color="#8A96AA"), width="stretch")

with tab_bands:
    if not is_champ:
        st.info("Phân nhóm quyết định chỉ áp dụng cho mô hình champion (LightGBM).")
    else:
        a, b = ui.artifact()["approve_below"], ui.artifact()["decline_from"]
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1, st.container(border=True):
            st.markdown("**Phân nhóm quyết định trên tập kiểm tra**")
            st.caption(f"PD sau hiệu chỉnh Platt: < {a:.0%} phê duyệt · {a:.0%}–{b:.0%} xem xét thủ công · ≥ {b:.0%} từ chối")
            bands = r["decision_bands_test"].set_index("band").reindex(list(ui.BAND_STYLE)).reset_index()
            show = pd.DataFrame({"Nhóm": bands.band, "Số hồ sơ": bands.n, "Tỷ trọng": 100 * bands.share,
                                 "Tỷ lệ vỡ nợ": 100 * bands.default_rate,
                                 "Phần vỡ nợ bắt được": 100 * bands.defaults_share})
            p1 = st.column_config.NumberColumn(format="%.1f%%")
            st.dataframe(show, hide_index=True, width="stretch",
                         column_config={"Tỷ trọng": p1, "Tỷ lệ vỡ nợ": p1, "Phần vỡ nợ bắt được": p1})
            cal = r["calibration_test"].set_index("version")
            st.caption(f"Hiệu chỉnh Platt giảm Brier từ {cal.brier.iloc[0]:.4f} xuống {cal.brier.iloc[1]:.4f}; "
                       f"PD trung bình {ui.pct(cal.mean_pd.iloc[1])} so với tỷ lệ vỡ nợ thực tế "
                       f"{ui.pct(cal.observed_rate.iloc[1])}.")
        with c2, st.container(border=True):
            st.markdown("**Hiệu chỉnh theo decile (Calibration)**")
            dec = r["calibration_deciles"].assign(decile=lambda d: d.decile + 1).melt("decile", ["mean_pd", "observed"], "Loại", "Tỷ lệ")
            dec["Loại"] = dec["Loại"].map({"mean_pd": "PD dự báo", "observed": "Vỡ nợ thực tế"})
            st.altair_chart(alt.Chart(dec).mark_bar().encode(
                x=alt.X("decile:O", title="Decile PD (1 = thấp nhất)", axis=alt.Axis(labelAngle=0)), xOffset="Loại:N",
                y=alt.Y("Tỷ lệ:Q", axis=alt.Axis(format="%")),
                color=alt.Color("Loại:N", scale=alt.Scale(range=[ui.TEAL, "#E8875A"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["decile", "Loại", alt.Tooltip("Tỷ lệ:Q", format=".1%")]).properties(height=300),
                width="stretch")
