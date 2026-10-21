"""Trang 2 - So sánh mô hình (Compare Models): quy tắc chọn champion và bảng 9 thuật toán."""
import altair as alt
import pandas as pd
import streamlit as st

import ui

sel = ui.selection_table()
n_valid = int(ui.results()["split"].set_index("split").loc["valid", "rows"])
champ = sel[sel.selected].iloc[0]
failed0 = ", ".join(sel.loc[~sel.step0_pass, "name"])
passed1 = sel.loc[sel.step1_pass, "name"].tolist()

ui.header(f"{len(sel)} thuật toán · tập kiểm định (validation) n = {n_valid}", "So sánh mô hình")

steps = [
    ("BƯỚC 0", "Khả năng giải thích (Explainability)",
     f"Đạt / không đạt: mô hình phải có SHAP chính xác (TreeSHAP hoặc tuyến tính). Bị loại: {failed0}."),
    ("BƯỚC 1", "AUC cao nhất + kiểm định DeLong",
     "Giữ các mô hình không khác biệt có ý nghĩa so với mô hình có AUC cao nhất. "
     f"Còn lại: {', '.join(passed1)}."),
    ("BƯỚC 2", "Phân xử bằng AUC CV 10-fold",
     "Chỉ dùng khi bước 1 còn từ hai mô hình. " + ("Lần chạy này không cần." if len(passed1) == 1 else "")),
]
cols = st.columns(3, gap="medium")
for col, (k, t, d) in zip(cols, steps):
    col.markdown(f"<div class='cs-card' style='height:100%'><div class='cs-step'>{k}</div>"
                 f"<div class='cs-step-t'>{ui.esc(t)}</div><div class='cs-step-d'>{ui.esc(d)}</div></div>",
                 unsafe_allow_html=True)

st.write("")
st.success(f"Champion: **{champ['name']}** — AUC validation {champ.auc:.3f} "
           f"[{champ.auc_lo:.3f}; {champ.auc_hi:.3f}], hạng trung bình {champ.avg_rank_all:.1f}.")

with st.container(border=True):
    st.markdown("**Bảng so sánh** &nbsp; <span style='color:#5B6475;font-size:.85rem'>"
                "xếp theo hạng trung bình trên 5 thước đo (AUC, KS, PR-AUC, Brier, MCC)</span>",
                unsafe_allow_html=True)
    tbl = pd.DataFrame({
        "Mô hình": sel.name + sel.selected.map({True: "  ★ champion", False: ""}),
        "Bước 0": sel.step0_pass.map({True: "Đạt", False: "Không đạt"}),
        "AUC": sel.auc, "KTC 95%": sel.apply(lambda r: f"[{r.auc_lo:.3f}; {r.auc_hi:.3f}]", axis=1),
        "ΔAUC": sel.delta_auc_vs_best,
        "p DeLong": [("—" if m == champ.model else f"{p:.2g}") for m, p in zip(sel.model, sel.p_delong)],
        "KS": sel.ks, "PR-AUC": sel.pr_auc, "Brier": sel.brier, "MCC": sel.mcc,
        "F1 CV": sel.cv_f1, "Hạng TB": sel.avg_rank_all, "PSI": sel.psi,
        "Điền khuyết": sel.imputer, "Mất cân bằng lớp": sel.sampler,
    })
    num3 = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "AUC": num3, "ΔAUC": st.column_config.NumberColumn(format="%+.3f"),
        "KS": num3, "PR-AUC": num3,
        "Brier": st.column_config.NumberColumn(format="%.4f"), "MCC": num3, "F1 CV": num3,
        "Hạng TB": st.column_config.NumberColumn(format="%.1f"), "PSI": num3})
    st.caption("Chỉ số trên tập kiểm định; F1 CV là F1 trung bình 10-fold khi tinh chỉnh siêu tham số. "
               "PSI tính giữa phân phối điểm trên tập kiểm định và tập kiểm tra.")

with st.container(border=True):
    st.markdown("**Hạng trung bình** (thấp hơn là tốt hơn)")
    rk = sel.assign(Nhóm=sel.selected.map({True: "Champion", False: "Mô hình khác"}))
    st.altair_chart(alt.Chart(rk).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("avg_rank_all:Q", title=None, scale=alt.Scale(domain=[0, len(sel)])),
        y=alt.Y("name:N", sort=rk.name.tolist(), title=None),
        color=alt.Color("Nhóm:N", scale=alt.Scale(domain=["Champion", "Mô hình khác"], range=[ui.TEAL, "#8A96AA"]),
                        legend=None),
        tooltip=["name", alt.Tooltip("avg_rank_all:Q", format=".1f")]).properties(height=30 * len(sel)),
        width="stretch")
