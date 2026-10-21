"""Trang 3 - Chấm điểm hồ sơ: PD đã hiệu chỉnh, SHAP theo 12 biến gốc và khuyến nghị cho cán bộ tín dụng."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui
from creditscore.data import CAT_FEATURES, DESCRIPTIONS, NUM_FEATURES, RAW_FEATURES, TARGET
from creditscore.decision_support import BANDS, LABELS_VI, fmt_pd, recommendation

art = ui.artifact()
te = art["test_applicants"]
examples = ui.results()["example_applicants"].set_index("case")
CASES = {"approve": "Hồ sơ mẫu: phê duyệt", "review": "Hồ sơ mẫu: xem xét", "decline": "Hồ sơ mẫu: từ chối"}

def fmt_value(v) -> str:
    if pd.isna(v):
        return "(thiếu)"
    if isinstance(v, (int, float, np.number)):
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"
    return str(v)


ui.header("Giai đoạn 4 · Hỗ trợ quyết định cho cán bộ tín dụng", "Chấm điểm hồ sơ")
source = st.segmented_control("Nguồn hồ sơ", ["Hồ sơ mẫu", "Hồ sơ trong tập kiểm tra", "Nhập hồ sơ mới"],
                              default="Hồ sơ mẫu", key="source") or "Hồ sơ mẫu"

left, right = st.columns([5, 6], gap="large")
actual = None
with left, st.container(border=True):
    st.markdown("**Thông tin hồ sơ**")
    if source == "Nhập hồ sơ mới":
        # Giá trị gợi ý là trung vị ĐÃ LÀM TRÒN: trung vị chính xác là giá trị điền khuyết, cây đọc nó như "thiếu".
        typical = {c: (round(v, -2) if v > 1000 else round(v, 1)) for c, v in art["medians"].items()}
        vals = {}
        grid = st.columns(2)
        for k, c in enumerate(NUM_FEATURES):
            with grid[k % 2]:
                lo, hi = art["ranges"][c]
                miss = st.checkbox(f"{c}: không có thông tin", key=f"m_{c}")
                vals[c] = np.nan if miss else st.number_input(
                    f"{LABELS_VI[c]} ({c})", float(lo), float(hi), float(typical[c]), key=c, help=DESCRIPTIONS[c])
        for k, c in enumerate(CAT_FEATURES):
            with grid[k % 2]:
                opts = ["(không có thông tin)"] + art["categories"][c]
                opt = st.selectbox(f"{LABELS_VI[c]} ({c})", opts, index=opts.index(art["modes"][c]), key=f"s_{c}")
                vals[c] = np.nan if opt.startswith("(") else opt
        row = pd.DataFrame([vals])[RAW_FEATURES]
        st.caption("Ô đánh dấu “không có thông tin” được giữ là thiếu: mô hình dùng biến chỉ báo thiếu "
                   "(ví dụ DEBTINC_MISSING) chứ không điền trung vị.")
    else:
        if source == "Hồ sơ mẫu":
            case = st.radio("Chọn hồ sơ mẫu", list(CASES), format_func=CASES.get, horizontal=True, key="case")
            i = int(examples.loc[case, "test_index"])
        else:
            i = int(st.number_input(f"Số thứ tự hồ sơ trong tập kiểm tra (0 – {len(te) - 1})", 0, len(te) - 1, 134,
                                    key="idx"))
        row = te.iloc[[i]][RAW_FEATURES].reset_index(drop=True)
        actual = int(te.iloc[i][TARGET])
        show = pd.DataFrame({"Biến": RAW_FEATURES, "Mô tả": [LABELS_VI[c] for c in RAW_FEATURES],
                             "Giá trị": [fmt_value(v) for v in row.iloc[0]]})
        st.dataframe(show, hide_index=True, width="stretch", height=460)
    with st.expander("Ngưỡng chính sách (minh họa)"):
        approve = st.slider("Phê duyệt khi PD dưới", 0.01, 0.40, float(art["approve_below"]), 0.01, key="thr_a")
        decline = st.slider("Từ chối khi PD từ", 0.20, 0.95, float(art["decline_from"]), 0.05, key="thr_d")

ex = ui.explainer(approve, decline)
rep = ex.report(row)
dec = max(decline, approve + 0.01)
bg, fg, _ = ui.BAND_STYLE[rep["band"]]

with right:
    with st.container(border=True):
        a, b = st.columns([3, 2], vertical_alignment="center")
        a.metric("Xác suất vỡ nợ (PD, đã hiệu chỉnh Platt)", fmt_pd(rep["pd"]),
                 f"{100 * (rep['pd'] - art['base_rate']):+.1f} điểm % so với tỷ lệ vỡ nợ chung", delta_color="inverse")
        b.markdown(f"<div style='text-align:right'><span class='cs-badge' style='background:{bg};color:{fg}'>"
                   f"{ui.esc(rep['band'])}</span></div>", unsafe_allow_html=True)
        z = [ui.BAND_STYLE[x][2] for x in BANDS]
        pos = min(max(rep["pd"], 0.0), 1.0) * 100
        st.markdown(
            f"<div class='cs-scale'><div style='width:{approve * 100:.1f}%;background:{z[0]}'></div>"
            f"<div style='width:{(dec - approve) * 100:.1f}%;background:{z[1]}'></div>"
            f"<div style='flex:1;background:{z[2]}'></div></div>"
            f"<div class='cs-marker'><svg width='16' height='12' viewBox='0 0 16 12' "
            f"style='position:absolute;left:calc({pos:.2f}% - 8px);top:2px' aria-hidden='true'>"
            f"<path d='M8 0 L16 12 H0 Z' fill='{ui.INK}'/></svg></div>"
            f"<div class='cs-ticks'><span style='position:absolute;left:0'>0%</span>"
            f"<span style='position:absolute;left:{approve * 100:.1f}%;transform:translateX(-50%)'>{approve:.0%}</span>"
            f"<span style='position:absolute;left:{dec * 100:.1f}%;transform:translateX(-50%)'>{dec:.0%}</span>"
            f"<span style='position:absolute;right:0'>100%</span></div>", unsafe_allow_html=True)
        if actual is not None:
            st.caption(f"Kết quả thực tế (chỉ để kiểm tra mô hình): **{'Vỡ nợ' if actual == 1 else 'Trả nợ tốt'}**")

    with st.container(border=True):
        st.markdown("**Yếu tố tác động đến hồ sơ này (SHAP)**")
        contrib = rep["contributions"].rename("phi").reset_index().rename(columns={"index": "feature"})
        contrib["Biến"] = contrib.feature + " – " + contrib.feature.map(LABELS_VI)
        contrib["Tác động"] = np.where(contrib.phi > 0, "Tăng rủi ro", "Giảm rủi ro")
        st.altair_chart(alt.Chart(contrib).mark_bar(cornerRadius=3, height=16).encode(
            x=alt.X("phi:Q", title="Đóng góp vào log-odds vỡ nợ"),
            y=alt.Y("Biến:N", sort=alt.EncodingSortField("phi", op="sum", order="descending"), title=None,
                    axis=alt.Axis(labelLimit=320, labelOverlap=False)),
            color=alt.Color("Tác động:N", scale=alt.Scale(domain=["Tăng rủi ro", "Giảm rủi ro"],
                                                          range=["#E8875A", "#3FA595"]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Biến", alt.Tooltip("phi:Q", format="+.3f")]).properties(height=34 * len(contrib)),
            width="stretch")
        st.caption("Đóng góp TreeSHAP trên thang log-odds đã hiệu chỉnh; tổng các đóng góp cộng giá trị nền bằng log-odds của PD.")

    items = "".join(f"<li>{ui.esc(x)}</li>" for x in recommendation(rep, row.iloc[0]))
    st.markdown(f"<div class='cs-rec' style='background:{bg};color:{fg}'><h4>Khuyến nghị cho cán bộ tín dụng "
                f"(minh họa học thuật)</h4><ul>{items}</ul></div>", unsafe_allow_html=True)

st.caption("Dữ liệu HMEQ (khoản vay thế chấp nhà tại Mỹ). Ngưỡng phê duyệt/từ chối là tham số chính sách minh họa; "
           "kết quả chỉ hỗ trợ, không thay thế quyết định của cán bộ tín dụng.")
