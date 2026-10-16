"""Ứng dụng hỗ trợ cán bộ tín dụng - giai đoạn 2 của khung luận văn.

Chạy trên máy:  streamlit run app/streamlit_app.py   (từ thư mục gốc của repository)
Chạy trên Colab: notebooks/03_run_app_colab.ipynb
Cần:   models/champion.joblib (tạo bởi scripts/finalize_champion.py).
"""
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from creditscore.data import DESCRIPTIONS, NUM_FEATURES, RAW_FEATURES, TARGET  # noqa: E402
from creditscore.decision_support import BANDS, LABELS_VI, Explainer, fmt_pd, recommendation  # noqa: E402

st.set_page_config(page_title="Chấm điểm tín dụng HMEQ", layout="wide")


@st.cache_resource
def load():
    art = joblib.load(ROOT / "models" / "champion.joblib")
    res = ROOT / "results" / "v3"
    return art, pd.read_csv(res / "shap_importance_test.csv"), pd.read_csv(res / "final_test.csv")


art, shap_imp, final_test = load()
COLOR = {BANDS[0]: "#2e7d32", BANDS[1]: "#ef6c00", BANDS[2]: "#c62828"}

# ------------------------------------------------------------------ sidebar: applicant + policy
st.sidebar.header("Hồ sơ khách hàng")
mode = st.sidebar.radio("Nguồn hồ sơ", ["Hồ sơ trong tập kiểm tra", "Nhập hồ sơ mới"])
te = art["test_applicants"]
actual = None
if mode == "Hồ sơ trong tập kiểm tra":
    i = st.sidebar.number_input("Số thứ tự hồ sơ (0 – %d)" % (len(te) - 1), 0, len(te) - 1, 134)
    row = te.iloc[[int(i)]][RAW_FEATURES].reset_index(drop=True)
    actual = int(te.iloc[int(i)][TARGET])
else:
    vals = {}
    # Pre-filled with ROUNDED typical values: the exact training medians are the imputation values, which the
    # trees read as "missing" (e.g. DEBTINC = 34.881 never occurs as a real value). Use the checkbox for missing.
    typical = {c: (round(v, -2) if v > 1000 else round(v, 1)) for c, v in art["medians"].items()}
    for c in NUM_FEATURES:
        lo, hi = art["ranges"][c]
        miss = st.sidebar.checkbox(f"{c}: không có thông tin", key=f"m_{c}")
        vals[c] = np.nan if miss else st.sidebar.number_input(
            f"{c} – {DESCRIPTIONS[c]}", float(lo), float(hi), float(typical[c]), key=c)
    for c in ["REASON", "JOB"]:
        opts = ["(không có thông tin)"] + art["categories"][c]
        opt = st.sidebar.selectbox(f"{c} – {DESCRIPTIONS[c]}", opts, index=opts.index(art["modes"][c]))
        vals[c] = np.nan if opt.startswith("(") else opt
    row = pd.DataFrame([vals])[RAW_FEATURES]

st.sidebar.header("Ngưỡng chính sách")
approve = st.sidebar.slider("Phê duyệt khi PD dưới", 0.01, 0.40, float(art["approve_below"]), 0.01)
decline = st.sidebar.slider("Từ chối khi PD từ", 0.20, 0.95, float(art["decline_from"]), 0.05)
ex = Explainer(art["pipeline"], art["platt_a"], art["platt_b"], approve, max(decline, approve + 0.01))

# ------------------------------------------------------------------ main panel
st.title("Hỗ trợ quyết định tín dụng: một hồ sơ vay")
rep = ex.report(row)
c1, c2, c3 = st.columns(3)
c1.metric("Xác suất vỡ nợ (PD, đã hiệu chỉnh)", fmt_pd(rep["pd"]),
          f"{100 * (rep['pd'] - art['base_rate']):+.1f} điểm % so với tỷ lệ vỡ nợ chung", delta_color="inverse")
c2.markdown(f"**Khuyến nghị (minh họa học thuật)**<br><span style='font-size:1.6rem;color:{COLOR[rep['band']]}'>"
            f"{rep['band']}</span>", unsafe_allow_html=True)
if actual is not None:
    c3.metric("Kết quả thực tế (chỉ để kiểm tra)", "Vỡ nợ" if actual == 1 else "Trả nợ tốt")

tab1, tab2, tab3 = st.tabs(["Giải thích hồ sơ", "Thông tin hồ sơ", "Về mô hình"])
with tab1:
    for line in recommendation(rep, row.iloc[0]):
        st.write("• " + line)
    contrib = rep["contributions"].rename("phi").reset_index().rename(columns={"index": "feature"})
    contrib["Biến"] = contrib["feature"].map(lambda v: f"{v} – {LABELS_VI[v]}")
    contrib["Tác động"] = np.where(contrib["phi"] > 0, "Tăng rủi ro", "Giảm rủi ro")
    chart = alt.Chart(contrib).mark_bar().encode(
        x=alt.X("phi:Q", title="Đóng góp vào log-odds vỡ nợ (TreeSHAP)"),
        y=alt.Y("Biến:N", sort=alt.EncodingSortField("phi", op="sum", order="descending"), title=None,
                axis=alt.Axis(labelLimit=320)),
        color=alt.Color("Tác động:N", scale=alt.Scale(domain=["Tăng rủi ro", "Giảm rủi ro"],
                                                      range=["#c62828", "#1565c0"])),
        tooltip=["Biến", alt.Tooltip("phi:Q", format="+.3f")]).properties(height=360)
    st.altair_chart(chart, width="stretch")
    st.caption("Mỗi thanh là phần đóng góp chính xác của một biến vào log-odds vỡ nợ của hồ sơ này; "
               "tổng các đóng góp cộng với giá trị nền bằng log-odds của PD.")
with tab2:
    show = row.T.rename(columns={0: "Giá trị"})
    show["Mô tả"] = [DESCRIPTIONS[c] for c in show.index]
    st.dataframe(show.astype(str), width="stretch")
with tab3:
    ft = final_test.iloc[0]
    st.write(f"Mô hình: {art['champion_label']} (mô hình được chọn ở Chương 4), huấn luyện trên 5,066 khoản vay "
             f"(train + validation). Trên tập kiểm tra: AUC {ft.auc:.3f}, F1 {ft.f1:.3f}, Recall {ft.recall:.3f}, "
             f"Precision {ft.precision:.3f}.")
    imp = shap_imp.assign(Biến=lambda d: d.feature.map(lambda v: f"{v} – {LABELS_VI[v]}"))
    st.altair_chart(alt.Chart(imp).mark_bar(color="#1565c0").encode(
        x=alt.X("share:Q", title="Tỷ trọng trung bình |SHAP| trên tập kiểm tra", axis=alt.Axis(format="%")),
        y=alt.Y("Biến:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320))).properties(height=330), width="stretch")
    st.info("Dữ liệu HMEQ (khoản vay thế chấp nhà tại Mỹ). Ngưỡng phê duyệt/từ chối là tham số chính sách minh họa, "
            "ngân hàng cần xác định theo khẩu vị rủi ro. Kết quả chỉ hỗ trợ, không thay thế quyết định của cán bộ tín dụng.")
