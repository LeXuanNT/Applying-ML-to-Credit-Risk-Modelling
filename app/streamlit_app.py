"""Ứng dụng CreditScore: hỗ trợ cán bộ tín dụng - giai đoạn 2 của khung luận văn.

Ba trang (bố cục tham khảo trang Model Manager của demo Taipy churn-classification):
  1. Quản lý mô hình   - chỉ số trên tập kiểm tra, ROC, ma trận nhầm lẫn, biến quan trọng, phân nhóm quyết định
  2. So sánh mô hình   - quy tắc chọn champion và bảng 9 thuật toán trên tập kiểm định
  3. Chấm điểm hồ sơ   - PD đã hiệu chỉnh, SHAP theo 12 biến gốc và khuyến nghị cho một hồ sơ

Chạy trên máy:  streamlit run app/streamlit_app.py   (từ thư mục gốc của repository, để đọc .streamlit/config.toml)
Chạy trên Colab: notebooks/03_run_app_colab.ipynb
Cần:   models/champion.joblib (tạo bởi scripts/finalize_champion.py) và results/v3/*.csv.
"""
from pathlib import Path

import streamlit as st

import ui

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="CreditScore · HMEQ", page_icon=str(HERE / "assets" / "logo_mark.svg"), layout="wide")
ui.inject_css()
st.logo(str(HERE / "assets" / "logo.svg"), size="large", icon_image=str(HERE / "assets" / "logo_mark.svg"))

pages = [
    st.Page(str(HERE / "views" / "model_manager.py"), title="Quản lý mô hình", icon=":material/dashboard:",
            default=True),
    st.Page(str(HERE / "views" / "compare.py"), title="So sánh mô hình", icon=":material/bar_chart:",
            url_path="compare-models"),
    st.Page(str(HERE / "views" / "score.py"), title="Chấm điểm hồ sơ", icon=":material/person_search:",
            url_path="score"),
]
page = st.navigation(pages, position="sidebar")
ui.sidebar_summary()
page.run()
