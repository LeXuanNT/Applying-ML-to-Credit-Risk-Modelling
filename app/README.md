# Ứng dụng CreditRisk (Streamlit)

Giai đoạn 2 của đồ án: hỗ trợ cán bộ tín dụng đánh giá một hồ sơ vay bằng mô hình cuối cùng (LightGBM + hiệu chỉnh Platt).

- **Trên Colab:** mở `notebooks/03_run_app_colab.ipynb` → Runtime → Run all.
- **Trên máy:** từ thư mục gốc của repository: `streamlit run app/streamlit_app.py` (xem README chính).

Ứng dụng nạp `models/champion.joblib` (pipeline tiền xử lý + LightGBM + hệ số Platt). Nếu file này không có, ứng dụng không chạy được:
tạo lại bằng `python scripts/finalize_champion.py` sau khi chạy `python scripts/run_v3.py`.

Chức năng: nhập 12 biến (cho phép thiếu) hoặc chọn hồ sơ trong tập kiểm tra theo số thứ tự; PD đã hiệu chỉnh; khuyến nghị theo ngưỡng
10% / 50% (điều chỉnh được ở thanh bên); đóng góp SHAP của 12 biến gốc; gợi ý xác minh khi thiếu thông tin.

**Giới hạn:** khuyến nghị là minh họa học thuật trên dữ liệu HMEQ (khoản vay thế chấp nhà tại Mỹ), không phải quyết định cấp tín dụng.
