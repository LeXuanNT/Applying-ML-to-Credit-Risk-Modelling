# Ứng dụng CreditScore (Streamlit)

Giai đoạn 2 của đồ án: hỗ trợ cán bộ tín dụng bằng mô hình cuối cùng (LightGBM + hiệu chỉnh Platt).
Bố cục tham khảo trang *Model Manager* của demo Taipy churn-classification: thanh điều hướng bên trái, bộ chọn mô hình,
thẻ chỉ số và các tab đổi chế độ xem.

- **Trên Colab:** mở `notebooks/03_run_app_colab.ipynb` → Runtime → Run all.
- **Trên máy:** từ thư mục gốc của repository: `streamlit run app/streamlit_app.py`
  (chạy từ thư mục gốc để Streamlit đọc giao diện trong `.streamlit/config.toml`).

## Ba trang

| Trang | Nội dung | Nguồn số liệu |
|---|---|---|
| **Quản lý mô hình** | Chọn 1 trong 9 mô hình; AUC, Gini, KS, F1, Brier, ΔAUC; tab *Chỉ số đánh giá* (ROC, ma trận nhầm lẫn), *Biến quan trọng*, *Phân nhóm quyết định* (bảng nhóm + hiệu chỉnh theo decile) | `results/v3/final_test.csv`, `evaluation.csv`, `gini_ks_test.csv`, `calibration_*.csv`, `decision_bands_test.csv`, `shap_importance_test.csv`, `feature_importance.csv`; ROC và KTC 95% của AUC tính trực tiếp từ champion trên tập kiểm tra |
| **So sánh mô hình** | Quy tắc chọn champion (bước 0 → 2), bảng 9 thuật toán trên tập kiểm định, hạng trung bình | `selection.csv`, `tuning.csv`, `stability.csv` |
| **Chấm điểm hồ sơ** | Hồ sơ mẫu (phê duyệt / xem xét / từ chối), hồ sơ trong tập kiểm tra hoặc nhập mới (cho phép thiếu); PD đã hiệu chỉnh, thang ngưỡng, SHAP theo 12 biến gốc, khuyến nghị | `models/champion.joblib`, `example_applicants.csv` |

ROC và độ quan trọng SHAP chỉ có cho champion, vì repository chỉ lưu mô hình champion.

## Tệp

```
app/streamlit_app.py      điểm vào: giao diện, điều hướng 3 trang
app/ui.py                 nạp dữ liệu/mô hình (có cache) và các thành phần giao diện dùng chung
app/views/model_manager.py, compare.py, score.py
app/assets/               logo
.streamlit/config.toml    màu, font (Be Vietnam Pro, IBM Plex Mono)
```

Ứng dụng nạp `models/champion.joblib` (pipeline tiền xử lý + LightGBM + hệ số Platt). Nếu file này không có, ứng dụng không chạy được:
tạo lại bằng `python scripts/finalize_champion.py` sau khi chạy `python scripts/run_v3.py`.

Kiểm thử: `pytest tests/test_app.py` (mở cả 3 trang, chọn từng mô hình, chấm 3 hồ sơ mẫu và khớp PD với `example_applicants.csv`,
chấm một hồ sơ thiếu toàn bộ thông tin).

**Giới hạn:** khuyến nghị là minh họa học thuật trên dữ liệu HMEQ (khoản vay thế chấp nhà tại Mỹ), không phải quyết định cấp tín dụng.
Ngưỡng 10% / 50% là tham số chính sách, điều chỉnh được ở mục *Ngưỡng chính sách* trên trang Chấm điểm hồ sơ.
