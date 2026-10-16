# Bảng liên kết: luận văn → mã → artifact → trạng thái kiểm chứng

Trạng thái (ngày 16/10/2026, Python 3.11, phiên bản thư viện trong `requirements.txt`):
- **Khớp – tự động**: notebook tính lại và so với con số in trong luận văn, tất cả ✓.
- **Khớp – file**: tính lại và so với file kết quả gốc trong `results/` (sai khác ≤ 1e-6).
- **Có artifact**: file kết quả có trong repository nhưng chưa có ô đối chiếu tự động.
- **Chưa kiểm chứng**: chưa chạy lại trong đợt này.

| Mục / bảng / hình trong luận văn | Notebook / module | Cấu hình | Artifact | Trạng thái |
|---|---|---|---|---|
| 3.2.2 Bảng 3.1–3.2: mô tả, giá trị khuyết | `notebooks/02_chapter3_check.ipynb` | `data` | (tính trong notebook) | Khớp – tự động |
| 3.2.3 Q1–Q4: mất cân bằng lớp, r_pb, φ, dendrogram, Bảng 3.5–3.7, Little | `02_chapter3_check.ipynb`; `src/creditscore/eda.py`; `scripts/missing_study.py` (A1–A3) | — | `results/missing_study/A*.csv` | Khớp – tự động (209/209 cho cả Chương 3) |
| 3.2.3 Q5: VIF | `02_chapter3_check.ipynb` mục 6 | — | — | Con số khớp, **phương pháp sai** (xem `deviation_log.md` D1) |
| 3.3 / 4.1 Bảng: phân chia dữ liệu | `01_thesis_pipeline.ipynb` Bước 3; `data.split_data` | `split` | `results/v3/split.csv` | Khớp – tự động |
| 4.1 Bảng: kiểm định chỉ báo khuyết | Bước 4; `protocol_v3.missing_indicator_tests` | `missing_data` | `results/v3/missing_tests.csv` | Khớp – tự động |
| 4.1 Bảng: cấu trúc ma trận đặc trưng | Bước 4, 8; `protocol_v3.V3Preprocessor` | `features` | `feature_matrix.csv` (fit trên train), `feature_matrix_final.csv` (mô hình cuối) | Khớp – tự động; xem D2 |
| 4.2 Bảng: so sánh chiến lược xử lý khuyết S0–S7; NRMSE | `scripts/missing_study.py` (B, C) | — | `results/missing_study/B_*, C_*` | Khớp – file (8/8 bảng, notebook kiểm tra Chương 4 cũ) |
| 4.1 Bảng: tỷ lệ vỡ nợ theo YOJ; Bảng IV | Bước 4; `features.WoEBinner` | `features` | `yoj_bins.csv`, `iv.csv`, `woe.csv` | Khớp – tự động (IV) |
| 4.3 Hình: F1/Recall theo kỹ thuật cân bằng lớp | Bước 5; `protocol_v3.imbalance_comparison` | `tuning` | `results/v3/imbalance.csv` | Có artifact; chạy lại: chưa kiểm chứng (`RUN_IMBALANCE=True`) |
| 4.4 Bảng: kết quả tối ưu siêu tham số | Bước 6; `protocol_v3.tune`; `scripts/run_v3.py` | `tuning` | `tuning.csv`, `tuning_options.csv` | Huấn luyện lại từ tham số đã lưu: khớp. **Chạy lại BO đầy đủ: chưa kiểm chứng** |
| 4.4 Bảng: đánh giá các mô hình trên tập kiểm tra; Hình ROC | `scripts/run_v3.py`, `finalize_champion.py` | — | `evaluation.csv`, `gini_ks_test.csv` | Có artifact |
| 4.5 Bảng: lựa chọn mô hình trên tập kiểm định | Bước 7; `selection.select_champion` | `selection` | `selection.csv`, `selection_oof.csv` | Khớp – tự động (36 giá trị) và file |
| 4.6 Bảng: LightGBM trước/sau huấn luyện lại; ma trận nhầm lẫn | Bước 8; `finalize_champion.py` | `final_model` | `final_test.csv` | Khớp – tự động |
| 4.7 Hình/Bảng: mức độ quan trọng, SHAP | Bước 10; `decision_support.Explainer` | — | `shap_importance_test.csv`, `feature_importance.csv` | Khớp – tự động; tính cộng tính ≤ 1e-12 |
| 4.8 Bảng/Hình: hiệu chỉnh Platt, biểu đồ độ tin cậy | Bước 9 | `final_model` | `calibration_test.csv`, `calibration_deciles.csv` | Khớp – tự động |
| 4.9 Bảng: ba nhóm khuyến nghị; ba hồ sơ minh họa | Bước 11 | `final_model.decision_bands` | `decision_bands_test.csv`, `example_applicants.csv` | Khớp – tự động |
| 4.9 Hình: giao diện ứng dụng | `app/streamlit_app.py`; `03_run_app_colab.ipynb` | — | `models/champion.joblib` | Chạy được (kiểm thử `tests/test_app.py`) |
| 4.10 Bảng: 21 kiểm thử | Bước 12; `scripts/run_model_tests_hmeq.py` | — | `model_tests.csv`, `model_tests.md` | Khớp – tự động (19/21); xem D2, D10 |
| Các hình Chương 3–4 (PNG/PDF) | `plots.py`, `plots_v3.py` (gọi trong `run_v3.py`, `finalize_champion.py`) | — | `figures/` (không đưa vào Git) | Chưa đối chiếu từng hình |
