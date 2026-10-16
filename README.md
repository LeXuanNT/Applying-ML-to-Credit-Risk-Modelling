# Apply ML algorithms in credit scoring modelling

Mã nguồn của luận văn thạc sĩ **"Ứng dụng của học máy trong mô hình rủi ro tín dụng"**, Lê Thị Thanh Xuân,
ngành Khoa học dữ liệu, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM, 2026.

Bài toán: dự báo xác suất vỡ nợ (PD) trên bộ dữ liệu HMEQ (5,960 khoản vay thế chấp bằng giá trị nhà; BAD = 1 là vỡ nợ).
So sánh 9 thuật toán; chọn mô hình theo quy trình ba bước (khả năng giải thích → kiểm định DeLong → hạng trung bình);
mô hình cuối cùng (LightGBM) được hiệu chỉnh Platt, giải thích bằng SHAP và đưa vào ứng dụng hỗ trợ cán bộ tín dụng.

## Cấu trúc

```
configs/thesis.yaml        giao thức thí nghiệm (chia dữ liệu, BO, lựa chọn, ngưỡng)
data/hmeq.csv              dữ liệu (công khai) + data/README.md (nguồn, checksum)
src/creditscore/           gói Python: tiền xử lý, mô hình, lựa chọn, hiệu chỉnh, SHAP, kiểm thử
scripts/                   chạy thí nghiệm từ dòng lệnh
notebooks/                 01 quy trình đầy đủ · 02 kiểm tra Chương 3 · 03 chạy ứng dụng trên Colab
app/streamlit_app.py       ứng dụng CreditRisk
models/champion.joblib     mô hình cuối cùng (pipeline + LightGBM + hệ số Platt)
results/v3/, results/missing_study/   kết quả gốc dùng trong luận văn (bảng CSV)
reports/                   bảng liên kết luận văn → mã, nhật ký sai khác, manifest môi trường
tests/                     kiểm thử tự động (pytest)
```

## Chạy nhanh trên Google Colab (không cần cài gì trên máy)

1. Vào <https://colab.research.google.com> → thẻ **GitHub** → dán đường dẫn repository → chọn `notebooks/01_thesis_pipeline.ipynb`.
   (Repository riêng tư: bật **Include private repos** và cấp quyền cho Colab.)
2. Ở ô **0.3**, sửa `REPO_URL` thành đường dẫn repository (nếu công khai). Nếu riêng tư: để nguyên; khi được hỏi, tải lên file ZIP
   lấy từ GitHub (**Code → Download ZIP**).
3. **Runtime → Run all.** Lần đầu, ô 0.1 cài đúng phiên bản thư viện và tự khởi động lại phiên; chọn **Run all** lần nữa.
4. Khoảng 4–6 phút. Ô cuối tổng hợp các giá trị đối chiếu với luận văn (✓ = khớp).

| Notebook | Nội dung | Thời gian |
|---|---|---|
| `01_thesis_pipeline.ipynb` | 13 bước: dữ liệu → EDA → chia dữ liệu → tiền xử lý → cân bằng lớp → 9 mô hình → lựa chọn → mô hình cuối → hiệu chỉnh → SHAP → khuyến nghị → kiểm thử → đối chiếu | 4–6 phút |
| `02_chapter3_check.ipynb` | Tính lại 209 con số của Chương 3 | < 1 phút |
| `03_run_app_colab.ipynb` | Chạy ứng dụng Streamlit và mở bằng đường dẫn tạm thời của Colab | 1–2 phút |

Tùy chọn trong ô 0.2 của notebook 01: `DATA_SOURCE` (`"hmeq"`, `"synthetic"`, `"upload"`), `TUNE` (chạy lại Bayesian Optimization),
`FAST_TUNE` (chạy thử nhanh), `RUN_IMBALANCE` (chạy lại so sánh cân bằng lớp). Chỉ cấu hình mặc định tái lập đúng số trong luận văn.

## Chạy trên máy cá nhân

Cần Python 3.11 (đã kiểm tra với 3.11.15). Mở Terminal (macOS) hoặc PowerShell (Windows) trong thư mục repository:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/step0_data_check.py # kiểm tra dữ liệu, ghi reports/run_manifest.json
python -m pytest -q                # 23 kiểm thử
streamlit run app/streamlit_app.py # mở ứng dụng tại http://localhost:8501
```

## Chạy lại toàn bộ thí nghiệm

| Lệnh | Việc làm | Thời gian (CPU 2 nhân) |
|---|---|---|
| `python scripts/run_v3.py --fast` | **Smoke test** (4 vòng × 3-fold). Ghi vào `results/v3_fast/`, không dùng để báo cáo | ~6 phút |
| `python scripts/run_v3.py` | Thí nghiệm đầy đủ: cân bằng lớp, BO 50 vòng × 10-fold cho 9 mô hình, lựa chọn. Lưu checkpoint từng mô hình ở `results/v3/checkpoints/`; chạy lại lệnh để tiếp tục khi bị ngắt | 1.5–2 giờ |
| `python scripts/finalize_champion.py` | (sau `run_v3.py`) Huấn luyện lại mô hình được chọn trên train + valid, hiệu chỉnh Platt (OOF), SHAP, nhóm khuyến nghị, lưu `models/champion.joblib` | ~1 phút |
| `python scripts/missing_study.py` | Little, kiểm tra cơ chế khuyết, NRMSE, so sánh S0–S7 | ~6 phút |
| `python scripts/run_model_tests_hmeq.py` | 21 kiểm thử mô hình (luận văn: đạt 19/21) | < 1 phút |

Trên Colab Free, chạy đầy đủ có thể bị ngắt (không tương tác ~90 phút). Nhờ checkpoint, chỉ cần chạy lại lệnh; các mô hình đã xong được nạp lại.

## Artifact nào nằm trong Git

- **Có:** mã, notebook (không kèm output), `data/hmeq.csv` (dữ liệu công khai, 0.4 MB), `models/champion.joblib` (1.2 MB,
  để ứng dụng chạy ngay), các bảng CSV trong `results/`, `reports/`.
- **Không** (xem `.gitignore`): `results/v3/run_v3.joblib` (7 MB, tạo lại bằng `run_v3.py`), checkpoint, `figures/`, kết quả smoke test,
  môi trường ảo, mọi token hoặc mật khẩu.

## Tái lập

- Seed `42` cho chia dữ liệu, CV, BO và mô hình. Phiên bản thư viện cố định trong `requirements.txt` và được ghi vào `reports/run_manifest.json`.
- Checksum dữ liệu: `dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa`.
- Đã kiểm chứng (16/10/2026, Linux, Python 3.11; **chưa chạy trên phiên Colab thật**): huấn luyện lại từ siêu tham số đã lưu cho kết quả trùng luận văn (72/72 giá trị trong notebook 01;
  209/209 trong notebook 02); PD trùng `models/champion.joblib` với sai khác ≤ 1e-7. **Chưa kiểm chứng:** chạy lại BO đầy đủ.
- Sai khác và hạn chế: `reports/deviation_log.md`. Liên kết bảng/hình → mã → artifact: `reports/traceability.md`.

## Ứng dụng CreditRisk

`app/streamlit_app.py` dùng đúng `models/champion.joblib`. Chạy trên Colab bằng notebook 03, hoặc trên máy bằng `streamlit run`.
Khuyến nghị Phê duyệt / Xem xét / Từ chối theo ngưỡng 10% và 50% là **minh họa học thuật**, không phải quyết định cấp tín dụng.

**Triển khai công khai (tùy chọn) trên Streamlit Community Cloud:** đăng nhập <https://share.streamlit.io> bằng GitHub →
**Create app** → chọn repository, nhánh `main`, file `app/streamlit_app.py`, Python 3.11 → **Deploy**.
Cần: `requirements.txt` ở thư mục gốc (đã có), `models/champion.joblib` trong repository (đã có). Ứng dụng ngủ khi không ai dùng và
có giới hạn tài nguyên; repository riêng tư cần cấp quyền cho Streamlit.

## Giới hạn

HMEQ là dữ liệu Mỹ, không có ngày tháng (không kiểm định ngoài thời gian), chỉ gồm hồ sơ đã được cấp vay (không có reject inference)
và không có biến vĩ mô. Kết quả không chuyển trực tiếp sang danh mục cho vay tại Việt Nam.
