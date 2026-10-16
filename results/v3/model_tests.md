HMEQ, champion LGBM (models/champion.joblib, sha256 c3e7046cd0af…). Tập kiểm tra n = 894, 178 hồ sơ vỡ nợ.

| Lớp kiểm thử | Kiểm thử | Tiêu chí | Kết quả | Đánh giá |
|---|---|---|---|---|
| Chất lượng dữ liệu | Cấu trúc (schema) | Đủ 12 biến của Bảng 3.1 | thiếu: không | Đạt |
| Chất lượng dữ liệu | Miền giá trị | Không vi phạm miền giá trị | không vi phạm | Đạt |
| Chất lượng dữ liệu | Tỷ lệ khuyết | ≤ tập huấn luyện + 5 điểm % | chênh lệch lớn nhất: NINQ +2.4 điểm % | Đạt |
| Mô hình | Khả năng phân biệt | AUC vượt mô hình cơ sở (DeLong p < 0.05) | AUC 0.982 [0.974; 0.988] so với 0.895; Gini 0.964; p = 2.5e-13 | Đạt |
| Mô hình | Hiệu chỉnh tổng thể | \|PD TB − tỷ lệ vỡ nợ\| ≤ 2 điểm % | 21.1% so với 19.9% (+1.2 điểm %) | Đạt |
| Mô hình | Hiệu chỉnh theo decile | Mỗi decile ≤ 10 điểm % | lệch lớn nhất: decile 9: 78.9% so với 67.4% (+11.5 điểm %) | Không đạt |
| Mô hình | Brier score | Sau hiệu chỉnh ≤ trước hiệu chỉnh | 0.0431 so với 0.0521 | Đạt |
| Mô hình | Ổn định (PSI) | PSI < 0.1 | PSI = 0.012 (kiểm định → kiểm tra; hai mẫu ngẫu nhiên cùng quần thể) | Đạt |
| Mô hình | Độ nhạy điền khuyết | \|ΔPD\| trung bình quanh trung vị ≤ 5 điểm % | lớn nhất: CLAGE 2.3 điểm % | Đạt |
| Tích hợp | Phiên bản mô hình | Mô hình ứng dụng = champion đã chọn, đúng siêu tham số | LGBM (selection.csv: LGBM); siêu tham số trùng tuning.csv; sha256 c3e7046cd0af… | Đạt |
| Tích hợp | Danh sách biến | Danh sách và thứ tự biến trùng khớp | 32 cột so với 33 cột trong feature_matrix.csv; lệch | Không đạt |
| Tích hợp | PD ứng dụng = PD pipeline | Sai khác < 1e-09 | 3 hồ sơ mẫu: sai khác lớn nhất 1.1e-16; PD TB 21.13% (r2: 21.13%) | Đạt |
| Quy tắc nghiệp vụ | Phân nhóm tại giá trị biên | PD = 10% → Xem xét; PD = 50% → Từ chối | đúng | Đạt |
| Quy tắc nghiệp vụ | Tỷ lệ vỡ nợ theo nhóm | Tăng đơn điệu Phê duyệt → Xem xét → Từ chối | Approve 0.8% [0.3; 1.8] (n=638) → Review 31.1% [21.7; 42.3] (n=74) → Decline 82.4% [76.2; 87.3] (n=182) | Đạt |
| Quy tắc nghiệp vụ | Khớp bảng phân nhóm báo cáo | Ứng dụng phân nhóm như quy tắc; số hồ sơ mỗi nhóm = Bảng phân nhóm r2 | trùng 100.0% hồ sơ; Phê duyệt: 638, Xem xét thủ công: 74, Từ chối: 182 | Đạt |
| Khả năng giải thích | Tính cộng SHAP (12 biến gốc) | \|cơ sở + ΣSHAP − đầu ra\| < 1e-06 | sai khác lớn nhất 8.0e-13 | Đạt |
| Khả năng giải thích | Chiều tác động | Cùng chiều với mục 3.2 (DEBTINC, DELINQ, DEROG, NINQ +; CLAGE −) | DEBTINC ρ=+0.57, DELINQ ρ=+0.72, DEROG ρ=+0.61, NINQ ρ=+0.38, CLAGE ρ=-0.75 | Đạt |
| Khả năng giải thích | Lý do từ chối | 3 lý do từ 12 biến đầu vào cho mỗi hồ sơ bị từ chối | 182/182 hồ sơ | Đạt |
| Hồi quy | PD trên golden dataset | \|ΔPD\| ≤ 0.001 | \|ΔPD\| lớn nhất 0.0e+00 | Đạt |
| Hồi quy | AUC sau huấn luyện lại | ΔAUC ≥ −0.005 | 0.982 so với 0.982 (-0.0000) | Đạt |
| Hồi quy | Đổi nhóm quyết định | ≤ 1% hồ sơ | 0.00% hồ sơ | Đạt |

Đạt 19/21 kiểm thử.
