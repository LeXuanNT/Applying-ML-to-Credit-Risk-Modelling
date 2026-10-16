"""Thesis §3.9 / Bảng 4.x: run the six-layer model tests on the r2 champion and HMEQ.

    python scripts/run_model_tests_hmeq.py

Uses models/champion.joblib exactly as app/streamlit_app.py loads it, the r2 split
(split_data(df, 42), 70/15/15) and the r2 tuned hyper-parameters (results/v3/tuning.csv).
Writes results/v3/model_tests.csv and results/v3/model_tests.md. Nothing in results/v3 is overwritten.
"""
import ast
import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from creditscore import decision_support as DS  # noqa: E402
from creditscore import model_testing as mt  # noqa: E402
from creditscore import protocol_v3 as P  # noqa: E402
from creditscore.data import RAW_FEATURES, TARGET, load_hmeq, split_data  # noqa: E402

R = ROOT / "results" / "v3"
BAND_EN = {DS.BANDS[0]: "Approve", DS.BANDS[1]: "Review", DS.BANDS[2]: "Decline"}


class ChampionAdapter:
    """The four members model_testing needs, around the r2 pipeline + Platt scaling."""

    def __init__(self, pipe, a, b):
        self.ex = DS.Explainer(pipe, a, b)
        self.pipe = pipe

    @property
    def feature_names(self):
        return list(self.pipe.named_steps["prep"].columns_)

    def margin(self, X):
        return DS._margin(self.pipe, X[RAW_FEATURES])

    def predict_pd(self, X):
        return self.ex.pd(X[RAW_FEATURES])

    def contributions(self, X):
        phi, base = DS.raw_contributions(self.pipe, X[RAW_FEATURES])
        return base, phi


def tuned(name):
    return ast.literal_eval(pd.read_csv(R / "tuning.csv").set_index("model").loc[name, "best_params"])


def main():
    df = load_hmeq(ROOT / "data" / "hmeq.csv")
    S = split_data(df, 42)
    tr, va, te = S["train"], S["valid"], S["test"]
    trva = pd.concat([tr, va], ignore_index=True)
    Xte, yte = te[RAW_FEATURES], te[TARGET].values

    # ---- the champion as the web app loads it
    art_path = ROOT / "models" / "champion.joblib"
    art = joblib.load(art_path)
    app = ChampionAdapter(art["pipeline"], art["platt_a"], art["platt_b"])
    pd_te = app.predict_pd(Xte)
    results = []

    # 1. data quality: test applicants against the train + valid data the champion was fitted on
    results += mt.test_data_quality(Xte, trva[RAW_FEATURES])

    # 2. model
    lr = P.build("LR", seed=42).set_params(**tuned("LR")).fit(tr[RAW_FEATURES], tr[TARGET].values)
    lgbm_train = P.build("LGBM", seed=42).set_params(**tuned("LGBM")).fit(tr[RAW_FEATURES], tr[TARGET].values)
    pd_uncal = P.score_of(art["pipeline"], Xte)          # before Platt (after resampling)
    results += mt.test_model(yte, pd_te, P.score_of(lr, Xte), pd_uncal, pd_te,
                             champion=app, X_ref=Xte, medians=pd.Series(art["medians"]))
    # PSI as in r2: train-fitted LGBM, validation scores vs test scores (the final model saw validation).
    v = mt.psi(P.score_of(lgbm_train, va[RAW_FEATURES]), P.score_of(lgbm_train, Xte))
    results = [r for r in results if r.test != "Ổn định (PSI)"]
    results.insert(len(results) - 1, mt.TestResult(
        "Mô hình", "Ổn định (PSI)", f"PSI < {mt.THRESHOLDS['psi_max']}",
        f"PSI = {v:.3f} (kiểm định → kiểm tra; hai mẫu ngẫu nhiên cùng quần thể)", v < mt.THRESHOLDS["psi_max"]))

    # 3. integration: the app's model is the selected, tuned champion and reproduces the reported results
    L = "Tích hợp"
    sel = pd.read_csv(R / "selection.csv").query("selected").iloc[0]["model"]
    meta = json.loads((ROOT / "results" / "run_meta.json").read_text())
    same_version = art["champion"] == sel == meta["champion"] and art["params"] == tuned(sel)
    sha = hashlib.sha256(art_path.read_bytes()).hexdigest()
    results.append(mt.TestResult(L, "Phiên bản mô hình", "Mô hình ứng dụng = champion đã chọn, đúng siêu tham số",
                                 f"{art['champion']} (selection.csv: {sel}); siêu tham số trùng tuning.csv; "
                                 f"sha256 {sha[:12]}…", same_version))
    fm = pd.read_csv(R / "feature_matrix.csv")["column"].tolist()
    same = app.feature_names == fm
    results.append(mt.TestResult(L, "Danh sách biến", "Danh sách và thứ tự biến trùng khớp",
                                 f"{len(app.feature_names)} cột so với {len(fm)} cột trong feature_matrix.csv; "
                                 f"{'trùng' if same else 'lệch'}", same))
    exm = pd.read_csv(R / "example_applicants.csv")
    d_ex = float(np.max(np.abs(pd_te[exm["test_index"].values] - exm["pd"].values)))
    cal = pd.read_csv(R / "calibration_test.csv").iloc[1]
    d_mean = abs(pd_te.mean() - cal["mean_pd"])
    ok = max(d_ex, d_mean) < mt.THRESHOLDS["integration_tol"]
    results.append(mt.TestResult(L, "PD ứng dụng = PD pipeline", f"Sai khác < {mt.THRESHOLDS['integration_tol']:g}",
                                 f"3 hồ sơ mẫu: sai khác lớn nhất {d_ex:.1e}; PD TB {pd_te.mean()*100:.2f}% "
                                 f"(r2: {cal['mean_pd']*100:.2f}%)", ok))

    # 4. business rules (on the bands the app itself assigns)
    br = mt.test_business_rules(yte, pd_te)
    app_bands = np.array([BAND_EN[b] for b in app.ex.band(pd_te)])
    agree = float(np.mean(app_bands == mt.assign_band(pd_te)))
    bands_r2 = pd.read_csv(R / "decision_bands_test.csv").set_index("band")["n"]
    now = pd.Series(app.ex.band(pd_te)).value_counts()
    match = all(int(now.get(b, 0)) == int(bands_r2[b]) for b in DS.BANDS)
    results += br
    results.append(mt.TestResult("Quy tắc nghiệp vụ", "Khớp bảng phân nhóm báo cáo",
                                 "Ứng dụng phân nhóm như quy tắc; số hồ sơ mỗi nhóm = Bảng phân nhóm r2",
                                 f"trùng {agree:.1%} hồ sơ; " + ", ".join(f"{b}: {int(now.get(b, 0))}"
                                                                    for b in DS.BANDS),
                                 agree == 1.0 and match))

    # 5. explainability
    results += mt.test_explainability(app, Xte)

    # 6. regression: rebuild the champion from code (same params, seed, data) and compare with the saved model
    rebuilt = P.build("LGBM", seed=42).set_params(**art["params"]).fit(trva[RAW_FEATURES], trva[TARGET].values)
    reb = ChampionAdapter(rebuilt, art["platt_a"], art["platt_b"])
    results += mt.test_regression(Xte, pd_te, reb)
    results += mt.test_regression(Xte, pd_te, reb, y=yte, baseline_auc=roc_auc_score(yte, pd_te), retrained=True)

    table = mt.to_frame(results)
    table.to_csv(R / "model_tests.csv", index=False)
    cols = ["Lớp kiểm thử", "Kiểm thử", "Tiêu chí", "Kết quả", "Đánh giá"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |" for _, r in table.iterrows()]
    md = (f"HMEQ, champion {art['champion']} (models/champion.joblib, sha256 {sha[:12]}…). "
          f"Tập kiểm tra n = {len(te):,}, {int(yte.sum())} hồ sơ vỡ nợ.\n\n" + "\n".join(lines)
          + f"\n\nĐạt {int(table['passed'].sum())}/{len(table)} kiểm thử.\n")
    (R / "model_tests.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
