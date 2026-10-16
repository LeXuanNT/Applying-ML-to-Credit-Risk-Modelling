"""Re-check the numeric model tests (Ch.4, Bảng kết quả kiểm thử) directly from models/champion.joblib.

Run: python scripts/verify_champion_tests.py
Reproduces: test AUC, mean PD vs default rate, Brier before/after Platt, decision bands,
decile calibration (incl. PD range of decile 9), SHAP additivity and the deployed feature list.
"""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, "src")
COLS = ["LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]

m = joblib.load("models/champion.joblib")
pipe, X = m["pipeline"], m["test_applicants"]
y = X["BAD"].values
Z = pipe.named_steps["prep"].transform(X[COLS])
clf = pipe.named_steps["clf"]
raw = clf.predict(Z, raw_score=True)
pd_cal = 1 / (1 + np.exp(-(m["platt_a"] * raw + m["platt_b"])))

print("deployed columns:", Z.shape[1], "| CLAGE_MISSING present:", "CLAGE_MISSING" in list(Z.columns))
print(f"AUC {roc_auc_score(y, pd_cal):.4f}")
print(f"mean PD {pd_cal.mean():.4f} vs default rate {y.mean():.4f}")
print(f"Brier raw {brier_score_loss(y, 1 / (1 + np.exp(-raw))):.4f} -> Platt {brier_score_loss(y, pd_cal):.4f}")
band = np.where(pd_cal < m["approve_below"], 0, np.where(pd_cal < m["decline_from"], 1, 2))
print("bands n:", np.bincount(band), "default rate:", [round(y[band == k].mean(), 3) for k in range(3)])
q = pd.qcut(pd_cal, 10, labels=False)
dec = pd.DataFrame({"q": q, "pd": pd_cal, "y": y}).groupby("q").mean()
dec["gap_pp"] = (dec["pd"] - dec["y"]) * 100
print(dec.round(3))
print(f"decile 9 PD range: {pd_cal[q == 8].min():.3f}-{pd_cal[q == 8].max():.3f}")
contrib = clf.predict(Z, pred_contrib=True)
print(f"SHAP additivity max error: {np.abs(contrib.sum(1) - raw).max():.1e}")
