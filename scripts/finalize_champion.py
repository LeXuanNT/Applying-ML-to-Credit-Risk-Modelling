"""Stage 1 -> stage 2: finalise the champion chosen by creditscore.selection and prepare it for the web app.

1. Refit the champion (tuned hyper-parameters) on train + validation; evaluate once on test.
2. Platt calibration of the PD from 10-fold out-of-fold margins on train + validation (test not used).
3. TreeSHAP contributions (LightGBM pred_contrib) on test, aggregated to the 12 application variables.
4. Decision bands (Approve / Manual review / Decline) on the calibrated PD; shares and observed default rates.
5. Save models/champion.joblib for the Streamlit app and refresh results/v3 tables and figures.
"""
import ast
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from creditscore import decision_support as DS  # noqa: E402
from creditscore import protocol_v3 as P  # noqa: E402
from creditscore.data import RAW_FEATURES, TARGET  # noqa: E402

R = ROOT / "results" / "v3"
FIG = ROOT / "figures" / "v3"
run = joblib.load(R / "run_v3.joblib")
S = run["splits"]
sel = pd.read_csv(R / "selection.csv")
champ = sel.query("selected").iloc[0]["model"]
assert champ in ("LGBM", "XGB"), f"decision_support explains LightGBM / XGBoost only, champion is {champ}"
params = ast.literal_eval(pd.read_csv(R / "tuning.csv").set_index("model").loc[champ, "best_params"])

trva = pd.concat([S["train"], S["valid"]], ignore_index=True)
X, y = trva[RAW_FEATURES], trva[TARGET].values
Xte, yte = S["test"][RAW_FEATURES], S["test"][TARGET].values

# 1. refit + test
t0 = time.time()
final = P.build(champ, seed=42).set_params(**params).fit(X, y)
secs = time.time() - t0
final_test = pd.DataFrame([{"model": champ, "trained_on": "train+valid", "fit_seconds": secs,
                            **P.metrics(yte, final.predict(Xte), P.score_of(final, Xte))}])
final_test.to_csv(R / "final_test.csv", index=False)
# the refit on train+valid re-runs the missing-indicator tests, so its column list can differ from the
# train-only fit in feature_matrix.csv (CLAGE_MISSING is significant on train only): record it separately
pd.DataFrame({"column": final.named_steps["prep"].columns_}).to_csv(R / "feature_matrix_final.csv", index=False)
(ROOT / "results" / "run_meta.json").write_text(json.dumps({"champion": champ, "params": params}, indent=2))


def margin(pipe, Xd):
    return DS._margin(pipe, Xd)


# 2. Platt calibration from OOF margins on train + valid
oof = np.zeros(len(y))
for tr, va in StratifiedKFold(10, shuffle=True, random_state=42).split(X, y):
    m = clone(final).fit(X.iloc[tr], y[tr])
    oof[va] = margin(m, X.iloc[va])
A, B = DS.fit_platt(oof, y)
ex = DS.Explainer(final, A, B)
p_raw = P.score_of(final, Xte)
p_cal = ex.pd(Xte)
cal = pd.DataFrame([{"version": "Chưa hiệu chỉnh (sau tái cân bằng)", "brier": brier_score_loss(yte, p_raw),
                     "mean_pd": p_raw.mean(), "observed_rate": yte.mean()},
                    {"version": "Hiệu chỉnh Platt", "brier": brier_score_loss(yte, p_cal),
                     "mean_pd": p_cal.mean(), "observed_rate": yte.mean()}])
cal["platt_a"], cal["platt_b"] = A, B
cal.to_csv(R / "calibration_test.csv", index=False)
q = pd.qcut(p_cal, 10, labels=False, duplicates="drop")
pd.DataFrame({"decile": q, "pd": p_cal, "y": yte}).groupby("decile").agg(
    mean_pd=("pd", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index() \
    .to_csv(R / "calibration_deciles.csv", index=False)

# 3. SHAP on test, aggregated to application variables
phi, base = ex.explain(Xte)
shap_imp = phi.abs().mean().sort_values(ascending=False)
pd.DataFrame({"feature": shap_imp.index, "mean_abs_shap": shap_imp.values,
              "share": shap_imp.values / shap_imp.sum()}).to_csv(R / "shap_importance_test.csv", index=False)
# gain importance of the refitted champion (engineered columns), as in the other tree models
fi = pd.read_csv(R / "feature_importance.csv")
fi = fi[~fi.model.str.endswith("_final")]
f = P.feature_importance(final, champ)
fi = pd.concat([fi, f.assign(model=f"{champ}_final", rank=np.arange(1, len(f) + 1))], ignore_index=True)
fi.to_csv(R / "feature_importance.csv", index=False)
# additivity check: base + sum(phi) reproduces the calibrated log-odds
lo = np.log(p_cal / (1 - p_cal))
assert np.allclose(base + phi.sum(axis=1).values, lo, atol=1e-6)

# 4. decision bands on test
band = ex.band(p_cal)
bands = pd.DataFrame({"band": band, "y": yte}).groupby("band").agg(n=("y", "size"), defaults=("y", "sum"),
                                                                    default_rate=("y", "mean"))
bands = bands.reindex(list(DS.BANDS)).reset_index()
bands["share"] = bands["n"] / len(yte)
bands["defaults_share"] = bands["defaults"] / yte.sum()
bands.to_csv(R / "decision_bands_test.csv", index=False)

# Gini / KS on test for every train-fitted model and for the refitted champion
rows = []
for m, est in list(run["models"].items()) + [(f"{champ}_final", final)]:
    sc = P.score_of(est, Xte)
    f_, t_, _ = roc_curve(yte, sc)
    au = float(roc_auc_score(yte, sc))
    rows.append({"model": m, "auc": au, "gini": 2 * au - 1, "ks": float(np.max(t_ - f_))})
pd.DataFrame(rows).to_csv(R / "gini_ks_test.csv", index=False)

# example applicants for the thesis / app: highest-PD default, a borderline case, a low-PD good loan
te = S["test"].reset_index(drop=True)
ex_rows = {"decline": int(np.argmax(np.where(yte == 1, p_cal, -1))),
           "review": int(np.argmin(np.abs(p_cal - 0.3))),
           "approve": int(np.argmin(np.where(yte == 0, p_cal, 2)))}
examples = []
for k, i in ex_rows.items():
    r = ex.report(te.iloc[[i]][RAW_FEATURES])
    examples.append({"case": k, "test_index": i, "BAD": int(yte[i]), "pd": r["pd"], "band": r["band"],
                     **{f"phi_{v}": r["contributions"][v] for v in RAW_FEATURES},
                     **{v: te.iloc[i][v] for v in RAW_FEATURES}})
pd.DataFrame(examples).to_csv(R / "example_applicants.csv", index=False)

# 5. artefacts
run.update({"final": final, "champion": champ})
joblib.dump(run, R / "run_v3.joblib", compress=3)
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump({"pipeline": final, "platt_a": A, "platt_b": B, "champion": champ, "params": params,
             "champion_label": P.LABELS[champ],
             "approve_below": DS.APPROVE_BELOW, "decline_from": DS.DECLINE_FROM,
             "medians": X.median(numeric_only=True).to_dict(),
             "ranges": {c: (float(X[c].min()), float(X[c].max())) for c in X.select_dtypes("number")},
             "categories": {c: sorted(X[c].dropna().unique().tolist()) for c in ["REASON", "JOB"]},
             "modes": {c: X[c].mode()[0] for c in ["REASON", "JOB"]},
             "test_applicants": te, "base_rate": float(y.mean())},
            ROOT / "models" / "champion.joblib", compress=3)
rt = pd.read_csv(R / "runtime.csv")
rt["champion"] = champ
rt.to_csv(R / "runtime.csv", index=False)
from creditscore import plots_v3  # noqa: E402
plots_v3.make_all(R, FIG, run["models"], final, champ, S)
print(final_test.round(4).to_string(index=False))
print(cal.round(4).to_string(index=False))
print(bands.round(4).to_string(index=False))
print(shap_imp.round(4).to_string())
print(pd.DataFrame(examples)[["case", "test_index", "BAD", "pd", "band"]])
