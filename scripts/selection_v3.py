"""Apply the weight-free selection rule (src/creditscore/selection.py) to the v3 models.
Reads results/v3/run_v3.joblib + tuning.csv, writes results/v3/selection.csv and selection_oof.csv."""
import ast, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from creditscore import protocol_v3 as P, selection as SEL
from creditscore.data import RAW_FEATURES, TARGET

R = joblib.load(ROOT / "results/v3/run_v3.joblib"); S = R["splits"]
tr, va = S["train"], S["valid"]
Xtr, ytr, Xva, yva = tr[RAW_FEATURES], tr[TARGET].values, va[RAW_FEATURES], va[TARGET].values
tun = pd.read_csv(ROOT / "results/v3/tuning.csv").set_index("model")
oof_path = ROOT / "results/v3/selection_oof.csv"
if oof_path.exists():
    oof = pd.read_csv(oof_path).to_dict("list"); oof = {k: np.array(v) for k, v in oof.items()}
else:
    oof, t0 = {}, time.time()
    for m in R["models"]:
        est = P.build(m, seed=42).set_params(**ast.literal_eval(tun.loc[m, "best_params"]))
        oof[m] = SEL.oof_scores(est, Xtr, ytr, P.score_of)
        print(m, "oof", round(time.time() - t0), "s", flush=True)
    pd.DataFrame(oof).to_csv(oof_path, index=False)
vs = {m: P.score_of(e, Xva) for m, e in R["models"].items()}
vp = {m: e.predict(Xva) for m, e in R["models"].items()}
sel = SEL.select_champion(yva, vs, vp, oof, ytr)
sel.to_csv(ROOT / "results/v3/selection.csv", index=False)
SEL.select_champion(yva, vs, vp, oof, ytr, require_explainable=False) \
    .to_csv(ROOT / "results/v3/selection_no_gate.csv", index=False)
# population stability of each train-fitted model's score: validation vs test (same model, both out-of-sample)
te = S["test"]
pd.DataFrame([{"model": m, "psi_valid_test": SEL.psi(vs[m], P.score_of(e, te[RAW_FEATURES]))}
              for m, e in R["models"].items()]).to_csv(ROOT / "results/v3/stability.csv", index=False)
pd.set_option("display.width", 250)
cols = ["model", "step0_pass", "auc", "auc_lo", "auc_hi", "delta_auc_vs_best", "p_delong", "cv_auc", "cv_auc_std", "ks", "pr_auc", "brier", "mcc",
        "step1_pass", "avg_rank_all", "avg_rank_eligible", "tie", "selected"]
print(sel[cols].round(4).to_string(index=False))
