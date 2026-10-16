"""Run the experiment exactly as designed in thesis draft v3 (Ch.1-3) and write results/v3/*.csv
and figures/v3/*.pdf|png.  python scripts/run_v3.py [--fast]"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from creditscore import protocol_v3 as P  # noqa: E402
from creditscore.data import RAW_FEATURES, TARGET, data_summary, load_hmeq, split_data  # noqa: E402
from creditscore.features import WoEBinner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true")
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()
# --fast is a smoke test only: it writes to results/v3_fast so the thesis results in results/v3 are never overwritten
OUT = ROOT / "results" / ("v3_fast" if a.fast else ("v3_synthetic" if a.synthetic else "v3"))
FIG = ROOT / "figures" / OUT.name
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
n_iter, folds, abl_folds = (4, 3, 3) if a.fast else (50, 10, 5)

if a.synthetic:
    from creditscore.synthetic import make_hmeq_like
    df = make_hmeq_like(2500)
else:
    df = load_hmeq()
S = split_data(df, 42)
tr, va, te = S["train"], S["valid"], S["test"]
X, y = tr[RAW_FEATURES], tr[TARGET]
T = {"split": data_summary(S)}
t_all = time.time()

# 4.1 preprocessing results
T["missing_tests"] = P.missing_indicator_tests(tr)
prep = P.V3Preprocessor().fit(X, y)
T["feature_matrix"] = pd.DataFrame({"column": prep.columns_})
for k, d in S.items():
    Z = prep.transform(d[RAW_FEATURES])
    T.setdefault("prep_check", []).append({"split": k, "rows": Z.shape[0], "columns": Z.shape[1],
                                           "missing_after": int(Z.isna().sum().sum())})
T["prep_check"] = pd.DataFrame(T["prep_check"])
yoj = pd.cut(tr["YOJ"], P.YOJ_EDGES, labels=P.YOJ_LABELS).astype(object).where(tr["YOJ"].notna(), "Missing")
T["yoj_bins"] = tr.groupby(yoj)[TARGET].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "bad_rate"}) \
    .reindex(P.YOJ_LABELS + ["Missing"]).reset_index().rename(columns={"YOJ": "bin"})
binner = WoEBinner().fit(tr[RAW_FEATURES], y)
T["iv"] = binner.iv_table()
T["woe"] = pd.concat([binner.woe_table(f).assign(feature=f) for f in binner.tables_])
print("prep done", flush=True)

# 4.2 imbalance techniques (Ch.1: evaluate each method)
T["imbalance"] = P.imbalance_comparison(X, y, P.MODELS, 42, abl_folds)
print("imbalance done", round(time.time() - t_all), flush=True)

# 4.3 tuning + training + evaluation
models, infos, opts, evals = {}, [], [], []
CKPT = OUT / "checkpoints"   # one file per tuned model: resume after a Colab disconnect
CKPT.mkdir(parents=True, exist_ok=True)
for m in P.MODELS:
    ck = CKPT / f"{m}.joblib"
    if ck.exists() and not a.synthetic:
        bp, info, op = joblib.load(ck)
        print(f"  {m:5s} loaded from checkpoint {ck.name}", flush=True)
    else:
        bp, info, op = P.tune(m, X, y, n_iter, folds)
        if not a.synthetic:
            joblib.dump((bp, info, op), ck)
    est, secs = P.fit_with(m, bp, X, y)
    info["fit_seconds"] = secs
    models[m] = est
    infos.append(info)
    opts.append(op)
    for k, d in S.items():
        Xd, yd = d[RAW_FEATURES], d[TARGET]
        evals.append({"model": m, "split": k, **P.metrics(yd, est.predict(Xd), P.score_of(est, Xd))})
    print(f"  {m:5s} cv_f1={info['cv_f1']:.4f} ({info['tuning_seconds']:.0f}s)", flush=True)
T["tuning"] = pd.DataFrame(infos)
T["tuning_options"] = pd.concat(opts, ignore_index=True)
T["evaluation"] = pd.DataFrame(evals)

# 4.4 selection on validation (creditscore.selection): DeLong eligibility + equal-weight average rank
#     -> final model, retrained on train+valid, evaluated on test
from creditscore import selection as SEL  # noqa: E402
tun = T["tuning"].set_index("model")
oof = {m: SEL.oof_scores(P.build(m, seed=42).set_params(**tun.loc[m, "best_params"]), tr[RAW_FEATURES],
                         tr[TARGET].values, P.score_of) for m in models}
T["selection_oof"] = pd.DataFrame(oof)
T["selection"] = SEL.select_champion(va[TARGET].values,
                                     {m: P.score_of(e, va[RAW_FEATURES]) for m, e in models.items()},
                                     {m: e.predict(va[RAW_FEATURES]) for m, e in models.items()},
                                     oof, tr[TARGET].values)
champ = T["selection"].query("selected").iloc[0]["model"]
T["stability"] = pd.DataFrame([{"model": m, "psi_valid_test": SEL.psi(P.score_of(e, va[RAW_FEATURES]),
                                                                    P.score_of(e, te[RAW_FEATURES]))}
                               for m, e in models.items()])
bp = T["tuning"].set_index("model").loc[champ, "best_params"]
trva = pd.concat([tr, va], ignore_index=True)
final, secs = P.fit_with(champ, bp, trva[RAW_FEATURES], trva[TARGET])
T["final_test"] = pd.DataFrame([{"model": champ, "trained_on": "train+valid", "fit_seconds": secs,
                                 **P.metrics(te[TARGET], final.predict(te[RAW_FEATURES]),
                                             P.score_of(final, te[RAW_FEATURES]))}])
print("champion", champ, flush=True)

# 4.5 feature importance
fi = []
for m in ["LR", "DT", "RF", "XGB", "LGBM"]:
    f = P.feature_importance(models[m], m)
    if len(f):
        fi.append(f.assign(model=m, rank=np.arange(1, len(f) + 1)))
f = P.feature_importance(final, champ)
if len(f):
    fi.append(f.assign(model=f"{champ}_final", rank=np.arange(1, len(f) + 1)))
T["feature_importance"] = pd.concat(fi, ignore_index=True)
T["runtime"] = pd.DataFrame([{"total_minutes": (time.time() - t_all) / 60, "champion": champ}])

for k, v in T.items():
    v.to_csv(OUT / f"{k}.csv", index=False)
joblib.dump({"models": models, "final": final, "champion": champ, "splits": S}, OUT / "run_v3.joblib", compress=3)

# figures
from creditscore import plots_v3  # noqa: E402
plots_v3.make_all(OUT, FIG, models, final, champ, S)
print(f"done in {(time.time() - t_all) / 60:.1f} min; champion {champ}")
