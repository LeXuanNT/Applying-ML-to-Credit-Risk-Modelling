"""End-to-end pipeline: run_champion(), run_challengers(), run_all().

Stage 1  data + features   : split, missing summary, WoE/IV, feature selection
Stage 2  (not used)        : HMEQ has no customer history to segment on; see README
Stage 3  PD models         : WoE scorecard (baseline) vs 7 ML challengers
Stage 4  decision layer    : PDO scaling, cut-offs, expected loss, profit

Selection rule (fixed before looking at the test set):
  1. rank all models by VALIDATION AUC;
  2. a model is "tied" with the best if the paired bootstrap test on VALIDATION gives p >= 0.05;
  3. among tied models pick the most explainable tier
     (1 WoE scorecard, 2 logistic raw, 3 decision tree, 4 tree ensembles, 5 MLP/SVM),
     then the highest validation AUC within that tier.
The champion is then calibrated on validation (Platt) and used for every downstream table.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import decision, explain, models, scorecard, validation
from .data import (RAW_FEATURES, TARGET, data_summary, load_hmeq, missing_summary, near_duplicate_groups,
                   near_duplicate_summary,
                   split_data)
from .features import WoEBinner, scorecard_frame, select_woe_features

TIER = {"LR_WoE": 1, "LR_raw": 2, "DT": 3, "RF": 4, "XGB": 4, "LGBM": 4, "MLP": 5, "SVM": 5}
TREE_MODELS = {"RF", "XGB", "LGBM", "DT"}


@dataclass
class Config:
    seed: int = 42
    n_iter: int = 50              # Bayesian optimisation iterations per model
    cv_folds: int = 10            # stratified folds for tuning (Chapter 3)
    ablation_folds: int = 5
    n_boot: int = 1000
    challengers: list = field(default_factory=lambda: list(models.CHALLENGERS))
    lgd: float = decision.LGD
    margin: float = decision.MARGIN
    neutralise: tuple = ()        # e.g. ("DEBTINC",) for the sensitivity run
    group_split: bool = False     # keep near-duplicate groups within one split and one CV fold
    group_radius: float = 0.5


@dataclass
class Run:
    cfg: Config
    splits: dict
    tables: dict = field(default_factory=dict)
    fitted: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)   # scores[model][split] -> np.ndarray
    objects: dict = field(default_factory=dict)


def _neutralise(splits, cols):
    if not cols:
        return splits
    out = {}
    for k, d in splits.items():
        d = d.copy()
        for c in cols:
            d[c] = 0.0                # constant: carries no information, no missing flag
        out[k] = d
    return out


# ---------------------------------------------------------------- champion baseline
def run_champion(run: Run) -> Run:
    s, cfg = run.splits, run.cfg
    tr = s["train"]
    binner = WoEBinner().fit(scorecard_frame(tr), tr[TARGET])
    iv = binner.iv_table()
    W = {k: binner.transform(scorecard_frame(d)) for k, d in s.items()}
    keep, sel_log = select_woe_features(W["train"], iv)
    iv = iv.merge(sel_log[["feature", "decision", "reason"]], on="feature", how="left")
    g = tr["group"] if "group" in tr else None
    lr, info = models.fit_scorecard(W["train"][keep], tr[TARGET], cfg.seed, groups=g)
    # drop features whose beta has the wrong sign (positive on WoE(good/bad)), refit
    wrong = [f for f, b in zip(keep, lr.coef_[0]) if b > 0]
    if wrong:
        keep = [f for f in keep if f not in wrong]
        iv.loc[iv.feature.isin(wrong), ["decision", "reason"]] = ["drop", "positive beta"]
        lr, info = models.fit_scorecard(W["train"][keep], tr[TARGET], cfg.seed, groups=g)
    pts = scorecard.points_table(binner, lr, keep)
    coef = pd.DataFrame({"feature": ["(intercept)"] + keep,
                         "beta": [lr.intercept_[0]] + list(lr.coef_[0])})
    coef = coef.merge(iv[["feature", "iv"]], on="feature", how="left")
    run.tables.update(iv_table=iv, scorecard_coef=coef, points_table=pts,
                      woe_tables=pd.concat([binner.woe_table(f).assign(feature=f) for f in binner.tables_]))
    run.fitted["LR_WoE"] = lr
    run.objects.update(binner=binner, woe_features=keep, W=W)
    run.scores["LR_WoE"] = {k: lr.predict_proba(W[k][keep])[:, 1] for k in s}
    run.tables.setdefault("tuning", []).append(info)
    # trivial baselines: constant PD and best single WoE feature
    best1 = iv.iloc[0]["feature"]
    run.scores["Single_" + best1] = {k: -W[k][best1].to_numpy() for k in s}
    run.objects["single_feature"] = best1
    return run


# ---------------------------------------------------------------- challengers
def run_challengers(run: Run, verbose: bool = True) -> Run:
    s, cfg = run.splits, run.cfg
    X, y = s["train"][RAW_FEATURES], s["train"][TARGET]
    g = s["train"]["group"] if "group" in s["train"] else None
    abl = models.imbalance_ablation(X, y, cfg.challengers, cfg.seed, cfg.ablation_folds, groups=g)
    strat = models.choose_strategy(abl)
    abl["chosen"] = [strat[m] == st for m, st in zip(abl["model"], abl["strategy"])]
    run.tables["imbalance_ablation"] = abl
    for m in cfg.challengers:
        t0 = time.time()
        est, info = models.tune(m, X, y, strat[m], cfg.n_iter, cfg.cv_folds, cfg.seed, groups=g)
        run.fitted[m] = est
        run.scores[m] = {k: models.scores_of(est, d[RAW_FEATURES]) for k, d in s.items()}
        run.tables["tuning"].append(info)
        if verbose:
            print(f"  {m:7s} strategy={strat[m]:12s} cv_auc={info['cv_auc']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return run


# ---------------------------------------------------------------- evaluation
def evaluate(run: Run) -> Run:
    s, cfg = run.splits, run.cfg
    names = ["LR_WoE"] + [m for m in cfg.challengers if m in run.scores]
    rows = []
    for m in list(run.scores):
        for k in ("train", "valid", "test"):
            y, p = s[k][TARGET].to_numpy(), run.scores[m][k]
            d = validation.discrimination(y, p, cfg.n_boot, cfg.seed)
            d.update(model=m, split=k, bad_rate=y.mean(),
                     brier=validation.brier(y, p) if (p.min() >= 0 and p.max() <= 1) else np.nan)
            rows.append(d)
    comp = pd.DataFrame(rows)
    run.tables["model_comparison"] = comp

    # champion selection on VALIDATION
    yv = s["valid"][TARGET].to_numpy()
    v = comp[(comp.split == "valid") & comp.model.isin(names)].sort_values("auc", ascending=False)
    best = v.iloc[0]["model"]
    sel = []
    for m in v["model"]:
        t = validation.paired_auc_test(yv, run.scores[m]["valid"], run.scores[best]["valid"],
                                       n_boot=cfg.n_boot, seed=cfg.seed)
        sel.append({"model": m, "valid_auc": float(v.set_index("model").loc[m, "auc"]),
                    "diff_vs_best": t["auc_diff"], "p_value": t["p_value"], "tier": TIER[m],
                    "tied_with_best": (m == best) or t["p_value"] >= 0.05})
    sel = pd.DataFrame(sel)
    tied = sel[sel.tied_with_best].sort_values(["tier", "valid_auc"], ascending=[True, False])
    champion = tied.iloc[0]["model"]
    sel["champion"] = sel["model"] == champion
    run.tables["selection"] = sel
    run.objects["champion"] = champion

    # significance on TEST vs scorecard and vs champion
    yt = s["test"][TARGET].to_numpy()
    sig = []
    for m in names:
        for ref in ("LR_WoE", champion):
            if m == ref:
                continue
            t = validation.paired_auc_test(yt, run.scores[m]["test"], run.scores[ref]["test"],
                                           n_boot=2 * cfg.n_boot, seed=cfg.seed)
            sig.append({"model": m, "reference": ref, **t})
    run.tables["significance_test"] = pd.DataFrame(sig)

    # threshold metrics: cut-off maximising F1 on VALID, applied to TEST
    th = []
    for m in names:
        thr = validation.best_f1_threshold(yv, run.scores[m]["valid"])
        th.append({"model": m, **validation.threshold_metrics(yt, run.scores[m]["test"], thr)})
    run.tables["threshold_metrics"] = pd.DataFrame(th)
    return run


def calibrate_and_stability(run: Run) -> Run:
    s, cfg = run.splits, run.cfg
    champ = run.objects["champion"]
    yv, yt = s["valid"][TARGET], s["test"][TARGET]
    pd_models = {}
    cal_rows, cal_tabs = [], []
    for m in dict.fromkeys(["LR_WoE", champ]):
        raw_v, raw_t = run.scores[m]["valid"], run.scores[m]["test"]
        cal = validation.PlattCalibrator().fit(raw_v, yv)
        pd_models[m] = cal
        for label, p in (("raw", raw_t), ("platt", cal.predict(raw_t))):
            hl = validation.hosmer_lemeshow(yt, p)
            cal_rows.append({"model": m, "version": label, "brier": validation.brier(yt, p),
                             "brier_base": float(np.mean((yt.mean() - yt) ** 2)),
                             "mean_pd": float(np.mean(p)), "observed": float(yt.mean()), **hl})
            ct = validation.calibration_table(yt, p).reset_index().assign(model=m, version=label)
            cal_tabs.append(ct)
    run.tables["calibration"] = pd.DataFrame(cal_rows)
    run.tables["calibration_bins"] = pd.concat(cal_tabs, ignore_index=True)
    run.objects["calibrators"] = pd_models

    # PD and score for every split (calibrated)
    pdv, score = {}, {}
    for m, cal in pd_models.items():
        pdv[m] = {k: cal.predict(run.scores[m][k]) for k in s}
        score[m] = {k: scorecard.scale_score(pdv[m][k]) for k in s}
    run.objects.update(pd=pdv, score=score)

    # PSI of the score, CSI of top features
    st = []
    for m in pd_models:
        for ref, k in (("train", "valid"), ("train", "test"), ("valid", "test")):
            v = validation.psi(score[m][ref], score[m][k])
            st.append({"model": m, "reference": ref, "vs": k, "psi": v, "status": validation.psi_label(v)})
    run.tables["psi"] = pd.DataFrame(st)
    top = run.tables["iv_table"]["feature"].head(6).tolist()
    top = [f for f in top if f in s["train"].columns]
    run.tables["csi"] = pd.concat([validation.csi_table(s["train"], s[k], top).assign(vs=k)
                                   for k in ("valid", "test")], ignore_index=True)
    run.tables["decile_table"] = validation.decile_table(yt, pdv[champ]["test"]).reset_index()
    run.tables["decile_table_scorecard"] = validation.decile_table(yt, pdv["LR_WoE"]["test"]).reset_index()
    return run


def explain_champion(run: Run) -> Run:
    s = run.splits
    champ = run.objects["champion"]
    shap_model = champ if champ in TREE_MODELS else next(
        (m for m in ("XGB", "LGBM", "RF") if m in run.fitted), None)
    if shap_model is None:
        return run
    vals, Z, _ = explain.shap_values(run.fitted[shap_model], s["test"][RAW_FEATURES])
    imp = explain.shap_importance(vals, Z)
    bysrc = explain.shap_by_source(vals, Z)
    iv = run.tables["iv_table"][["feature", "iv"]].copy()
    iv["iv_rank"] = np.arange(1, len(iv) + 1)
    comp = bysrc.merge(iv, left_on="source", right_on="feature", how="left").drop(columns="feature")
    comp["source"] = comp["source"].astype(str)
    run.tables.update(shap_importance=imp, shap_vs_iv=comp)
    run.objects.update(shap_values=vals, shap_Z=Z, shap_model=shap_model)

    # reason-code examples: 3 declined applicants on test (highest champion PD)
    pdt = run.objects["pd"][champ]["test"]
    idx = np.argsort(-pdt)[[0, 10, 40]]
    pts = run.tables["points_table"]
    Wt = run.objects["W"]["test"][run.objects["woe_features"]]
    ex = []
    Zfull = run.fitted[shap_model].named_steps["prep"].transform(s["test"][RAW_FEATURES])
    import shap
    explainer = shap.TreeExplainer(run.fitted[shap_model].named_steps["clf"])
    sv = np.asarray(explainer.shap_values(Zfull.iloc[idx]))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    G = explain.grouped_shap(sv, Zfull.iloc[idx])
    raw = s["test"][RAW_FEATURES]
    for j, i in enumerate(idx):
        rc = scorecard.reason_codes(Wt.iloc[i], pts)
        g = G.iloc[j].sort_values(ascending=False).head(3)
        le = pd.DataFrame({"feature": g.index, "shap": g.values,
                           "value": [raw.iloc[i][f] if f in raw.columns else Zfull.iloc[i][f] for f in g.index]})
        ex.append({"test_row": int(i), "bad": int(s["test"][TARGET].iloc[i]),
                   "pd_champion": float(pdt[i]),
                   "score_scorecard": float(run.objects["score"]["LR_WoE"]["test"][i]),
                   "scorecard_reasons": "; ".join(f"{r.feature} {r.bin} (-{r.points_lost:.0f})"
                                                  for r in rc.itertuples()),
                   "shap_top": "; ".join(f"{r.feature}={'Missing' if pd.isna(r.value) else (f'{r.value:.3g}' if isinstance(r.value, (int, float, np.floating)) else r.value)} ({r.shap:+.2f})"
                                         for r in le.itertuples())})
    run.tables["reason_codes"] = pd.DataFrame(ex)
    return run


def decision_layer(run: Run) -> Run:
    s, cfg = run.splits, run.cfg
    champ = run.objects["champion"]
    yt, ead = s["test"][TARGET].to_numpy(), s["test"]["LOAN"].to_numpy()
    tabs = []
    for m in dict.fromkeys(["LR_WoE", champ]):
        t = decision.cutoff_table(yt, run.objects["pd"][m]["test"], ead, np.arange(0.30, 1.0001, 0.05),
                                  lgd=cfg.lgd, margin=cfg.margin)
        t["score_cutoff"] = scorecard.scale_score(t["pd_cutoff"])
        tabs.append(t.assign(model=m))
    cut = pd.concat(tabs, ignore_index=True)
    run.tables["cutoff_table"] = cut
    edges = [-np.inf, 520, 560, 600, 640, 680, np.inf]
    run.tables["score_bands"] = decision.score_band_table(
        yt, run.objects["score"][champ]["test"], ead, run.objects["pd"][champ]["test"], edges, cfg.lgd)
    run.tables["score_bands_scorecard"] = decision.score_band_table(
        yt, run.objects["score"]["LR_WoE"]["test"], ead, run.objects["pd"]["LR_WoE"]["test"], edges, cfg.lgd)
    return run


def run_all(df: pd.DataFrame | None = None, cfg: Config | None = None, verbose: bool = True) -> Run:
    cfg = cfg or Config()
    df = load_hmeq() if df is None else df
    # groups and splits are defined on the full data, so a sensitivity run (neutralise) uses the
    # SAME rows in train/valid/test as the main run and isolates the effect of the removed feature
    groups = near_duplicate_groups(df, cfg.group_radius) if cfg.group_split else None
    splits = _neutralise(split_data(df, cfg.seed, groups=groups), cfg.neutralise)
    run = Run(cfg, splits)
    run.tables["data_summary"] = data_summary(splits)
    run.tables["missing_summary"] = missing_summary(df)
    run.tables["near_duplicates"] = near_duplicate_summary(df, None if cfg.group_split else splits, cfg.group_radius)
    for step in (run_champion, lambda r: run_challengers(r, verbose), evaluate,
                 calibrate_and_stability, explain_champion, decision_layer):
        run = step(run)
    run.tables["tuning"] = pd.DataFrame(run.tables["tuning"])
    return run


def save(run: Run, out: str | Path = "results") -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in run.tables.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"{k}.csv", index=False)
    meta = {"champion": run.objects.get("champion"), "woe_features": run.objects.get("woe_features"),
            "shap_model": run.objects.get("shap_model"),
            "single_feature_baseline": run.objects.get("single_feature"),
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(run.cfg).items()}}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    rows = []
    for m, d in run.scores.items():
        for k, v in d.items():
            rows.append(pd.DataFrame({"model": m, "split": k, "row": np.arange(len(v)), "score": v,
                                      "y": run.splits[k][TARGET].to_numpy()}))
    pd.concat(rows).to_csv(out / "scores.csv.gz", index=False)
    import joblib
    joblib.dump(run, out / "run.joblib", compress=3)
