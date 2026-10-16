"""Figures for Chapter 4 (protocol v3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .data import RAW_FEATURES, TARGET
from .plots import INK, INK2, MUTED, SLOT, _save, plt
from .protocol_v3 import LABELS, MODELS, score_of

COL = dict(zip(MODELS, SLOT + ["#7a5c2e"]))   # 9th series: brown (only in line charts with legend)
STRAT = {"none": "Không xử lý", "class_weight": "class_weight", "ros": "ROS", "rus": "RUS",
         "smote": "SMOTE", "adasyn": "ADASYN", "smote_tomek": "SMOTE-Tomek", "smote_enn": "SMOTE-ENN"}


def fig_iv(R, out):
    iv = pd.read_csv(R / "iv.csv").sort_values("iv")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(iv["feature"], iv["iv"], color=np.where(iv["iv"] >= 0.02, SLOT[0], MUTED), height=0.65)
    for yv, v in enumerate(iv["iv"]):
        ax.text(v + 0.02, yv, f"{v:.3f}", va="center", fontsize=7)
    for x in (0.02, 0.1, 0.3, 0.5):
        ax.axvline(x, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("Information Value (IV)")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig4_iv")


def fig_imbalance(R, out):
    d = pd.read_csv(R / "imbalance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    strats = list(STRAT)
    w = 0.1
    for ax, met, title in [(axes[0], "f1", "F1 (CV 5-fold, tập train)"), (axes[1], "recall", "Recall (CV 5-fold, tập train)")]:
        for i, st in enumerate(strats):
            v = [d[(d.model == m) & (d.strategy == st)][met].mean() for m in MODELS]
            ax.bar(np.arange(len(MODELS)) + (i - (len(strats) - 1) / 2) * w, v, width=w * 0.9, color=SLOT[i],
                   label=STRAT[st])
        ax.set_xticks(range(len(MODELS)), MODELS, fontsize=7.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=8, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 1.22))
    fig.tight_layout()
    _save(fig, out, "fig4_imbalance")


def fig_roc(models, S, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, k, t in [(axes[0], "valid", "Tập kiểm định (validation)"), (axes[1], "test", "Tập kiểm tra (test)")]:
        y = S[k][TARGET]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
        for m in MODELS:
            s = score_of(models[m], S[k][RAW_FEATURES])
            fpr, tpr, _ = roc_curve(y, s)
            from sklearn.metrics import roc_auc_score
            ax.plot(fpr, tpr, lw=1.4, color=COL[m], label=f"{m} (AUC {roc_auc_score(y, s):.3f})")
        ax.set_title(t)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR (Recall)")
        ax.legend(fontsize=6.5, loc="lower right")
        ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, out, "fig4_roc")


def fig_metrics(R, out):
    e = pd.read_csv(R / "evaluation.csv")
    e = e[e.split == "test"].set_index("model").loc[MODELS]
    mets = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    w = 0.13
    for i, met in enumerate(mets):
        ax.bar(np.arange(len(MODELS)) + (i - 2.5) * w, e[met], width=w * 0.9, color=(SLOT + ["#7a5c2e"])[i],
               label=met.upper() if met in ("mcc", "auc") else met.capitalize())
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=6, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.15))
    ax.grid(axis="x", visible=False)
    _save(fig, out, "fig4_metrics_test")


def fig_confusion(R, out, champ):
    e = pd.read_csv(R / "evaluation.csv")
    f = pd.read_csv(R / "final_test.csv").iloc[0]
    lr = e[(e.model == "LR") & (e.split == "test")].iloc[0]
    items = [("Logistic Regression (WoE)\nhuấn luyện trên train", lr), (f"{LABELS[champ]}\nhuấn luyện lại trên train+valid", f)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, (t, r) in zip(axes, items):
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]]).astype(int)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=11, color=INK)
        ax.set_xticks([0, 1], ["Dự báo 0", "Dự báo 1"])
        ax.set_yticks([0, 1], ["Thực tế 0", "Thực tế 1"])
        ax.grid(False)
        ax.set_title(t, fontsize=8.5)
    fig.tight_layout()
    _save(fig, out, "fig4_confusion")


def fig_selection(R, out):
    """AUC with 95% DeLong CI on validation; models not significantly below the best are highlighted."""
    r = pd.read_csv(R / "selection.csv").sort_values("auc")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for yv, x in enumerate(r.itertuples()):
        c = SLOT[0] if x.selected else (SLOT[1] if x.step1_pass else MUTED)
        ax.errorbar(x.auc, yv, xerr=[[x.auc - x.auc_lo], [x.auc_hi - x.auc]], fmt="o", color=c, ms=5, capsize=3)
        ax.text(x.auc_hi + 0.004, yv, f"{x.auc:.3f}" + ("" if x.step1_pass else f"  (p={x.p_delong:.3f})"
                                                         if x.p_delong >= 0.001 else "  (p<0.001)"),
                va="center", fontsize=7)
    best = r.loc[r.auc.idxmax()]
    ax.axvline(best.auc, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(range(len(r)), [LABELS[m] for m in r["model"]])
    ax.set_xlabel("AUC trên tập kiểm định (khoảng tin cậy 95% DeLong)")
    ax.set_xlim(r.auc_lo.min() - 0.01, 1.03)
    ax.set_xticks([t for t in ax.get_xticks() if t <= 1.0001])
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig4_selection")


def fig_importance(R, out, champ):
    fi = pd.read_csv(R / "feature_importance.csv")
    tree = importance_model(R, champ)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4))
    for ax, m, t, col in [(axes[0], tree, f"{LABELS[tree.replace('_final', '')]} (tỷ trọng gain)", "importance"),
                          (axes[1], "LR", "Logistic Regression (WoE): |hệ số|", "importance")]:
        d = fi[fi.model == m].head(12).iloc[::-1]
        ax.barh(d["feature"], d[col], color=SLOT[0] if m != "LR" else SLOT[1], height=0.6)
        ax.set_title(t)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig4_importance")


def importance_model(R, champ):
    """Champion if it has native importances, otherwise the best-ranked tree model in the selection table."""
    fi = pd.read_csv(Path(R) / "feature_importance.csv")
    if f"{champ}_final" in set(fi.model):
        return f"{champ}_final"
    rk = pd.read_csv(Path(R) / "selection.csv")              # eligible models first, by average rank
    return next(m for m in rk["model"] if m in set(fi.model) and m != "LR")


def fig_shap(R, out):
    """Global importance of the champion: mean |TreeSHAP| on test, summed to the 12 application variables."""
    d = pd.read_csv(R / "shap_importance_test.csv").iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(d["feature"], d["mean_abs_shap"], color=SLOT[0], height=0.6)
    for yv, (v, sh) in enumerate(zip(d["mean_abs_shap"], d["share"])):
        ax.text(v + 0.02, yv, f"{100 * sh:.1f}%", va="center", fontsize=7.5)
    ax.set_xlabel("Trung bình |SHAP| (log-odds đã hiệu chỉnh), tập kiểm tra")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig4_shap")


def fig_calibration(R, out):
    d = pd.read_csv(R / "calibration_deciles.csv")
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax.plot(d["mean_pd"], d["observed"], "o-", color=SLOT[0], ms=4)
    ax.set_xlabel("PD dự báo trung bình (nhóm thập phân vị)")
    ax.set_ylabel("Tỷ lệ vỡ nợ thực tế")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _save(fig, out, "fig4_calibration")


def fig_example(R, out):
    """Per-applicant contributions for the three example applicants (calibrated log-odds)."""
    ex = pd.read_csv(R / "example_applicants.csv")
    title = {"decline": "Từ chối", "review": "Xem xét thủ công", "approve": "Phê duyệt"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, (_, r) in zip(axes, ex.iterrows()):
        v = pd.Series({f: r[f"phi_{f}"] for f in RAW_FEATURES}).reindex(RAW_FEATURES[::-1])
        ax.barh(v.index, v.values, color=[SLOT[1] if x > 0 else SLOT[0] for x in v.values], height=0.6)
        ax.axvline(0, color=INK2, lw=0.6)
        pdt = "> 99.9%" if r["pd"] > 0.999 else ("< 0.1%" if r["pd"] < 0.001 else f"{100 * r['pd']:.1f}%")
        ax.set_title(f"{title[r['case']]}: PD {pdt}", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("Đóng góp vào log-odds vỡ nợ (dương: tăng rủi ro)")
    fig.tight_layout()
    _save(fig, out, "fig4_example")


def make_all(R, out, models, final, champ, S):
    R, out = Path(R), Path(out)
    fig_iv(R, out)
    fig_imbalance(R, out)
    fig_roc(models, S, out)
    fig_metrics(R, out)
    fig_confusion(R, out, champ)
    fig_selection(R, out)
    fig_importance(R, out, champ)
    if (R / "shap_importance_test.csv").exists():
        fig_shap(R, out)
        fig_calibration(R, out)
        fig_example(R, out)
