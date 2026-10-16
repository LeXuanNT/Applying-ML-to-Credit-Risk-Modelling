"""Thesis figures (Vietnamese labels), saved as PDF (for LaTeX) and PNG (for preview)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .data import TARGET  # noqa: E402
from .models import MODEL_LABELS  # noqa: E402

# categorical slots in fixed order (validated reference palette, light mode)
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#a8a7a2"
MODEL_ORDER = ["LR_WoE", "LR_raw", "DT", "RF", "XGB", "LGBM", "MLP", "SVM"]
MODEL_COLOR = dict(zip(MODEL_ORDER, SLOT))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2,
})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=170)
    plt.close(fig)


def fig_iv(run, out):
    iv = run.tables["iv_table"].sort_values("iv")
    kept = iv["decision"].eq("keep")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(iv["feature"], iv["iv"], color=np.where(kept, SLOT[0], MUTED), height=0.65)
    for x, lab in [(0.02, ""), (0.1, "0,1"), (0.3, "0,3"), (0.5, "0,5")]:
        ax.axvline(x, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.text(x + 0.01, -0.9, lab, ha="left", va="center", fontsize=7, color=INK2)
    for y, (v, k) in enumerate(zip(iv["iv"], kept)):
        ax.text(v + 0.02, y, f"{v:.3f}", va="center", fontsize=7, color=INK if k else INK2)
    ax.set_xlabel("Information Value (IV)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, iv["iv"].max() * 1.15)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SLOT[0], label="Giữ lại trong scorecard"),
                       Patch(color=MUTED, label="Loại (IV thấp / tương quan / dấu β)")],
              loc="lower right", fontsize=8)
    _save(fig, out, "fig4_iv")


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}" if abs(v) >= 10 else f"{v:.2g}"


def _bin_label(s: str) -> str:
    if s == "Missing":
        return "Thiếu"
    lo, hi = s.strip("([]").split(", ")
    lo, hi = float(lo), float(hi)
    if lo == -np.inf:
        return f"≤ {_fmt(hi)}"
    if hi == np.inf:
        return f"> {_fmt(lo)}"
    return f"{_fmt(lo)} – {_fmt(hi)}"


def fig_woe(run, out, feats=None):
    b = run.objects["binner"]
    feats = feats or run.objects["woe_features"][:4]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, f in zip(axes.ravel(), feats):
        t = b.woe_table(f)
        lab = [_bin_label(s) for s in t["bin"].astype(str)]
        cols = [MUTED if s == "Missing" else SLOT[0] for s in t["bin"]]
        ax.bar(range(len(t)), t["bad_rate"] * 100, color=cols, width=0.7)
        for i, (br, w) in enumerate(zip(t["bad_rate"], t["woe"])):
            ax.text(i, br * 100 + 1.2, f"WoE {w:+.2f}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=6.5)
        ax.set_title(f)
        ax.set_ylabel("Tỷ lệ vỡ nợ (%)")
        ax.set_ylim(0, max(t["bad_rate"]) * 100 * 1.25)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig4_woe_top")


def fig_imbalance(run, out):
    a = run.tables["imbalance_ablation"]
    strat_lab = {"none": "Không xử lý", "class_weight": "Class weight", "smote": "SMOTE", "adasyn": "ADASYN"}
    models_ = [m for m in MODEL_ORDER if m in set(a["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=False)
    w = 0.2
    for j, (metric, title) in enumerate([("cv_auc", "AUC (CV 5-fold, tập train)"),
                                         ("cv_f1_at_default_threshold", "F1 tại ngưỡng mặc định")]):
        ax = axes[j]
        for i, st in enumerate(["none", "class_weight", "smote", "adasyn"]):
            v = [a[(a.model == m) & (a.strategy == st)][metric].mean() for m in models_]
            ax.bar(np.arange(len(models_)) + (i - 1.5) * w, v, width=w * 0.9, color=SLOT[i], label=strat_lab[st])
        ax.set_xticks(range(len(models_)))
        ax.set_xticklabels([m.replace("_", "\n") for m in models_], fontsize=7.5)
        ax.set_title(title)
        lo = np.nanmin(a[metric]) - 0.05
        ax.set_ylim(max(0, lo), min(1, np.nanmax(a[metric]) + 0.03))
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=4, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.22))
    fig.tight_layout()
    _save(fig, out, "fig4_imbalance")


def fig_roc(run, out):
    y = run.splits["test"][TARGET]
    champ = run.objects["champion"]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
    comp = run.tables["model_comparison"].query("split == 'test'").set_index("model")
    for m in MODEL_ORDER:
        if m not in run.scores:
            continue
        fpr, tpr, _ = roc_curve(y, run.scores[m]["test"])
        hi = m in ("LR_WoE", champ)
        ax.plot(fpr, tpr, color=MODEL_COLOR[m], lw=2.2 if hi else 1.2, alpha=1 if hi else 0.75,
                label=f"{MODEL_LABELS[m]} (AUC {comp.loc[m, 'auc']:.3f})", zorder=3 if hi else 2)
    ax.set_xlabel("Tỷ lệ dương tính giả (FPR)")
    ax.set_ylabel("Tỷ lệ dương tính thật (TPR)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_aspect("equal")
    _save(fig, out, "fig4_roc_test")


def fig_gini_ci(run, out):
    c = run.tables["model_comparison"]
    c = c[c.model.isin(MODEL_ORDER)]
    order = c.query("split=='test'").sort_values("gini")["model"].tolist()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for k, (split, col, off) in enumerate([("valid", SLOT[2], 0.17), ("test", SLOT[0], -0.17)]):
        d = c[c.split == split].set_index("model").loc[order]
        yy = np.arange(len(order)) + off
        ax.errorbar(d["gini"], yy, xerr=[d["gini"] - d["gini_lo"], d["gini_hi"] - d["gini"]], fmt="o",
                    color=col, ms=5, capsize=2.5, lw=1.4,
                    label="Tập kiểm định (validation)" if split == "valid" else "Tập kiểm tra (test)")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in order])
    ax.set_xlabel("Gini (khoảng tin cậy 95%, bootstrap)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig4_gini_ci")


def fig_calibration(run, out):
    cb = run.tables["calibration_bins"]
    champ = run.objects["champion"]
    models_ = list(dict.fromkeys(["LR_WoE", champ]))
    fig, axes = plt.subplots(1, len(models_), figsize=(4 * len(models_), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    cal = run.tables["calibration"].set_index(["model", "version"])
    for ax, m in zip(axes, models_):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
        for v, col, lab in [("raw", SLOT[1], "Trước hiệu chuẩn"), ("platt", SLOT[0], "Sau hiệu chuẩn Platt")]:
            d = cb[(cb.model == m) & (cb.version == v)]
            ax.plot(d["mean_pd"], d["observed"], "-o", color=col, ms=4, lw=1.6,
                    label=f"{lab} (Brier {cal.loc[(m, v), 'brier']:.3f})")
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("PD dự báo trung bình (theo decile)")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Tỷ lệ vỡ nợ thực tế")
    fig.tight_layout()
    _save(fig, out, "fig4_calibration")


def fig_score_dist(run, out):
    champ = run.objects["champion"]
    sc = run.objects["score"][champ]
    psi = run.tables["psi"].set_index(["model", "reference", "vs"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    bins = np.linspace(min(np.min(v) for v in sc.values()), max(np.max(v) for v in sc.values()), 35)
    ax = axes[0]
    for k, col, lab in [("train", SLOT[0], "Train"), ("valid", SLOT[2], "Validation"), ("test", SLOT[1], "Test")]:
        ax.hist(sc[k], bins=bins, density=True, histtype="step", lw=1.6, color=col, label=lab)
    ax.set_title(f"PSI train→test = {psi.loc[(champ, 'train', 'test'), 'psi']:.3f}; "
                 f"valid→test = {psi.loc[(champ, 'valid', 'test'), 'psi']:.3f}", fontsize=9)
    ax.set_xlabel("Điểm tín dụng (thang PDO)")
    ax.legend(fontsize=8)
    ax = axes[1]
    y = run.splits["test"][TARGET].to_numpy()
    for v, col, lab in [(0, SLOT[2], "Khách hàng tốt (BAD=0)"), (1, SLOT[7], "Vỡ nợ (BAD=1)")]:
        ax.hist(sc["test"][y == v], bins=bins, density=True, alpha=0.55, color=col, label=lab)
    ax.set_title("Phân tách tốt / xấu trên tập test")
    ax.set_xlabel("Điểm tín dụng (thang PDO)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig4_score_distribution")


def fig_shap(run, out):
    if "shap_values" not in run.objects:
        return
    import shap
    vals, Z = run.objects["shap_values"], run.objects["shap_Z"]
    plt.figure(figsize=(6.5, 4.8))
    shap.summary_plot(vals, Z, max_display=12, show=False, plot_size=None, color_bar_label="Giá trị đặc trưng")
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Giá trị SHAP (tác động lên log-odds vỡ nợ)")
    _save(fig, out, "fig4_shap_summary")

    c = run.tables["shap_vs_iv"].dropna(subset=["iv_rank"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([1, len(c)], [1, len(c)], color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.scatter(c["iv_rank"], c["shap_rank"], s=40, color=SLOT[0], zorder=3, edgecolor="white", lw=1.5)
    for r in c.itertuples():
        ax.annotate(r.source, (r.iv_rank, r.shap_rank), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Thứ hạng theo IV (scorecard)")
    ax.set_ylabel(f"Thứ hạng theo |SHAP| ({MODEL_LABELS[run.objects['shap_model']]})")
    ax.invert_xaxis()
    ax.invert_yaxis()
    _save(fig, out, "fig4_shap_vs_iv")


def fig_cutoff(run, out):
    t = run.tables["cutoff_table"]
    champ = run.objects["champion"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for m in dict.fromkeys(["LR_WoE", champ]):
        d = t[t.model == m]
        axes[0].plot(d["approval_rate"] * 100, d["bad_rate_approved"] * 100, "-o", ms=4,
                     color=MODEL_COLOR[m], label=MODEL_LABELS[m])
        axes[1].plot(d["approval_rate"] * 100, d["profit"] / 1e3, "-o", ms=4, color=MODEL_COLOR[m],
                     label=MODEL_LABELS[m])
    axes[0].set_xlabel("Tỷ lệ phê duyệt (%)")
    axes[0].set_ylabel("Tỷ lệ vỡ nợ trong nhóm được duyệt (%)")
    axes[1].set_xlabel("Tỷ lệ phê duyệt (%)")
    axes[1].set_ylabel("Lợi nhuận trên tập test (nghìn USD)")
    axes[1].axhline(0, color=INK2, lw=0.8)
    for ax in axes:
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, out, "fig4_cutoff")


def fig_confusion(run, out):
    th = run.tables["threshold_metrics"].set_index("model")
    champ = run.objects["champion"]
    models_ = list(dict.fromkeys(["LR_WoE", champ]))
    fig, axes = plt.subplots(1, len(models_), figsize=(3.6 * len(models_), 3.1))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models_):
        r = th.loc[m]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]]).astype(int)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=11, color=INK)
        ax.set_xticks([0, 1], ["Dự báo 0", "Dự báo 1"])
        ax.set_yticks([0, 1], ["Thực tế 0", "Thực tế 1"])
        ax.grid(False)
        ax.set_title(f"{MODEL_LABELS[m]}\nngưỡng PD = {r.threshold:.3f}", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig4_confusion")


def make_all(run, out="figures"):
    out = Path(out)
    for f in (fig_iv, fig_woe, fig_imbalance, fig_roc, fig_gini_ci, fig_calibration, fig_score_dist,
              fig_shap, fig_cutoff, fig_confusion):
        f(run, out)


def fig_split_comparison(results="results", out="figures"):
    """Test Gini with 95% CI: random split vs near-duplicate-aware group split."""
    R = Path(results)
    a = pd.read_csv(R / "model_comparison.csv").query("split == 'test'").set_index("model")
    b = pd.read_csv(R / "group_split" / "model_comparison.csv").query("split == 'test'").set_index("model")
    order = a.loc[MODEL_ORDER].sort_values("gini").index.tolist()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for k, (d, col, lab, off) in enumerate([(a, SLOT[1], "Chia ngẫu nhiên", 0.17),
                                            (b, SLOT[0], "Chia theo nhóm gần trùng", -0.17)]):
        d = d.loc[order]
        yy = np.arange(len(order)) + off
        ax.errorbar(d["gini"], yy, xerr=[d["gini"] - d["gini_lo"], d["gini_hi"] - d["gini"]], fmt="o",
                    color=col, ms=5, capsize=2.5, lw=1.4, label=lab)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in order])
    ax.set_xlabel("Gini trên tập test (khoảng tin cậy 95%)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, Path(out), "fig4_split_comparison")


# ============================================================== Chapter 3 figures
def fig3_target_categorical(df, out="figures"):
    from .data import TARGET
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), gridspec_kw={"width_ratios": [0.8, 1, 1.6]})
    n = df[TARGET].value_counts().sort_index()
    axes[0].bar(["Không vỡ nợ\n(BAD=0)", "Vỡ nợ\n(BAD=1)"], n.values, color=[SLOT[0], SLOT[7]], width=0.6)
    for i, v in enumerate(n.values):
        axes[0].text(i, v + 60, f"{v:,}\n({v / n.sum():.1%})", ha="center", fontsize=7.5)
    axes[0].set_ylim(0, n.max() * 1.25)
    axes[0].set_title("Biến mục tiêu BAD")
    axes[0].grid(axis="x", visible=False)
    base = df[TARGET].mean()
    for ax, c in zip(axes[1:], ["REASON", "JOB"]):
        g = df.groupby(df[c].fillna("Thiếu"))[TARGET].agg(["size", "mean"]).sort_values("mean")
        ax.barh(g.index, g["mean"] * 100, color=[MUTED if i == "Thiếu" else SLOT[0] for i in g.index], height=0.6)
        for y, (r, s) in enumerate(zip(g["mean"], g["size"])):
            ax.text(r * 100 + 0.6, y, f"{r:.1%} (n={s:,})", va="center", fontsize=7)
        ax.axvline(base * 100, color=INK2, lw=0.9, ls=(0, (3, 3)))
        ax.set_xlim(0, g["mean"].max() * 100 * 1.45)
        ax.set_xlabel("Tỷ lệ vỡ nợ (%)")
        ax.set_title(f"Tỷ lệ vỡ nợ theo {c}")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, Path(out), "fig3_target_categorical")


def fig3_numeric_by_bad(df, out="figures"):
    from .data import NUM_FEATURES, TARGET
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6))
    for ax, c in zip(axes.ravel(), NUM_FEATURES):
        data = [df.loc[df[TARGET] == k, c].dropna() for k in (0, 1)]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3, "markeredgecolor": INK2},
                        medianprops={"color": INK, "lw": 1.2})
        for patch, col in zip(bp["boxes"], [SLOT[0], SLOT[7]]):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        ax.set_xticks([1, 2], ["BAD=0", "BAD=1"], fontsize=7)
        ax.set_title(c, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        if c in ("LOAN", "MORTDUE", "VALUE", "DEBTINC"):
            ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, Path(out), "fig3_numeric_by_bad")


def fig3_missing(df, out="figures"):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from .data import RAW_FEATURES, TARGET
    from .eda import missing_table
    m = missing_table(df)
    m = m[m.n_missing > 0].sort_values("pct_missing")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(m))
    axes[0].barh(y + 0.2, m["pct_missing_good"] * 100, height=0.38, color=SLOT[0], label="BAD = 0")
    axes[0].barh(y - 0.2, m["pct_missing_bad"] * 100, height=0.38, color=SLOT[7], label="BAD = 1")
    axes[0].set_yticks(y, m["feature"])
    axes[0].set_xlabel("Tỷ lệ giá trị khuyết (%)")
    axes[0].set_title("Tỷ lệ khuyết theo trạng thái vỡ nợ")
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].grid(axis="y", visible=False)
    M = df[RAW_FEATURES].isna()
    M = M.loc[:, M.any()].astype(bool)
    Z = linkage(M.T.to_numpy(), method="average", metric="jaccard")
    from scipy.cluster import hierarchy
    hierarchy.set_link_color_palette([SLOT[0], SLOT[2], SLOT[1]])
    dendrogram(Z, labels=list(M.columns), ax=axes[1], color_threshold=0.75, above_threshold_color=MUTED,
               leaf_font_size=7.5, leaf_rotation=45)
    axes[1].set_ylabel("Khoảng cách Jaccard")
    axes[1].set_title("Phân cụm mẫu hình thiếu dữ liệu")
    axes[1].grid(False)
    fig.tight_layout()
    _save(fig, Path(out), "fig3_missing")


def fig3_corr(df, out="figures"):
    from .data import NUM_FEATURES
    c = df[NUM_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    mask = np.triu(np.ones_like(c, bool), 1)
    im = ax.imshow(np.ma.array(c, mask=mask), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(c)):
        for j in range(i + 1):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(c.iloc[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(c)), c.index, fontsize=7.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Hệ số Pearson")
    _save(fig, Path(out), "fig3_corr")


def fig3_pipeline(out="figures"):
    """Four-stage pipeline with the data split and the evaluation of each stage."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, title, body, col, fill="#f4f7fb"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=1.3,
                                    edgecolor=col, facecolor=fill))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.9, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.1})

    box(1, 30, 16, 14, "Dữ liệu HMEQ", "5,960 khoản vay\nBAD = vỡ nợ quan sát\nchia phân tầng\n70 / 15 / 15", SLOT[6])
    stages = [
        ("1. Đặc trưng", "khuyết mang thông tin\ntỷ lệ LTV, cờ thiếu\nWoE đơn điệu, IV\nchọn biến cho scorecard"),
        ("2. Phân khúc*", "(không sử dụng)\nHMEQ không có\nlịch sử hành vi\nđể phân khúc"),
        ("3. Mô hình PD", "Scorecard WoE (chuẩn)\nvs LR, DT, RF, XGBoost,\nLightGBM, MLP, SVM\nBayesSearchCV 50×10-fold"),
        ("4. Ra quyết định", "thang điểm PDO\nngưỡng phê duyệt\nEL = PD×LGD×EAD\nlợi nhuận, reason codes"),
    ]
    xs = [21, 41, 61, 81]
    for (t, b), x in zip(stages, xs):
        box(x, 30, 18, 14, t, b, MUTED if t.startswith("2.") else SLOT[0],
            fill="#f6f6f4" if t.startswith("2.") else "#f4f7fb")
    for a, b in zip([17, 39, 59, 79], [21, 41, 61, 81]):
        arrow(a + 0.6, 37, b - 0.6, 37)
    evals = ["IV, đơn điệu WoE,\nCSI các biến", "—", "AUC, Gini, KS, PR-AUC\nCI bootstrap, kiểm định cặp\nBrier, HL, PSI",
             "tỷ lệ duyệt vs tỷ lệ vỡ nợ\nEL, lợi nhuận theo cut-off"]
    for e, x in zip(evals, xs):
        ax.text(x + 9, 26.5, e, ha="center", va="top", fontsize=6.8, color=INK2, linespacing=1.3)
        ax.annotate("", (x + 9, 26.8), (x + 9, 29.4), arrowprops={"arrowstyle": "-", "color": GRID, "lw": 1})
    ax.text(1, 26.5, "Đánh giá từng giai đoạn →", fontsize=7.5, color=INK, fontweight="bold", va="top")
    # split roles
    roles = [("Train (70%)", "fit điền khuyết, WoE, chuẩn hóa,\nSMOTE/ADASYN, tối ưu siêu tham số (CV)", SLOT[0]),
             ("Validation (15%)", "chọn champion, chọn ngưỡng,\nhiệu chuẩn Platt, PSI tham chiếu", SLOT[2]),
             ("Test (15%)", "chỉ dùng một lần để báo cáo\nkết quả cuối cùng (Chương 4)", SLOT[1])]
    for k, (t, b, col) in enumerate(roles):
        x = 1 + k * 33
        ax.add_patch(FancyBboxPatch((x, 2), 31, 9.5, boxstyle="round,pad=0.4,rounding_size=1", lw=1.1,
                                    edgecolor=col, facecolor="white"))
        ax.text(x + 1.5, 9.6, t, fontsize=8.2, fontweight="bold", color=col, va="top")
        ax.text(x + 1.5, 6.3, b, fontsize=6.9, color=INK2, va="top", linespacing=1.3)
    ax.text(1, 14.2, "Vai trò của ba tập dữ liệu (kiểm tra độ bền: chia theo nhóm bản ghi gần trùng)",
            fontsize=7.5, color=INK, fontweight="bold")
    _save(fig, Path(out), "fig3_pipeline")


def make_chapter3(df, out="figures"):
    fig3_pipeline(out)
    for f in (fig3_target_categorical, fig3_numeric_by_bad, fig3_missing, fig3_corr):
        f(df, out)
