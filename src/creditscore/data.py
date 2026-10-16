"""Stage 1a - load HMEQ, define the target, split into train / validation / test.

HMEQ (Home Equity) has 5,960 home-equity loans and a fixed target:
BAD = 1 if the applicant defaulted or was seriously delinquent, 0 otherwise.
The dataset has no application date, so an out-of-time (OOT) split is not
possible; we use a stratified 70/15/15 split (Chapter 3) and repeated CV on train.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "BAD"
NUM_FEATURES = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ",
                "CLAGE", "NINQ", "CLNO", "DEBTINC"]
CAT_FEATURES = ["REASON", "JOB"]
RAW_FEATURES = NUM_FEATURES + CAT_FEATURES

DESCRIPTIONS = {
    "BAD": "1 = vỡ nợ hoặc quá hạn nghiêm trọng; 0 = trả nợ tốt",
    "LOAN": "Số tiền vay đề nghị (USD)",
    "MORTDUE": "Dư nợ thế chấp hiện tại (USD)",
    "VALUE": "Giá trị tài sản bảo đảm (USD)",
    "REASON": "Mục đích vay: DebtCon / HomeImp",
    "JOB": "Nhóm nghề nghiệp",
    "YOJ": "Số năm làm công việc hiện tại",
    "DEROG": "Số báo cáo xấu (derogatory reports)",
    "DELINQ": "Số hạn mức tín dụng quá hạn",
    "CLAGE": "Tuổi của hạn mức tín dụng lâu nhất (tháng)",
    "NINQ": "Số lần truy vấn tín dụng gần đây",
    "CLNO": "Số hạn mức tín dụng hiện có",
    "DEBTINC": "Tỷ lệ nợ trên thu nhập (%)",
}


def default_data_path() -> Path:
    """data/hmeq.csv relative to the repo root (works from notebooks/ and scripts/)."""
    here = Path(__file__).resolve()
    for parent in [Path.cwd(), *Path.cwd().parents, *here.parents]:
        p = parent / "data" / "hmeq.csv"
        if p.exists():
            return p
    return Path("data/hmeq.csv")


def load_hmeq(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or default_data_path())
    missing = set([TARGET] + RAW_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df[TARGET] = df[TARGET].astype(int)
    for c in CAT_FEATURES:
        df[c] = df[c].astype("object")
    return df


def near_duplicate_groups(df: pd.DataFrame, radius: float = 0.5) -> np.ndarray:
    """Group id per row: connected components of rows lying within `radius` of each other
    in the standardised feature space (LOAN excluded). HMEQ contains many such near-copies
    (same job, tenure, history; property values within a few %), almost always with the same
    label, so a random split puts one copy in train and its twin in test."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import radius_neighbors_graph
    from sklearn.preprocessing import StandardScaler

    from .features import HMEQPreprocessor

    Z = HMEQPreprocessor().fit(df).transform(df).drop(columns=["LOAN"])
    X = StandardScaler().fit_transform(Z)
    _, lab = connected_components(radius_neighbors_graph(X, radius), directed=False)
    return lab


def split_data(df: pd.DataFrame, seed: int = 42, test_size: float = 0.15,
               valid_size: float = 0.15, groups: np.ndarray | None = None) -> dict[str, pd.DataFrame]:
    """Stratified train / valid / test split (default 70/15/15, as in Chapter 3).
    With `groups`, whole groups go to one split (StratifiedGroupKFold with 20 folds:
    3 folds test, 3 folds valid, 14 folds train) and a `group` column is kept."""
    if groups is not None:
        from sklearn.model_selection import StratifiedGroupKFold
        d = df.assign(group=groups).reset_index(drop=True)
        fold = np.empty(len(d), int)
        sgk = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
        for k, (_, idx) in enumerate(sgk.split(d, d[TARGET], d["group"])):
            fold[idx] = k
        n_te, n_va = round(20 * test_size), round(20 * valid_size)
        part = np.where(fold < n_te, "test", np.where(fold < n_te + n_va, "valid", "train"))
        return {k: d[part == k].reset_index(drop=True) for k in ("train", "valid", "test")}
    train, temp = train_test_split(df, test_size=test_size + valid_size,
                                   stratify=df[TARGET], random_state=seed)
    valid, test = train_test_split(temp, test_size=test_size / (test_size + valid_size),
                                   stratify=temp[TARGET], random_state=seed)
    return {"train": train.reset_index(drop=True), "valid": valid.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def data_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, d in splits.items():
        rows.append({"split": name, "rows": len(d), "bads": int(d[TARGET].sum()),
                     "bad_rate": d[TARGET].mean(),
                     "share": np.nan})
    out = pd.DataFrame(rows)
    out["share"] = out["rows"] / out["rows"].sum()
    total = pd.DataFrame([{"split": "total", "rows": out["rows"].sum(), "bads": out["bads"].sum(),
                           "bad_rate": out["bads"].sum() / out["rows"].sum(), "share": 1.0}])
    return pd.concat([out, total], ignore_index=True)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Missing share per feature and bad rate with / without the value."""
    rows = []
    for c in RAW_FEATURES:
        m = df[c].isna()
        rows.append({"feature": c, "n_missing": int(m.sum()), "pct_missing": m.mean(),
                     "bad_rate_missing": df.loc[m, TARGET].mean() if m.any() else np.nan,
                     "bad_rate_present": df.loc[~m, TARGET].mean()})
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def near_duplicate_summary(df: pd.DataFrame, splits: dict | None = None, radius: float = 0.5) -> pd.DataFrame:
    """Size of the near-duplicate problem: groups, rows covered, label purity, and (for a given
    random split) how often a test row's nearest train neighbour has the same label."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    from .features import HMEQPreprocessor

    g = near_duplicate_groups(df, radius)
    sz = pd.Series(g).value_counts()
    multi = sz[sz > 1].index
    pure = df[TARGET].groupby(g).mean()[multi].isin([0, 1]).mean()
    out = {"radius": radius, "groups_2plus": len(multi), "rows_in_groups": int(sz[multi].sum()),
           "share_rows_in_groups": sz[multi].sum() / len(df), "largest_group": int(sz.max()),
           "label_pure_share": pure}
    if splits is not None:
        p = HMEQPreprocessor().fit(splits["train"])
        A, B = p.transform(splits["train"]), p.transform(splits["test"])
        sc = StandardScaler().fit(A)
        _, idx = NearestNeighbors(n_neighbors=1).fit(sc.transform(A)).kneighbors(sc.transform(B))
        ytr, yte = splits["train"][TARGET].to_numpy(), splits["test"][TARGET].to_numpy()
        br = ytr.mean()
        out.update(nn1_label_agreement=(ytr[idx[:, 0]] == yte).mean(),
                   random_pairing_agreement=br ** 2 + (1 - br) ** 2)
    return pd.DataFrame([out])
