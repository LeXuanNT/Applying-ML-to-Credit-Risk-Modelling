"""Stage 4a - score scaling (PDO) and scorecard points table.

Score = Offset + Factor * ln(odds_good), Factor = PDO / ln 2.
Default anchors: 600 points at good:bad odds 50:1, PDO = 20.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .credit_metrics import scale_score, scaling  # noqa: F401

ANCHOR = {"target_score": 600, "target_odds": 50, "pdo": 20}


def points_table(binner, lr, features: list[str], **anchor) -> pd.DataFrame:
    a = {**ANCHOR, **anchor}
    factor, offset = scaling(a["target_score"], a["target_odds"], a["pdo"])
    n = len(features)
    alpha = float(lr.intercept_[0])
    rows = []
    for f, beta in zip(features, lr.coef_[0]):
        t = binner.woe_table(f)
        for _, r in t.iterrows():
            rows.append({"feature": f, "bin": r["bin"], "n": int(r["n"]), "bad_rate": r["bad_rate"],
                         "woe": r["woe"], "beta": beta,
                         "points": -(beta * r["woe"] + alpha / n) * factor + offset / n})
    out = pd.DataFrame(rows)
    out["points_rounded"] = out["points"].round().astype(int)
    return out


def score_applicants(W: pd.DataFrame, pts: pd.DataFrame) -> pd.Series:
    """Sum of bin points per applicant, using the WoE value as the bin key."""
    total = pd.Series(0.0, index=W.index)
    for f in W.columns:
        m = pts[pts.feature == f].drop_duplicates("woe").set_index("woe")["points"]
        total += W[f].map(m).fillna(0.0)
    return total


def reason_codes(W_row: pd.Series, pts: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Features where the applicant lost the most points vs the best bin of that feature."""
    rows = []
    for f, w in W_row.items():
        t = pts[pts.feature == f]
        got = t.loc[(t["woe"] - w).abs().idxmin()]
        rows.append({"feature": f, "bin": got["bin"], "points": got["points"],
                     "points_lost": t["points"].max() - got["points"]})
    return pd.DataFrame(rows).sort_values("points_lost", ascending=False).head(k)
