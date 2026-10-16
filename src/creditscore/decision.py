"""Stage 4b - decision layer: cut-off analysis with approval rate, bad rate,
expected loss and a simple profit figure.

EL_i = PD_i x LGD x EAD_i with EAD = LOAN (HMEQ has no exposure field) and LGD = 45%
(Basel foundation-IRB senior unsecured value; HMEQ loans are secured by home equity,
so this is conservative - change `lgd` to run a sensitivity).
Profit_i = (1 - y_i) x margin x EAD_i - y_i x LGD x EAD_i uses the REALISED outcome on test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LGD = 0.45
MARGIN = 0.05


def cutoff_table(y, pd_, ead, approval_rates=np.arange(0.5, 1.0001, 0.05),
                 lgd: float = LGD, margin: float = MARGIN) -> pd.DataFrame:
    """Approve the `a` share with the lowest PD, for each target approval rate a."""
    y, pd_, ead = (np.asarray(v, float) for v in (y, pd_, ead))
    order = np.argsort(pd_, kind="mergesort")
    rows = []
    for a in approval_rates:
        k = int(round(a * len(y)))
        ok = np.zeros(len(y), bool)
        ok[order[:k]] = True
        rows.append({
            "approval_rate": k / len(y),
            "pd_cutoff": pd_[order[k - 1]] if k else np.nan,
            "n_approved": k,
            "bad_rate_approved": y[ok].mean() if k else np.nan,
            "bads_approved": int(y[ok].sum()),
            "bads_rejected_share": y[~ok].sum() / y.sum(),
            "expected_loss": float(np.sum(pd_[ok] * lgd * ead[ok])),
            "realised_loss": float(np.sum(y[ok] * lgd * ead[ok])),
            "profit": float(np.sum((1 - y[ok]) * margin * ead[ok] - y[ok] * lgd * ead[ok])),
        })
    return pd.DataFrame(rows)


def score_band_table(y, score, ead, pd_, edges, lgd: float = LGD) -> pd.DataFrame:
    """Bad rate and EL per score band (higher score = safer)."""
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score), "ead": np.asarray(ead),
                      "pd": np.asarray(pd_)})
    d["band"] = pd.cut(d["s"], edges, right=False)
    g = d.groupby("band", observed=False).agg(n=("y", "size"), bads=("y", "sum"), mean_pd=("pd", "mean"),
                                              el=("pd", lambda s: float((s * lgd * d.loc[s.index, "ead"]).sum())))
    g["share"] = g["n"] / g["n"].sum()
    g["bad_rate"] = g["bads"] / g["n"].replace(0, np.nan)
    return g.reset_index()


def max_profit_cutoff(tab: pd.DataFrame) -> pd.Series:
    return tab.loc[tab["profit"].idxmax()]
