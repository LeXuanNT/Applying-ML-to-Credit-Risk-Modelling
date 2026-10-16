"""Synthetic HMEQ-like data so tests and the notebook run offline.

Same columns, dtypes and roughly the same default rate (~20%) and missingness
pattern as HMEQ (DEBTINC missing ~21% and strongly linked to default).
Results on synthetic data only prove that the code runs - not that the model works.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_hmeq_like(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    value = np.exp(rng.normal(11.4, 0.5, n))
    mortdue = value * rng.beta(4, 3, n)
    loan = np.clip(np.exp(rng.normal(9.7, 0.55, n)), 1100, 90000).round(-2)
    yoj = np.clip(rng.gamma(1.6, 5.5, n), 0, 41).round(1)
    derog = rng.poisson(0.25, n)
    delinq = rng.poisson(0.45, n)
    clage = np.clip(rng.normal(180, 85, n), 0, 1100)
    ninq = rng.poisson(1.2, n)
    clno = np.clip(rng.normal(21, 10, n), 0, 71).round()
    debtinc = np.clip(rng.normal(34, 8, n), 0.5, 200)
    reason = rng.choice(["DebtCon", "HomeImp"], n, p=[0.69, 0.31])
    job = rng.choice(["Other", "ProfExe", "Office", "Mgr", "Self", "Sales"], n,
                     p=[0.42, 0.22, 0.16, 0.13, 0.04, 0.03])

    logit = (-2.2 + 0.9 * delinq + 0.6 * derog + 0.12 * ninq - 0.006 * (clage - 180)
             + 0.06 * (debtinc - 34) + 0.6 * (loan / value) - 0.02 * yoj
             + 0.4 * np.isin(job, ["Self", "Sales"]))
    debtinc_missing = rng.random(n) < 1 / (1 + np.exp(-(logit + 0.3)))  # informative
    logit = logit + 1.6 * debtinc_missing
    bad = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({"BAD": bad, "LOAN": loan.astype(int), "MORTDUE": mortdue.round(),
                       "VALUE": value.round(), "REASON": reason, "JOB": job, "YOJ": yoj,
                       "DEROG": derog.astype(float), "DELINQ": delinq.astype(float),
                       "CLAGE": clage, "NINQ": ninq.astype(float), "CLNO": clno,
                       "DEBTINC": np.where(debtinc_missing, np.nan, debtinc)})
    for c, rate in {"MORTDUE": .09, "VALUE": .02, "REASON": .04, "JOB": .05, "YOJ": .09,
                    "DEROG": .12, "DELINQ": .10, "CLAGE": .05, "NINQ": .09, "CLNO": .04}.items():
        df.loc[rng.random(n) < rate, c] = np.nan
    joint = rng.random(n) < 0.03                      # CLAGE-CLNO-DELINQ-NINQ missing together
    df.loc[joint, ["CLAGE", "CLNO", "DELINQ", "NINQ"]] = np.nan
    return df
