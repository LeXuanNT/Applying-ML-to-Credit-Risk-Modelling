"""Step 0: record the environment and verify the raw HMEQ file before any modelling."""
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/hmeq.csv") if Path("data/hmeq.csv").exists() else Path("hmeq.csv")
EXPECTED_COLUMNS = ["BAD", "LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ",
                    "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost",
            "imbalanced-learn", "scikit-optimize", "shap", "statsmodels"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


df = pd.read_csv(DATA_PATH)
checks = {
    "columns_match": list(df.columns) == EXPECTED_COLUMNS,
    "n_rows": len(df),
    "n_duplicates": int(df.duplicated().sum()),
    "bad_values": sorted(df["BAD"].unique().tolist()),
    "bad_rate": round(float(df["BAD"].mean()), 4),
    "reason_levels": sorted(df["REASON"].dropna().unique().tolist()),
    "job_levels": sorted(df["JOB"].dropna().unique().tolist()),
    "negative_numeric": {c: int((df[c] < 0).sum()) for c in df.select_dtypes("number") if (df[c] < 0).any()},
    "missing_share": df.isna().mean().round(4).sort_values(ascending=False).to_dict(),
}
manifest = {"data_file": str(DATA_PATH), "data_sha256": sha256(DATA_PATH),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": package_versions(), "checks": checks}
Path("reports").mkdir(exist_ok=True)
Path("reports/run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(json.dumps(manifest, indent=2, ensure_ascii=False))
