"""Tests on the saved champion artefact (models/champion.joblib) used by the notebook and the web app.

They check that the artefact loads, produces valid calibrated PDs, explains them additively on the
calibrated log-odds scale, handles missing / unseen inputs, and reproduces the stored test-set results.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from creditscore import decision_support as DS
from creditscore.data import RAW_FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "models" / "champion.joblib"
pytestmark = pytest.mark.skipif(not ART.exists(), reason="models/champion.joblib not built yet")


@pytest.fixture(scope="module")
def art():
    return joblib.load(ART)


@pytest.fixture(scope="module")
def ex(art):
    return DS.Explainer(art["pipeline"], art["platt_a"], art["platt_b"], art["approve_below"], art["decline_from"])


def test_pd_is_finite_probability(art, ex):
    p = ex.pd(art["test_applicants"][RAW_FEATURES])
    assert np.isfinite(p).all() and (p > 0).all() and (p < 1).all()


def test_all_missing_and_unseen_category_are_scored(art, ex):
    row = pd.DataFrame([{c: np.nan for c in RAW_FEATURES}])
    assert 0 < ex.pd(row)[0] < 1
    row = art["test_applicants"][RAW_FEATURES].iloc[[0]].copy()
    row["JOB"] = "Astronaut"                      # category never seen in training
    assert 0 < ex.pd(row)[0] < 1


def test_thresholds_are_valid_and_bands_monotone(art, ex):
    assert 0 < art["approve_below"] < art["decline_from"] < 1
    p = np.linspace(0.001, 0.999, 200)
    order = {b: i for i, b in enumerate(DS.BANDS)}
    codes = np.array([order[b] for b in ex.band(p)])
    assert (np.diff(codes) >= 0).all()           # higher PD never gets a milder recommendation


def test_shap_additive_on_calibrated_scale(art, ex):
    X = art["test_applicants"][RAW_FEATURES].iloc[:200]
    phi, base = ex.explain(X)
    p = ex.pd(X)
    assert list(phi.columns) == RAW_FEATURES      # aggregated to the 12 application variables
    assert np.allclose(base + phi.sum(axis=1).values, np.log(p / (1 - p)), atol=1e-6)


def test_artifact_reproduces_stored_test_results(art, ex):
    te = art["test_applicants"]
    auc = roc_auc_score(te[TARGET], ex.pd(te[RAW_FEATURES]))
    stored = pd.read_csv(ROOT / "results" / "v3" / "final_test.csv").iloc[0]["auc"]
    assert abs(auc - stored) < 1e-9


def test_save_load_round_trip_keeps_predictions(art, tmp_path):
    X = art["test_applicants"][RAW_FEATURES]
    joblib.dump(art["pipeline"], tmp_path / "p.joblib")
    again = joblib.load(tmp_path / "p.joblib")
    assert np.array_equal(art["pipeline"].predict_proba(X), again.predict_proba(X))


def test_feature_names_of_final_pipeline_are_recorded(art):
    """The refit on train+valid may keep a different set of missing indicators than the fit on train
    (CLAGE_MISSING: significant on train only), so the deployed column list is saved separately."""
    cols = list(art["pipeline"].named_steps["prep"].columns_)
    f = ROOT / "results" / "v3" / "feature_matrix_final.csv"
    assert f.exists()
    assert pd.read_csv(f)["column"].tolist() == cols
