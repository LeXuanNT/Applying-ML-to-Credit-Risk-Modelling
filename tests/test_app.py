"""Smoke tests of the Streamlit app: every page renders from the saved champion and results/v3 without raising,
and the scoring page scores an example, a test applicant and an applicant with every field missing."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "champion.joblib").exists(), reason="no model artefact")
APP = str(ROOT / "app" / "streamlit_app.py")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=180)


def _pd(at) -> float:
    v = at.metric[0].value
    assert v.endswith("%")
    return float(v.lstrip("<> ").rstrip("%"))


def test_model_manager_every_model():
    at = _app().run()
    assert not at.exception
    assert at.selectbox[0].value == "LGBM"
    for code in at.selectbox[0].options:
        at.selectbox[0].set_value(code).run()
        assert not at.exception, code


def test_compare_page():
    at = _app().run()
    at.switch_page("views/compare.py").run()
    assert not at.exception
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 9


def test_score_examples_match_saved_pd():
    sys.path.insert(0, str(ROOT / "src"))
    ex = pd.read_csv(ROOT / "results" / "v3" / "example_applicants.csv").set_index("case")
    at = _app().run()
    at.switch_page("views/score.py").run()
    assert not at.exception
    for case in ["approve", "review", "decline"]:
        at.radio(key="case").set_value(case).run()
        assert not at.exception
        shown = _pd(at)
        p = 100 * ex.loc[case, "pd"]
        assert shown == pytest.approx(min(max(p, 0.1), 99.9), abs=0.06), case


def test_score_test_applicant_and_all_missing():
    at = _app().run()
    at.switch_page("views/score.py").run()
    at.segmented_control(key="source").set_value("Hồ sơ trong tập kiểm tra").run()
    assert not at.exception and 0 <= _pd(at) <= 100
    at.segmented_control(key="source").set_value("Nhập hồ sơ mới").run()
    for cb in at.checkbox:
        cb.check()
    for sb in at.selectbox:
        sb.set_value("(không có thông tin)")
    at.run()
    assert not at.exception
    assert 0 < _pd(at) < 100
