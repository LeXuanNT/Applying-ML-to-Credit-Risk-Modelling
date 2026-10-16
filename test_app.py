"""Smoke test of the Streamlit app: it loads the saved champion and scores a test applicant and an
applicant with every field missing, without raising."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "champion.joblib").exists(), reason="no model artefact")


def test_app_scores_test_and_all_missing_applicant():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].value.endswith("%")
    at.sidebar.radio[0].set_value("Nhập hồ sơ mới").run()
    for cb in at.sidebar.checkbox:
        cb.check()
    at.run()
    assert not at.exception
    pd_value = float(at.metric[0].value.rstrip("%"))
    assert 0 < pd_value < 100
