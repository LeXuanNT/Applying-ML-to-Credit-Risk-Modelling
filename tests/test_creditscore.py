import numpy as np
import pandas as pd
import pytest

from creditscore import credit_metrics as cm
from creditscore import decision, scorecard, validation
from creditscore.data import RAW_FEATURES, TARGET, split_data
from creditscore.features import HMEQPreprocessor, WoEBinner, scorecard_frame
from creditscore.synthetic import make_hmeq_like


@pytest.fixture(scope="module")
def df():
    return make_hmeq_like(2500, seed=1)


def test_split_is_stratified_and_disjoint(df):
    s = split_data(df)
    sizes = [len(s[k]) for k in ("train", "valid", "test")]
    assert sum(sizes) == len(df) and abs(sizes[0] / len(df) - 0.70) < 0.01
    rates = [s[k][TARGET].mean() for k in s]
    assert max(rates) - min(rates) < 0.01


def test_preprocessor_fits_on_train_only(df):
    s = split_data(df)
    p = HMEQPreprocessor().fit(s["train"])
    Zt = p.transform(s["test"])
    assert Zt.isna().sum().sum() == 0
    assert list(Zt.columns) == p.feature_names_out_
    # medians come from train, not test
    assert p.medians_["DEBTINC"] == pytest.approx(s["train"]["DEBTINC"].median())
    assert Zt["DEBTINC_MISSING"].sum() == s["test"]["DEBTINC"].isna().sum()


def test_woe_missing_bin_and_monotonic(df):
    s = split_data(df)
    b = WoEBinner().fit(scorecard_frame(s["train"]), s["train"][TARGET])
    t = b.woe_table("DEBTINC")
    assert "Missing" in set(t["bin"])
    br = t[t["bin"] != "Missing"]["bad_rate"].to_numpy()
    d = np.diff(br)
    assert (d >= -1e-12).all() or (d <= 1e-12).all()        # monotonic
    iv = b.iv_table()
    assert iv.iloc[0]["iv"] >= iv.iloc[-1]["iv"]


def test_points_add_up_to_scaled_score(df):
    from sklearn.linear_model import LogisticRegression
    s = split_data(df)
    b = WoEBinner().fit(scorecard_frame(s["train"]), s["train"][TARGET])
    feats = b.iv_table()["feature"].head(5).tolist()
    W = b.transform(scorecard_frame(s["train"]))[feats]
    lr = LogisticRegression(max_iter=2000).fit(W, s["train"][TARGET])
    pts = scorecard.points_table(b, lr, feats)
    total = scorecard.score_applicants(W, pts)
    assert np.allclose(total, scorecard.scale_score(lr.predict_proba(W)[:, 1]), atol=1e-6)


def test_ks_gini_psi():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    p = y * 0.3 + rng.random(4000)
    assert 0 < cm.ks(y, p) < 1
    assert cm.gini(y, p) == pytest.approx(2 * validation.discrimination(y, p, 50)["auc"] - 1)
    assert cm.psi(p, p) < 1e-9 and cm.psi(p, p + 1) > 0.25
    t = validation.paired_auc_test(y, p, rng.random(4000), n_boot=200)
    assert t["diff_lo"] > 0 and t["p_value"] < 0.05


def test_platt_and_cutoff():
    rng = np.random.default_rng(0)
    z = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(z - 1.5)))).astype(int)
    raw = 1 / (1 + np.exp(-(2 * z)))                         # mis-calibrated score
    cal = validation.PlattCalibrator().fit(raw, y)
    assert validation.brier(y, cal.predict(raw)) < validation.brier(y, raw)
    tab = decision.cutoff_table(y, cal.predict(raw), np.ones(5000))
    assert tab["bad_rate_approved"].is_monotonic_increasing
    assert tab.iloc[-1]["approval_rate"] == pytest.approx(1.0)


def test_pipeline_smoke(df, tmp_path):
    from creditscore.pipeline import Config, run_all, save
    cfg = Config(n_iter=2, cv_folds=3, ablation_folds=3, n_boot=50, challengers=["LR_raw", "XGB"])
    run = run_all(df, cfg, verbose=False)
    save(run, tmp_path)
    comp = run.tables["model_comparison"]
    assert {"LR_WoE", "LR_raw", "XGB"} <= set(comp["model"])
    assert run.objects["champion"] in {"LR_WoE", "LR_raw", "XGB"}
    assert (tmp_path / "cutoff_table.csv").exists()
    assert comp.query("split=='test'")["auc"].between(0.5, 1).all()


def test_group_split_keeps_groups_together(df):
    from creditscore.data import near_duplicate_groups
    dup = pd.concat([df, df.iloc[:300].assign(LOAN=df.iloc[:300]["LOAN"] + 100)], ignore_index=True)
    g = near_duplicate_groups(dup)
    assert (g[:300] == g[len(df):]).all()                    # each copy grouped with its original
    s = split_data(dup, groups=g)
    tr, te = set(s["train"]["group"]), set(s["test"]["group"])
    assert not tr & te and not tr & set(s["valid"]["group"])
    assert abs(s["test"][TARGET].mean() - dup[TARGET].mean()) < 0.03


def test_eda_tables(df):
    from creditscore.eda import missing_table, numeric_table, vif_table
    v = vif_table(df)
    assert (v["vif"] >= 1).all() and (v["vif_no_intercept"] >= v["vif"] - 1e-9).all()
    m = missing_table(df).set_index("feature")
    assert m.loc["DEBTINC", "phi"] > 0 and m.loc["LOAN", "n_missing"] == 0
    assert set(numeric_table(df)["feature"]) == set(m.index) - {"REASON", "JOB"}


# ---------------------------------------------------------------- model selection (DeLong + average rank)
def test_delong_matches_sklearn_and_detects_difference():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from creditscore import selection as SEL
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    good = y + rng.normal(0, 0.6, 600)
    weak = y + rng.normal(0, 2.0, 600)
    auc, lo, hi = SEL.delong_ci(y, good)
    assert abs(auc - roc_auc_score(y, good)) < 1e-12 and lo < auc < hi
    assert SEL.delong_test(y, good, good)[1] == 1.0
    diff, p = SEL.delong_test(y, weak, good)
    assert diff < 0 and p < 0.001


def test_psi_zero_for_same_distribution():
    import numpy as np
    from creditscore import selection as SEL
    x = np.random.default_rng(1).normal(size=2000)
    assert SEL.psi(x, x) < 1e-9


def test_select_champion_drops_significantly_worse_model():
    import numpy as np
    from creditscore import selection as SEL
    rng = np.random.default_rng(2)
    yt, yv = rng.integers(0, 2, 800), rng.integers(0, 2, 500)
    mk = lambda y, sd: y + rng.normal(0, sd, len(y))  # noqa: E731
    scores = {"RF": mk(yv, 0.5), "LR": mk(yv, 3.0)}
    oof = {"RF": mk(yt, 0.5), "LR": mk(yt, 3.0)}
    pred = {m: (s > 0.5).astype(int) for m, s in scores.items()}
    sel = SEL.select_champion(yv, scores, pred, oof, yt).set_index("model")
    assert sel.loc["RF", "selected"] and not sel.loc["LR", "step1_pass"]


# ---------------------------------------------------------------- decision support (web app back end)
def _small_champion(kind="LGBM"):
    import numpy as np
    from creditscore import protocol_v3 as P
    from creditscore.data import RAW_FEATURES, TARGET
    d = make_hmeq_like(1200, seed=3)
    pipe = P.build(kind, sampler="none", imputer="simple_ind", seed=0)
    pipe.set_params(clf__n_estimators=60)
    pipe.fit(d[RAW_FEATURES], d[TARGET])
    return pipe, d


def test_source_variables_mapping():
    from creditscore.decision_support import source_variables
    assert source_variables("LOAN^2") == ["LOAN"]
    assert source_variables("LOAN VALUE") == ["LOAN", "VALUE"]
    assert source_variables("DEBTINC_MISSING") == ["DEBTINC"]
    assert source_variables("JOB_Office") == ["JOB"] and source_variables("YOJ_BIN_0-5") == ["YOJ"]


def test_contributions_are_additive_on_calibrated_scale():
    import numpy as np
    from creditscore.data import RAW_FEATURES
    from creditscore.decision_support import Explainer
    for kind in ("LGBM", "XGB"):
        pipe, d = _small_champion(kind)
        ex = Explainer(pipe, 0.5, -0.3)
        X = d[RAW_FEATURES].head(50)
        phi, base = ex.explain(X)
        p = ex.pd(X)
        assert np.allclose(base + phi.sum(axis=1).values, np.log(p / (1 - p)), atol=1e-5)


def test_bands_and_recommendation_text():
    import numpy as np
    from creditscore.data import RAW_FEATURES
    from creditscore.decision_support import BANDS, Explainer, recommendation
    pipe, d = _small_champion()
    ex = Explainer(pipe, 1.0, 0.0, approve_below=0.1, decline_from=0.5)
    assert list(ex.band([0.05, 0.3, 0.7])) == list(BANDS)
    row = d[RAW_FEATURES].head(1).copy()
    row.loc[row.index[0], "DEBTINC"] = np.nan
    rep = ex.report(row)
    txt = recommendation(rep, row.iloc[0])
    assert txt[0].startswith("Đề xuất") and any("tăng rủi ro" in t for t in txt)
