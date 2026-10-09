"""Smoke tests: run with `pytest -q`. They check the pieces work together,
not that the model is good (that is what reports/metrics.json is for)."""
import numpy as np
import pandas as pd
import pytest

from src import config, data, rag, recommend, roi, segment
from src.forecast import PayGapAdder

pytestmark = pytest.mark.skipif(not config.RAW_DATA.exists(), reason="dataset not downloaded")


@pytest.fixture(scope="module")
def df():
    return data.prepare(data.load_raw())


def test_data_shape_and_cleaning(df):
    assert len(df) == 1470
    assert set(df[config.TARGET].unique()) == {0, 1}
    assert not set(config.DROP_COLS) & set(df.columns)
    assert df.isna().sum().sum() == 0


def test_sensitive_features_excluded():
    lists = data.feature_lists(exclude_sensitive=True)
    used = set(lists["numeric"] + lists["categorical"])
    assert not used & set(config.SENSITIVE_FEATURES)


def test_split_is_stratified(df):
    train, test = data.split(df)
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.02


def test_pay_gap_adder(df):
    out = PayGapAdder().fit(df).transform(df)
    assert "PayGapPct" in out and abs(out.PayGapPct.median()) < 0.2


def test_cost_model_logic():
    y = np.array([1, 0])
    repl, interv = np.array([100.0, 100.0]), np.array([10.0, 10.0])
    # nobody flagged: pay replacement for the leaver only
    assert roi.realised_cost(y, [False, False], repl, interv, 0.5) == 100
    # everyone flagged: 2 interventions + half the leaver's replacement
    assert roi.realised_cost(y, [True, True], repl, interv, 0.5) == 70


def test_rfm_scores_range(df):
    s = segment.rfm_style_scores(df)
    assert s.RFM_total.between(3, 15).all()


def test_recommender_outputs(df):
    R = recommend.simulate_response_matrix(df.head(200))
    pred = recommend.FunkSVD(epochs=5).fit(R).predict_all()
    assert pred.shape == R.shape and pred.notna().all().all()


def test_rag_retrieves_right_section():
    r = rag.Retriever(rag.load_chunks(), prefer_transformer=False)
    hits = r.retrieve("Can she work from home because of a long commute?", k=3)
    assert "Flexible and Remote Work" in hits.section.tolist()


# --------------------------------------------------------------------------
# Phase 2: risk-model decision logic
# --------------------------------------------------------------------------
def _cv(mean, std, folds):
    return {"repeated_cv": {"pr_auc": {"mean": mean, "std": std, "folds": folds}}}


def test_select_model_prefers_simplest_when_tied():
    from src.model import select_model
    comp = {
        "XGBoost": _cv(0.64, 0.05, [0.64] * 15),
        "Logistic Regression": _cv(0.62, 0.05, [0.62] * 15),   # within 1 std of best
        "Random Forest": _cv(0.50, 0.05, [0.50] * 15),          # clearly worse
    }
    chosen, info = select_model(comp)
    assert info["highest_mean"] == "XGBoost"
    assert chosen == "Logistic Regression"
    assert "Random Forest" not in info["tied_with_best"]


def test_select_model_keeps_clear_winner():
    from src.model import select_model
    comp = {"XGBoost": _cv(0.70, 0.02, [0.70] * 15),
            "Logistic Regression": _cv(0.60, 0.02, [0.60] * 15)}
    assert select_model(comp)[0] == "XGBoost"


def test_break_even_threshold_matches_cost_model():
    # 1 month intervention, replacement = 12 months, 40% success -> 1/4.8
    assert roi.break_even_threshold(1.0, 1.0, 0.4) == pytest.approx(1 / 4.8)
    # at exactly p*, flagging and not flagging cost the same for one person
    p, R, C, s = roi.break_even_threshold(1.0, 1.0, 0.4), 1200.0, 100.0, 0.4
    assert p * R == pytest.approx(C + p * R * (1 - s))


def test_wilson_interval_and_ece():
    from src.model import expected_calibration_error, wilson_interval
    lo, hi = wilson_interval(0, 6)          # 0 of 6 caught: CI must not collapse to [0, 0]
    assert lo == 0 and hi > 0.3
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)


def test_fairness_age_bands_match_labels():
    from src.model import fairness_report
    d = pd.DataFrame({config.TARGET: [1, 1, 1, 1], "Gender": [1, 0, 1, 0], "Age": [29, 30, 49, 50]})
    r = fairness_report(d, np.array([0.9, 0.9, 0.9, 0.9]), 0.5)
    bands = r[r.attribute == "AgeBand"].set_index("group").n
    assert bands["<30"] == 1 and bands["30-39"] == 1 and bands["40-49"] == 1 and bands["50+"] == 1


# --------------------------------------------------------------------- Phase 3
def test_knee_index_finds_the_bend():
    y = np.r_[np.linspace(0, 1, 90), np.linspace(1.5, 10, 10)]  # flat, then a sharp rise
    assert 85 <= segment.knee_index(y) <= 92


def test_persona_names_follow_the_profile_not_the_number(df):
    X, _ = segment.scale(df)
    prof = segment.profile_clusters(df, segment.fit_kmeans(X, 3).labels_)
    names = segment.persona_names(prof)
    assert sorted(names.values()) == sorted(segment.PERSONA_RULES.values())
    assert names[int(prof.JobLevel.idxmax())] == segment.PERSONA_RULES["senior"]
    shuffled = prof.rename(index={0: 2, 1: 0, 2: 1})  # renumber the clusters
    assert segment.persona_names(shuffled)[int(shuffled.JobLevel.idxmax())] == segment.PERSONA_RULES["senior"]


def test_cf_does_not_start_from_the_simulated_noise(df):
    """Regression test for the Phase 3 seed-collision leak."""
    R = recommend.simulate_response_matrix(df, observed_frac=1.0)
    m = recommend.FunkSVD(n_factors=R.shape[1], epochs=0).fit(R)
    noise = np.random.default_rng(config.RANDOM_STATE).normal(0, 0.6, R.shape)
    assert abs(np.corrcoef(m.P_.ravel(), noise.ravel())[0, 1]) < 0.05


def test_tfidf_cosine_equals_euclidean_and_empty_query_flagged():
    r = rag.Retriever(rag.load_chunks(), prefer_transformer=False)
    q = "My team member has been doing long hours for weeks"
    assert r.retrieve(q, 5, "cosine").chunk_id.tolist() == r.retrieve(q, 5, "euclidean").chunk_id.tolist()
    assert not r.retrieve("zzzz qqqq", 3).matched.iloc[0]


def test_driver_query_uses_handbook_words():
    q = rag.driver_query("What can I offer?", ["OverTime", "MonthlyIncome"])
    assert "overtime" in q and "salary" in q


def test_handbook_has_no_placeholders():
    text = config.HANDBOOK.read_text(encoding="utf-8")
    assert "[N]" not in text and "[amount]" not in text


def test_naive_forecast_and_d_selection():
    from src import forecast
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 0.1, 120)) + 2,
                  index=pd.date_range("2010-01-01", periods=120, freq="MS"))
    fc = forecast.forecast_naive(s, 12)
    assert (fc.forecast == s.iloc[-1]).all() and (fc.upper.diff().dropna() > 0).all()
    assert forecast.choose_d(s)[0] == 1  # a random walk needs one difference
