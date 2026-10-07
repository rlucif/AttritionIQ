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
