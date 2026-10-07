import numpy as np
import pandas as pd
import pytest

from fraud.evaluate import pick_config, rolling_validation
from fraud.features import build_features
from fraud.model import FraudScorer, fit_scorer, prior_correct


@pytest.fixture(scope="module")
def feats(synth):
    return build_features(synth)


def test_prior_correct_shrinks_balanced_probabilities():
    p = np.array([0.5, 0.9])
    c = prior_correct(p, 0.01)
    assert (c < p).all() and c[0] == pytest.approx(0.01)  # 0.5 under balanced weights == the base rate
    assert (np.diff(c) > 0).all()


def test_fit_save_load_roundtrip_gives_identical_scores(feats, tmp_path):
    cutoff = pd.Timestamp("2017-10-01", tz="UTC")
    train = feats[feats["transactionTime"] < cutoff]
    sc = fit_scorer(train, cutoff)
    path = tmp_path / "m.joblib"
    sc.save(path)
    new = feats[feats["transactionTime"] >= cutoff]
    s1, s2 = sc.score(new), FraudScorer.load(path).score(new)
    pd.testing.assert_frame_equal(s1, s2)
    assert s1["fraud_probability"].between(0, 1).all() and len(sc.explain(new.head(5))) == 5


def test_unseen_categories_do_not_crash(feats):
    cutoff = pd.Timestamp("2017-10-01", tz="UTC")
    sc = fit_scorer(feats[feats["transactionTime"] < cutoff], cutoff)
    odd = feats.head(3).copy()
    odd["mcc"], odd["merchantCountry"] = 9999, "ZZ"
    assert np.isfinite(sc.score(odd)["rank_score"]).all()


def test_labels_unreported_at_cutoff_are_not_used(feats):
    cutoff = pd.Timestamp("2017-10-01", tz="UTC")
    train = feats[feats["transactionTime"] < cutoff]
    sc = fit_scorer(train, cutoff)
    late = ((train["is_fraud"] == 1) & (train["reportedTime"] >= cutoff)).sum()
    assert sc.meta["n_known_fraud"] == int(train["is_fraud"].sum()) - int(late)


def test_rolling_validation_and_pick_config(feats):
    val = rolling_validation(feats, ["2017-08", "2017-09"], monthly_k=40, b_grid=(1.0, 1.5))
    assert {"raw", "prior_corrected"} == set(val["calibration"])
    cfg, table = pick_config(val)
    assert cfg["b"] in (1.0, 1.5) and isinstance(cfg["calibrate"], (bool, np.bool_))
    assert len(table) == 4
