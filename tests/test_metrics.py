import numpy as np
import pandas as pd
import pytest

from fraud.metrics import (capture_pct, cluster_bootstrap, fraud_value_captured, known_label,
                           month_budgets, monthly_capture, score_fn)


def _df():
    return pd.DataFrame({"transactionAmount": [100.0, 50.0, 10.0, 1000.0],
                         "is_fraud": [1, 0, 1, 1]})


def test_capture_pct_known_values():
    df = _df()  # fraud value = 100 + 10 + 1000 = 1110
    assert capture_pct(df, df["transactionAmount"].to_numpy(), k=1) == pytest.approx(1000 / 1110 * 100)
    assert capture_pct(df, df["transactionAmount"].to_numpy(), k=2) == pytest.approx(1100 / 1110 * 100)
    assert capture_pct(df, np.arange(4)[::-1], k=4) == pytest.approx(100.0)


def test_capture_pct_no_fraud_is_nan_and_money_matches():
    df = _df().assign(is_fraud=0)
    assert np.isnan(capture_pct(df, "transactionAmount"))
    assert fraud_value_captured(_df(), "transactionAmount", k=1) == 1000


def test_capture_pct_ties_are_stable():
    df = _df()
    s = np.ones(4)
    assert capture_pct(df, s, k=1) == capture_pct(df, s.copy(), k=1)  # first row wins, deterministic


def test_score_fn():
    assert score_fn(0.5, 100, b=1) == pytest.approx(50)
    assert score_fn(0.5, 100, b=0) == pytest.approx(0.5)
    assert score_fn(0.5, 0, b=1.5) > 0  # amount floored, never collapses to 0
    assert score_fn(np.array([0.1, 0.2]), np.array([10, 10]), 2).shape == (2,)


def test_known_label_uses_only_reported_before_asof():
    t = pd.Timestamp
    df = pd.DataFrame({"is_fraud": [1, 1, 0, 1],
                       "reportedTime": [t("2017-01-05", tz="UTC"), t("2017-02-05", tz="UTC"),
                                        pd.NaT, t("2017-01-10", tz="UTC")]})
    out = known_label(df, t("2017-01-10", tz="UTC"))
    assert out.tolist() == [1, 0, 0, 0]  # strict '<': reported exactly at as_of is not yet known


def test_month_budgets_scale_partial_months():
    b = month_budgets("2017-11-21", "2018-01-31", monthly_k=400)
    assert [str(m) for m in b["month"]] == ["2017-11", "2017-12", "2018-01"]
    assert b.loc[0, "k"] == round(400 * 10 / 30)      # 21..30 Nov = 10 days
    assert b.loc[1, "k"] == 400                       # full December
    assert b.loc[2, "k"] == round(400 * 30 / 31)      # 1..30 Jan
    assert list(month_budgets("2017-11-21", "2018-01-31", min_coverage=0.9)["month"].astype(str)) == ["2017-12", "2018-01"]


def test_month_budgets_accepts_tz_aware():
    b = month_budgets(pd.Timestamp("2017-12-01", tz="UTC"), pd.Timestamp("2018-01-01", tz="UTC"))
    assert b["k"].tolist() == [400]


def test_cluster_bootstrap_shapes_and_bounds(synth):
    df = synth.copy()
    df["month"] = df["transactionTime"].dt.tz_localize(None).dt.to_period("M")
    df = df[df["month"] == pd.Period("2017-06")].reset_index(drop=True)
    b = month_budgets("2017-06-01", "2017-07-01", monthly_k=50)
    d = cluster_bootstrap(df, {"amt": df["transactionAmount"]}, b, n_boot=30, seed=1)
    assert d["amt"].shape == (30,)
    v = d["amt"][~np.isnan(d["amt"])]
    assert ((v >= 0) & (v <= 100 + 1e-9)).all()
    d2 = cluster_bootstrap(df, {"amt": df["transactionAmount"]}, b, n_boot=30, seed=1)
    assert np.allclose(d["amt"], d2["amt"], equal_nan=True)  # reproducible


def test_monthly_capture_uses_budget(synth):
    df = synth.copy()
    df["month"] = df["transactionTime"].dt.tz_localize(None).dt.to_period("M")
    b = month_budgets("2017-03-01", "2017-05-01", monthly_k=10)
    out = monthly_capture(df, df["transactionAmount"].to_numpy(), b)
    assert list(out.index) == ["2017-03", "2017-04"]
