import numpy as np
import pandas as pd
import pytest

from fraud.features import FEATURE_COLS, build_features


def _brute(df, i, win):
    r = df.iloc[i]
    prev = df[(df["accountNumber"] == r["accountNumber"]) & (df["transactionTime"] < r["transactionTime"])
              & (df["transactionTime"] >= r["transactionTime"] - pd.Timedelta(win))]
    return len(prev), prev["transactionAmount"].sum()


def test_all_feature_columns_exist_and_order_kept(synth):
    out = build_features(synth)
    assert set(FEATURE_COLS) <= set(out.columns)
    assert (out["eventId"].to_numpy() == synth["eventId"].to_numpy()).all()


def test_rolling_features_match_brute_force(synth):
    out = build_features(synth)
    for i in np.random.default_rng(0).choice(len(out), 60, replace=False):
        c24, s24 = _brute(out, i, "24h")
        c1, _ = _brute(out, i, "1h")
        assert out.loc[i, "txn_count_24h"] == c24
        assert out.loc[i, "txn_count_1h"] == c1
        assert out.loc[i, "spend_24h"] == pytest.approx(s24)


def test_causality_future_rows_do_not_change_past_features(synth):
    cut = synth["transactionTime"].quantile(0.6)
    full = build_features(synth)
    part = build_features(synth[synth["transactionTime"] < cut].reset_index(drop=True))
    a = full[full["transactionTime"] < cut].sort_values("eventId").reset_index(drop=True)
    b = part.sort_values("eventId").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[FEATURE_COLS], b[FEATURE_COLS], check_dtype=False)


def test_first_transaction_of_account_has_no_history(synth):
    out = build_features(synth)
    first = out.sort_values("transactionTime").groupby("accountNumber").head(1)
    assert (first["n_prev"] == 0).all()
    assert first["amt_vs_card_mean"].isna().all()
    assert first["secs_since_last"].isna().all()
    assert (first["txn_count_24h"] == 0).all() and (first["is_new_merchant"] == 1).all()


def test_missing_columns_raise(synth):
    with pytest.raises(ValueError):
        build_features(synth.drop(columns=["merchantId"]))
