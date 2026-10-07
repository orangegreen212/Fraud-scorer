"""Single source of truth for features. EDA and modelling both call build_features()."""
import numpy as np
import pandas as pd

RAW_REQUIRED = ["eventId", "accountNumber", "transactionTime", "transactionAmount",
                "availableCash", "mcc", "merchantCountry", "posEntryMode", "merchantId"]
BASE_FEATURES = ["transactionAmount", "availableCash", "amount_to_cash_ratio", "hour", "dayofweek",
                 "is_weekend", "is_night", "is_foreign", "txn_count_24h", "is_ecommerce",
                 "is_stripe_fallback", "mcc", "merchantCountry", "posEntryMode"]
HISTORY_FEATURES = ["amt_vs_card_mean", "secs_since_last", "is_new_merchant", "is_new_mcc",
                    "country_share", "txn_count_1h", "spend_24h", "n_prev"]
FEATURE_COLS = BASE_FEATURES + HISTORY_FEATURES


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Return `tx` (same row order) plus every model feature.

    Causality contract: a row's features use only that account's transactions
    strictly before it (rolling windows use closed="left"). Features of
    row t therefore never change when later rows are added or edited.
    Same-timestamp rows of one account are ordered by eventId.
    """
    missing = [c for c in RAW_REQUIRED if c not in tx.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = tx.copy()
    df["transactionTime"] = pd.to_datetime(df["transactionTime"], utc=True)
    df["_pos"] = np.arange(len(df))
    df = df.sort_values(["accountNumber", "transactionTime", "eventId"], kind="stable").reset_index(drop=True)
    g = df.groupby("accountNumber", sort=False)
    amt = df["transactionAmount"]

    df["hour"] = df["transactionTime"].dt.hour
    df["dayofweek"] = df["transactionTime"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_ecommerce"] = (df["posEntryMode"] == 81).astype(int)
    df["is_stripe_fallback"] = (df["posEntryMode"] == 80).astype(int)
    df["amount_to_cash_ratio"] = amt / (df["availableCash"] + 1)

    df["n_prev"] = g.cumcount()
    n_prev0 = df["n_prev"].replace(0, np.nan)
    same_country = df.groupby(["accountNumber", "merchantCountry"], sort=False).cumcount()
    df["country_share"] = same_country / n_prev0
    df["is_foreign"] = (df["country_share"] < 0.1).astype(int)
    prev_mean = (g["transactionAmount"].cumsum() - amt) / n_prev0
    df["amt_vs_card_mean"] = amt / prev_mean.where(prev_mean > 0)
    df["secs_since_last"] = g["transactionTime"].diff().dt.total_seconds()
    df["is_new_merchant"] = (df.groupby(["accountNumber", "merchantId"], sort=False).cumcount() == 0).astype(int)
    df["is_new_mcc"] = (df.groupby(["accountNumber", "mcc"], sort=False).cumcount() == 0).astype(int)

    roll = df.set_index("transactionTime").groupby("accountNumber", sort=False)["transactionAmount"]
    for name, win, how in [("txn_count_24h", "24h", "count"), ("txn_count_1h", "1h", "count"),
                           ("spend_24h", "24h", "sum")]:
        df[name] = getattr(roll.rolling(win, closed="left"), how)().to_numpy()
        df[name] = df[name].fillna(0)

    return df.sort_values("_pos").drop(columns="_pos").reset_index(drop=True)
