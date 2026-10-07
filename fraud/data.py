"""Data loading. The data directory comes from --data-dir or $FRAUD_DATA_DIR, never a hardcoded path."""
import os
from pathlib import Path

import pandas as pd


def resolve_data_dir(data_dir=None) -> Path:
    d = data_dir or os.environ.get("FRAUD_DATA_DIR") or "data"
    d = Path(d).expanduser()
    if not (d / "transactions_obf.csv").exists():
        raise FileNotFoundError(f"transactions_obf.csv not found in {d.resolve()}. "
                                "Pass --data-dir or set FRAUD_DATA_DIR.")
    return d


def load_labelled(data_dir=None) -> pd.DataFrame:
    d = resolve_data_dir(data_dir)
    tx = pd.read_csv(d / "transactions_obf.csv", parse_dates=["transactionTime"])
    labels = pd.read_csv(d / "labels_obf.csv", parse_dates=["reportedTime"])
    tx["transactionTime"] = pd.to_datetime(tx["transactionTime"], utc=True)
    tx["is_fraud"] = tx["eventId"].isin(set(labels["eventId"])).astype(int)
    tx["reportedTime"] = pd.to_datetime(tx["eventId"].map(labels.set_index("eventId")["reportedTime"]), utc=True)
    return tx
