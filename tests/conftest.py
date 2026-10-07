import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_synthetic(n_accounts=120, per_account=60, start="2017-01-01", days=400, seed=0):
    """Small synthetic card dataset with the real schema; fraud = big, night, new-country bursts."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz="UTC")
    rows = []
    for a in range(n_accounts):
        home = rng.choice(["GB", "FR", "DE"])
        compromised = rng.random() < 0.15
        times = np.sort(rng.uniform(0, days * 86400, per_account))
        for s in times:
            ts = t0 + pd.Timedelta(seconds=float(s))
            fraud = compromised and ts.hour < 6 and rng.random() < 0.5
            rows.append(dict(
                accountNumber=f"A{a:04d}", transactionTime=ts,
                transactionAmount=float(rng.lognormal(6.0 if fraud else 3.0, 1.0)),
                availableCash=float(rng.lognormal(7, 1)),
                mcc=int(rng.choice([5411, 5812, 5999, 4829])), merchantId=f"M{rng.integers(0, 40)}",
                merchantCountry=("US" if fraud else home), posEntryMode=int(rng.choice([5, 81, 80, 7])),
                is_fraud=int(fraud),
                reportedTime=(ts + pd.Timedelta(days=float(rng.uniform(0.5, 6)))) if fraud else pd.NaT))
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "eventId", [f"E{i:06d}" for i in range(len(df))])
    df["reportedTime"] = pd.to_datetime(df["reportedTime"], utc=True)
    return df


@pytest.fixture(scope="session")
def synth():
    return make_synthetic()
