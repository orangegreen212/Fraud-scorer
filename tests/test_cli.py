import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_train_then_score_end_to_end(synth, tmp_path):
    d = tmp_path / "data"; d.mkdir()
    tx = synth.drop(columns=["is_fraud", "reportedTime"])
    tx.to_csv(d / "transactions_obf.csv", index=False)
    lab = synth.loc[synth["is_fraud"] == 1, ["eventId", "reportedTime"]]
    lab.to_csv(d / "labels_obf.csv", index=False)
    art = tmp_path / "art"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--data-dir", str(d), "--cutoff", "2017-11-21",
                        "--val-months", "2017-08", "2017-09", "--monthly-k", "60", "--n-boot", "20", "--out", str(art)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-1500:]
    assert (art / "fraud_scorer.joblib").exists() and (art / "metrics.json").exists()

    cut = pd.Timestamp("2018-01-01", tz="UTC")
    t = pd.to_datetime(tx["transactionTime"], utc=True)
    tx[t < cut].to_csv(tmp_path / "hist.csv", index=False)
    tx[t >= cut].to_csv(tmp_path / "new.csv", index=False)
    q = tmp_path / "q.csv"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/score.py"), "--model", str(art / "fraud_scorer.joblib"),
                        "--history", str(tmp_path / "hist.csv"), "--new", str(tmp_path / "new.csv"),
                        "--k", "25", "--out", str(q)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-1500:]
    out = pd.read_csv(q)
    assert len(out) == 25 and out["rank_score"].is_monotonic_decreasing
