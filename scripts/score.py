"""Produce the monthly review queue.

    python scripts/score.py --model artifacts/fraud_scorer.joblib --history old.csv --new new.csv --k 400

--history = earlier transactions of the same cards (features need them); --new = rows to score.
For a window shorter than a month pass a scaled --k (400 * days_observed / days_in_month).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from fraud.model import FraudScorer
from fraud.queue import build_queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/fraud_scorer.joblib")
    ap.add_argument("--new", required=True)
    ap.add_argument("--history")
    ap.add_argument("--k", type=int, default=400)
    ap.add_argument("--out", default="review_queue.csv")
    a = ap.parse_args()

    new = pd.read_csv(a.new, parse_dates=["transactionTime"])
    hist = pd.read_csv(a.history, parse_dates=["transactionTime"]) if a.history else None
    q = build_queue(FraudScorer.load(a.model), new, hist, a.k)
    q.to_csv(a.out, index=False)
    print(f"wrote {len(q)} rows to {a.out}")


if __name__ == "__main__":
    main()
