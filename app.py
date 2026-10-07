"""Streamlit app: upload transactions -> monthly review queue.

    streamlit run app.py
"""
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud.features import RAW_REQUIRED
from fraud.model import FraudScorer
from fraud.queue import build_queue

# Only a model bundled with the app is loaded. Never unpickle an uploaded file (pickle can run code).
MODEL_PATH = Path(os.environ.get("FRAUD_MODEL", "models/fraud_scorer.joblib"))

st.set_page_config(page_title="Fraud review queue", page_icon="🔎", layout="wide")


@st.cache_resource
def load_scorer(path: str) -> FraudScorer:
    return FraudScorer.load(path)


@st.cache_data(show_spinner=False)
def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), parse_dates=["transactionTime"])


st.title("🔎 Fraud review queue")
st.caption("Ranks transactions by expected fraud value (P(fraud) × amount^b) so a fixed review budget catches the most money.")

if not MODEL_PATH.exists():
    st.error(f"Model not found at `{MODEL_PATH}`. Run `python scripts/train.py` and copy "
             "`artifacts/fraud_scorer.joblib` to `models/`.")
    st.stop()
scorer = load_scorer(str(MODEL_PATH))

with st.sidebar:
    st.header("Settings")
    k = st.number_input("Review budget (K)", 1, 10_000, 400, step=50,
                        help="For a window shorter than a month, scale it: 400 × days observed / days in month.")
    st.markdown("**Model**")
    st.write(f"b = {scorer.b}, calibrated = {scorer.calibrate}")
    st.write(f"trained as of {scorer.meta.get('as_of', '?')[:10]}, {scorer.meta.get('n_known_fraud', '?')} known frauds")

with st.expander("Required columns"):
    st.code(", ".join(RAW_REQUIRED))

new_file = st.file_uploader("Transactions to score (CSV)", type="csv")
hist_file = st.file_uploader("Earlier transactions of the same cards (CSV, optional but improves the history features)",
                             type="csv")

if new_file is None:
    st.info("Upload a CSV to build the queue.")
    st.stop()

try:
    new = read_csv(new_file.getvalue())
    hist = read_csv(hist_file.getvalue()) if hist_file else None
    with st.spinner("Scoring..."):
        queue = build_queue(scorer, new, hist, int(k))
except Exception as e:  # show a readable message instead of a stack trace
    st.error(f"Could not score the file: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Transactions scored", f"{new['eventId'].nunique():,}")
c2.metric("Queue size", f"{len(queue):,}")
c3.metric("Value in queue", f"£{queue['transactionAmount'].sum():,.0f}")

st.dataframe(queue, use_container_width=True, hide_index=True,
             column_config={"fraud_probability": st.column_config.NumberColumn(format="%.4f"),
                            "rank_score": st.column_config.NumberColumn(format="%.2f")})
st.download_button("Download review_queue.csv", queue.to_csv(index=False).encode(), "review_queue.csv", "text/csv")
st.caption("Reasons list the features pushing each transaction's score up most. They support the analyst; they are not proof of fraud.")
