"""Build the review queue from raw transactions (shared by scripts/score.py and the Streamlit app)."""
import pandas as pd

from .features import RAW_REQUIRED, build_features

OUT_COLS = ["rank", "eventId", "accountNumber", "transactionTime", "transactionAmount",
            "fraud_probability", "rank_score", "reasons"]


def build_queue(scorer, new: pd.DataFrame, history: pd.DataFrame = None, k: int = 400) -> pd.DataFrame:
    """Top-k transactions of `new` by rank_score. `history` = earlier rows of the same cards (optional)."""
    for name, d in (("new", new), ("history", history)):
        if d is not None:
            miss = [c for c in RAW_REQUIRED if c not in d.columns]
            if miss:
                raise ValueError(f"{name} file is missing columns: {miss}")
    parts = [history] if history is not None else []
    allrows = pd.concat(parts + [new], ignore_index=True).drop_duplicates("eventId", keep="last")
    feats = build_features(allrows)
    feats = feats[feats["eventId"].isin(set(new["eventId"]))].reset_index(drop=True)
    q = feats.join(scorer.score(feats)).assign(reasons=scorer.explain(feats))
    q = q.sort_values("rank_score", ascending=False, kind="stable").head(int(k)).reset_index(drop=True)
    q.insert(0, "rank", range(1, len(q) + 1))
    return q[OUT_COLS]
