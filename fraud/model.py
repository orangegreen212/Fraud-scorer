"""Deployable logistic-regression scorer: one sklearn Pipeline + ranking rule, persisted with joblib."""
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import FEATURE_COLS
from .metrics import known_label, score_fn

CAT_COLS = ["mcc", "merchantCountry", "posEntryMode"]
LOG_COLS = ["transactionAmount", "availableCash", "txn_count_24h", "txn_count_1h",
            "spend_24h", "n_prev", "secs_since_last", "amt_vs_card_mean"]


def _log1p_clip(X):
    return np.log1p(np.clip(X, 0, None))


def _to_str(X):
    return pd.DataFrame(X).astype(str)


def build_pipeline(feature_cols=None, C=0.3, min_category_count=200):
    """log1p(skewed) -> median impute -> scale; one-hot for common categories; logistic regression.

    Rare categories (< min_category_count in the training data) share one
    'infrequent' column instead of creating near-empty columns.
    """
    cols = list(feature_cols or FEATURE_COLS)
    log_cols = [c for c in LOG_COLS if c in cols]
    num_cols = [c for c in cols if c not in CAT_COLS and c not in LOG_COLS]
    cat_cols = [c for c in CAT_COLS if c in cols]
    pre = ColumnTransformer([
        ("log", Pipeline([("log", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
                          ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), log_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
                          ("oh", OneHotEncoder(min_frequency=min_category_count, sparse_output=False,
                                               handle_unknown="infrequent_if_exist"))]), cat_cols),
    ])
    clf = LogisticRegression(C=C, max_iter=500, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def prior_correct(p, prior):
    """Undo the odds inflation from class_weight='balanced'.

    Balanced weighting multiplies the model's odds by (1-prior)/prior, so the
    raw output is NOT a probability. True odds = model odds * prior/(1-prior).
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    odds = p / (1 - p) * prior / (1 - prior)
    return odds / (1 + odds)


@dataclass
class FraudScorer:
    pipeline: Pipeline
    feature_cols: list
    b: float = 1.5                 # amount exponent in the ranking score
    calibrate: bool = True         # apply prior_correct before ranking
    prior: float = 0.0             # fraud rate (known labels) in the training data
    meta: dict = field(default_factory=dict)

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        p = self.pipeline.predict_proba(feats[self.feature_cols])[:, 1]
        return prior_correct(p, self.prior) if self.calibrate else p

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_proba(feats)
        return pd.DataFrame({"fraud_probability": p,
                             "rank_score": score_fn(p, feats["transactionAmount"], self.b)}, index=feats.index)

    def explain(self, feats: pd.DataFrame, top_n=3) -> list:
        """Reason codes: features pushing each row's score up most (coef x scaled value)."""
        Z = self.pipeline.named_steps["pre"].transform(feats[self.feature_cols])
        names = self.pipeline.named_steps["pre"].get_feature_names_out()
        contrib = Z * self.pipeline.named_steps["clf"].coef_[0]
        top = np.argsort(-contrib, axis=1)[:, :top_n]
        return [", ".join(names[j].split("__", 1)[-1] for j in row) for row in top]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def fit_scorer(train_feats: pd.DataFrame, as_of, b=1.5, calibrate=True, feature_cols=None, C=0.3):
    """Fit on rows before `as_of`, using only labels reported before `as_of`."""
    cols = list(feature_cols or FEATURE_COLS)
    y = known_label(train_feats, as_of)
    pipe = build_pipeline(cols, C=C).fit(train_feats[cols], y)
    return FraudScorer(pipe, cols, b=b, calibrate=calibrate, prior=float(y.mean()),
                       meta=dict(as_of=str(as_of), n_train=len(y), n_known_fraud=int(y.sum())))
