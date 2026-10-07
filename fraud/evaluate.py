"""Rolling validation: choose calibration + ranking exponent b on training-only folds."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .metrics import capture_pct, known_label, score_fn
from .model import fit_scorer, prior_correct

B_GRID = (0.5, 1.0, 1.5, 2.0, 2.5)


def rolling_validation(feats, fold_months, monthly_k=400, b_grid=B_GRID, feature_cols=None):
    """Train on months before each fold month, score that month. Returns long table.

    Each fold's budget is scaled by observed coverage of that month in `feats`
    (so a partial fold is not given a full 400). Probabilities are evaluated
    both raw (balanced-weight output) and prior-corrected.
    """
    feats = feats.copy()
    feats["month"] = feats["transactionTime"].dt.tz_localize(None).dt.to_period("M")
    rows = []
    for m in fold_months:
        m = pd.Period(m)
        tr, va = feats[feats["month"] < m], feats[feats["month"] == m]
        if va.empty or va["is_fraud"].sum() == 0:
            continue
        as_of = m.start_time.tz_localize("UTC")
        sc = fit_scorer(tr, as_of, feature_cols=feature_cols)
        raw = sc.pipeline.predict_proba(va[sc.feature_cols])[:, 1]
        cor = prior_correct(raw, sc.prior)
        y_va = va["is_fraud"].to_numpy()
        days = (va["transactionTime"].max() - va["transactionTime"].min()).total_seconds() / 86400
        k = int(round(monthly_k * min(1.0, max(days, 1) / m.days_in_month)))
        base = capture_pct(va, va["transactionAmount"].to_numpy(), k=k)
        for calib, p in (("raw", raw), ("prior_corrected", cor)):
            for b in b_grid:
                rows.append(dict(fold=str(m), calibration=calib, b=b, k=k, baseline=base,
                                 capture=capture_pct(va, score_fn(p, va["transactionAmount"], b), k=k),
                                 brier=brier_score_loss(y_va, p)))
    return pd.DataFrame(rows)


def pick_config(val: pd.DataFrame):
    """Config with best mean uplift over the amount-only baseline; ties -> more folds won, then simpler (lower b)."""
    val = val.assign(uplift=val["capture"] - val["baseline"])
    g = (val.groupby(["calibration", "b"])
            .agg(uplift=("uplift", "mean"), capture=("capture", "mean"), baseline=("baseline", "mean"),
                 brier=("brier", "mean"), folds_won=("uplift", lambda s: int((s > 0).sum())),
                 folds=("uplift", "size"))
            .reset_index().sort_values(["uplift", "folds_won", "b"], ascending=[False, False, True]))
    best = g.iloc[0]
    return dict(calibrate=best["calibration"] == "prior_corrected", b=float(best["b"])), g
