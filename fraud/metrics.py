"""Business metrics. Everything here is pure numpy/pandas (no model deps)."""
import calendar

import numpy as np
import pandas as pd


def _as_scores(df, score):
    return df[score].to_numpy() if isinstance(score, str) else np.asarray(score)


def capture_pct(df, score, k=400):
    """% of the group's fraud value found in the k highest-scored rows."""
    score = _as_scores(df, score)
    total = df.loc[df["is_fraud"] == 1, "transactionAmount"].sum()
    if total <= 0:
        return np.nan
    top = df.iloc[np.argsort(-score, kind="stable")[: int(k)]]
    return top.loc[top["is_fraud"] == 1, "transactionAmount"].sum() / total * 100


def fraud_value_captured(df, score, k=400):
    """Same ranking as capture_pct, but returns the money caught."""
    score = _as_scores(df, score)
    top = df.iloc[np.argsort(-score, kind="stable")[: int(k)]]
    return top.loc[top["is_fraud"] == 1, "transactionAmount"].sum()


def score_fn(probability, amount, b=1.0):
    """Ranking score = P(fraud) * amount**b (amount floored at 0.01)."""
    amount = np.clip(np.asarray(amount, dtype=float), 0.01, None)
    return np.asarray(probability) * amount ** b


def known_label(df, as_of):
    """Fraud label as known at `as_of`: reported strictly before it."""
    return ((df["is_fraud"] == 1) & (df["reportedTime"] < as_of)).astype(int).to_numpy()


def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo else ts


def month_budgets(window_start, window_end, monthly_k=400, min_coverage=0.0):
    """Review budget per calendar month, scaled by the share of the month observed.

    A month observed for 10 of 30 days gets round(400 * 10/30) = 133 reviews,
    not 400. `window_end` is exclusive. Months with coverage < `min_coverage`
    are dropped (e.g. 0.9 = evaluate (almost) full months only).

    Returns DataFrame[month, days_observed, days_in_month, coverage, k].
    """
    start, end = _naive(window_start), _naive(window_end)
    rows = []
    for period in pd.period_range(start, end - pd.Timedelta(seconds=1), freq="M"):
        m_start = period.start_time
        m_end = (period + 1).start_time
        lo, hi = max(m_start, start), min(m_end, end)
        days = max((hi - lo).total_seconds() / 86400, 0.0)
        dim = calendar.monthrange(period.year, period.month)[1]
        cov = min(days / dim, 1.0)
        if cov > 0 and cov >= min_coverage:
            rows.append(dict(month=period, days_observed=round(days, 2), days_in_month=dim,
                             coverage=round(cov, 3), k=int(round(monthly_k * cov))))
    return pd.DataFrame(rows, columns=["month", "days_observed", "days_in_month", "coverage", "k"])


def monthly_capture(df, scores, budgets):
    """Per-month capture % using the scaled budgets. `df` needs a `month` Period column."""
    scores = np.asarray(scores)
    out = {}
    for _, r in budgets.iterrows():
        idx = np.flatnonzero((df["month"] == r["month"]).to_numpy())
        if len(idx):
            out[str(r["month"])] = capture_pct(df.iloc[idx], scores[idx], k=r["k"])
    return pd.Series(out, dtype=float)


def cluster_bootstrap(df, score_cols, budgets, n_boot=1000, seed=0, group_col="accountNumber"):
    """Card/account-level (cluster) bootstrap of the month-averaged capture %.

    Whole accounts are resampled with replacement, because fraud clusters on a
    few cards: a row-level bootstrap treats correlated rows as independent and
    gives intervals that are too narrow. `score_cols` maps name -> column/array.
    Returns {name: array(n_boot)} of month-averaged capture %.
    """
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(df[group_col])
    n_acc = len(uniq)
    month_idx = [(np.flatnonzero((df["month"] == r["month"]).to_numpy()), int(r["k"]))
                 for _, r in budgets.iterrows()]
    amt = df["transactionAmount"].to_numpy(float)
    fr = df["is_fraud"].to_numpy(int)
    scores = {n: _as_scores(df, s).astype(float) for n, s in score_cols.items()}
    draws = {n: np.empty(n_boot) for n in scores}
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n_acc, n_acc), minlength=n_acc)[codes]  # copies of each row
        for n, s in scores.items():
            vals = []
            for idx, k in month_idx:
                rep = np.repeat(idx, w[idx])
                if rep.size == 0:
                    continue
                tot = (amt[rep] * fr[rep]).sum()
                if tot <= 0:
                    continue
                top = rep[np.argsort(-s[rep], kind="stable")[:k]]
                vals.append((amt[top] * fr[top]).sum() / tot * 100)
            draws[n][i] = np.mean(vals) if vals else np.nan
    return draws


def summarize_draws(draws, alpha=0.05):
    return {n: dict(mean=float(np.nanmean(v)), lo=float(np.nanpercentile(v, 100 * alpha / 2)),
                    hi=float(np.nanpercentile(v, 100 * (1 - alpha / 2)))) for n, v in draws.items()}
