"""Train, validate, evaluate once on the test window, and persist the scorer.

    python scripts/train.py --data-dir "./Fraud is in the Air" --cutoff 2017-11-21
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from fraud.data import load_labelled
from fraud.evaluate import pick_config, rolling_validation
from fraud.features import build_features
from fraud.metrics import (cluster_bootstrap, month_budgets, monthly_capture, score_fn, summarize_draws)
from fraud.model import fit_scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--cutoff", default="2017-11-21 22:51:30", help="train < cutoff <= test (UTC date)")
    ap.add_argument("--val-months", nargs="+", default=["2017-07", "2017-08", "2017-09", "2017-10"])
    ap.add_argument("--monthly-k", type=int, default=400)
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="drop test months observed less than this share (e.g. 0.9)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    feats = build_features(load_labelled(a.data_dir))
    cutoff = pd.Timestamp(a.cutoff, tz="UTC")
    train, test = feats[feats["transactionTime"] < cutoff], feats[feats["transactionTime"] >= cutoff].copy()

    # 1) choose calibration + b on training-only rolling folds
    val = rolling_validation(train, a.val_months, a.monthly_k)
    cfg, table = pick_config(val)
    table.to_csv(out / "validation_grid.csv", index=False)
    print("validation grid (best first):\n", table.round(2).to_string(index=False), "\nchosen:", cfg)

    # 2) fit the final model on all pre-cutoff data and persist
    scorer = fit_scorer(train, cutoff, **cfg)
    scorer.save(out / "fraud_scorer.joblib")

    # 3) test window, touched once. K scaled to observed days per month
    test["month"] = test["transactionTime"].dt.tz_localize(None).dt.to_period("M")
    test["model"] = scorer.score(test)["rank_score"].to_numpy()
    test["baseline"] = test["transactionAmount"].to_numpy()
    window_end = test["transactionTime"].max().tz_localize(None) + pd.Timedelta(seconds=1)
    budgets = month_budgets(cutoff, window_end, a.monthly_k, a.min_coverage)
    res = pd.DataFrame({"model_%": monthly_capture(test, test["model"], budgets),
                        "baseline_%": monthly_capture(test, test["baseline"], budgets)})
    res["uplift_pp"] = res["model_%"] - res["baseline_%"]
    res = budgets.assign(month=budgets["month"].astype(str)).set_index("month").join(res)
    print("\nper-month results (K scaled to coverage):\n", res.round(1).to_string())

    # 4) account-level bootstrap for the SELECTED model
    draws = cluster_bootstrap(test, {"model": test["model"], "baseline": test["baseline"]},
                              budgets, n_boot=a.n_boot)
    draws["uplift"] = draws["model"] - draws["baseline"]
    summ = summarize_draws(draws)
    print("\naccount-level bootstrap, 95% CI of month-averaged capture %:")
    for n, s in summ.items():
        print(f"  {n}: {s['mean']:.1f}  [{s['lo']:.1f}, {s['hi']:.1f}]")

    point = dict(model=float(res["model_%"].mean()), baseline=float(res["baseline_%"].mean()))
    (out / "metrics.json").write_text(json.dumps(dict(config=cfg, point=point, bootstrap=summ,
                                                      months=res.reset_index().to_dict("records")), indent=2, default=str))


if __name__ == "__main__":
    main()
