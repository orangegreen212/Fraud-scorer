# Fraud is in the Air: monthly review-queue ranking

Ranks transactions so that the fraud team's monthly review budget (default 400) catches as much
**fraud value (£)** as possible. Model: logistic regression, score = P(fraud) x amount^b.

## Run
```bash
pip install -r requirements.txt
export FRAUD_DATA_DIR="./Fraud is in the Air"        # or pass --data-dir (no hardcoded paths)
python scripts/train.py --cutoff 2017-11-21           # validate -> fit -> test once -> bootstrap CI
python scripts/score.py --new new.csv --history old.csv --k 400   # -> review_queue.csv
pytest -q
```
`train.py` writes `artifacts/fraud_scorer.joblib` (the whole pipeline), `metrics.json` and `validation_grid.csv`.
Raw data is git-ignored; do not commit the archives.

## Layout
| file | role |
|---|---|
| `fraud/features.py` | `build_features()`: the one feature contract (causal, `closed="left"`); EDA and model both call it |
| `fraud/model.py` | sklearn `Pipeline` (log1p, impute, scale, one-hot, LR), `FraudScorer` (save/load, score, reason codes) |
| `fraud/metrics.py` | `capture_pct`, `known_label`, `score_fn`, scaled monthly budgets, **account-level bootstrap** |
| `fraud/evaluate.py` | rolling validation: picks calibration and exponent `b` on training-only folds |
| `scripts/` | `train.py`, `score.py` |
| `tests/` | metrics, features (brute-force + causality check), model round-trip, end-to-end CLI |

## Review point -> change
1. **No deployable pipeline** -> `Pipeline` + `FraudScorer` persisted with joblib; `score.py` builds features from history + new rows and writes the queue with reason codes.
2. **Unsupported CI** -> `cluster_bootstrap` resamples whole accounts for the *selected* LR model and the baseline (`metrics.json`). If it does not reproduce 94.7%, quote the new number, not the old one.
3. **K vs partial months** -> `month_budgets` scales K by the share of each month observed (21 Nov start -> ~133 reviews, not 400); `--min-coverage 0.9` drops partial months instead.
4. **Reproducibility** -> `FRAUD_DATA_DIR`/`--data-dir`, `requirements*.txt`, `.gitignore`, tests.
5. **EDA vs features** -> `hour`/`dayofweek`/rolling counts are created in `build_features()` only; delete the feature-creating lines from the EDA cells (4.3, 4.5, section 5) and use `tx = build_features(tx)` once after loading.
6. **Calibration and `b`** -> `class_weight="balanced"` inflates odds by (1-p0)/p0, so the raw output is not a probability. `rolling_validation` compares raw vs `prior_correct`ed probabilities and `b` in {0.5..2.5} on the Jul-Oct folds (not the test months), with Brier score.
7. **Business discussion** -> see the talking points below.

## Business talking points (lead with these)
- **Recommendation:** run the model in shadow mode next to today's rules for 2-3 months; the analysts still review 400, the queue just changes.
- **Value:** report £ caught vs the amount-only rule, month by month, with the account-level interval, not a single average.
- **Workload:** the queue size is fixed by capacity; show how capture changes at K = 300 / 400 / 500.
- **False positives:** at 400 reviews most queue items are genuine; the cost is analyst time, so track £ caught per review.
- **Go-live criteria (agree in advance):** model beats baseline in most shadow months and the lower CI bound is above zero; no drift in score distribution; label delay (median 3.5 days, up to 95) accounted for in monitoring.
- **Monitoring:** monthly capture vs baseline, score/feature drift, share of queue from new cards, retrain schedule.

## Results on the real data (cutoff 2017-11-21 22:51, K scaled to observed days)
| month | days | K | model % | amount-only % |
|---|---|---|---|---|
| 2017-11 | 9 | 121 | 64.1 | 42.5 |
| 2017-12 | 31 | 400 | 87.3 | 33.6 |
| 2018-01 | 30 | 387 | 93.1 | 63.9 |

Month average **81.5%** (account-level bootstrap 95% CI **[57.5, 92.4]**) vs 44.0% for the amount-only rule;
uplift **+34.8 pp, CI [14.8, 56.9]**. Chosen on validation folds only: prior-corrected probabilities, b = 2.5.
The earlier 94.7% used a full 400 reviews for the 9-day November; Dec/Jan reproduce (89.9 / 95.0 with the old
raw b=1.5 setting). Test has only 96 frauds on 31 cards, so differences of a few points between configs are noise.
Models in `models/` were trained with pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0 (Python 3.11+); keep those pins.
