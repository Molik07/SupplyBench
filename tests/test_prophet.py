"""
test_prophet.py — Phase 7 verification script for SupplyBench
===============================================================
Tests the Prophet baseline model on the HOBBIES_1_001 product from
the M5 dataset, computes evaluation metrics, and compares Prophet's
performance against the four existing classical baselines.

Run from the project root:
    python test_prophet.py
"""
import sys
# Force UTF-8 stdout so Unicode box-drawing chars work on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_m5_data
from baselines import (
    NaiveForecaster,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    ARIMAForecaster,
    ProphetForecaster,
)
from evaluator import MetricsEvaluator

SEPARATOR = "=" * 65
THIN_SEP  = "─" * 65
TEST_ITEM = "HOBBIES_1_001"
TEST_DAYS = 28          # evaluation horizon (matches config.yaml)
PREVIEW_N = 7           # days to show in comparison table


def section(title: str) -> None:
    print(f"\n{THIN_SEP}")
    print(f"  {title}")
    print(THIN_SEP)


def main() -> None:
    print(SEPARATOR)
    print("  SupplyBench — Prophet Baseline Test")
    print(SEPARATOR)

    # ------------------------------------------------------------------ #
    # Step 1 — Load M5 data and select HOBBIES_1_001                      #
    # ------------------------------------------------------------------ #
    section("Step 1 — Load Data & Select Product")
    df = load_m5_data()
    print(f"  Data loaded — shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    product_df = df[df["item_id"] == TEST_ITEM].sort_values("date").reset_index(drop=True)

    if product_df.empty:
        print(f"  ERROR: item '{TEST_ITEM}' not found in dataset.")
        sys.exit(1)

    print(f"  Product      : {TEST_ITEM}")
    print(f"  Product rows : {len(product_df):,}")
    print(f"  Date range   : {product_df['date'].min().date()} → {product_df['date'].max().date()}")

    # Train / test split
    train_df = product_df.iloc[:-TEST_DAYS]
    test_df  = product_df.iloc[-TEST_DAYS:]

    train_series = train_df.set_index("date")["sales"]
    actual       = test_df["sales"].values.astype(np.float64)

    print(f"  Train : {len(train_df):,} days  "
          f"({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"  Test  : {len(test_df):,} days  "
          f"({test_df['date'].min().date()} → {test_df['date'].max().date()})")

    # ------------------------------------------------------------------ #
    # Step 2 — Fit and predict with Prophet                               #
    # ------------------------------------------------------------------ #
    section("Step 2 — Prophet Fit + Predict")
    prophet_model = ProphetForecaster()

    t0 = time.perf_counter()
    prophet_model.fit(train_series)
    prophet_preds = prophet_model.predict(TEST_DAYS)
    elapsed = time.perf_counter() - t0

    # Sanity checks
    assert isinstance(prophet_preds, np.ndarray), "predict() must return np.ndarray"
    assert len(prophet_preds) == TEST_DAYS, f"Expected {TEST_DAYS} preds, got {len(prophet_preds)}"

    print(f"  Prophet fitted and predicted in {elapsed:.2f}s")
    print(f"  Predictions shape : {prophet_preds.shape}")
    print(f"  Min prediction    : {prophet_preds.min():.3f}")
    print(f"  Max prediction    : {prophet_preds.max():.3f}")

    # ------------------------------------------------------------------ #
    # Step 3 — Side-by-side: first 7 days of actuals vs predictions       #
    # ------------------------------------------------------------------ #
    section(f"Step 3 — Prophet vs Actual (first {PREVIEW_N} days)")

    col_w = 14
    header = f"  {'Date':<12}" + f"{'Actual':>{col_w}}" + f"{'Prophet':>{col_w}}" + f"{'Error':>{col_w}}"
    print(header)
    print("  " + "─" * (12 + col_w * 3))

    test_dates = test_df["date"].values[:PREVIEW_N]
    for i in range(PREVIEW_N):
        date_str = str(pd.Timestamp(test_dates[i]).date())
        err = prophet_preds[i] - actual[i]
        print(f"  {date_str:<12}{actual[i]:>{col_w}.1f}{prophet_preds[i]:>{col_w}.3f}{err:>{col_w}.3f}")

    # ------------------------------------------------------------------ #
    # Step 4 — Compute Prophet metrics                                    #
    # ------------------------------------------------------------------ #
    section("Step 4 — Prophet Metrics")
    evaluator = MetricsEvaluator()
    prophet_scores = evaluator.evaluate(actual, prophet_preds)

    print(f"  RMSE : {prophet_scores['rmse']:.4f}")
    print(f"  MAE  : {prophet_scores['mae']:.4f}")
    mape_str = f"{prophet_scores['mape']:.2f}%" if prophet_scores['mape'] is not None else "N/A"
    print(f"  MAPE : {mape_str}")

    # ------------------------------------------------------------------ #
    # Step 5 — Compare against all four classical baselines                #
    # ------------------------------------------------------------------ #
    section("Step 5 — Prophet vs Classical Baselines")

    baseline_models = [
        NaiveForecaster(),
        MovingAverageForecaster(),
        SeasonalNaiveForecaster(),
        ARIMAForecaster(),
    ]

    all_scores: dict[str, dict] = {"Prophet": prophet_scores}

    for model in baseline_models:
        t0 = time.perf_counter()
        model.fit(train_series)
        preds = model.predict(TEST_DAYS)
        elapsed_m = time.perf_counter() - t0
        scores = evaluator.evaluate(actual, preds)
        all_scores[model.name] = scores
        print(f"  {model.name:<22} fitted in {elapsed_m:.4f}s")

    # Rank by RMSE
    ranked = sorted(all_scores.items(), key=lambda kv: kv[1]["rmse"])

    print()
    header = f"  {'Rank':<6}{'Model':<22}{'RMSE':>{col_w}}{'MAE':>{col_w}}{'MAPE(%)':>{col_w}}"
    print(header)
    print("  " + "─" * (6 + 22 + col_w * 3))

    ordinals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}
    prophet_rank = None

    for rank, (name, scores) in enumerate(ranked, start=1):
        label = ordinals.get(rank, f"{rank}th")
        rmse_s = f"{scores['rmse']:.4f}"
        mae_s  = f"{scores['mae']:.4f}"
        mape_s = f"{scores['mape']:.2f}" if scores['mape'] is not None else "N/A"
        marker = "  ◀ Prophet" if name == "Prophet" else ""
        print(f"  {label:<6}{name:<22}{rmse_s:>{col_w}}{mae_s:>{col_w}}{mape_s:>{col_w}}{marker}")
        if name == "Prophet":
            prophet_rank = rank

    # ------------------------------------------------------------------ #
    # Step 6 — Final confirmation                                         #
    # ------------------------------------------------------------------ #
    section("Step 6 — Confirmation")

    print(f"  Prophet model          : WORKING")
    print(f"  Fit + predict time     : {elapsed:.2f}s")
    print(f"  Predictions returned   : {len(prophet_preds)} (expected {TEST_DAYS})")
    print(f"  All predictions >= 0   : {bool(np.all(prophet_preds >= 0))}")
    print(f"  Prophet RMSE           : {prophet_scores['rmse']:.4f}")
    print(f"  Prophet ranking        : {ordinals.get(prophet_rank, f'{prophet_rank}th')} out of 5 models")

    winner_name, winner_scores = ranked[0]
    print(f"  Best model overall     : {winner_name} (RMSE = {winner_scores['rmse']:.4f})")

    print(f"\n{SEPARATOR}")
    print(f"  PROPHET TEST PASSED — Ready for integration")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
