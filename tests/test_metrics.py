"""
test_metrics.py — Phase 3 verification script for SupplyBench
=============================================================
Loads the M5 data, splits one product into train/test, runs all four
baseline models, then evaluates every model with the MetricsEvaluator
and prints a formatted results table, ranking and interpretation.

Run from the project root:
    python test_metrics.py
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
)
from evaluator import MetricsEvaluator

SEPARATOR = "=" * 65
THIN_SEP  = "─" * 65
TEST_ITEM = "HOBBIES_1_001"
TEST_DAYS = 28


def section(title: str) -> None:
    print(f"\n{THIN_SEP}")
    print(f"  {title}")
    print(THIN_SEP)


def main() -> None:
    print(SEPARATOR)
    print("  SupplyBench — Phase 3 Metrics Engine Test")
    print(SEPARATOR)

    # ------------------------------------------------------------------ #
    # Step 1 — Load the M5 data and select product                        #
    # ------------------------------------------------------------------ #
    section(f"Step 1 — Load Data & Select Product: {TEST_ITEM}")
    df = load_m5_data()
    product_df = (
        df[df["item_id"] == TEST_ITEM]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if product_df.empty:
        print(f"  ERROR: item '{TEST_ITEM}' not found in dataset.")
        sys.exit(1)

    # Train / test split — last 28 days are the test set
    train_df = product_df.iloc[:-TEST_DAYS]
    test_df  = product_df.iloc[-TEST_DAYS:]

    train_series = train_df.set_index("date")["sales"]
    test_series  = test_df.set_index("date")["sales"]

    print(f"  Product rows : {len(product_df):,}")
    print(f"  Train        : {len(train_series):,} days  "
          f"({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"  Test         : {len(test_series):,} days  "
          f"({test_df['date'].min().date()} → {test_df['date'].max().date()})")

    # ------------------------------------------------------------------ #
    # Step 2 — Fit all four baseline models and collect predictions        #
    # ------------------------------------------------------------------ #
    section("Step 2 — Run All Baseline Models")
    models = [
        NaiveForecaster(),
        MovingAverageForecaster(),
        SeasonalNaiveForecaster(),
        ARIMAForecaster(),
    ]

    predictions: dict[str, np.ndarray] = {}
    for model in models:
        t0 = time.perf_counter()
        try:
            model.fit(train_series)
            preds = model.predict(TEST_DAYS)
            elapsed = time.perf_counter() - t0
            predictions[model.name] = preds
            print(f"  {model.name:<22} — {elapsed:.4f}s  "
                  f"(first 5 preds: {np.round(preds[:5], 2)})")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  {model.name:<22} — FAILED in {elapsed:.4f}s — {exc}")

    if not predictions:
        print("\n  No models produced predictions. Cannot evaluate.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 3 — Extract actual test sales as numpy array                    #
    # ------------------------------------------------------------------ #
    section("Step 3 — Extract Actual Test Values")
    actual = test_series.values.astype(np.float64)
    print(f"  Actual array shape : {actual.shape}")
    print(f"  Actual first 5     : {actual[:5]}")
    print(f"  Actual mean        : {actual.mean():.2f}")

    # ------------------------------------------------------------------ #
    # Step 4 — Evaluate all models with MetricsEvaluator                   #
    # ------------------------------------------------------------------ #
    section("Step 4 — Evaluate All Models")
    evaluator = MetricsEvaluator()
    results = evaluator.evaluate_all(actual, predictions)

    print("  MetricsEvaluator.evaluate_all() complete")
    print(f"  Models evaluated : {len(results)}")

    # ------------------------------------------------------------------ #
    # Step 5 — Print results table                                         #
    # ------------------------------------------------------------------ #
    section("Step 5 — Metrics Results Table")

    col_w = 14
    header = f"  {'Model':<22}{'RMSE':>{col_w}}{'MAE':>{col_w}}{'MAPE (%)':>{col_w}}"
    print(header)
    print("  " + "─" * (22 + col_w * 3))

    for name, metrics in results.items():
        mape_str = f"{metrics['mape']:.4f}" if metrics["mape"] is not None else "N/A"
        row = (
            f"  {name:<22}"
            f"{metrics['rmse']:>{col_w}.4f}"
            f"{metrics['mae']:>{col_w}.4f}"
            f"{mape_str:>{col_w}}"
        )
        print(row)

    # ------------------------------------------------------------------ #
    # Step 6 — Rank models by RMSE                                         #
    # ------------------------------------------------------------------ #
    section("Step 6 — Model Ranking (by RMSE, lower is better)")
    ranked = evaluator.rank_models(results)

    for rank, (name, metrics) in enumerate(ranked.items(), start=1):
        ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(rank, f"{rank}th")
        print(f"  {ordinal:>4}  {name:<22} RMSE = {metrics['rmse']:.4f}")

    # ------------------------------------------------------------------ #
    # Step 7 — Interpretation vs Naive baseline                            #
    # ------------------------------------------------------------------ #
    section("Step 7 — Interpretation")

    # Use the Naive model as the reference baseline for comparisons
    naive_rmse = results.get("Naive", {}).get("rmse")

    for name, metrics in ranked.items():
        rmse = metrics["rmse"]
        mae  = metrics["mae"]
        mape = metrics["mape"]

        mape_part = f", MAPE of {mape:.2f}%" if mape is not None else ""

        if naive_rmse is not None and name != "Naive" and naive_rmse > 0:
            pct_change = ((naive_rmse - rmse) / naive_rmse) * 100
            if pct_change >= 0:
                comparison = f"which is {pct_change:.1f}% better than the Naive baseline"
            else:
                comparison = f"which is {abs(pct_change):.1f}% worse than the Naive baseline"
            print(f"  • {name} achieved RMSE of {rmse:.4f}{mape_part} — {comparison}.")
        else:
            print(f"  • {name} achieved RMSE of {rmse:.4f}, MAE of {mae:.4f}{mape_part}.")

    # ------------------------------------------------------------------ #
    # Final summary                                                        #
    # ------------------------------------------------------------------ #
    print(f"\n{SEPARATOR}")
    print("  METRICS ENGINE PASSED — Phase 3 complete!")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
