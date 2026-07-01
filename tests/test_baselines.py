"""
test_baselines.py — Phase 2 verification script for SupplyBench
===============================================================
Loads the M5 data, splits one product into train/test, runs all four
baseline models and prints a side-by-side comparison table.

Run from the project root:
    python test_baselines.py
"""
import sys, io
# Force UTF-8 stdout so Unicode box-drawing chars work on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys
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

SEPARATOR  = "=" * 65
THIN_SEP   = "─" * 65
TEST_ITEM  = "HOBBIES_1_001"
TEST_DAYS  = 28          # evaluation horizon (matches config.yaml)
PREVIEW_N  = 7           # days to show in comparison table


def section(title: str) -> None:
    print(f"\n{THIN_SEP}")
    print(f"  {title}")
    print(THIN_SEP)


def main() -> None:
    print(SEPARATOR)
    print("  SupplyBench — Phase 2 Baseline Models Test")
    print(SEPARATOR)

    # ------------------------------------------------------------------ #
    # Step 1 — Load the M5 data                                           #
    # ------------------------------------------------------------------ #
    section("Step 1 — Load Data")
    df = load_m5_data()
    print(f"  Data loaded — shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ------------------------------------------------------------------ #
    # Step 2 — Select one product                                         #
    # ------------------------------------------------------------------ #
    section(f"Step 2 — Select Product: {TEST_ITEM}")
    product_df = df[df["item_id"] == TEST_ITEM].sort_values("date").reset_index(drop=True)

    if product_df.empty:
        print(f"  ERROR: item '{TEST_ITEM}' not found in dataset.")
        sys.exit(1)

    print(f"  Product rows : {len(product_df):,}")
    print(f"  Date range   : {product_df['date'].min().date()} → {product_df['date'].max().date()}")

    # ------------------------------------------------------------------ #
    # Step 3 — Train / test split                                         #
    # ------------------------------------------------------------------ #
    section("Step 3 — Train / Test Split")
    train_df = product_df.iloc[:-TEST_DAYS]
    test_df  = product_df.iloc[-TEST_DAYS:]

    print(f"  Train : {len(train_df):,} days  "
          f"({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"  Test  : {len(test_df):,} days  "
          f"({test_df['date'].min().date()} → {test_df['date'].max().date()})")

    # ------------------------------------------------------------------ #
    # Step 4 — Extract pandas Series with datetime index                  #
    # ------------------------------------------------------------------ #
    section("Step 4 — Extract Sales Series")
    train_series = train_df.set_index("date")["sales"]
    test_series  = test_df.set_index("date")["sales"]

    print(f"  train_series shape : {train_series.shape}")
    print(f"  test_series  shape : {test_series.shape}")
    print(f"  train dtype        : {train_series.dtype}")

    # ------------------------------------------------------------------ #
    # Step 5 — Instantiate all four models                                #
    # ------------------------------------------------------------------ #
    section("Step 5 — Instantiate Models")
    models = [
        NaiveForecaster(),
        MovingAverageForecaster(),
        SeasonalNaiveForecaster(),
        ARIMAForecaster(),
    ]
    for m in models:
        print(f"  {m.name:<20} ready")

    # ------------------------------------------------------------------ #
    # Step 6 — Fit + Predict each model, time it                         #
    # ------------------------------------------------------------------ #
    section("Step 6 — Fit + Predict (timed)")
    results: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    failures: list[str] = []

    for model in models:
        t0 = time.perf_counter()
        try:
            model.fit(train_series)
            preds = model.predict(TEST_DAYS)
            elapsed = time.perf_counter() - t0

            # Sanity checks
            assert isinstance(preds, np.ndarray), f"{model.name}: predict() must return np.ndarray"
            assert len(preds) == TEST_DAYS,        f"{model.name}: expected {TEST_DAYS} preds, got {len(preds)}"

            results[model.name] = preds
            timings[model.name] = elapsed

            print(f"\n  [{model.name}]")
            print(f"    Time          : {elapsed:.4f}s")
            print(f"    First 7 preds : {np.round(preds[:PREVIEW_N], 3)}")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            failures.append(model.name)
            print(f"\n  [{model.name}]  FAILED in {elapsed:.4f}s — {exc}")

    # ------------------------------------------------------------------ #
    # Step 7 — Side-by-side comparison table (first 7 days)              #
    # ------------------------------------------------------------------ #
    section(f"Step 7 — Comparison Table (first {PREVIEW_N} days of test period)")

    actuals = test_series.values[:PREVIEW_N]
    model_names = list(results.keys())

    # Build header
    col_w = 14
    header = f"  {'Date':<12}" + f"{'Actual':>{col_w}}"
    for name in model_names:
        header += f"{name:>{col_w}}"
    print(header)
    print("  " + "─" * (12 + col_w * (1 + len(model_names))))

    # Build rows
    test_dates = test_df["date"].values[:PREVIEW_N]
    for i in range(PREVIEW_N):
        date_str = str(pd.Timestamp(test_dates[i]).date())
        row = f"  {date_str:<12}" + f"{actuals[i]:>{col_w}.1f}"
        for name in model_names:
            val = results[name][i] if name in results else float("nan")
            row += f"{val:>{col_w}.3f}"
        print(row)

    # ------------------------------------------------------------------ #
    # Step 8 — Summary                                                    #
    # ------------------------------------------------------------------ #
    section("Step 8 — Summary")
    passed = len(results)
    failed = len(failures)

    print(f"  Models run successfully : {passed} / {len(models)}")
    if failures:
        print(f"  Failed models          : {', '.join(failures)}")
    for name, t in timings.items():
        print(f"    {name:<22} — {t:.4f}s")

    print(f"\n{SEPARATOR}")
    if failed == 0:
        print("  ALL MODELS PASSED — Phase 2 complete!")
    else:
        print(f"  {failed} model(s) failed — review output above.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
