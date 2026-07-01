"""
test_cv.py — Rolling Origin Cross Validation test script
=========================================================
Tests RollingOriginCV on 3 products with 3 folds to verify the
cross validation engine works correctly before running it on the
full 50-product suite.

Run from the project root:
    python test_cv.py
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
from evaluator import MetricsEvaluator, RollingOriginCV

SEPARATOR = "=" * 65
THIN_SEP  = "─" * 65
N_TEST_PRODUCTS = 3       # quick test — just 3 products
N_FOLDS         = 3       # 3 folds instead of 5 to keep test fast
TEST_DAYS       = 28


def section(title: str) -> None:
    print(f"\n{THIN_SEP}")
    print(f"  {title}")
    print(THIN_SEP)


def main() -> None:
    t_start = time.perf_counter()

    print(SEPARATOR)
    print("  SupplyBench — Rolling Origin Cross Validation Test")
    print(SEPARATOR)

    # ------------------------------------------------------------------ #
    # Step 1 — Load M5 data and select 3 products                        #
    # ------------------------------------------------------------------ #
    section("Step 1 — Load Data & Select Products")
    df = load_m5_data()
    print(f"  Data loaded — shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    all_items = df["item_id"].unique().tolist()
    test_items = all_items[:N_TEST_PRODUCTS]

    print(f"  Testing with {N_TEST_PRODUCTS} products: {test_items}")
    test_df = df[df["item_id"].isin(test_items)].copy()
    print(f"  Test subset: {test_df.shape[0]:,} rows")

    # ------------------------------------------------------------------ #
    # Step 2 — Instantiate all 5 models                                  #
    # ------------------------------------------------------------------ #
    section("Step 2 — Instantiate Models")
    models = [
        NaiveForecaster(),
        MovingAverageForecaster(),
        SeasonalNaiveForecaster(),
        ARIMAForecaster(),
        ProphetForecaster(),
    ]
    for m in models:
        print(f"  {m.name:<22} ready")

    # ------------------------------------------------------------------ #
    # Step 3 — Run RollingOriginCV with 3 folds on 3 products            #
    # ------------------------------------------------------------------ #
    section(f"Step 3 — Rolling Origin CV ({N_FOLDS} folds × {TEST_DAYS} days)")

    cv = RollingOriginCV(
        n_folds=N_FOLDS,
        test_size=TEST_DAYS,
        min_train_size=365,
    )

    cv_results = cv.evaluate_all_items(
        test_df, models, show_progress=True
    )

    # ------------------------------------------------------------------ #
    # Step 4 — Print CV results table                                    #
    # ------------------------------------------------------------------ #
    section("Step 4 — CV Results Table")

    col_w = 14
    header = (f"  {'Model':<20}"
              f"{'Mean RMSE':>{col_w}}"
              f"{'Std RMSE':>{col_w}}"
              f"{'Mean MAE':>{col_w}}"
              f"{'Std MAE':>{col_w}}"
              f"{'Mean MAPE':>{col_w}}"
              f"{'Std MAPE':>{col_w}}")
    print(header)
    print("  " + "─" * (20 + col_w * 6))

    # Sort by mean_rmse
    ranked = sorted(
        cv_results["models"].items(),
        key=lambda kv: kv[1]["mean_rmse"] or float("inf"),
    )

    ordinals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}

    for rank, (name, m) in enumerate(ranked, start=1):
        rmse_s = f"{m['mean_rmse']:.4f}" if m["mean_rmse"] is not None else "N/A"
        rmse_std = f"±{m['std_rmse']:.4f}" if m["std_rmse"] is not None else "N/A"
        mae_s = f"{m['mean_mae']:.4f}" if m["mean_mae"] is not None else "N/A"
        mae_std = f"±{m['std_mae']:.4f}" if m["std_mae"] is not None else "N/A"
        mape_s = f"{m['mean_mape']:.2f}%" if m["mean_mape"] is not None else "N/A"
        mape_std = f"±{m['std_mape']:.2f}%" if m["std_mape"] is not None else "N/A"
        print(f"  {name:<20}{rmse_s:>{col_w}}{rmse_std:>{col_w}}"
              f"{mae_s:>{col_w}}{mae_std:>{col_w}}"
              f"{mape_s:>{col_w}}{mape_std:>{col_w}}")

    # ------------------------------------------------------------------ #
    # Step 5 — Print fold-level results for one model (ARIMA)            #
    # ------------------------------------------------------------------ #
    section("Step 5 — Fold-Level Detail (single product, ARIMA)")

    # Run CV on just the first product to show fold-level detail
    first_item = test_items[0]
    product_df = (
        test_df[test_df["item_id"] == first_item]
        .sort_values("date")
        .reset_index(drop=True)
    )
    series = product_df.set_index("date")["sales"]

    single_cv = RollingOriginCV(n_folds=N_FOLDS, test_size=TEST_DAYS)
    arima_model = [ARIMAForecaster()]
    single_result = single_cv.evaluate(series, arima_model)

    arima_data = single_result["models"]["ARIMA"]
    print(f"  Product: {first_item}")
    print(f"  Folds completed: {len(arima_data['fold_results'])}\n")

    fold_header = f"  {'Fold':<8}{'RMSE':>{col_w}}{'MAE':>{col_w}}{'MAPE(%)':>{col_w}}"
    print(fold_header)
    print("  " + "─" * (8 + col_w * 3))

    for fr in arima_data["fold_results"]:
        mape_str = f"{fr['mape']:.2f}" if fr["mape"] is not None else "N/A"
        print(f"  Fold {fr['fold']:<3}{fr['rmse']:>{col_w}.4f}"
              f"{fr['mae']:>{col_w}.4f}{mape_str:>{col_w}}")

    print(f"\n  Mean RMSE: {arima_data['mean_rmse']:.4f} ± {arima_data['std_rmse']:.4f}")
    print(f"  Mean MAE:  {arima_data['mean_mae']:.4f} ± {arima_data['std_mae']:.4f}")
    if arima_data["mean_mape"] is not None:
        print(f"  Mean MAPE: {arima_data['mean_mape']:.2f}% ± {arima_data['std_mape']:.2f}%")

    # ------------------------------------------------------------------ #
    # Step 6 — Final confirmation                                        #
    # ------------------------------------------------------------------ #
    t_total = time.perf_counter() - t_start

    section("Step 6 — Confirmation")

    n_models = len(cv_results["models"])
    all_have_results = all(
        m["mean_rmse"] is not None
        for m in cv_results["models"].values()
    )

    print(f"  RollingOriginCV      : WORKING")
    print(f"  Products tested      : {N_TEST_PRODUCTS}")
    print(f"  Folds per product    : {N_FOLDS}")
    print(f"  Models evaluated     : {n_models}")
    print(f"  All models produced  : {'YES' if all_have_results else 'NO — check warnings above'}")
    print(f"  Total time           : {t_total:.1f}s")

    winner_name, winner_data = ranked[0]
    print(f"  Best model (CV RMSE) : {winner_name} "
          f"({winner_data['mean_rmse']:.4f} ± {winner_data['std_rmse']:.4f})")

    print(f"\n{SEPARATOR}")
    if all_have_results:
        print(f"  CV TEST PASSED — Rolling Origin Cross Validation working")
    else:
        print(f"  CV TEST PARTIAL — Some models may have failed, review output above")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
