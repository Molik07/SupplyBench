"""
test_loader.py — Phase 1 verification script for SupplyBench
=============================================================
Run this from the project root to confirm the M5 data loader
is working correctly end-to-end.

Usage:
    python test_loader.py
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path so `data.loader` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_m5_data

SEPARATOR = "=" * 60


def main():
    print(SEPARATOR)
    print("  SupplyBench — Phase 1 Data Loader Test")
    print(SEPARATOR)

    # ------------------------------------------------------------------ #
    # Run the loader                                                       #
    # ------------------------------------------------------------------ #
    df = load_m5_data()

    # ------------------------------------------------------------------ #
    # 1. Shape                                                             #
    # ------------------------------------------------------------------ #
    print(f"\n{'─' * 40}")
    print("  DataFrame Shape")
    print(f"{'─' * 40}")
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")

    # ------------------------------------------------------------------ #
    # 2. Column dtypes                                                     #
    # ------------------------------------------------------------------ #
    print(f"\n{'─' * 40}")
    print("  Column Dtypes")
    print(f"{'─' * 40}")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<12}  →  {dtype}")

    # ------------------------------------------------------------------ #
    # 3. First 5 rows                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n{'─' * 40}")
    print("  First 5 Rows")
    print(f"{'─' * 40}")
    print(df.head(5).to_string(index=False))

    # ------------------------------------------------------------------ #
    # 4. Last 5 rows                                                       #
    # ------------------------------------------------------------------ #
    print(f"\n{'─' * 40}")
    print("  Last 5 Rows")
    print(f"{'─' * 40}")
    print(df.tail(5).to_string(index=False))

    # ------------------------------------------------------------------ #
    # 5. Descriptive statistics for the sales column                       #
    # ------------------------------------------------------------------ #
    print(f"\n{'─' * 40}")
    print("  Sales Column — Descriptive Stats")
    print(f"{'─' * 40}")
    desc = df["sales"].describe()
    for stat, val in desc.items():
        print(f"  {stat:<8} : {val:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Unique items & date range                                         #
    # ------------------------------------------------------------------ #
    print(f"\n{'─' * 40}")
    print("  Coverage Summary")
    print(f"{'─' * 40}")
    print(f"  Unique items : {df['item_id'].nunique()}")
    print(f"  Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total days   : {(df['date'].max() - df['date'].min()).days + 1}")

    # ------------------------------------------------------------------ #
    # 7. Missing value check                                               #
    # ------------------------------------------------------------------ #
    print(f"\n{'─' * 40}")
    print("  Missing Values")
    print(f"{'─' * 40}")
    nulls = df.isnull().sum()
    any_nulls = nulls.sum() > 0
    for col, count in nulls.items():
        status = " MISSING" if count > 0 else "OK"
        print(f"  {col:<12}  :  {count}  {status}")

    # ------------------------------------------------------------------ #
    # 8. Column order assertion                                            #
    # ------------------------------------------------------------------ #
    print(f"\n{'─' * 40}")
    print("  Schema Validation")
    print(f"{'─' * 40}")
    expected_cols = ["item_id", "date", "sales"]
    actual_cols = list(df.columns)
    if actual_cols == expected_cols:
        print(f"  Column order correct: {actual_cols}")
    else:
        print(f"  Column order MISMATCH")
        print(f"    Expected : {expected_cols}")
        print(f"    Got      : {actual_cols}")

    # Sort order check
    is_sorted = (
        df[["item_id", "date"]]
        .equals(df[["item_id", "date"]].sort_values(["item_id", "date"]))
    )
    sort_status = "Sorted correctly" if is_sorted else "NOT sorted"
    print(f"  {sort_status} (by item_id, then date)")

    # date dtype check
    date_ok = str(df["date"].dtype) == "datetime64[ns]"
    print(f"  {'[OK]' if date_ok else '[FAIL]'} date column is datetime64[ns]: {df['date'].dtype}")

    # sales dtype check
    sales_ok = str(df["sales"].dtype) == "float64"
    print(f"  {'[OK]' if sales_ok else '[FAIL]'} sales column is float64: {df['sales'].dtype}")

    print(f"\n{SEPARATOR}")
    if not any_nulls and actual_cols == expected_cols and is_sorted and date_ok and sales_ok:
        print("  ALL CHECKS PASSED — Phase 1 complete!")
    else:
        print("  SOME CHECKS FAILED — review output above.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
