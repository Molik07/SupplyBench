"""
SupplyBench - data/loader.py
============================
Handles downloading, extracting, and preprocessing the M5 Forecasting
Accuracy dataset from Kaggle into a clean, analysis-ready pandas DataFrame.

Usage:
    from data.loader import load_m5_data
    df = load_m5_data()
"""

import os
import zipfile
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config.yaml from the project root (two levels up from this file)."""
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _ensure_raw_dir(raw_path: Path) -> None:
    """Create the raw data directory if it does not already exist."""
    raw_path.mkdir(parents=True, exist_ok=True)


def _required_files_present(raw_path: Path) -> bool:
    """Return True if all three required M5 CSV files are already on disk."""
    required = [
        "sales_train_validation.csv",
        "calendar.csv",
        "sell_prices.csv",
    ]
    return all((raw_path / f).exists() for f in required)


def _download_m5_dataset(raw_path: Path) -> None:
    """
    Download the M5 competition dataset via the Kaggle CLI.

    Prerequisites:
        - kaggle Python package installed  (`pip install kaggle`)
        - ~/.kaggle/kaggle.json with valid API credentials
          OR KAGGLE_USERNAME / KAGGLE_KEY environment variables set.
    """
    print("=" * 60)
    print("[SupplyBench] M5 dataset not found locally.")
    print("[SupplyBench] Downloading from Kaggle — this may take a few minutes...")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "kaggle",
        "competitions", "download",
        "-c", "m5-forecasting-accuracy",
        "-p", str(raw_path),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise RuntimeError(
            "\n[SupplyBench] Kaggle download failed.\n"
            "Please make sure:\n"
            "  1. The `kaggle` package is installed  (pip install kaggle)\n"
            "  2. Your API credentials are set up at ~/.kaggle/kaggle.json\n"
            "     or via KAGGLE_USERNAME / KAGGLE_KEY environment variables.\n"
            "  3. You have accepted the M5 competition rules on Kaggle.\n"
            f"Original error: {e.stderr}"
        ) from e

    # Unzip all zip files found in the raw directory
    _unzip_all(raw_path)

    print("[SupplyBench] Download and extraction complete.\n")


def _unzip_all(raw_path: Path) -> None:
    """Extract every .zip file inside raw_path."""
    zip_files = list(raw_path.glob("*.zip"))
    if not zip_files:
        print("[SupplyBench] Warning: No zip files found after download.")
        return

    for zip_file in zip_files:
        print(f"[SupplyBench] Extracting {zip_file.name} ...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(raw_path)
        # Remove the zip after extraction to save disk space
        zip_file.unlink()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_m5_data(config: dict | None = None) -> pd.DataFrame:
    """
    Download (if needed), load, and preprocess the M5 dataset.

    Parameters
    ----------
    config : dict, optional
        Pre-loaded configuration dictionary. If None, config.yaml is read
        from the project root automatically.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with exactly three columns:
            - item_id  (str)       : product identifier
            - date     (datetime)  : calendar date
            - sales    (float64)   : daily unit sales
        Sorted by item_id, then date (ascending).
    """
    if config is None:
        config = _load_config()

    cfg_data = config["dataset"]
    raw_path = Path(cfg_data["raw_data_path"])
    store_filter: str = cfg_data["store_filter"]      # e.g. "CA_1"
    max_items: int = int(cfg_data["max_items"])        # e.g. 50

    _ensure_raw_dir(raw_path)

    # ------------------------------------------------------------------
    # Step 1 — Download if not already present
    # ------------------------------------------------------------------
    if not _required_files_present(raw_path):
        _download_m5_dataset(raw_path)
    else:
        print(f"[SupplyBench] M5 dataset already present at '{raw_path}'. Skipping download.")

    # ------------------------------------------------------------------
    # Step 2 — Load raw CSVs
    # ------------------------------------------------------------------
    print("[SupplyBench] Loading raw CSV files ...")

    sales_path = raw_path / "sales_train_validation.csv"
    calendar_path = raw_path / "calendar.csv"
    prices_path = raw_path / "sell_prices.csv"

    sales_df = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)
    # sell_prices loaded but not used in Phase 1 — reserved for future phases
    # prices_df = pd.read_csv(prices_path)

    print(f"[SupplyBench] sales_train_validation.csv loaded — {sales_df.shape[0]:,} rows")
    print(f"[SupplyBench] calendar.csv loaded — {calendar_df.shape[0]:,} rows")

    # ------------------------------------------------------------------
    # Step 3 — Filter to store CA_1, first N items
    # ------------------------------------------------------------------
    print(f"[SupplyBench] Filtering to store '{store_filter}', first {max_items} items ...")

    store_sales = sales_df[sales_df["store_id"] == store_filter].copy()

    if store_sales.empty:
        raise ValueError(
            f"[SupplyBench] No rows found for store_id='{store_filter}'. "
            "Check the store_filter value in config.yaml."
        )

    # Take the first max_items unique item_ids (preserving original order)
    unique_items = store_sales["item_id"].unique()[:max_items]
    store_sales = store_sales[store_sales["item_id"].isin(unique_items)]

    print(f"[SupplyBench] Retained {store_sales.shape[0]:,} rows for {len(unique_items)} items.")

    # ------------------------------------------------------------------
    # Step 4 — Melt wide → long format
    # ------------------------------------------------------------------
    # Identify the day columns (d_1, d_2, … d_1913)
    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in store_sales.columns if c.startswith("d_") and c[2:].isdigit()]

    print(f"[SupplyBench] Melting {len(day_cols)} day-columns into long format ...")

    long_df = store_sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",      # will be replaced with actual dates
        value_name="sales",
    )

    # Keep only the columns we need going forward
    long_df = long_df[["item_id", "d", "sales"]]

    # ------------------------------------------------------------------
    # Step 5 — Merge calendar to get real dates
    # ------------------------------------------------------------------
    print("[SupplyBench] Merging calendar to attach real dates ...")

    # calendar_df has columns: date, wm_yr_wk, weekday, wday, month, year, d, ...
    cal_slim = calendar_df[["d", "date"]].copy()

    long_df = long_df.merge(cal_slim, on="d", how="left")

    # Convert date strings to proper datetime objects
    long_df["date"] = pd.to_datetime(long_df["date"], format="%Y-%m-%d")

    # Drop the 'd' helper column now that we have real dates
    long_df.drop(columns=["d"], inplace=True)

    # ------------------------------------------------------------------
    # Step 6 — Handle missing values
    # ------------------------------------------------------------------
    missing_before = long_df["sales"].isna().sum()
    if missing_before > 0:
        print(f"[SupplyBench] Filling {missing_before:,} missing sales values with 0 ...")
        long_df["sales"] = long_df["sales"].fillna(0.0)
    else:
        print("[SupplyBench] No missing sales values found.")

    # ------------------------------------------------------------------
    # Step 7 — Final clean-up and type enforcement
    # ------------------------------------------------------------------
    long_df["sales"] = long_df["sales"].astype(np.float64)
    long_df["item_id"] = long_df["item_id"].astype(str)

    # Sort by item_id (lexicographic) then date ascending
    long_df = long_df.sort_values(["item_id", "date"]).reset_index(drop=True)

    # Guarantee column order
    long_df = long_df[["item_id", "date", "sales"]]

    print(
        f"\n[SupplyBench] Dataset ready — "
        f"{long_df.shape[0]:,} rows × {long_df.shape[1]} columns | "
        f"{long_df['item_id'].nunique()} items | "
        f"Date range: {long_df['date'].min().date()} → {long_df['date'].max().date()}"
    )

    return long_df
