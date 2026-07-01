"""
evaluator/cross_validator.py
=============================
Rolling Origin Cross Validation for SupplyBench.

Instead of evaluating models on a single train/test split, this module
tests each model across multiple time windows with a growing training
set and fixed-size test window. This produces mean ± std metrics that
are statistically more reliable than a single-split evaluation.

Fold layout for 1913-day series with 5 folds of 28 days:
    Fold 1: Train days 1→1485  |  Test days 1486→1513
    Fold 2: Train days 1→1585  |  Test days 1586→1613
    Fold 3: Train days 1→1685  |  Test days 1686→1713
    Fold 4: Train days 1→1785  |  Test days 1786→1813
    Fold 5: Train days 1→1885  |  Test days 1886→1913

Usage:
    from evaluator import RollingOriginCV
    cv = RollingOriginCV(n_folds=5, test_size=28)
    results = cv.evaluate(series, models)
"""

import sys
import time
import copy

import numpy as np
import pandas as pd

from evaluator.metrics import MetricsEvaluator


class RollingOriginCV:
    """
    Rolling Origin Cross Validation engine.

    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds (default 5).
    test_size : int
        Number of days in each test window (default 28).
    min_train_size : int
        Minimum number of training days required before the first
        fold is allowed (default 365).
    """

    def __init__(
        self,
        n_folds: int = 5,
        test_size: int = 28,
        min_train_size: int = 365,
    ) -> None:
        self.n_folds = n_folds
        self.test_size = test_size
        self.min_train_size = min_train_size
        self._evaluator = MetricsEvaluator()

    # ------------------------------------------------------------------ #
    # Single-item cross validation                                        #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        series: pd.Series,
        models: list,
    ) -> dict:
        """
        Run rolling origin CV on a single product's time series.

        Parameters
        ----------
        series : pd.Series
            Full time series with datetime index, sorted ascending.
        models : list
            List of instantiated model objects following BaseForecaster.

        Returns
        -------
        dict
            CV results with mean/std metrics and per-fold breakdowns
            for each model.
        """
        n = len(series)

        # ── Step 1: Calculate fold split points ──────────────────────── #
        # Work backwards from the end of the series.
        # Last fold ends at position n, test starts at n - test_size.
        # Each prior fold steps back by test_size days.
        folds = []
        for fold_idx in range(self.n_folds):
            # How far back from the end this fold's test window starts
            offset = fold_idx * self.test_size
            test_end = n - offset            # exclusive end
            test_start = test_end - self.test_size
            train_end = test_start           # exclusive end of train

            # Skip if not enough training data
            if train_end < self.min_train_size:
                continue

            folds.append((train_end, test_start, test_end))

        # Reverse so fold 1 is the earliest window
        folds.reverse()

        if not folds:
            raise ValueError(
                f"Series has {n} days — not enough data for {self.n_folds} "
                f"folds with test_size={self.test_size} and "
                f"min_train_size={self.min_train_size}."
            )

        # ── Step 2–3: Run each fold for each model ───────────────────── #
        # Structure: {model_name: [fold_result_dict, ...]}
        model_fold_results: dict[str, list[dict]] = {
            m.name: [] for m in models
        }

        for fold_num, (train_end, test_start, test_end) in enumerate(folds, start=1):
            train = series.iloc[:train_end]
            test = series.iloc[test_start:test_end]
            actual = test.values.astype(np.float64)

            for model in models:
                try:
                    # Create a fresh copy of the model for each fold
                    model_copy = copy.deepcopy(model)
                    model_copy.fit(train)
                    preds = model_copy.predict(self.test_size)

                    scores = self._evaluator.evaluate(actual, preds)

                    model_fold_results[model.name].append({
                        "fold": fold_num,
                        "rmse": scores["rmse"],
                        "mae": scores["mae"],
                        "mape": scores["mape"],
                    })
                except Exception as exc:
                    print(
                        f"[SupplyBench] CV fold {fold_num} failed for "
                        f"{model.name}: {exc}"
                    )
                    # Skip this fold for this model — do not crash

        # ── Step 4–5: Aggregate results ──────────────────────────────── #
        result = {
            "n_folds": len(folds),
            "test_size": self.test_size,
            "models": {},
        }

        for model_name, fold_results in model_fold_results.items():
            if not fold_results:
                result["models"][model_name] = {
                    "mean_rmse": None, "std_rmse": None,
                    "mean_mae": None, "std_mae": None,
                    "mean_mape": None, "std_mape": None,
                    "fold_results": [],
                }
                continue

            rmse_vals = [f["rmse"] for f in fold_results]
            mae_vals = [f["mae"] for f in fold_results]
            mape_vals = [f["mape"] for f in fold_results if f["mape"] is not None]

            result["models"][model_name] = {
                "mean_rmse": round(float(np.mean(rmse_vals)), 4),
                "std_rmse": round(float(np.std(rmse_vals)), 4),
                "mean_mae": round(float(np.mean(mae_vals)), 4),
                "std_mae": round(float(np.std(mae_vals)), 4),
                "mean_mape": round(float(np.mean(mape_vals)), 4) if mape_vals else None,
                "std_mape": round(float(np.std(mape_vals)), 4) if mape_vals else None,
                "fold_results": fold_results,
            }

        return result

    # ------------------------------------------------------------------ #
    # Multi-item cross validation                                         #
    # ------------------------------------------------------------------ #

    def evaluate_all_items(
        self,
        df: pd.DataFrame,
        models: list,
        show_progress: bool = True,
    ) -> dict:
        """
        Run rolling origin CV across all items in the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Full M5 dataframe with columns: item_id, date, sales.
        models : list
            List of instantiated model objects.
        show_progress : bool
            Whether to print progress for each item.

        Returns
        -------
        dict
            Aggregated CV results across all items and folds.
        """
        all_items = df["item_id"].unique().tolist()
        n_items = len(all_items)

        # Collect per-model metrics across ALL items and ALL folds
        global_fold_results: dict[str, list[dict]] = {
            m.name: [] for m in models
        }
        skipped_items: list[str] = []

        for idx, item_id in enumerate(all_items, start=1):
            t0 = time.perf_counter()

            if show_progress:
                width = len(str(n_items))
                sys.stdout.write(
                    f"  [{idx:>{width}}/{n_items}]  {item_id:<30}"
                )
                sys.stdout.flush()

            try:
                product_df = (
                    df[df["item_id"] == item_id]
                    .sort_values("date")
                    .reset_index(drop=True)
                )

                series = product_df.set_index("date")["sales"]

                # Check if series is long enough for at least one fold
                min_required = self.min_train_size + self.test_size
                if len(series) < min_required:
                    elapsed = time.perf_counter() - t0
                    if show_progress:
                        print(f"SKIPPED — only {len(series)} days ({elapsed:.2f}s)")
                    skipped_items.append(item_id)
                    continue

                # Create fresh model instances for each item
                fresh_models = [copy.deepcopy(m) for m in models]
                item_result = self.evaluate(series, fresh_models)

                # Merge fold results into global accumulator
                for model_name, model_data in item_result["models"].items():
                    global_fold_results[model_name].extend(
                        model_data.get("fold_results", [])
                    )

                elapsed = time.perf_counter() - t0
                if show_progress:
                    print(f"done ({elapsed:.2f}s)")

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                if show_progress:
                    print(f"ERROR ({elapsed:.2f}s) — {exc}")
                skipped_items.append(item_id)
                continue

        # ── Aggregate across all items and folds ─────────────────────── #
        aggregate = {
            "n_folds": self.n_folds,
            "test_size": self.test_size,
            "n_items": n_items,
            "n_items_skipped": len(skipped_items),
            "models": {},
        }

        for model_name, fold_results in global_fold_results.items():
            if not fold_results:
                aggregate["models"][model_name] = {
                    "mean_rmse": None, "std_rmse": None,
                    "mean_mae": None, "std_mae": None,
                    "mean_mape": None, "std_mape": None,
                    "n_successful_folds": 0,
                }
                continue

            rmse_vals = [f["rmse"] for f in fold_results]
            mae_vals = [f["mae"] for f in fold_results]
            mape_vals = [f["mape"] for f in fold_results if f["mape"] is not None]

            aggregate["models"][model_name] = {
                "mean_rmse": round(float(np.mean(rmse_vals)), 4),
                "std_rmse": round(float(np.std(rmse_vals)), 4),
                "mean_mae": round(float(np.mean(mae_vals)), 4),
                "std_mae": round(float(np.std(mae_vals)), 4),
                "mean_mape": round(float(np.mean(mape_vals)), 4) if mape_vals else None,
                "std_mape": round(float(np.std(mape_vals)), 4) if mape_vals else None,
                "n_successful_folds": len(fold_results),
            }

        return aggregate
