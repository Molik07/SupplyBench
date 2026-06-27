"""
evaluator/metrics.py
====================
The metrics engine that turns SupplyBench from a forecasting tool into a
benchmarking tool. Takes actual vs. predicted arrays and computes three
standard error metrics — RMSE, MAE and MAPE — that objectively score
every model in the benchmark.

Usage:
    from evaluator import MetricsEvaluator

    evaluator = MetricsEvaluator()
    scores    = evaluator.evaluate(actual, predicted)
    all_scores = evaluator.evaluate_all(actual, {'Naive': naive_preds, ...})
    ranked     = evaluator.rank_models(all_scores)
"""

from collections import OrderedDict

import numpy as np


class MetricsEvaluator:
    """
    Computes RMSE, MAE and MAPE for forecast evaluation.

    Methods
    -------
    evaluate(actual, predicted)
        Score a single model's predictions against actual values.
    evaluate_all(actual, predictions_dict)
        Score multiple models in one call.
    rank_models(results)
        Rank models from best (lowest RMSE) to worst.
    """

    # ------------------------------------------------------------------ #
    # Single-model evaluation                                             #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict:
        """
        Compute RMSE, MAE and MAPE between actual and predicted arrays.

        Parameters
        ----------
        actual : np.ndarray
            1-D array of ground-truth sales values.
        predicted : np.ndarray
            1-D array of forecasted values (same length as *actual*).

        Returns
        -------
        dict
            Keys: 'rmse', 'mae', 'mape'. All values rounded to four
            decimal places. MAPE is ``None`` when every actual value
            is zero (division undefined).
        """
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)

        if actual.shape != predicted.shape:
            raise ValueError(
                f"Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}"
            )

        # ---- RMSE ----
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        # ---- MAE ----
        mae = float(np.mean(np.abs(actual - predicted)))

        # ---- MAPE (with zero-actual guard) ----
        nonzero_mask = actual != 0
        if nonzero_mask.sum() == 0:
            # Every actual value is zero — MAPE is undefined
            print("[SupplyBench] MAPE undefined — all actual values are zero")
            mape = None
        else:
            mape = float(
                np.mean(
                    np.abs((actual[nonzero_mask] - predicted[nonzero_mask])
                           / actual[nonzero_mask])
                ) * 100
            )
            mape = round(mape, 4)

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": mape,
        }

    # ------------------------------------------------------------------ #
    # Multi-model evaluation                                              #
    # ------------------------------------------------------------------ #

    def evaluate_all(
        self,
        actual: np.ndarray,
        predictions: dict[str, np.ndarray],
    ) -> dict[str, dict]:
        """
        Evaluate every model in *predictions* against the same actuals.

        Parameters
        ----------
        actual : np.ndarray
            Ground-truth sales values.
        predictions : dict[str, np.ndarray]
            Mapping of model name → predicted array.

        Returns
        -------
        dict[str, dict]
            Mapping of model name → metrics dictionary.
        """
        return {
            name: self.evaluate(actual, preds)
            for name, preds in predictions.items()
        }

    # ------------------------------------------------------------------ #
    # Ranking                                                             #
    # ------------------------------------------------------------------ #

    def rank_models(
        self,
        results: dict[str, dict],
    ) -> OrderedDict:
        """
        Rank models from best (lowest RMSE) to worst.

        Parameters
        ----------
        results : dict[str, dict]
            Output of :meth:`evaluate_all`.

        Returns
        -------
        OrderedDict
            Same data as *results* but ordered by ascending RMSE so that
            the first entry is the best-performing model.
        """
        sorted_items = sorted(results.items(), key=lambda kv: kv[1]["rmse"])
        return OrderedDict(sorted_items)
