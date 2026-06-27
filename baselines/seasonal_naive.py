"""
baselines/seasonal_naive.py
============================
Seasonal Naive Forecast — captures weekly retail patterns.

Instead of repeating a single value, this model repeats the last full
week (7 days) of training data cyclically. Monday gets last Monday's
value, Tuesday gets last Tuesday's value, and so on.

This captures the dominant weekly seasonality in retail data — stores
sell more on weekends than weekdays. Far more realistic than plain Naive
for any business with a weekly demand cycle.
"""

import numpy as np
import pandas as pd

from baselines.base import BaseForecaster

# One full weekly seasonal cycle (daily retail data)
SEASON_LENGTH = 7


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Repeats the last full seasonal cycle (7 days) to cover the forecast horizon.

    Example
    -------
    If last week's daily sales were [1, 0, 2, 3, 5, 7, 4] and we need
    14 predictions, we return [1, 0, 2, 3, 5, 7, 4, 1, 0, 2, 3, 5, 7, 4].
    """

    name: str = "SeasonalNaive"

    def __init__(self, season_length: int = SEASON_LENGTH) -> None:
        self.season_length = season_length
        self.season: np.ndarray | None = None

    def fit(self, train: pd.Series) -> None:
        """Store the last season_length values as the seasonal template."""
        if train.empty:
            raise ValueError("Training series is empty.")
        # Use whatever is available if series shorter than season_length
        recent = train.tail(self.season_length)
        self.season = recent.to_numpy(dtype=np.float64)

    def predict(self, n: int) -> np.ndarray:
        """
        Tile the seasonal pattern to cover n days exactly.

        np.tile repeats the array enough times, then we slice to n.
        """
        if self.season is None:
            raise RuntimeError("Call fit() before predict().")

        # Compute how many full repeats are needed to cover n days
        repeats = -(-n // len(self.season))  # ceiling division
        tiled = np.tile(self.season, repeats)
        return tiled[:n].astype(np.float64)
