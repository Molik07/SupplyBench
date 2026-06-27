"""
baselines/moving_average.py
============================
Moving Average Forecast — smoother than Naive, still very simple.

Takes the average of the last 28 days of training data and predicts
that same flat average for every future day. The 28-day window matches
our test horizon (4 weeks), making it a natural comparison period.

Averages out day-to-day noise, but cannot capture trends or seasonality.
"""

import numpy as np
import pandas as pd

from baselines.base import BaseForecaster

# Window size matches the evaluation test horizon (28 days = 4 weeks)
WINDOW = 28


class MovingAverageForecaster(BaseForecaster):
    """
    Predicts a constant value equal to the mean of the last WINDOW training days.

    Example
    -------
    If the last 28 days averaged 2.5 units/day, every predicted
    day gets sales = 2.5.
    """

    name: str = "MovingAverage"

    def __init__(self, window: int = WINDOW) -> None:
        self.window = window
        self.mean_value: float | None = None

    def fit(self, train: pd.Series) -> None:
        """Compute the mean of the last `window` training values."""
        if train.empty:
            raise ValueError("Training series is empty.")
        # Use whatever is available if series shorter than window
        recent = train.tail(self.window)
        self.mean_value = float(recent.mean())

    def predict(self, n: int) -> np.ndarray:
        """Return an array of length n where every value is self.mean_value."""
        if self.mean_value is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(n, self.mean_value, dtype=np.float64)
