"""
baselines/naive.py
==================
Naive Forecast — the simplest possible forecasting strategy.

Whatever the last observed sales value was, predict that exact same
value for every future day. No math. No learning. Just repetition.

This is the absolute performance floor. If your model cannot beat
this, your model is not adding value.
"""

import numpy as np
import pandas as pd

from baselines.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """
    Predicts a constant value equal to the last observed training value.

    Example
    -------
    If the last day of training data had sales = 3, then every
    predicted day also gets sales = 3.
    """

    name: str = "Naive"

    def __init__(self) -> None:
        self.last_value: float | None = None

    def fit(self, train: pd.Series) -> None:
        """Store the last observed sales value from the training series."""
        if train.empty:
            raise ValueError("Training series is empty.")
        self.last_value = float(train.iloc[-1])

    def predict(self, n: int) -> np.ndarray:
        """Return an array of length n where every value is self.last_value."""
        if self.last_value is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(n, self.last_value, dtype=np.float64)
