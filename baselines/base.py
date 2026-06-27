"""
baselines/base.py
=================
Abstract base class that defines the standard interface every forecasting
model in SupplyBench must follow. The evaluation engine in Phase 3 will
call fit() and predict() on every model without knowing what's inside.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """
    Contract that all SupplyBench baseline models must satisfy.

    Subclasses MUST:
        - Set  self.name  to a descriptive string (e.g. 'Naive')
        - Implement  fit(train)   to learn from historical data
        - Implement  predict(n)   to produce n-step-ahead forecasts
    """

    # Subclasses should override this with a human-readable label
    name: str = "BaseForecaster"

    @abstractmethod
    def fit(self, train: pd.Series) -> None:
        """
        Learn from historical sales data.

        Parameters
        ----------
        train : pd.Series
            Daily sales values sorted by date ascending.
            The Series index should be datetime dates.
        """
        ...

    @abstractmethod
    def predict(self, n: int) -> np.ndarray:
        """
        Generate n-step-ahead point forecasts.

        Parameters
        ----------
        n : int
            Number of future days to forecast.

        Returns
        -------
        np.ndarray
            1-D array of length exactly n containing float predictions.
            Always returns a numpy array, never a pandas Series.
        """
        ...
