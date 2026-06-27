"""
baselines/arima.py
==================
ARIMA Forecast — classical statistical time series model.

ARIMA (AutoRegressive Integrated Moving Average) captures three things:
  - AR (AutoRegressive): how much past values influence future values
  - I  (Integrated):     how many times to difference the series to
                         remove trends and make it stationary
  - MA (Moving Average): how past forecast errors influence future values

Order (1, 1, 1): 1 AR lag, difference once, 1 MA lag.
This is the standard general-purpose configuration for retail demand.

Includes graceful fallback to a constant prediction if ARIMA fitting
fails (common on zero-heavy or completely flat series).
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from baselines.base import BaseForecaster

# Standard general-purpose ARIMA order for daily retail data
ARIMA_ORDER = (1, 1, 1)


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA(1,1,1) model wrapped with graceful fallback.

    If statsmodels fails to fit (e.g. on a flat all-zero series),
    the model silently falls back to repeating the last observed value.
    """

    name: str = "ARIMA"

    def __init__(self, order: tuple[int, int, int] = ARIMA_ORDER) -> None:
        self.order = order
        self.model_fit = None
        self.fallback_value: float = 0.0

    def fit(self, train: pd.Series) -> None:
        """
        Fit ARIMA to the training series.

        Falls back to storing the last value if fitting fails.
        """
        if train.empty:
            raise ValueError("Training series is empty.")

        self.fallback_value = float(train.iloc[-1])

        try:
            # Suppress convergence warnings that are noisy but non-fatal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima_model = ARIMA(train.values, order=self.order)
                self.model_fit = arima_model.fit()
        except Exception as exc:
            print(
                f"[SupplyBench] ARIMA fit failed for this series, "
                f"will use fallback on predict. Reason: {exc}"
            )
            self.model_fit = None

    def predict(self, n: int) -> np.ndarray:
        """
        Produce n-step-ahead forecasts.

        Falls back to constant prediction if model is not fitted or
        if forecasting itself raises an exception.
        """
        if self.model_fit is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = self.model_fit.forecast(steps=n)
                return np.asarray(forecast, dtype=np.float64)
            except Exception as exc:
                print(
                    f"[SupplyBench] ARIMA forecast failed, using fallback. "
                    f"Reason: {exc}"
                )

        # Fallback: constant prediction using last known value
        return np.full(n, self.fallback_value, dtype=np.float64)
