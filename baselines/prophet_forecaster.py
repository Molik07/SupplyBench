"""
baselines/prophet_forecaster.py
================================
Prophet Forecast — Meta's production-grade time series model.

Prophet handles daily data with strong seasonality and trend components
automatically. Unlike ARIMA which requires manual parameter tuning,
Prophet figures out trend and seasonality on its own.

This is the first genuine ML model in SupplyBench's benchmark suite.
The four existing baselines are all classical statistical methods;
Prophet represents the modern forecasting tools that real supply chain
teams actually use in production.

Includes graceful fallback to the mean of the last 28 training values
if Prophet fitting or prediction fails.
"""

import sys
import logging

import numpy as np
import pandas as pd
from prophet import Prophet

from baselines.base import BaseForecaster


# --------------------------------------------------------------------------- #
# Prophet baseline model                                                       #
# --------------------------------------------------------------------------- #

class ProphetForecaster(BaseForecaster):
    """
    Meta Prophet model wrapped with graceful fallback.

    Prophet is configured with:
      - yearly_seasonality  = True
      - weekly_seasonality  = True
      - daily_seasonality   = False
      - seasonality_mode    = 'multiplicative'

    If Prophet fails to fit (e.g. on sparse or degenerate series),
    the model falls back to the mean of the last 28 training values.
    """

    name: str = "Prophet"

    def __init__(self) -> None:
        self.model = None
        self.last_train_date = None
        self.fallback_value: float = 0.0

    def fit(self, train: pd.Series) -> None:
        """
        Fit Prophet to the training series.

        Parameters
        ----------
        train : pd.Series
            Daily sales values with a datetime index, sorted ascending.
        """
        if train.empty:
            raise ValueError("Training series is empty.")

        # Store fallback value — mean of last 28 days (or fewer if series is short)
        tail_n = min(28, len(train))
        self.fallback_value = float(train.iloc[-tail_n:].mean())

        # Store last training date for future date generation in predict()
        self.last_train_date = train.index[-1]

        try:
            # Convert to Prophet's required format: columns 'ds' and 'y'
            df = pd.DataFrame({
                'ds': train.index,
                'y': train.values
            })

            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )

            # Suppress Prophet's verbose Stan output
            logging.getLogger('prophet').setLevel(logging.ERROR)
            logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
            logging.getLogger('stan').setLevel(logging.ERROR)

            import warnings
            warnings.filterwarnings('ignore')

            self.model.fit(df)

        except Exception as exc:
            print(
                f"[SupplyBench] Prophet fit failed, using fallback. "
                f"Reason: {exc}"
            )
            self.model = None

    def predict(self, n: int) -> np.ndarray:
        """
        Produce n-step-ahead forecasts.

        Parameters
        ----------
        n : int
            Number of future days to forecast.

        Returns
        -------
        np.ndarray
            1-D array of length n with non-negative float predictions.
        """
        if self.model is not None:
            try:
                # Build future dataframe manually (exclude training dates)
                future_dates = pd.date_range(
                    start=self.last_train_date + pd.Timedelta(days=1),
                    periods=n,
                    freq='D'
                )
                future_df = pd.DataFrame({'ds': future_dates})

                forecast = self.model.predict(future_df)
                predictions = forecast['yhat'].values.clip(0)

                return np.asarray(predictions[:n], dtype=np.float64)

            except Exception as exc:
                print(
                    f"[SupplyBench] Prophet predict failed, using fallback. "
                    f"Reason: {exc}"
                )

        # Fallback: constant prediction using mean of last 28 training values
        return np.full(n, self.fallback_value, dtype=np.float64)
