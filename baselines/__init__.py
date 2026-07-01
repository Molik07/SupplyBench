# This file marks baselines/ as a Python package and exposes
# all five model classes for clean top-level imports.
#
# Usage:
#   from baselines import NaiveForecaster, MovingAverageForecaster
#   from baselines import SeasonalNaiveForecaster, ARIMAForecaster
#   from baselines import ProphetForecaster

from baselines.naive import NaiveForecaster
from baselines.moving_average import MovingAverageForecaster
from baselines.seasonal_naive import SeasonalNaiveForecaster
from baselines.arima import ARIMAForecaster
from baselines.prophet_forecaster import ProphetForecaster

__all__ = [
    "NaiveForecaster",
    "MovingAverageForecaster",
    "SeasonalNaiveForecaster",
    "ARIMAForecaster",
    "ProphetForecaster",
]
