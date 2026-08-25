"""
Per-series statistical tier: Holt-Winters exponential smoothing, with a
graceful downgrade path for series that don't have enough history for a
seasonal fit. This tier exists to give the global LightGBM model (which
pools information across all series) a per-series-specialized competitor —
if a store/liquor_type series has an idiosyncratic pattern the pooled
model smooths away, this tier is where that should show up as a win.

SARIMAX is available for series where Holt-Winters' additive structure is
a poor fit (see `sarimax_forecast`), but Holt-Winters is the default: it's
faster to fit ~200 times per backtest fold and, empirically on this
dataset, no worse for weekly bottle counts with a strong single seasonal
period.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def exponential_smoothing_forecast(history: pd.Series, horizon: int, season_length: int = 52) -> np.ndarray:
    values = history.to_numpy(dtype=float)
    # Additive trend clips at zero for series that are trending down toward
    # zero bottles/week; multiplicative seasonality needs strictly positive
    # values, which a real zero-sale week violates — additive throughout.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(values) >= 2 * season_length:
            try:
                model = ExponentialSmoothing(
                    values,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=season_length,
                    initialization_method="estimated",
                ).fit(optimized=True)
                forecast = model.forecast(horizon)
            except (ValueError, np.linalg.LinAlgError):
                forecast = _damped_trend_only(values, horizon)
        else:
            forecast = _damped_trend_only(values, horizon)
    return np.clip(forecast, a_min=0.0, a_max=None)


def _damped_trend_only(values: np.ndarray, horizon: int) -> np.ndarray:
    """Fallback for series too short for a reliable seasonal fit: trend
    only, damped so a short run of growth/decline doesn't get extrapolated
    unboundedly across the whole horizon."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            values, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
        return model.forecast(horizon)


def sarimax_forecast(
    history: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 52),
) -> np.ndarray:
    values = history.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            values,
            order=order,
            seasonal_order=seasonal_order if len(values) >= 2 * seasonal_order[3] else (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        forecast = model.forecast(horizon)
    return np.clip(forecast, a_min=0.0, a_max=None)
