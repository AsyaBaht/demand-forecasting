from __future__ import annotations

import numpy as np
import pandas as pd

from demand_forecasting.evaluation.eval_suite import wape
from demand_forecasting.models.baselines import naive_forecast, seasonal_naive_forecast


def test_naive_forecast_repeats_last_value():
    history = pd.Series([10.0, 20.0, 30.0, 15.0])
    forecast = naive_forecast(history, horizon=5)
    assert forecast.shape == (5,)
    assert np.all(forecast == 15.0)


def test_seasonal_naive_recovers_exact_periodic_pattern():
    # three perfect cycles of period 4: the next cycle should be predicted exactly.
    cycle = [10.0, 20.0, 30.0, 40.0]
    history = pd.Series(cycle * 3)
    forecast = seasonal_naive_forecast(history, horizon=4, season_length=4)
    assert np.allclose(forecast, cycle)


def test_seasonal_naive_falls_back_to_naive_without_enough_history():
    history = pd.Series([5.0, 6.0, 7.0])
    forecast = seasonal_naive_forecast(history, horizon=3, season_length=52)
    # season_length steps back would reach before the series starts for every step
    assert np.all(forecast == 7.0)


def test_seasonal_naive_beats_naive_on_strongly_seasonal_series():
    # one noisy sine cycle repeated, matching the shape of the real fixture's
    # seasonal component: seasonal-naive should have much lower WAPE than
    # naive against the true continuation.
    rng = np.random.default_rng(0)
    t = np.arange(104)
    true_values = 100 + 30 * np.sin(2 * np.pi * t / 52)
    noisy = true_values + rng.normal(0, 2, size=104)
    history = pd.Series(noisy[:100])
    actual_continuation = true_values[100:104]

    naive_fc = naive_forecast(history, horizon=4)
    seasonal_fc = seasonal_naive_forecast(history, horizon=4, season_length=52)

    assert wape(actual_continuation, seasonal_fc) < wape(actual_continuation, naive_fc)
