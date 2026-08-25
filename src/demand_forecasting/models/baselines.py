"""
Naive baselines. These aren't scaffolding to delete once "real" models
exist — they're the reference frame the statistical and global tiers have
to actually beat. A model that loses to seasonal-naive on liquor sales
(which has a strong, well-known weekly-seasonal pattern around holidays)
isn't adding value, no matter how sophisticated it is.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every step of the horizon."""
    if len(history) == 0:
        raise ValueError("naive_forecast requires at least one observation")
    return np.full(horizon, float(history.iloc[-1]), dtype=float)


def seasonal_naive_forecast(history: pd.Series, horizon: int, season_length: int = 52) -> np.ndarray:
    """Forecast step h as the value observed `season_length` steps before
    it — for weekly data with season_length=52, "same week last year."
    Falls back to the plain naive value for any step that would need to
    reach further back than the available history."""
    if len(history) == 0:
        raise ValueError("seasonal_naive_forecast requires at least one observation")
    values = history.to_numpy(dtype=float)
    n = len(values)
    last_value = values[-1]
    out = np.empty(horizon, dtype=float)
    for h in range(horizon):
        source_idx = n + h - season_length
        out[h] = values[source_idx] if source_idx >= 0 else last_value
    return out
