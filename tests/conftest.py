"""
Synthetic multi-series fixture with a known, fixed data-generating
process — same role the simulator's ground-truth generator plays in the
journey-attribution project. Every test in this suite runs against this
fixture; nothing here touches BigQuery.

The generating process, per (store, liquor_type) series:
  value(t) = base_level
           + base_level * 0.35 * sin(2*pi * week_of_year / 52)   [seasonal]
           + 0.15 * t                                             [trend]
           + noise(t)                                             [N(0, 5% of base)]
  ... then multiplied by 1.6x during the 2020-03-01..2020-12-31 window, a
  deliberate regime shift standing in for the real COVID disruption —
  this is what lets test_pipeline_offline.py exercise the eval suite's
  regime-shift-aware fold scoring without needing the real dataset.

The seasonal component is a strong, exact sinusoid with small noise, so
seasonal-naive reliably beats naive in isolated, no-trend cases (see
test_baselines.py). The full fixture also carries a linear trend and the
COVID shock, both of which can flip that ordering over a short horizon or
a year-later "same week last year" reference — test_pipeline_offline.py
checks for that honestly instead of assuming seasonal-naive always wins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.schemas import DemandObservation, DemandSeries

STORES = [10, 20, 30]
LIQUOR_TYPES = ["whiskey", "vodka", "rum"]
BASE_LEVELS = {
    (10, "whiskey"): 120.0, (10, "vodka"): 90.0, (10, "rum"): 40.0,
    (20, "whiskey"): 200.0, (20, "vodka"): 150.0, (20, "rum"): 60.0,
    (30, "whiskey"): 60.0, (30, "vodka"): 45.0, (30, "rum"): 20.0,
}
SEASONAL_AMPLITUDE_FRAC = 0.35
TREND_PER_WEEK = 0.15
NOISE_FRAC = 0.05
COVID_SHOCK_MULTIPLIER = 1.6

START_DATE = pd.Timestamp("2019-01-07")  # a Monday
N_WEEKS = 190
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2020-12-31")


def _generate_values(base_level: float, seed: int, weeks: pd.DatetimeIndex) -> np.ndarray:
    rng = np.random.default_rng(seed)
    week_of_year = weeks.isocalendar().week.to_numpy(dtype=float)
    t = np.arange(len(weeks), dtype=float)
    seasonal = base_level * SEASONAL_AMPLITUDE_FRAC * np.sin(2 * np.pi * week_of_year / 52)
    trend = TREND_PER_WEEK * t
    noise = rng.normal(0.0, base_level * NOISE_FRAC, size=len(weeks))
    values = base_level + seasonal + trend + noise
    covid_mask = (weeks >= COVID_START) & (weeks <= COVID_END)
    values = np.where(covid_mask, values * COVID_SHOCK_MULTIPLIER, values)
    return np.clip(values, a_min=0.0, a_max=None)


@pytest.fixture
def synthetic_calendar() -> pd.DatetimeIndex:
    return pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")


@pytest.fixture
def synthetic_series(synthetic_calendar: pd.DatetimeIndex) -> list[DemandSeries]:
    weeks = synthetic_calendar
    series_list = []
    for i, store in enumerate(STORES):
        for j, liquor in enumerate(LIQUOR_TYPES):
            base = BASE_LEVELS[(store, liquor)]
            seed = 42 + i * len(LIQUOR_TYPES) + j
            values = _generate_values(base, seed, weeks)
            observations = [
                DemandObservation(week_start=w.date(), bottles_sold=round(float(v), 1))
                for w, v in zip(weeks, values)
            ]
            series_list.append(DemandSeries(store_number=store, liquor_type=liquor, observations=observations))
    return series_list


@pytest.fixture
def synthetic_raw_frame(synthetic_series: list[DemandSeries]) -> pd.DataFrame:
    """Raw-extract-shaped frame — as if it came straight out of BigQuery —
    for exercising ingestion/aggregate.py with zero BigQuery access."""
    rows = [
        {
            "week_start": o.week_start,
            "store_number": s.store_number,
            "liquor_type": s.liquor_type,
            "bottles_sold": o.bottles_sold,
        }
        for s in synthetic_series
        for o in s.observations
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def small_backtest_kwargs() -> dict:
    """A backtest configuration sized for the synthetic fixture: enough
    folds and history to exercise every tier plus a COVID-spanning fold,
    small enough (few weak learners, few folds) to run in seconds."""
    return {
        "lag_weeks": [1, 2, 4, 8],
        "rolling_windows": [4, 8],
        "horizon_weeks": 4,
        "n_folds": 4,
        "step_weeks": 40,
        "conformal_alpha": 0.1,
        "covid_start": COVID_START.date(),
        "covid_end": COVID_END.date(),
        "min_series_length_weeks": 40,
        "lgb_params": {"n_estimators": 40, "num_leaves": 15},
    }
