"""
Structured types for the whole pipeline. Every stage — ingestion, each tier
of the model stack, conformal calibration, evaluation — passes typed
objects instead of raw dicts or bare DataFrames, so a mistake (wrong
column, mismatched units, a forecast for the wrong series) fails at
construction time instead of silently propagating into a metric.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DemandObservation(BaseModel):
    week_start: date
    bottles_sold: float = Field(ge=0)


class DemandSeries(BaseModel):
    """One (store_number, liquor_type) time series, weekly bottles sold.
    Observations must be contiguous, gap-filled weeks (see
    ingestion/aggregate.py) — every model downstream assumes no missing
    weeks in the middle of a series."""

    store_number: int
    liquor_type: str
    observations: list[DemandObservation]

    @property
    def series_id(self) -> str:
        return f"{self.store_number}::{self.liquor_type}"

    def as_series(self) -> pd.Series:
        """Weekly bottles_sold indexed by week_start, sorted ascending."""
        idx = pd.DatetimeIndex([o.week_start for o in self.observations])
        vals = [o.bottles_sold for o in self.observations]
        return pd.Series(vals, index=idx, name="bottles_sold").sort_index()

    def truncate(self, through: date) -> DemandSeries:
        """Return a copy containing only observations up to and including
        `through` — the standard operation for carving a rolling-origin
        training window out of a full series."""
        kept = [o for o in self.observations if o.week_start <= through]
        return DemandSeries(store_number=self.store_number, liquor_type=self.liquor_type, observations=kept)


class ForecastPoint(BaseModel):
    week_start: date
    point_forecast: float


class ForecastResult(BaseModel):
    """Output of any single tier (naive, statistical, global) for one series."""

    series_id: str
    model_name: str
    points: list[ForecastPoint]

    def as_series(self) -> pd.Series:
        idx = pd.DatetimeIndex([p.week_start for p in self.points])
        vals = [p.point_forecast for p in self.points]
        return pd.Series(vals, index=idx, name="point_forecast").sort_index()


class ConformalInterval(BaseModel):
    """A calibrated prediction interval wrapping one tier's point forecast.
    `alpha` is the target miscoverage rate — a well-calibrated interval
    should contain the true value in roughly (1 - alpha) of cases."""

    series_id: str
    week_start: date
    model_name: str
    point_forecast: float
    lower: float
    upper: float
    alpha: float

    model_config = ConfigDict(protected_namespaces=())

    def covers(self, actual: float) -> bool:
        return self.lower <= actual <= self.upper


class EvalResult(BaseModel):
    """One scored line item from the eval suite — a metric value plus
    enough context (dimension, fold, whether the fold spans COVID) to
    build the backtest report from a flat list of these."""

    dimension: str
    detail: str
    value: float | None = None
    fold_id: int | None = None
    model_name: str | None = None
    spans_covid: bool | None = None
