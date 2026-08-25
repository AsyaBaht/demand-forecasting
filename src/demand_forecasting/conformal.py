"""
Split conformal prediction wrapping the tiered stack's point forecasts in
calibrated intervals.

This is the piece the rest of the stack exists to feed: a point forecast
alone can't tell a planner how much safety stock to hold, and a naive
"+/- 2 standard deviations" interval only means what it claims to mean if
the residuals are normally distributed and homoscedastic — neither of
which is a safe assumption for bottle counts across stores that differ by
two orders of magnitude in volume. Split conformal makes no distributional
assumption: calibrate a nonconformity quantile on held-out residuals, and
the resulting interval has (very close to, given finite-sample
correction) the target marginal coverage by construction.

Calibration is grouped by liquor_type rather than pooled globally, because
residual scale varies enormously across categories (whiskey vs.
cordial_liqueur) — a single pooled quantile would be too wide for
low-volume categories and too narrow for high-volume ones. A group with
too few calibration points falls back to the global quantile rather than
overfitting an unstable per-group interval width.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from demand_forecasting.schemas import ConformalInterval, ForecastResult


def _conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    n = len(abs_residuals)
    if n == 0:
        raise ValueError("cannot calibrate a conformal quantile from zero residuals")
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(abs_residuals, level, method="higher"))


@dataclass
class ConformalCalibrator:
    alpha: float
    min_group_size: int = 20
    group_q_hat: dict[str, float] = field(default_factory=dict)
    fallback_q_hat: float | None = None

    def calibrate(self, residual_frame: pd.DataFrame, group_col: str | None = "liquor_type") -> None:
        """residual_frame needs columns `actual` and `point_forecast`, plus
        `group_col` if grouping is requested."""
        abs_resid = (residual_frame["actual"] - residual_frame["point_forecast"]).abs().to_numpy()
        self.fallback_q_hat = _conformal_quantile(abs_resid, self.alpha)

        self.group_q_hat = {}
        if group_col is not None:
            for group, sub in residual_frame.groupby(group_col):
                if len(sub) >= self.min_group_size:
                    sub_abs = (sub["actual"] - sub["point_forecast"]).abs().to_numpy()
                    self.group_q_hat[group] = _conformal_quantile(sub_abs, self.alpha)

    def interval(self, point_forecast: float, group: str | None = None) -> tuple[float, float]:
        if self.fallback_q_hat is None:
            raise RuntimeError("ConformalCalibrator must be calibrated before producing intervals")
        q = self.group_q_hat.get(group, self.fallback_q_hat) if group is not None else self.fallback_q_hat
        return max(point_forecast - q, 0.0), point_forecast + q

    def wrap(self, forecast: ForecastResult, liquor_type: str | None = None) -> list[ConformalInterval]:
        intervals = []
        for point in forecast.points:
            lower, upper = self.interval(point.point_forecast, group=liquor_type)
            intervals.append(
                ConformalInterval(
                    series_id=forecast.series_id,
                    week_start=point.week_start,
                    model_name=forecast.model_name,
                    point_forecast=point.point_forecast,
                    lower=lower,
                    upper=upper,
                    alpha=self.alpha,
                )
            )
        return intervals


def empirical_coverage(intervals: list[ConformalInterval], actuals: dict[tuple[str, str], float]) -> float:
    """actuals keyed by (series_id, isoformat week_start) -> true value."""
    if not intervals:
        raise ValueError("cannot compute coverage over zero intervals")
    hits = 0
    for interval in intervals:
        key = (interval.series_id, interval.week_start.isoformat())
        if key not in actuals:
            continue
        hits += int(interval.covers(actuals[key]))
    return hits / len(intervals)
