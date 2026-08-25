from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.conformal import ConformalCalibrator, empirical_coverage
from demand_forecasting.schemas import ConformalInterval, ForecastPoint, ForecastResult


def _residual_frame(n: int, scale: float, group: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    point_forecast = rng.uniform(50, 150, size=n)
    actual = point_forecast + rng.normal(0, scale, size=n)
    return pd.DataFrame({"actual": actual, "point_forecast": point_forecast, "liquor_type": group})


def test_calibration_achieves_close_to_target_coverage_on_held_out_data():
    alpha = 0.1
    calib = _residual_frame(2000, scale=5.0, group="whiskey", seed=1)
    test = _residual_frame(2000, scale=5.0, group="whiskey", seed=2)

    calibrator = ConformalCalibrator(alpha=alpha)
    calibrator.calibrate(calib, group_col="liquor_type")

    hits = 0
    for _, row in test.iterrows():
        lower, upper = calibrator.interval(row["point_forecast"], group="whiskey")
        hits += int(lower <= row["actual"] <= upper)
    coverage = hits / len(test)

    # split conformal's guarantee is marginal and finite-sample, not exact —
    # allow a few points of slack either side of the 90% target.
    assert 0.87 <= coverage <= 0.93


def test_grouped_calibration_produces_different_widths_for_different_scales():
    calib = pd.concat(
        [
            _residual_frame(200, scale=2.0, group="rum", seed=3),
            _residual_frame(200, scale=20.0, group="whiskey", seed=4),
        ]
    )
    calibrator = ConformalCalibrator(alpha=0.1, min_group_size=20)
    calibrator.calibrate(calib, group_col="liquor_type")

    rum_lower, rum_upper = calibrator.interval(100.0, group="rum")
    whiskey_lower, whiskey_upper = calibrator.interval(100.0, group="whiskey")

    assert (whiskey_upper - whiskey_lower) > (rum_upper - rum_lower)


def test_small_group_falls_back_to_global_quantile():
    calib = pd.concat(
        [
            _residual_frame(500, scale=5.0, group="common", seed=5),
            _residual_frame(3, scale=5.0, group="rare", seed=6),  # below min_group_size
        ]
    )
    calibrator = ConformalCalibrator(alpha=0.1, min_group_size=20)
    calibrator.calibrate(calib, group_col="liquor_type")

    assert "rare" not in calibrator.group_q_hat
    lower, upper = calibrator.interval(100.0, group="rare")
    fallback_lower, fallback_upper = calibrator.interval(100.0, group=None)
    assert (lower, upper) == (fallback_lower, fallback_upper)


def test_interval_never_goes_negative_for_low_point_forecasts():
    calib = _residual_frame(500, scale=50.0, group="vodka", seed=7)
    calibrator = ConformalCalibrator(alpha=0.1)
    calibrator.calibrate(calib, group_col="liquor_type")
    lower, _ = calibrator.interval(1.0, group="vodka")
    assert lower >= 0.0


def test_wrap_produces_one_interval_per_forecast_point():
    calib = _residual_frame(500, scale=5.0, group="gin", seed=8)
    calibrator = ConformalCalibrator(alpha=0.1)
    calibrator.calibrate(calib, group_col="liquor_type")

    forecast = ForecastResult(
        series_id="10::gin",
        model_name="global",
        points=[
            ForecastPoint(week_start=date(2024, 1, 1), point_forecast=100.0),
            ForecastPoint(week_start=date(2024, 1, 8), point_forecast=110.0),
        ],
    )
    intervals = calibrator.wrap(forecast, liquor_type="gin")
    assert len(intervals) == 2
    assert all(isinstance(i, ConformalInterval) for i in intervals)
    assert all(i.lower <= i.point_forecast <= i.upper for i in intervals)


def test_empirical_coverage_counts_hits_correctly():
    intervals = [
        ConformalInterval(
            series_id="10::gin", week_start=date(2024, 1, 1), model_name="global",
            point_forecast=100.0, lower=90.0, upper=110.0, alpha=0.1,
        ),
        ConformalInterval(
            series_id="10::gin", week_start=date(2024, 1, 8), model_name="global",
            point_forecast=100.0, lower=90.0, upper=110.0, alpha=0.1,
        ),
    ]
    actuals = {("10::gin", "2024-01-01"): 105.0, ("10::gin", "2024-01-08"): 500.0}
    assert empirical_coverage(intervals, actuals) == 0.5


def test_calibrate_raises_before_use():
    calibrator = ConformalCalibrator(alpha=0.1)
    with pytest.raises(RuntimeError):
        calibrator.interval(100.0)
