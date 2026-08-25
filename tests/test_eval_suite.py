from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from demand_forecasting.evaluation.eval_suite import (
    EvalReport,
    generate_folds,
    rmse,
    run_backtest,
    wape,
    write_report,
)
from demand_forecasting.schemas import DemandObservation, DemandSeries, EvalResult


def test_wape_basic():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 40.0])
    assert wape(y_true, y_pred) == pytest.approx(10.0 / 60.0)


def test_wape_zero_actuals_is_nan():
    assert np.isnan(wape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))


def test_rmse_basic():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx((3.0**2 + 4.0**2) ** 0.5 / (2**0.5))


def _flat_series(store: int, liquor: str, start: date, n_weeks: int, value: float = 50.0) -> DemandSeries:
    import pandas as pd

    weeks = pd.date_range(start, periods=n_weeks, freq="W-MON")
    return DemandSeries(
        store_number=store,
        liquor_type=liquor,
        observations=[DemandObservation(week_start=w.date(), bottles_sold=value) for w in weeks],
    )


def test_generate_folds_chronological_order_and_counts():
    series = [_flat_series(1, "whiskey", date(2019, 1, 7), 100)]
    folds = generate_folds(
        series, horizon_weeks=4, n_folds=3, step_weeks=10, covid_start=date(2020, 3, 1), covid_end=date(2020, 12, 31)
    )
    assert len(folds) == 3
    assert [f.fold_id for f in folds] == [0, 1, 2]
    # chronological: each fold's train_end should be earlier than the next's
    assert folds[0].train_end < folds[1].train_end < folds[2].train_end
    for fold in folds:
        assert len(fold.horizon_dates) == 4
        assert fold.horizon_dates == sorted(fold.horizon_dates)
        assert fold.train_end < fold.horizon_dates[0]


def test_generate_folds_flags_covid_spanning_fold():
    # 190 weeks from 2019-01-07 crosses the COVID window around week ~60-103
    series = [_flat_series(1, "whiskey", date(2019, 1, 7), 190)]
    folds = generate_folds(
        series,
        horizon_weeks=4,
        n_folds=4,
        step_weeks=40,
        covid_start=date(2020, 3, 1),
        covid_end=date(2020, 12, 31),
    )
    assert any(f.spans_covid for f in folds)
    assert any(not f.spans_covid for f in folds)


def test_generate_folds_empty_series_raises():
    with pytest.raises(ValueError):
        generate_folds([], horizon_weeks=4, n_folds=1, step_weeks=4, covid_start=date(2020, 1, 1), covid_end=date(2020, 2, 1))


def test_eval_report_summary_and_records():
    report = EvalReport(
        results=[
            EvalResult(dimension="point_accuracy_wape", detail="fold 0: naive WAPE=0.5000", value=0.5, fold_id=0),
        ]
    )
    assert "point_accuracy_wape" in report.summary()
    records = report.to_records()
    assert records[0]["value"] == 0.5


def test_write_report_writes_valid_json(tmp_path):
    report = EvalReport(
        results=[EvalResult(dimension="point_accuracy_wape", detail="x", value=0.1, fold_id=0)]
    )
    path = write_report(report, tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["dimension"] == "point_accuracy_wape"


def test_run_backtest_end_to_end(synthetic_series, small_backtest_kwargs):
    report = run_backtest(series_list=synthetic_series, **small_backtest_kwargs)
    assert report.results

    model_names = {r.model_name for r in report.results if r.dimension == "point_accuracy_wape"}
    assert {"naive", "seasonal_naive", "statistical", "global"} <= model_names

    for r in report.results:
        if r.dimension in ("point_accuracy_wape", "point_accuracy_rmse", "interval_coverage"):
            assert r.value is not None
            assert np.isfinite(r.value)

    coverage_results = [r for r in report.results if r.dimension == "interval_coverage"]
    assert coverage_results
    for r in coverage_results:
        assert 0.0 <= r.value <= 1.0

    assert any(r.spans_covid for r in report.results)
