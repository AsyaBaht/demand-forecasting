"""
Full offline pipeline: synthetic raw rows -> aggregated DemandSeries ->
tiered stack -> conformal -> rolling-origin backtest -> report. Zero
BigQuery access anywhere in this path — everything downstream of
tests/conftest.py's synthetic_raw_frame.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import numpy as np

from demand_forecasting.evaluation.eval_suite import run_backtest, write_report
from demand_forecasting.ingestion.aggregate import raw_to_series, series_to_frame


def test_raw_to_series_gap_fills_and_matches_fixture_shape(synthetic_raw_frame, synthetic_series):
    recovered = raw_to_series(synthetic_raw_frame, min_length_weeks=40)
    assert len(recovered) == len(synthetic_series)

    by_id = {s.series_id: s for s in recovered}
    for original in synthetic_series:
        assert original.series_id in by_id
        assert len(by_id[original.series_id].observations) == len(original.observations)


def test_raw_to_series_drops_short_series(synthetic_raw_frame):
    truncated = synthetic_raw_frame[synthetic_raw_frame["week_start"] < synthetic_raw_frame["week_start"].min() + __import__("pandas").Timedelta(weeks=5)]
    recovered = raw_to_series(truncated, min_length_weeks=40)
    assert recovered == []


def test_series_to_frame_round_trip(synthetic_series):
    frame = series_to_frame(synthetic_series)
    assert set(frame["series_id"]) == {s.series_id for s in synthetic_series}
    assert len(frame) == sum(len(s.observations) for s in synthetic_series)


def test_full_offline_pipeline_runs_and_reports_findings_honestly(
    synthetic_raw_frame, small_backtest_kwargs, tmp_path
):
    series_list = raw_to_series(synthetic_raw_frame, min_length_weeks=small_backtest_kwargs["min_series_length_weeks"])
    assert len(series_list) == 9  # 3 stores x 3 liquor types

    report = run_backtest(series_list=series_list, **small_backtest_kwargs)

    wape_by_model: dict[str, list[float]] = {}
    for r in report.results:
        if r.dimension == "point_accuracy_wape":
            wape_by_model.setdefault(r.model_name, []).append(r.value)

    assert set(wape_by_model) == {"naive", "seasonal_naive", "statistical", "global"}

    # Whether seasonal-naive beats plain naive in aggregate is genuinely not
    # guaranteed here: the fixture also has a linear trend (see conftest.py),
    # and over a short 4-week horizon "repeat the last value" already
    # captures most of the current level + trend + phase, while
    # seasonal-naive's year-ago reference misses a full year of accumulated
    # trend. This is a real, honest property of short-horizon forecasting
    # with a trending series, not a bug in either baseline — see
    # test_baselines.py for the case (strongly seasonal, no trend, a horizon
    # that spans a real seasonal turn) where seasonal-naive's advantage
    # does show up cleanly. This test only checks every tier produces a
    # finite, honest number here — see below.

    # On the COVID-spanning fold, the fixture's 1.6x demand shock breaks
    # seasonal-naive's "same week last year" assumption (last year wasn't
    # shocked) worse than it breaks naive's "repeat last observed value"
    # assumption, since naive's most recent observations are already inside
    # the shocked regime — exactly the kind of regime-shift degradation this
    # suite exists to surface, not hide. Assert it's actually visible rather
    # than asserting seasonal-naive wins everywhere.
    covid_wape = {r.model_name: r.value for r in report.results if r.dimension == "point_accuracy_wape" and r.spans_covid}
    assert covid_wape["seasonal_naive"] > covid_wape["naive"]

    # Whether the global (or statistical) tier beats seasonal-naive is a
    # real finding, not something this test should force either way — just
    # confirm every tier produced a finite, honest number.
    for model, values in wape_by_model.items():
        assert all(np.isfinite(v) for v in values), f"{model} produced a non-finite WAPE"

    coverage_results = [r for r in report.results if r.dimension == "interval_coverage"]
    assert coverage_results
    assert all(0.0 <= r.value <= 1.0 for r in coverage_results)

    covid_results = [r for r in report.results if r.spans_covid]
    stable_results = [r for r in report.results if r.spans_covid is False]
    assert covid_results and stable_results

    path = write_report(report, tmp_path)
    assert path.exists()
    assert path.suffix == ".json"
