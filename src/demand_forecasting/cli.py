"""
Command-line entry point. Three subcommands, each a stage of the pipeline:

    demand-forecasting extract --project YOUR_GCP_PROJECT
    demand-forecasting aggregate
    demand-forecasting backtest

`extract` and `aggregate` require GCP credentials and touch data/raw and
data/processed; `backtest` runs the rolling-origin eval suite against
whatever is in data/processed and writes a timestamped JSON report to
reports/eval_runs/. None of this is exercised by pytest — see
tests/test_pipeline_offline.py for the equivalent run against the
synthetic fixture.
"""
from __future__ import annotations

import argparse
import sys

from config.settings import settings

from demand_forecasting.evaluation.eval_suite import run_backtest, write_report


def cmd_extract(args: argparse.Namespace) -> None:
    from demand_forecasting.ingestion.extract_bigquery import extract_and_save

    path = extract_and_save(project_id=args.project)
    print(f"Wrote raw extract to {path}")


def cmd_aggregate(args: argparse.Namespace) -> None:
    from demand_forecasting.ingestion.aggregate import load_and_aggregate, save_processed

    series_list = load_and_aggregate()
    path = save_processed(series_list)
    print(f"Aggregated {len(series_list)} series -> {path}")


def cmd_backtest(args: argparse.Namespace) -> None:
    from demand_forecasting.ingestion.aggregate import load_and_aggregate

    series_list = load_and_aggregate()
    report = run_backtest(
        series_list=series_list,
        lag_weeks=settings.lag_weeks,
        rolling_windows=settings.rolling_windows,
        horizon_weeks=settings.backtest_horizon_weeks,
        n_folds=settings.backtest_n_folds,
        step_weeks=settings.backtest_step_weeks,
        conformal_alpha=settings.conformal_alpha,
        covid_start=settings.covid_window_start,
        covid_end=settings.covid_window_end,
        min_series_length_weeks=settings.min_series_length_weeks,
    )
    path = write_report(report, settings.eval_reports_dir)
    print(report.summary())
    print(f"\nWrote report to {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="demand-forecasting", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="pull raw data from BigQuery")
    p_extract.add_argument("--project", required=True, help="GCP project ID to bill the query against")
    p_extract.set_defaults(func=cmd_extract)

    p_aggregate = sub.add_parser("aggregate", help="turn raw rows into weekly DemandSeries")
    p_aggregate.set_defaults(func=cmd_aggregate)

    p_backtest = sub.add_parser("backtest", help="run the rolling-origin eval suite")
    p_backtest.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
