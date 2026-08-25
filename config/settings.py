"""
Externalized configuration — every knob that affects what gets extracted,
how series are aggregated, or how the eval suite is scoped lives here, not
scattered across scripts as magic numbers.

Override any field with an env var prefixed `DEMAND_FORECASTING_`, e.g.
`DEMAND_FORECASTING_TOP_N_STORES=10`, or via a `.env` file in the repo root.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

from datetime import date

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEMAND_FORECASTING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- BigQuery source ---
    gcp_project_id: str = "bigquery-public-data"
    bq_dataset: str = "iowa_liquor_sales"
    bq_table: str = "sales"

    # --- extraction scope ---
    # 2019-onward keeps enough pre-COVID history for seasonal-naive and
    # SARIMAX baselines to have a real year-over-year comparison, while
    # skipping the dataset's sparse/inconsistent 2012-2018 early years.
    start_date: date = date(2019, 1, 1)
    top_n_stores: int = 25

    # category code -> liquor_type grouping. Iowa's category codes are far
    # more granular than this (dozens of whiskey sub-codes alone); grouping
    # by the first three digits collapses them into buckets with enough
    # volume per store to actually forecast, instead of 200+ near-empty
    # per-SKU series.
    category_prefix_map: dict[str, str] = {
        "101": "whiskey",
        "102": "tequila",
        "103": "vodka",
        "104": "gin",
        "105": "brandy",
        "106": "rum",
        "108": "cordial_liqueur",
    }
    other_category_label: str = "other"

    # --- aggregation ---
    # Weekly, not daily: daily bottle counts per store/liquor_type are
    # dominated by zero-sale days and day-of-week ordering effects (most
    # Iowa liquor stores don't sell every category every day), which drowns
    # the actual demand signal in noise a forecaster has to learn around
    # for no benefit. Weekly aggregation trades temporal resolution nobody
    # needs (this is a stocking/planning problem, not same-day fulfillment)
    # for a signal-to-noise ratio that makes the statistical and global
    # tiers meaningfully comparable.
    week_start_day: str = "MONDAY"

    # --- modeling ---
    lag_weeks: list[int] = [1, 2, 3, 4, 8, 52]
    rolling_windows: list[int] = [4, 8, 12]
    min_series_length_weeks: int = 26

    # --- conformal prediction ---
    conformal_alpha: float = 0.10  # 1 - alpha = target coverage, e.g. 90%
    conformal_calibration_frac: float = 0.2

    # --- rolling-origin backtesting ---
    backtest_horizon_weeks: int = 8
    backtest_n_folds: int = 6
    backtest_step_weeks: int = 8

    # COVID stress-test window used to split backtest folds into
    # "spans regime shift" vs. "doesn't", per fold's horizon dates.
    covid_window_start: date = date(2020, 3, 1)
    covid_window_end: date = date(2020, 12, 31)

    # --- paths ---
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    eval_reports_dir: str = "reports/eval_runs"
    sql_path: str = "sql/extract_demand.sql"


settings = Settings()
