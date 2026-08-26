"""
Shared loading/constants for the exploration/ pipeline — a separate,
standalone track exploring monthly category-level forecasting (Prophet,
Optuna-tuned LightGBM, auto_arima SARIMAX, a Ridge ensemble, MAPIE
intervals), distinct from the tested production pipeline in
src/demand_forecasting (weekly, per-store-x-liquor_type, naive/Holt-
Winters/global-LightGBM/conformal).

Why a separate track instead of extending the production one: this
exploration works at a different granularity (monthly, not weekly) and a
different aggregation level (per liquor_type category summed across all
25 stores, not per-store), and pulls in a much heavier dependency stack
(prophet, optuna, pmdarima, mapie, shap) that the production CLI has no
reason to carry. Nothing here is covered by pytest or wired into
`demand-forecasting`.

Data reality check: demand_series.parquet has exactly four columns —
series_id, store_number, liquor_type, week_start, bottles_sold. There is
no price, promotion, temperature, payday, unemployment, consumer
confidence, or CPI data anywhere in this project. Those inputs are real
BigQuery public data or real public macro series in principle, but
pulling them in is a deliberate scope decision, not a silent one — see
stage1_eda.py's EDA summary table for which requested inputs are
available here vs. flagged NOT AVAILABLE.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import settings

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# Pandemic stress-test window, per the prompt's "flag pandemic period
# (2020-2021) separately" — wider than the production pipeline's COVID
# window (settings.covid_window_{start,end}, Mar-Dec 2020) since this
# exploration was asked to cover both 2020 and 2021.
PANDEMIC_START = date(2020, 3, 1)
PANDEMIC_END = date(2021, 12, 31)

# Real, publicly documented dates for the requested liquor-relevant US
# holidays that don't fall on a fixed calendar date. Fixed-date holidays
# (New Year's Day, July 4th, Christmas Eve/Day, New Year's Eve) are
# computed directly; Thanksgiving (4th Thursday of November) is computed;
# Super Bowl Sunday has no formula, so its actual historical/scheduled
# dates are hardcoded here for the years this dataset spans.
SUPER_BOWL_DATES = {
    2019: date(2019, 2, 3),
    2020: date(2020, 2, 2),
    2021: date(2021, 2, 7),
    2022: date(2022, 2, 13),
    2023: date(2023, 2, 12),
    2024: date(2024, 2, 11),
    2025: date(2025, 2, 9),
    2026: date(2026, 2, 8),
}


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """weekday: Monday=0 ... Sunday=6. n is 1-indexed (1st, 2nd, ... occurrence)."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return date(year, month, 1 + offset + 7 * (n - 1))


def major_holidays(years: range) -> pd.DataFrame:
    """One row per (year, holiday_name, holiday_date) for the liquor-
    relevant holidays called out in the prompt: New Year's Day, New
    Year's Eve, July 4th, Thanksgiving, Christmas Eve, Christmas, and
    Super Bowl Sunday."""
    rows = []
    for year in years:
        rows.append({"holiday_name": "new_years_day", "date": date(year, 1, 1)})
        rows.append({"holiday_name": "july_4th", "date": date(year, 7, 4)})
        rows.append({"holiday_name": "thanksgiving", "date": _nth_weekday_of_month(year, 11, 3, 4)})
        rows.append({"holiday_name": "christmas_eve", "date": date(year, 12, 24)})
        rows.append({"holiday_name": "christmas", "date": date(year, 12, 25)})
        rows.append({"holiday_name": "new_years_eve", "date": date(year, 12, 31)})
        if year in SUPER_BOWL_DATES:
            rows.append({"holiday_name": "super_bowl_sunday", "date": SUPER_BOWL_DATES[year]})
    return pd.DataFrame(rows)


def load_weekly_frame(path: Path | None = None) -> pd.DataFrame:
    """The processed weekly (store, liquor_type) frame, as written by
    demand_forecasting.ingestion.aggregate.save_processed."""
    path = path or (REPO_ROOT / settings.processed_data_dir / "demand_series.parquet")
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `demand-forecasting aggregate` first "
            "(after `demand-forecasting extract --project YOUR_GCP_PROJECT`)."
        )
    df = pd.read_parquet(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def load_monthly_category_frame(path: Path | None = None) -> pd.DataFrame:
    """Aggregate the weekly per-store series up to monthly, summed across
    all stores within each liquor_type — this exploration's target grain
    ("weekly/monthly unit sales volume by ... category") is category-level
    demand, not per-store demand, so store_number is deliberately summed
    away here rather than kept as a feature."""
    weekly = load_weekly_frame(path)
    weekly["month"] = weekly["week_start"].dt.to_period("M")
    monthly = (
        weekly.groupby(["liquor_type", "month"])["bottles_sold"]
        .sum()
        .reset_index()
        .rename(columns={"bottles_sold": "units_sold"})
    )
    monthly["month_start"] = monthly["month"].dt.to_timestamp()
    return monthly.sort_values(["liquor_type", "month_start"]).reset_index(drop=True)
