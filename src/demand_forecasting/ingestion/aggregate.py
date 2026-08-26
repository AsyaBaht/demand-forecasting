"""
Raw extraction rows -> one DemandSeries per (store_number, liquor_type).

The extraction query already aggregates to week/store/liquor_type, so the
real work here is gap-filling: a store that sells zero bottles of gin in a
given week doesn't produce a row at all, but a zero-sale week is a real
observation, not missing data, and every downstream model (lag features,
seasonal windows) assumes a contiguous weekly index. Series that are too
short to be worth forecasting (a store/category pair that only shows up
for a few months) are dropped rather than gap-filled across a mostly-empty
history.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from config.settings import settings

from demand_forecasting.schemas import DemandObservation, DemandSeries

REQUIRED_COLUMNS = {"week_start", "store_number", "liquor_type", "bottles_sold"}


def read_raw_parquet(path: Path) -> pd.DataFrame:
    """Read the raw extract parquet without requiring `db_dtypes` to be
    installed or imported. extract_bigquery.py normalizes week_start to a
    plain datetime64 before writing, so this is mostly a no-op for new
    extracts — but it also reads older files (or ones handed over from
    someone else) that still carry BigQuery's "dbdate" pandas extension
    type, by casting the column to plain pyarrow date32 and stripping the
    stale pandas metadata that would otherwise try (and fail) to
    reconstruct that extension type without `db_dtypes` imported."""
    table = pq.read_table(path)
    week_start_idx = table.schema.get_field_index("week_start")
    if week_start_idx != -1:
        table = table.set_column(week_start_idx, "week_start", table.column("week_start").cast(pa.date32()))
        table = table.replace_schema_metadata(None)
    return table.to_pandas()


def raw_to_series(raw: pd.DataFrame, min_length_weeks: int | None = None) -> list[DemandSeries]:
    """Group raw rows by (store_number, liquor_type), gap-fill each group to
    a contiguous weekly index (zero-filling weeks with no sale), and drop
    any resulting series shorter than min_length_weeks."""
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"raw extract is missing required columns: {sorted(missing)}")

    min_length_weeks = min_length_weeks if min_length_weeks is not None else settings.min_series_length_weeks
    raw = raw.copy()
    raw["week_start"] = pd.to_datetime(raw["week_start"])

    series_list: list[DemandSeries] = []
    for (store_number, liquor_type), group in raw.groupby(["store_number", "liquor_type"], sort=True):
        weekly = group.set_index("week_start")["bottles_sold"].sort_index()
        full_index = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON")
        weekly = weekly.reindex(full_index, fill_value=0.0)

        if len(weekly) < min_length_weeks:
            continue

        observations = [
            DemandObservation(week_start=ts.date(), bottles_sold=float(v)) for ts, v in weekly.items()
        ]
        series_list.append(
            DemandSeries(store_number=int(store_number), liquor_type=str(liquor_type), observations=observations)
        )
    return series_list


def series_to_frame(series_list: list[DemandSeries]) -> pd.DataFrame:
    """Long-format frame (one row per series/week) — convenient for writing
    the processed dataset to disk and for feature engineering in
    models/global_model.py."""
    rows = []
    for s in series_list:
        for o in s.observations:
            rows.append(
                {
                    "series_id": s.series_id,
                    "store_number": s.store_number,
                    "liquor_type": s.liquor_type,
                    "week_start": o.week_start,
                    "bottles_sold": o.bottles_sold,
                }
            )
    return pd.DataFrame(rows)


def load_and_aggregate(raw_path: str | None = None) -> list[DemandSeries]:
    """Read the raw parquet extract and turn it into DemandSeries."""
    path = Path(raw_path or settings.raw_data_dir) / "demand_raw.parquet"
    raw = read_raw_parquet(path)
    return raw_to_series(raw)


def save_processed(series_list: list[DemandSeries], out_path: str | None = None) -> Path:
    """Write the long-format series frame to
    `<out_path or settings.processed_data_dir>/demand_series.parquet`."""
    frame = series_to_frame(series_list)
    out = Path(out_path or settings.processed_data_dir) / "demand_series.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return out
