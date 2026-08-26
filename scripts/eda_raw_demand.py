"""
Exploratory data analysis on the raw BigQuery extract
(data/raw/demand_raw.parquet — week_start, store_number, liquor_type,
bottles_sold, one row per non-zero-sale week per series). This runs
*before* ingestion/aggregate.py's gap-filling, so it's also where data
quality issues (negative bottle counts from return-heavy weeks, missing
weeks, duplicate rows) should actually be caught — aggregate.py assumes
they've already been looked at, not that it will catch them itself.

Standalone script, not part of the installed package: it's a one-off
analysis tool, not something the forecasting pipeline imports or the test
suite covers. Needs a real raw extract to run against — see README.md for
how to produce one (`demand-forecasting extract --project ...`).

Usage:
    python scripts/eda_raw_demand.py
    python scripts/eda_raw_demand.py --input data/raw/demand_raw.parquet --output-dir reports/eda

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import settings

SECTION_RULE = "=" * 78


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw extract parquet. BigQuery's DATE columns round-trip
    through the `db-dtypes` package's "dbdate" pandas extension type, which
    pandas can't reconstruct unless db-dtypes happens to be imported first —
    so instead of requiring that (and the rest of the bigquery extra) just
    to run an EDA script, cast the column to a plain pyarrow date32 and
    strip the stale pandas metadata that would otherwise try to rebuild it."""
    table = pq.read_table(path)
    week_start_idx = table.schema.get_field_index("week_start")
    if week_start_idx == -1:
        raise ValueError(f"{path} has no 'week_start' column — is this the raw extract file?")
    table = table.set_column(week_start_idx, "week_start", table.column("week_start").cast(pa.date32()))
    table = table.replace_schema_metadata(None)

    df = table.to_pandas()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["store_number"] = df["store_number"].astype(int)
    df["liquor_type"] = df["liquor_type"].astype(str)
    df["bottles_sold"] = df["bottles_sold"].astype(float)
    return df


def schema_overview(df: pd.DataFrame) -> str:
    lines = [
        "SCHEMA & COVERAGE",
        f"  rows: {len(df):,}",
        (
            f"  date range: {df['week_start'].min().date()} to {df['week_start'].max().date()} "
            f"({(df['week_start'].max() - df['week_start'].min()).days // 7} weeks)"
        ),
        f"  distinct stores: {df['store_number'].nunique()}",
        f"  liquor types ({df['liquor_type'].nunique()}): {', '.join(sorted(df['liquor_type'].unique()))}",
        f"  distinct (store, liquor_type) series: {df.groupby(['store_number', 'liquor_type']).ngroups}",
    ]
    return "\n".join(lines)


def data_quality_checks(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Nulls, duplicate (store, liquor_type, week) rows, and negative/zero
    bottles_sold. The extraction query aggregates to one row per
    (week, store, liquor_type), so any duplicates here would mean the
    query or a re-run of it double-counted something."""
    nulls = df.isna().sum()
    dupes = df.duplicated(subset=["week_start", "store_number", "liquor_type"]).sum()
    negative_rows = df[df["bottles_sold"] < 0].sort_values("bottles_sold")
    zero_rows = (df["bottles_sold"] == 0).sum()

    lines = [
        "DATA QUALITY",
        f"  null values: {'none' if nulls.sum() == 0 else nulls[nulls > 0].to_dict()}",
        f"  duplicate (store, liquor_type, week) rows: {dupes}",
        (
            f"  zero-bottles-sold rows: {zero_rows} ({zero_rows / len(df):.2%}) — expected to be rare here, "
            "since a true zero-sale week produces no row at all in this extract; a zero row means the "
            "store recorded a transaction that net to exactly zero bottles."
        ),
        (
            f"  negative bottles_sold rows: {len(negative_rows)} ({len(negative_rows) / len(df):.3%}) — "
            "returns/corrections exceeding that week's sales. Real, not a bug: ingestion/aggregate.py "
            "does not clip these, so a downstream model can see them."
        ),
    ]
    if len(negative_rows):
        lines.append("  most negative rows:")
        lines.append(_indent(negative_rows.head(5).to_string(index=False)))
    return "\n".join(lines), negative_rows


def descriptive_stats(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    overall = df["bottles_sold"].describe()
    by_type = df.groupby("liquor_type")["bottles_sold"].describe().sort_values("mean", ascending=False)
    lines = [
        "DESCRIPTIVE STATISTICS",
        "  overall bottles_sold:",
        _indent(overall.to_string()),
        "",
        "  by liquor_type (sorted by mean):",
        _indent(by_type.round(1).to_string()),
    ]
    return "\n".join(lines), by_type


def detect_outliers(df: pd.DataFrame, iqr_multiplier: float) -> tuple[str, pd.DataFrame]:
    """Tukey-fence outliers, computed per liquor_type since bottles-sold
    scale differs by an order of magnitude or more across categories — a
    single global IQR would flag half of whiskey as outliers and miss
    every real spike in a low-volume category like cordial_liqueur."""

    def _flag(group: pd.DataFrame) -> pd.Series:
        q1, q3 = group["bottles_sold"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        return (group["bottles_sold"] < lower) | (group["bottles_sold"] > upper)

    is_outlier = df.groupby("liquor_type", group_keys=False).apply(_flag, include_groups=False)
    outliers = df.loc[is_outlier].copy()
    counts_by_type = outliers.groupby("liquor_type").size().sort_values(ascending=False)

    lines = [
        f"OUTLIERS (Tukey fence, {iqr_multiplier}x IQR, per liquor_type)",
        f"  total flagged: {len(outliers):,} of {len(df):,} rows ({len(outliers) / len(df):.2%})",
        "  by liquor_type:",
        _indent(counts_by_type.to_string()) if len(counts_by_type) else "    none",
    ]
    if len(outliers):
        top_outliers = outliers.reindex(outliers["bottles_sold"].abs().sort_values(ascending=False).index).head(10)
        lines.append("  10 most extreme outlier rows:")
        lines.append(_indent(top_outliers.to_string(index=False)))
    return "\n".join(lines), outliers


def monthly_view(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Aggregate weekly bottles_sold into calendar months, both overall and
    per liquor_type — the view that actually shows seasonality (December
    spikes, etc.) and the COVID-era demand shift, without the week-to-week
    noise of the raw series."""
    monthly = df.copy()
    monthly["month"] = monthly["week_start"].dt.to_period("M")
    total_by_month = monthly.groupby("month")["bottles_sold"].sum()
    by_type_by_month = monthly.pivot_table(
        index="month", columns="liquor_type", values="bottles_sold", aggfunc="sum", fill_value=0
    )
    mom_pct_change = total_by_month.pct_change() * 100

    top3 = total_by_month.sort_values(ascending=False).head(3)
    bottom3 = total_by_month.sort_values().head(3)
    biggest_swings = mom_pct_change.abs().sort_values(ascending=False).head(5)

    # The first and last calendar-month buckets are frequently partial: a
    # Monday-truncated week can pull a few days from the previous/next
    # month in (e.g. the week of 2019-01-01 starts Monday 2018-12-31, so
    # "2018-12" gets one day's worth of one week). Flag it rather than let
    # a partial month read as a real demand low.
    weeks_per_month = monthly.groupby("month")["week_start"].nunique()
    typical_weeks = weeks_per_month.iloc[1:-1].median() if len(weeks_per_month) > 2 else weeks_per_month.median()
    partial_months = weeks_per_month[weeks_per_month < typical_weeks * 0.5]

    lines = [
        "MONTHLY VIEW",
        f"  {len(total_by_month)} calendar months, total bottles_sold {total_by_month.sum():,.0f}",
        "  highest-volume months:",
        _indent(top3.to_string()),
        "  lowest-volume months:",
        _indent(bottom3.to_string()),
        "  largest month-over-month % swings (up or down):",
        _indent(mom_pct_change.loc[biggest_swings.index].round(1).to_string()),
    ]
    if len(partial_months):
        lines.append(
            "  NOTE: partial calendar-month buckets detected (fewer than half a typical month's weeks) — "
            "a week-boundary artifact of Monday-truncation at the start/end of the extraction range, not a "
            "real demand signal. Discount these before reading the highest/lowest lists above:"
        )
        lines.append(_indent(partial_months.rename("weeks_present").to_string()))
    return "\n".join(lines), by_type_by_month


def store_concentration(df: pd.DataFrame, top_n: int = 5) -> str:
    by_store = df.groupby("store_number")["bottles_sold"].sum().sort_values(ascending=False)
    total = by_store.sum()
    top = by_store.head(top_n)
    lines = [
        "STORE CONCENTRATION",
        f"  top {top_n} of {len(by_store)} stores hold {top.sum() / total:.1%} of total volume:",
        _indent((top.to_frame("bottles_sold").assign(share=lambda d: d['bottles_sold'] / total)).round(3).to_string()),
    ]
    return "\n".join(lines)


def series_completeness(df: pd.DataFrame, min_length_weeks: int) -> str:
    """How many observed (not gap-filled) weeks each (store, liquor_type)
    series actually has, vs. the calendar span it covers — a series with a
    short calendar span or a lot of zero-sale gaps between its first and
    last observed week is one that would get dropped or heavily gap-filled
    by ingestion/aggregate.py."""
    per_series = df.groupby(["store_number", "liquor_type"]).agg(
        observed_weeks=("week_start", "count"),
        first_week=("week_start", "min"),
        last_week=("week_start", "max"),
    )
    per_series["calendar_span_weeks"] = ((per_series["last_week"] - per_series["first_week"]).dt.days // 7) + 1
    per_series["fill_rate"] = per_series["observed_weeks"] / per_series["calendar_span_weeks"]
    below_threshold = (per_series["calendar_span_weeks"] < min_length_weeks).sum()

    lines = [
        "SERIES COMPLETENESS",
        f"  {len(per_series)} (store, liquor_type) series",
        (
            f"  calendar span (weeks): min={per_series['calendar_span_weeks'].min()}, "
            f"median={per_series['calendar_span_weeks'].median():.0f}, "
            f"max={per_series['calendar_span_weeks'].max()}"
        ),
        (
            f"  fill rate (observed / calendar span): min={per_series['fill_rate'].min():.2f}, "
            f"median={per_series['fill_rate'].median():.2f}"
        ),
        (
            f"  series shorter than min_series_length_weeks ({min_length_weeks}): {below_threshold} "
            "— these would be dropped by ingestion/aggregate.raw_to_series."
        ),
    ]
    return "\n".join(lines)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def make_plots(df: pd.DataFrame, monthly_by_type: pd.DataFrame, outliers: pd.DataFrame, out_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plots; `pip install -e '.[eda]'` to enable them)")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Monthly total volume trend, with the COVID window shaded.
    fig, ax = plt.subplots(figsize=(11, 4))
    total_by_month = monthly_by_type.sum(axis=1)
    x = total_by_month.index.to_timestamp()
    ax.plot(x, total_by_month.values, marker="o", markersize=3)
    ax.axvspan(settings.covid_window_start, settings.covid_window_end, color="orange", alpha=0.15, label="COVID window")
    ax.set_title("Total bottles sold by month, all stores/types")
    ax.set_ylabel("bottles sold")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "monthly_total.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(path)

    # 2. Monthly volume by liquor_type.
    fig, ax = plt.subplots(figsize=(11, 5))
    for liquor_type in monthly_by_type.columns:
        ax.plot(monthly_by_type.index.to_timestamp(), monthly_by_type[liquor_type], label=liquor_type)
    ax.set_title("Monthly bottles sold by liquor_type")
    ax.set_ylabel("bottles sold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "monthly_by_liquor_type.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(path)

    # 3. Boxplot of bottles_sold by liquor_type (log scale) — outlier shape.
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("liquor_type")["bottles_sold"].median().sort_values(ascending=False).index
    ax.boxplot(
        [df.loc[df["liquor_type"] == lt, "bottles_sold"].clip(lower=1) for lt in order],
        tick_labels=order,
        showfliers=True,
    )
    ax.set_yscale("log")
    ax.set_title("Weekly bottles_sold distribution by liquor_type (log scale)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = out_dir / "boxplot_by_liquor_type.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(path)

    # 4. Top-store volume bar chart.
    fig, ax = plt.subplots(figsize=(9, 5))
    by_store = df.groupby("store_number")["bottles_sold"].sum().sort_values(ascending=False)
    ax.bar(by_store.index.astype(str), by_store.values)
    ax.set_title("Total bottles sold by store")
    ax.set_ylabel("bottles sold")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    path = out_dir / "store_volume.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(path)

    return paths


def build_report(df: pd.DataFrame, input_path: Path, iqr_multiplier: float, min_length_weeks: int) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    sections = [f"EDA REPORT — {input_path}", f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z"]

    schema_section = schema_overview(df)
    quality_section, _negative_rows = data_quality_checks(df)
    stats_section, _ = descriptive_stats(df)
    outlier_section, outliers = detect_outliers(df, iqr_multiplier)
    monthly_section, monthly_by_type = monthly_view(df)
    concentration_section = store_concentration(df)
    completeness_section = series_completeness(df, min_length_weeks)

    for section in (
        schema_section,
        quality_section,
        stats_section,
        outlier_section,
        monthly_section,
        concentration_section,
        completeness_section,
    ):
        sections.append(SECTION_RULE)
        sections.append(section)

    return "\n\n".join(sections), outliers, monthly_by_type


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=f"{settings.raw_data_dir}/demand_raw.parquet", type=Path)
    parser.add_argument("--output-dir", default=settings.eda_reports_dir, type=Path)
    parser.add_argument("--iqr-multiplier", default=settings.outlier_iqr_multiplier, type=float)
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(
            f"{args.input} not found — run `demand-forecasting extract --project YOUR_GCP_PROJECT` first."
        )

    df = load_raw(args.input)
    report_text, outliers, monthly_by_type = build_report(
        df, args.input, args.iqr_multiplier, settings.min_series_length_weeks
    )

    print(report_text)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.output_dir / f"eda_report_{timestamp}.txt"
    report_path.write_text(report_text)

    plot_paths = make_plots(df, monthly_by_type, outliers, args.output_dir)

    print(f"\n{SECTION_RULE}")
    print(f"Wrote report to {report_path}")
    for p in plot_paths:
        print(f"Wrote plot to {p}")


if __name__ == "__main__":
    main()
