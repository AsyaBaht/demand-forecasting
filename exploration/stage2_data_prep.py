"""
Stage 2 — Data preparation for the monthly, category-level forecasting
exploration. Everything here is done SEPARATELY per category (per the
explicit instruction after Stage 1) — no step pools categories together,
even though several operations (the walk-forward split boundaries, the
holiday calendar) happen to land on the same dates for every category
since they share one calendar.

Data reality check carried over from Stage 1: price, promotions,
temperature, payday weeks, unemployment, consumer confidence, and CPI are
still not available in this dataset. Feature engineering below only uses
units_sold (real) and calendar features computed from real US holiday
dates — no external or fabricated inputs. Two other substitutions from
the original weekly-granularity spec, stated plainly:

- The requested weekly lags (1, 2, 4, 8, 12, 52 weeks) don't translate
  literally to monthly data; used here as 1, 2, 3, 6, 12 months instead
  (12 months is the exact analogue of a 52-week lag: same week/month last
  year).
- "is_weekend", "days_to_holiday", "days_after_holiday" are day/week-level
  concepts with no monthly equivalent; substituted with
  months_since_last_holiday / months_until_next_holiday.

Usage:
    python exploration/stage2_data_prep.py
    (or via run_pipeline.py, which passes a PipelineConfig — see pipeline_config.py)

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from common import OUTPUT_ROOT, PANDEMIC_END, PANDEMIC_START, load_monthly_category_frame, major_holidays
from pipeline_config import PipelineConfig
from sklearn.preprocessing import RobustScaler

OUT = OUTPUT_ROOT / "stage2_data_prep"

LAG_MONTHS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]
OUTLIER_WINDOW = 12
OUTLIER_K = 3.0


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def compute_split_boundaries(monthly: pd.DataFrame, test_months: int, val_months: int) -> dict:
    """Walk-forward split boundaries, identical across categories since
    they share one calendar: test = last `test_months` months, validation
    = the `val_months` immediately before that, train = everything
    earlier."""
    all_months = pd.date_range(monthly["month_start"].min(), monthly["month_start"].max(), freq="MS")
    n = len(all_months)
    if n <= test_months + val_months:
        raise ValueError(
            f"only {n} months of data available, but test_months={test_months} + val_months={val_months} "
            f"= {test_months + val_months} leaves no room for any training data"
        )
    test_start = all_months[n - test_months]
    val_start = all_months[n - test_months - val_months]
    train_end = all_months[n - test_months - val_months - 1]
    return {
        "full_range": all_months,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": test_start - pd.DateOffset(months=1),
        "test_start": test_start,
        "test_end": all_months[-1],
        "test_months": test_months,
        "val_months": val_months,
    }


def section_1_handle_missing_values(monthly: pd.DataFrame, boundaries: dict) -> tuple[str, pd.DataFrame]:
    """Reindex each category to a continuous monthly calendar, and for any
    gap: interpolate linearly if <=2 months, else fill from the same
    calendar month one year earlier ("seasonal" fill) if that value
    exists, else fall back to linear. A gap inside the test window is
    never imputed — it's left NaN so the holdout evaluation only ever
    scores real observations, per the "never impute target in test" rule.
    """
    categories = sorted(monthly["liquor_type"].unique())
    full_index = boundaries["full_range"]
    frames, gap_log = [], []

    for category in categories:
        raw = monthly[monthly["liquor_type"] == category].set_index("month_start")["units_sold"].reindex(full_index)
        raw.index.name = "month_start"
        is_missing = raw.isna()
        filled = raw.copy()

        if is_missing.any():
            run_id = (is_missing != is_missing.shift()).cumsum()
            for _, run in raw[is_missing].groupby(run_id[is_missing]):
                run_dates = run.index
                if (run_dates >= boundaries["test_start"]).any():
                    gap_log.append({"liquor_type": category, "n_months": len(run_dates), "start": run_dates[0].date(), "action": "left NaN (test period)"})
                    continue
                if len(run_dates) > 2:
                    prior_year_vals = raw.reindex(run_dates - pd.DateOffset(years=1))
                    if prior_year_vals.notna().all():
                        filled.loc[run_dates] = prior_year_vals.to_numpy()
                        gap_log.append({"liquor_type": category, "n_months": len(run_dates), "start": run_dates[0].date(), "action": "seasonal fill (same month, prior year)"})
                        continue
                    gap_log.append({"liquor_type": category, "n_months": len(run_dates), "start": run_dates[0].date(), "action": "seasonal fill unavailable -> linear interpolation fallback"})
                else:
                    gap_log.append({"liquor_type": category, "n_months": len(run_dates), "start": run_dates[0].date(), "action": "linear interpolation"})

        test_mask = filled.index >= boundaries["test_start"]
        to_interpolate = filled.mask(test_mask)  # protect the test window from interpolation entirely
        interpolated = to_interpolate.interpolate(method="linear", limit_area="inside")
        interpolated[test_mask] = raw[test_mask]  # restore true (possibly still-NaN) test values

        is_imputed = raw.isna() & interpolated.notna()
        frames.append(
            pd.DataFrame(
                {
                    "liquor_type": category,
                    "month_start": interpolated.index,
                    "units_sold": interpolated.values,
                    "is_imputed": is_imputed.values,
                }
            )
        )

    result = pd.concat(frames, ignore_index=True)
    gap_df = pd.DataFrame(gap_log)
    total_imputed = int(result["is_imputed"].sum())
    total_still_missing = int(result["units_sold"].isna().sum())

    lines = [
        "SECTION 1 — MISSING VALUES (per category)",
        "",
        (
            f"  Reindexed every category to the full {len(full_index)}-month calendar "
            f"({full_index[0].date()} to {full_index[-1].date()})."
        ),
        f"  Total imputed values across all categories: {total_imputed}",
        f"  Total still-missing values (all in the test window, left NaN by design): {total_still_missing}",
    ]
    if gap_df.empty:
        lines.append("  No gaps found in any category — every category has a value for every calendar month")
        lines.append("  (expected: totals are summed across 25 stores, so a true all-store zero-report month")
        lines.append("  would be a real, valid zero, not a gap).")
    else:
        lines.append("  Gap log:")
        lines.append(_indent(gap_df.to_string(index=False)))
    return "\n".join(lines), result


def section_2_treat_outliers(filled: pd.DataFrame, boundaries: dict) -> tuple[str, pd.DataFrame]:
    """Clip values beyond `OUTLIER_K`x the rolling IQR from a rolling
    median, computed separately per category. Two carve-outs, both
    deliberate: pandemic-window months are flagged (is_pandemic) but never
    clipped, so the model can still learn the regime shift instead of
    having it smoothed away; and the test window's actual values are never
    modified, since evaluation has to score the model against what really
    happened, not a cleaned version of it.
    """
    frames, clip_log = [], []
    min_periods = max(3, OUTLIER_WINDOW // 2)

    for category, sub in filled.groupby("liquor_type"):
        sub = sub.sort_values("month_start").reset_index(drop=True)
        s = sub["units_sold"]
        rolling_median = s.rolling(OUTLIER_WINDOW, min_periods=min_periods, center=True).median()
        q1 = s.rolling(OUTLIER_WINDOW, min_periods=min_periods, center=True).quantile(0.25)
        q3 = s.rolling(OUTLIER_WINDOW, min_periods=min_periods, center=True).quantile(0.75)
        iqr = q3 - q1
        lower = rolling_median - OUTLIER_K * iqr
        upper = rolling_median + OUTLIER_K * iqr

        is_pandemic = (sub["month_start"].dt.date >= PANDEMIC_START) & (sub["month_start"].dt.date <= PANDEMIC_END)
        is_test = sub["month_start"] >= boundaries["test_start"]
        would_flag = ((s < lower) | (s > upper)) & lower.notna() & upper.notna()
        clip_mask = would_flag & ~is_pandemic & ~is_test

        clipped = s.copy()
        clip_low = clip_mask & (s < lower)
        clip_high = clip_mask & (s > upper)
        clipped[clip_low] = lower[clip_low]
        clipped[clip_high] = upper[clip_high]

        sub = sub.assign(units_sold_raw=s, units_sold=clipped, was_clipped=clip_mask, is_pandemic=is_pandemic)
        clip_log.append(
            {
                "liquor_type": category,
                "n_clipped": int(clip_mask.sum()),
                "n_pandemic_months": int(is_pandemic.sum()),
                "n_pandemic_outliers_left_unclipped": int((would_flag & is_pandemic).sum()),
                "n_test_outliers_left_unclipped": int((would_flag & is_test).sum()),
            }
        )
        frames.append(sub)

    result = pd.concat(frames, ignore_index=True)
    clip_df = pd.DataFrame(clip_log)
    lines = [
        f"SECTION 2 — OUTLIER TREATMENT (rolling median +/- {OUTLIER_K}x rolling IQR, {OUTLIER_WINDOW}-month window, per category)",
        "",
        _indent(clip_df.to_string(index=False)),
        "",
        "  Pandemic-window and test-window points are never clipped even if flagged, by design",
        "  (see docstring) — those counts show how many WOULD have been clipped otherwise.",
    ]
    return "\n".join(lines), result


def section_3_feature_engineering(treated: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Calendar, lag, and rolling-window features, all leakage-guarded
    (every lag/rolling feature is built from `shift(1)` before the rolling
    window is applied, so a row's features only ever see data through the
    previous month — same discipline as the production pipeline's
    global_model.py)."""
    categories = sorted(treated["liquor_type"].unique())
    years = range(treated["month_start"].dt.year.min(), treated["month_start"].dt.year.max() + 1)
    holidays = major_holidays(years)
    holiday_months = pd.DatetimeIndex(sorted(pd.Timestamp(d).replace(day=1) for d in holidays["date"])).unique()

    frames = []
    for category in categories:
        sub = treated[treated["liquor_type"] == category].sort_values("month_start").set_index("month_start")
        df = sub.copy()
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
        df["quarter"] = df.index.quarter
        df["time_index"] = (df.index.year - df.index.year.min()) * 12 + df.index.month

        holidays_per_month = holidays.groupby(pd.PeriodIndex(holidays["date"], freq="M")).size()
        df["n_holidays_in_month"] = df.index.to_period("M").map(holidays_per_month).fillna(0).astype(int)
        df["is_holiday_month"] = (df["n_holidays_in_month"] > 0).astype(int)

        months_since = []
        months_until = []
        for month_start in df.index:
            past = holiday_months[holiday_months <= month_start]
            future = holiday_months[holiday_months >= month_start]
            months_since.append(int((month_start.to_period("M") - past[-1].to_period("M")).n) if len(past) else np.nan)
            months_until.append(int((future[0].to_period("M") - month_start.to_period("M")).n) if len(future) else np.nan)
        df["months_since_last_holiday"] = months_since
        df["months_until_next_holiday"] = months_until

        for lag in LAG_MONTHS:
            df[f"lag_{lag}"] = df["units_sold"].shift(lag)
        for window in ROLLING_WINDOWS:
            shifted = df["units_sold"].shift(1)
            df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"rolling_std_{ROLLING_WINDOWS[0]}"] = df["units_sold"].shift(1).rolling(ROLLING_WINDOWS[0]).std()

        df["liquor_type"] = category
        df["month_start"] = df.index
        frames.append(df.reset_index(drop=True))

    result = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in result.columns if c not in ("liquor_type", "month_start", "units_sold", "units_sold_raw", "is_imputed", "was_clipped")]
    lines = [
        "SECTION 3 — FEATURE ENGINEERING (per category)",
        "",
        f"  Features created ({len(feature_cols)}): {', '.join(feature_cols)}",
        "",
        "  NOT AVAILABLE, carried over from Stage 1 (no fabricated substitutes): price_per_unit,",
        "  price_vs_category_avg, price_change_pct, is_promotion, promotion_depth_pct,",
        "  days_in_promotion, unemployment_rate, consumer_confidence_index, cpi_beverages.",
        "  is_weekend / days_to_holiday / days_after_holiday have no monthly-grain equivalent —",
        "  substituted months_since_last_holiday / months_until_next_holiday instead.",
    ]
    return "\n".join(lines), result


def section_4_split(features: pd.DataFrame, boundaries: dict) -> tuple[str, pd.DataFrame]:
    def label(month_start: pd.Timestamp) -> str:
        if month_start >= boundaries["test_start"]:
            return "test"
        if month_start >= boundaries["val_start"]:
            return "validation"
        return "train"

    result = features.copy()
    result["split"] = result["month_start"].apply(label)

    counts = result.groupby(["liquor_type", "split"]).size().unstack(fill_value=0)
    counts = counts[[c for c in ("train", "validation", "test") if c in counts.columns]]
    lines = [
        "SECTION 4 — WALK-FORWARD TRAIN/VALIDATION/TEST SPLIT (per category, shared calendar cut points)",
        "",
        f"  train: through {boundaries['train_end'].date()}",
        f"  validation: {boundaries['val_start'].date()} to {boundaries['val_end'].date()} ({boundaries['val_months']} months)",
        f"  test (holdout): {boundaries['test_start'].date()} to {boundaries['test_end'].date()} ({boundaries['test_months']} months)",
        "",
        _indent(counts.to_string()),
    ]
    return "\n".join(lines), result


def section_5_scale_features(split_df: pd.DataFrame, feature_cols: list[str]) -> tuple[str, pd.DataFrame, dict]:
    """RobustScaler (median/IQR-based — robust to the outliers Section 2
    deliberately left in the pandemic/test windows), fit on each
    category's TRAIN rows only, applied to that category's validation and
    test rows with the same fitted scaler. Binary/cyclical features
    (already bounded or 0/1) are left unscaled."""
    scale_cols = [c for c in feature_cols if not c.startswith(("is_", "month_sin", "month_cos"))]
    frames, scalers = [], {}

    for category, sub in split_df.groupby("liquor_type"):
        sub = sub.copy()
        train_mask = sub["split"] == "train"
        scaler = RobustScaler()
        train_values = sub.loc[train_mask, scale_cols]
        scaler.fit(train_values.fillna(train_values.median()))
        scalers[category] = scaler

        filled_for_scaling = sub[scale_cols].fillna(train_values.median())
        scaled = scaler.transform(filled_for_scaling)
        for i, col in enumerate(scale_cols):
            sub[f"{col}_scaled"] = np.where(sub[col].isna(), np.nan, scaled[:, i])
        frames.append(sub)

    result = pd.concat(frames, ignore_index=True)
    lines = [
        "SECTION 5 — SCALING (RobustScaler, fit on train only, per category)",
        "",
        f"  Scaled columns ({len(scale_cols)}): {', '.join(scale_cols)}",
        "  Not scaled (already bounded/binary): is_holiday_month, is_pandemic, is_imputed,",
        "  was_clipped, month_sin, month_cos.",
        "  NaN feature values (early-history lag/rolling columns with no history yet) are filled with",
        "  that category's own TRAIN median before scaling, purely so the scaler has no missing input —",
        "  the original NaN is preserved in the unscaled column and in a per-row NaN in the _scaled",
        "  column, so downstream modeling can still tell a real value from a filled placeholder.",
    ]
    return "\n".join(lines), result, scalers


def section_6_summary(monthly: pd.DataFrame, final: pd.DataFrame, gap_report: str, outlier_report: str) -> str:
    rows_before = monthly.groupby("liquor_type").size()
    rows_after = final.groupby("liquor_type").size()
    imputed = final.groupby("liquor_type")["is_imputed"].sum()
    clipped = final.groupby("liquor_type")["was_clipped"].sum()
    summary = pd.DataFrame(
        {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "imputed_values": imputed,
            "outliers_clipped": clipped,
        }
    )
    lines = [
        "SECTION 6 — DATA PREPARATION SUMMARY",
        "",
        _indent(summary.to_string()),
        "",
        f"  Total rows: {rows_before.sum()} -> {rows_after.sum()} (reindexing to a full calendar per",
        "  category doesn't drop rows here since no category had missing months to begin with).",
    ]
    return "\n".join(lines)


def main(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = load_monthly_category_frame()
    if config.categories is not None:
        available = set(monthly["liquor_type"].unique())
        unknown = set(config.categories) - available
        if unknown:
            raise ValueError(f"unknown categor{'y' if len(unknown) == 1 else 'ies'} in config: {sorted(unknown)} — available: {sorted(available)}")
        monthly = monthly[monthly["liquor_type"].isin(config.categories)].reset_index(drop=True)
    boundaries = compute_split_boundaries(monthly, config.test_months, config.val_months)

    report = [f"STAGE 2 DATA PREPARATION REPORT — {datetime.now(tz=timezone.utc).date().isoformat()}", "=" * 78]

    missing_section, filled = section_1_handle_missing_values(monthly, boundaries)
    report.append(missing_section)
    report.append("=" * 78)

    outlier_section, treated = section_2_treat_outliers(filled, boundaries)
    report.append(outlier_section)
    report.append("=" * 78)

    feature_section, features = section_3_feature_engineering(treated)
    report.append(feature_section)
    report.append("=" * 78)

    split_section, split_df = section_4_split(features, boundaries)
    report.append(split_section)
    report.append("=" * 78)

    feature_cols = [
        c
        for c in features.columns
        if c not in ("liquor_type", "month_start", "units_sold", "units_sold_raw", "is_imputed", "was_clipped")
    ]
    scale_section, final, _scalers = section_5_scale_features(split_df, feature_cols)
    report.append(scale_section)
    report.append("=" * 78)

    summary_section = section_6_summary(monthly, final, missing_section, outlier_section)
    report.append(summary_section)

    full_report = "\n\n".join(report)
    print(full_report)
    (OUT / "stage2_data_prep_report.txt").write_text(full_report)
    final.to_parquet(OUT / "processed_features.parquet", index=False)
    print(f"\nWrote report and processed_features.parquet ({len(final)} rows, {len(feature_cols)} features) to {OUT}")


if __name__ == "__main__":
    main()
