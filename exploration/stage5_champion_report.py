"""
Stage 5 — Champion model report for the monthly, category-level
forecasting exploration. Unlike Stages 1-4 (which evaluate against
already-observed history), this stage produces a genuine forward
forecast: each category's champion model (selected in Stage 4 by lowest
test WAPE) is refit on ALL 89 months of real data and used to forecast
the 12 months beyond the last observed month — real future, not a
backtest window with known answers.

Conformal intervals are recalibrated here too, pooling the validation AND
test residuals (18 points instead of Stage 4's 6) — a real improvement in
calibration precision now that both periods have known outcomes to learn
from. 95% is still dropped: split conformal needs >=20 calibration points
for a 95% guarantee (1/(1-0.95)=20), and 18 doesn't clear that bar either,
so this stage stays consistent with Stage 4's decision rather than
declaring victory just because a stricter library (MAPIE) isn't in the
loop to enforce the check.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import OUTPUT_ROOT
from scipy.stats import spearmanr
from stage3_modeling import LightGBMCategoryModel

from demand_forecasting.conformal import ConformalCalibrator
from demand_forecasting.models.baselines import seasonal_naive_forecast

STAGE2_DIR = OUTPUT_ROOT / "stage2_data_prep"
STAGE3_DIR = OUTPUT_ROOT / "stage3_modeling"
STAGE4_DIR = OUTPUT_ROOT / "stage4_evaluation"
OUT = OUTPUT_ROOT / "stage5_champion_report"
FORECAST_HORIZON = 12
HIGH_UNCERTAINTY_THRESHOLD = 0.30  # PI width > 30% of the forecast


def refit_and_forecast(cat_df: pd.DataFrame, champion: str, lgb_params: dict | None, year_min: int, holiday_years: range) -> np.ndarray:
    """Refit `champion` on the category's FULL history (all 89 months —
    train+validation+test) and forecast FORECAST_HORIZON months beyond the
    last observed month. Only naive and lightgbm are needed here since
    those are the only models Stage 4 found as champions."""
    full_history = cat_df.sort_values("month_start").set_index("month_start")["units_sold"]

    if champion == "naive":
        return seasonal_naive_forecast(full_history, FORECAST_HORIZON, season_length=12)

    if champion == "lightgbm":
        train_rows = cat_df[cat_df["lag_1"].notna()]
        model = LightGBMCategoryModel(lgb_params).fit(train_rows)
        return model.recursive_forecast(full_history, FORECAST_HORIZON, year_min, holiday_years)

    raise ValueError(f"no refit path implemented for champion model '{champion}' — only naive/lightgbm occur in this run")


def build_conformal_calibrator(predictions: pd.DataFrame, category: str, champion: str, alpha: float = 0.20) -> ConformalCalibrator:
    """Calibrate on pooled validation+test residuals for this category's
    champion (18 points total) — more informative than Stage 4's
    validation-only 6-point calibration, now that both periods have
    known, out-of-sample outcomes to learn from."""
    cat_preds = predictions[(predictions["liquor_type"] == category) & (predictions["model"] == champion)]
    residual_frame = cat_preds.rename(columns={"predicted": "point_forecast"})[["actual", "point_forecast"]]
    calibrator = ConformalCalibrator(alpha=alpha)
    calibrator.calibrate(residual_frame, group_col=None)
    return calibrator


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def section_1_champion_selection(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    wape_pivot = metrics.pivot(index="liquor_type", columns="model", values="WAPE")
    rows = []
    for category in wape_pivot.index:
        ranked = wape_pivot.loc[category].sort_values()
        champion, champion_wape = ranked.index[0], ranked.iloc[0]
        naive_wape = wape_pivot.loc[category, "naive"]
        next_best_wape = ranked.iloc[1]
        vs_baseline_pct = 0.0 if champion == "naive" else (naive_wape - champion_wape) / naive_wape * 100
        vs_next_best_pct = (next_best_wape - champion_wape) / next_best_wape * 100
        rows.append(
            {
                "liquor_type": category,
                "champion": champion,
                "champion_wape": champion_wape,
                "vs_baseline_pct_improvement": vs_baseline_pct,
                "next_best_model": ranked.index[1],
                "vs_next_best_pct_improvement": vs_next_best_pct,
            }
        )
    selection = pd.DataFrame(rows)

    lines = ["## 1. Champion Selection", ""]
    lines.append("| Category | Champion | Test WAPE | vs. naive baseline | Next best | vs. next best |")
    lines.append("|---|---|---:|---:|---|---:|")
    for _, r in selection.iterrows():
        lines.append(
            f"| {r['liquor_type']} | **{r['champion']}** | {r['champion_wape']:.3f} | "
            f"{r['vs_baseline_pct_improvement']:+.1f}% | {r['next_best_model']} | {r['vs_next_best_pct_improvement']:+.1f}% |"
        )
    lines.append("")
    n_naive = (selection["champion"] == "naive").sum()
    lines.append(
        f"**Why**: champion = lowest test-holdout WAPE (Stage 4), not validation WAPE — the honest, "
        f"genuinely out-of-sample number. The naive seasonal baseline wins {n_naive} of 8 categories "
        f"outright; for those, no more sophisticated model earns its complexity, and the \"champion\" "
        f"is simply the model nobody managed to beat. LightGBM wins the remaining 2 (`other`, `vodka`) "
        f"with real double-digit improvements over naive."
    )
    return "\n".join(lines), selection


def section_2_forecast_tables(cat_forecasts: dict[str, pd.DataFrame]) -> str:
    lines = ["## 2. 12-Month Forecast (beyond the last observed month)", ""]
    lines.append(
        "95% intervals are omitted here, consistently with Stage 4: split conformal needs at least "
        "1/(1-confidence) calibration points for a valid guarantee (20 for 95%), and pooling "
        "validation+test residuals only provides 18 per category — still short. Only 80% is reported "
        "(needs 5, comfortably met)."
    )
    lines.append("")
    for category, table in cat_forecasts.items():
        lines.append(f"### {category}")
        lines.append("")
        lines.append("| Period | Forecast (units) | Lower 80 | Upper 80 | Lower 95 | Upper 95 | High uncertainty? |")
        lines.append("|---|---:|---:|---:|---|---|---|")
        for _, r in table.iterrows():
            flag = "**yes**" if r["high_uncertainty"] else ""
            lines.append(
                f"| {r['period']} | {r['forecast_units']:.0f} | {r['lower_80']:.0f} | {r['upper_80']:.0f} | "
                f"N/A | N/A | {flag} |"
            )
        lines.append("")
    return "\n".join(lines)


def section_3_key_drivers(selection: pd.DataFrame, importance: pd.DataFrame, cat_dfs: dict[str, pd.DataFrame]) -> str:
    lines = ["## 3. Key Drivers", ""]
    lgb_categories = selection[selection["champion"] == "lightgbm"]["liquor_type"].tolist()
    naive_categories = selection[selection["champion"] == "naive"]["liquor_type"].tolist()

    lines.append(
        f"For the {len(naive_categories)} naive-champion categories ({', '.join(naive_categories)}), "
        "there is no driver analysis to report in the usual sense — the champion forecast IS last "
        "year's same-month value, full stop. That's itself information: it means no engineered "
        "feature (lag, rolling window, calendar/holiday signal) earned enough test-set accuracy to "
        "beat that one number for these categories."
    )
    lines.append("")

    for category in lgb_categories:
        top5 = importance[category].sort_values(ascending=False).head(5)
        train_rows = cat_dfs[category][cat_dfs[category]["lag_1"].notna()]
        lines.append(f"**{category}** (LightGBM, gain-based importance):")
        lines.append("")
        lines.append("| Feature | Gain (relative) | Direction (Spearman r vs. units_sold) |")
        lines.append("|---|---:|---:|")
        total_gain = importance[category].sum()
        for feature, gain in top5.items():
            r, _ = spearmanr(train_rows[feature], train_rows["units_sold"], nan_policy="omit")
            direction = "higher -> higher sales" if r > 0 else "higher -> lower sales"
            lines.append(f"| {feature} | {gain / total_gain:.1%} | r={r:+.2f} ({direction}) |")
        lines.append("")
        lines.append(
            "  Direction is a simple Spearman correlation between the raw feature and units_sold, not "
            "a true SHAP marginal-effect attribution (that would need the `shap` package, not installed "
            "for this exploration) — a reasonable, honest approximation, not a claim of causal effect size."
        )
        lines.append("")
    return "\n".join(lines)


def section_4_risk_flags(selection: pd.DataFrame, cat_forecasts: dict[str, pd.DataFrame]) -> str:
    lines = ["## 4. Risk Flags", ""]

    flag_counts = {category: int(table["high_uncertainty"].sum()) for category, table in cat_forecasts.items()}
    total_flagged = sum(flag_counts.values())
    total_months = sum(len(table) for table in cat_forecasts.values())
    fully_flagged = [c for c, n in flag_counts.items() if n == 12]
    partially_flagged = {c: n for c, n in flag_counts.items() if 0 < n < 12}
    clean = [c for c, n in flag_counts.items() if n == 0]

    lines.append(f"**Low-confidence forecast periods** (80% interval width > {HIGH_UNCERTAINTY_THRESHOLD:.0%} of the forecast):")
    lines.append("")
    lines.append(
        f"{total_flagged} of {total_months} category-months ({total_flagged / total_months:.0%}) are flagged — this is "
        "expected, not a red flag on its own: at 80% confidence, an interval width proportional to a "
        "category's historical test-period WAPE (9-24% here) routinely exceeds 30% of the point "
        "forecast for a monthly series this volatile. Reported per category instead of listing all "
        f"{total_flagged} individual months:"
    )
    lines.append("")
    if fully_flagged:
        lines.append(f"- **All 12 months flagged**: {', '.join(fully_flagged)}")
    if partially_flagged:
        for category, n in partially_flagged.items():
            months = ", ".join(cat_forecasts[category].loc[cat_forecasts[category]["high_uncertainty"], "period"])
            lines.append(f"- **{n} of 12 months flagged** ({category}): {months}")
    if clean:
        lines.append(f"- **No months flagged** (genuinely tight 80% intervals): {', '.join(clean)}")
    lines.append("")

    lines.append("**Structural blind spots** (real limitations, not periods — apply to every forecast in this report):")
    lines.append("")
    lines.append(
        "- No promotion, price, or discount data exists in this dataset — a real promotional push or "
        "price change in the forecast window is invisible to every model here."
    )
    lines.append(
        "- No macro data (unemployment, consumer confidence, CPI) — a real economic shift affecting "
        "discretionary alcohol spending would not be anticipated by any of these models."
    )
    lines.append(
        "- The synthetic-scale COVID-style disruption used to stress-test the PRODUCTION pipeline's "
        "backtest doesn't apply here — this exploration's real data showed only a mild pandemic-era "
        "effect (Stage 1), but a disruption of that magnitude, if it recurred, is not something any "
        "of these models were built or validated to handle."
    )
    lines.append("")

    low_confidence = selection[(selection["champion"] == "lightgbm") & (selection["vs_baseline_pct_improvement"] < 5)]
    lines.append("**Categories where the champion beat baseline by <5%** (flag for manual review before trusting over naive):")
    lines.append("")
    if len(low_confidence):
        for _, r in low_confidence.iterrows():
            lines.append(f"- {r['liquor_type']}: +{r['vs_baseline_pct_improvement']:.1f}% over naive")
    else:
        lines.append("- None among the LightGBM-champion categories — both (`other`, `vodka`) clear 5% by a wide margin.")
    lines.append("")
    lines.append(
        "- Separately: the 6 naive-champion categories are, by construction, cases where nothing beat "
        "baseline at all — not a >=5%-but-not-enough case, but a 0%-improvement case. Listed in Section 1, "
        "not repeated here to avoid double-flagging the same finding two ways."
    )
    return "\n".join(lines)


def section_5_business_summary(selection: pd.DataFrame, cat_forecasts: dict[str, pd.DataFrame]) -> str:
    lines = ["## 5. Business Summary", ""]
    n_naive = (selection["champion"] == "naive").sum()
    total_12mo = {cat: table["forecast_units"].sum() for cat, table in cat_forecasts.items()}
    biggest = max(total_12mo, key=total_12mo.get)
    smallest = min(total_12mo, key=total_12mo.get)

    lines.append(
        f"- **Simple beats fancy here, for most categories.** For {n_naive} of 8 liquor categories, "
        f"\"expect the same volume as this month last year\" is still the best forecast we could "
        f"build — more complex models (Prophet, LightGBM, SARIMAX, and an ensemble of all three) were "
        f"tried and none of them improved on it over a real 12-month test. That's not a failure of "
        f"the exercise; it's a finding worth trusting over a fancier-looking number."
    )
    lines.append(
        "- **Two categories are worth the extra modeling complexity**: `other` and `vodka` — LightGBM "
        "beats the naive baseline by a real, double-digit margin for both. For purchasing decisions on "
        "these two, use the LightGBM forecast; for the other six, the naive (same-month-last-year) "
        "number is just as good and far simpler to explain to anyone reviewing the plan."
    )
    lines.append(
        f"- **Highest projected 12-month volume: `{biggest}` ({total_12mo[biggest]:,.0f} bottles); "
        f"lowest: `{smallest}` ({total_12mo[smallest]:,.0f} bottles).** Use these totals as a starting "
        f"point for annual purchasing volume by category, not month-by-month commitments — see the "
        f"per-month intervals in Section 2 before locking in any single month's order."
    )
    lines.append(
        "- **This forecast can't see price changes, promotions, or the economy** — none of that data "
        "exists in what's available. If a major promotion or price change is planned for the next 12 "
        "months, treat this forecast as a pre-promotion baseline, not the final number."
    )
    lines.append(
        "- **Trust the interval, not just the point number.** Every forecast here comes with an 80% "
        "range (Section 2) precisely because a single number invites over-precision. Where that range "
        "is wide (flagged in Section 4), budget for the upper end rather than the point forecast if "
        "a stockout is more costly than overstock for that category."
    )
    return "\n".join(lines)


def make_forecast_plot(category: str, full_history: pd.Series, forecast_dates: pd.DatetimeIndex, forecast: np.ndarray, lower: np.ndarray, upper: np.ndarray, champion: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(full_history.index, full_history.values, color="black", label="actual (89 months)")
    ax.plot(forecast_dates, forecast, color="tab:red", marker="o", markersize=3, label=f"forecast ({champion})")
    ax.fill_between(forecast_dates, lower, upper, color="tab:red", alpha=0.2, label="80% interval")
    ax.axvline(full_history.index[-1], color="grey", linestyle="--", linewidth=1)
    ax.set_title(f"{category}: 12-month forward forecast")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"forecast_{category}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(STAGE2_DIR / "processed_features.parquet")
    predictions = pd.read_parquet(STAGE3_DIR / "predictions.parquet")
    metrics = pd.read_csv(STAGE4_DIR / "model_comparison_metrics.csv")
    importance = pd.read_csv(STAGE3_DIR / "feature_importance_gain.csv", index_col=0)
    lgb_params_all = json.loads((STAGE3_DIR / "lgb_best_params.json").read_text())

    categories = sorted(features["liquor_type"].unique())
    year_min = features["month_start"].dt.year.min()
    holiday_years = range(year_min, features["month_start"].dt.year.max() + 3)

    selection_section, selection = section_1_champion_selection(metrics)
    champion_by_category = selection.set_index("liquor_type")["champion"]

    cat_forecasts = {}
    cat_dfs = {}
    for category in categories:
        cat_df = features[features["liquor_type"] == category].sort_values("month_start").reset_index(drop=True)
        cat_dfs[category] = cat_df
        champion = champion_by_category[category]

        forecast = refit_and_forecast(cat_df, champion, lgb_params_all.get(category), year_min, holiday_years)
        last_date = cat_df["month_start"].max()
        forecast_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq="MS")

        calibrator = build_conformal_calibrator(predictions, category, champion, alpha=0.20)
        lower = np.array([calibrator.interval(f)[0] for f in forecast])
        upper = np.array([calibrator.interval(f)[1] for f in forecast])
        width_pct = np.divide(upper - lower, forecast, out=np.full_like(forecast, np.inf), where=forecast != 0)

        table = pd.DataFrame(
            {
                "period": forecast_dates.strftime("%Y-%m"),
                "forecast_units": forecast,
                "lower_80": lower,
                "upper_80": upper,
                "high_uncertainty": width_pct > HIGH_UNCERTAINTY_THRESHOLD,
            }
        )
        cat_forecasts[category] = table

        full_history = cat_df.set_index("month_start")["units_sold"]
        make_forecast_plot(category, full_history, forecast_dates, forecast, lower, upper, champion)

    forecast_table_section = section_2_forecast_tables(cat_forecasts)
    drivers_section = section_3_key_drivers(selection, importance, cat_dfs)
    risk_section = section_4_risk_flags(selection, cat_forecasts)
    business_section = section_5_business_summary(selection, cat_forecasts)

    doc = [
        "# Champion Model Report — Monthly Category-Level Liquor Demand Forecast",
        "",
        (
            f"*Generated {datetime.now(tz=timezone.utc).date().isoformat()}. Exploration track — see "
            "exploration/common.py for scope and how this relates to the production pipeline.*"
        ),
        "",
        selection_section,
        "",
        forecast_table_section,
        "",
        drivers_section,
        "",
        risk_section,
        "",
        business_section,
        "",
        "## Charts",
        "",
        *[f"![{category} forecast](forecast_{category}.png)" for category in categories],
    ]
    full_doc = "\n".join(doc)
    (OUT / "champion_report.md").write_text(full_doc)
    print(full_doc)
    print(f"\nWrote champion_report.md and {len(categories)} forecast plots to {OUT}")


if __name__ == "__main__":
    main()
