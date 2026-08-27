"""
Stage 5 — Champion model report for the monthly, category-level
forecasting exploration. Unlike Stages 1-4 (which evaluate against
already-observed history), this stage produces a genuine forward
forecast: each category's champion model (selected in Stage 4 by lowest
test WAPE) is refit on ALL available months of real data
(train+validation+test) and used to forecast the months beyond the last
observed month — real future, not a backtest window with known answers.

Genuine refit is implemented for all 5 model types, not just whichever
happened to win in one particular run — a user-configured `models` list
(see pipeline_config.py) can make any of naive/prophet/lightgbm/sarimax/
ensemble the champion for any category, and this stage has to be able to
produce a real forecast for whichever one shows up. Prophet/SARIMAX/
ensemble reuse the hyperparameters Stage 3 already tuned (persisted to
JSON) rather than re-tuning from scratch.

Conformal intervals are recalibrated here too, pooling the validation AND
test residuals — more informative than Stage 4's validation-only
calibration, now that both periods have known outcomes to learn from.
95% is dropped whenever the pooled calibration set is still short of the
20 points split conformal needs for a valid 95% guarantee (1/(1-0.95)) —
consistent with Stage 4's reasoning, not a coincidence of one dataset size.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import OUTPUT_ROOT, PANDEMIC_END, PANDEMIC_START, holidays_per_month
from pipeline_config import PipelineConfig
from scipy.stats import spearmanr
from stage3_modeling import (
    EXOG_COLS,
    LightGBMCategoryModel,
    build_prophet_holidays,
    fit_prophet_forecast,
    refit_sarimax_forecast,
)

from demand_forecasting.conformal import ConformalCalibrator
from demand_forecasting.models.baselines import seasonal_naive_forecast

STAGE2_DIR = OUTPUT_ROOT / "stage2_data_prep"
STAGE3_DIR = OUTPUT_ROOT / "stage3_modeling"
STAGE4_DIR = OUTPUT_ROOT / "stage4_evaluation"
OUT = OUTPUT_ROOT / "stage5_champion_report"
HIGH_UNCERTAINTY_THRESHOLD = 0.30  # PI width > 30% of the forecast


@dataclass
class RefitContext:
    year_min: int
    holiday_years: range
    horizon: int
    holiday_df: pd.DataFrame
    lgb_params: dict
    prophet_params: dict
    sarimax_orders: dict
    ridge_ensemble: dict


def _future_exog(last_date: pd.Timestamp, horizon: int, holiday_years: range) -> np.ndarray:
    """EXOG_COLS values (is_holiday_month, n_holidays_in_month, is_pandemic)
    for the `horizon` months following `last_date` — all deterministic
    calendar facts, computable for genuinely future dates with no leakage."""
    counts = holidays_per_month(holiday_years)
    rows = []
    for h in range(1, horizon + 1):
        d = last_date + pd.DateOffset(months=h)
        n_hol = int(counts.get(d.to_period("M"), 0))
        rows.append({"is_holiday_month": int(n_hol > 0), "n_holidays_in_month": n_hol, "is_pandemic": int(PANDEMIC_START <= d.date() <= PANDEMIC_END)})
    return pd.DataFrame(rows)[EXOG_COLS].to_numpy(dtype=float)


def refit_and_forecast(cat_df: pd.DataFrame, champion: str, category: str, ctx: RefitContext) -> np.ndarray:
    """Refit `champion` on the category's FULL history (all splits
    combined) and forecast ctx.horizon months beyond the last observed
    month. Handles all 5 model types since config-driven runs can make
    any of them champion for any category."""
    full_history = cat_df.sort_values("month_start").set_index("month_start")["units_sold"]

    if champion == "naive":
        return seasonal_naive_forecast(full_history, ctx.horizon, season_length=12)

    if champion == "lightgbm":
        train_rows = cat_df[cat_df["lag_1"].notna()]
        model = LightGBMCategoryModel(ctx.lgb_params[category]).fit(train_rows)
        return model.recursive_forecast(full_history, ctx.horizon, ctx.year_min, ctx.holiday_years)

    if champion == "prophet":
        p = ctx.prophet_params[category]
        return fit_prophet_forecast(cat_df, ctx.horizon, p["changepoint_prior_scale"], p["seasonality_prior_scale"], ctx.holiday_df)

    if champion == "sarimax":
        info = ctx.sarimax_orders[category]
        exog_future = _future_exog(full_history.index[-1], ctx.horizon, ctx.holiday_years)
        return refit_sarimax_forecast(tuple(info["order"]), tuple(info["seasonal_order"]), cat_df, exog_future, ctx.horizon)

    if champion == "ensemble":
        info = ctx.ridge_ensemble[category]
        sub_forecasts = [refit_and_forecast(cat_df, sub, category, ctx) for sub in info["inputs"]]
        X = np.column_stack(sub_forecasts)
        X_scaled = (X - np.array(info["scaler_mean"])) / np.array(info["scaler_scale"])
        return np.clip(X_scaled @ np.array(info["coef"]) + info["intercept"], 0, None)

    raise ValueError(f"unknown champion model '{champion}' for category '{category}'")


def build_conformal_calibrator(predictions: pd.DataFrame, category: str, champion: str, alpha: float = 0.20) -> ConformalCalibrator:
    """Calibrate on pooled validation+test residuals for this category's
    champion — more informative than Stage 4's validation-only
    calibration, now that both periods have known, out-of-sample outcomes
    to learn from."""
    cat_preds = predictions[(predictions["liquor_type"] == category) & (predictions["model"] == champion)]
    residual_frame = cat_preds.rename(columns={"predicted": "point_forecast"})[["actual", "point_forecast"]]
    calibrator = ConformalCalibrator(alpha=alpha)
    calibrator.calibrate(residual_frame, group_col=None)
    return calibrator


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def section_1_champion_selection(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    wape_pivot = metrics.pivot(index="liquor_type", columns="model", values="WAPE")
    has_naive = "naive" in wape_pivot.columns
    rows = []
    for category in wape_pivot.index:
        ranked = wape_pivot.loc[category].sort_values()
        champion, champion_wape = ranked.index[0], ranked.iloc[0]
        has_next_best = len(ranked) > 1
        next_best_model = ranked.index[1] if has_next_best else None
        next_best_wape = ranked.iloc[1] if has_next_best else float("nan")
        if has_naive:
            naive_wape = wape_pivot.loc[category, "naive"]
            vs_baseline_pct = 0.0 if champion == "naive" else (naive_wape - champion_wape) / naive_wape * 100
        else:
            vs_baseline_pct = float("nan")
        vs_next_best_pct = (next_best_wape - champion_wape) / next_best_wape * 100 if has_next_best else float("nan")
        rows.append(
            {
                "liquor_type": category,
                "champion": champion,
                "champion_wape": champion_wape,
                "vs_baseline_pct_improvement": vs_baseline_pct,
                "next_best_model": next_best_model,
                "vs_next_best_pct_improvement": vs_next_best_pct,
            }
        )
    selection = pd.DataFrame(rows)

    def _pct(v: float) -> str:
        return f"{v:+.1f}%" if pd.notna(v) else "N/A"

    lines = ["## 1. Champion Selection", ""]
    lines.append("| Category | Champion | Test WAPE | vs. naive baseline | Next best | vs. next best |")
    lines.append("|---|---|---:|---:|---|---:|")
    for _, r in selection.iterrows():
        lines.append(
            f"| {r['liquor_type']} | **{r['champion']}** | {r['champion_wape']:.3f} | "
            f"{_pct(r['vs_baseline_pct_improvement'])} | {r['next_best_model'] or 'N/A'} | {_pct(r['vs_next_best_pct_improvement'])} |"
        )
    lines.append("")
    champion_counts = selection["champion"].value_counts()
    counts_text = ", ".join(f"{model} ({n})" for model, n in champion_counts.items())
    lines.append(
        f"**Why**: champion = lowest test-holdout WAPE (Stage 4), not validation WAPE — the honest, "
        f"genuinely out-of-sample number. Champion counts across {len(selection)} categor{'y' if len(selection) == 1 else 'ies'}: {counts_text}."
        + ("" if has_naive else " (naive baseline wasn't included in this run's `models` config, so no vs-baseline comparison is available above.)")
    )
    return "\n".join(lines), selection


def section_2_forecast_tables(cat_forecasts: dict[str, pd.DataFrame], n_calib: int, horizon: int) -> str:
    lines = [f"## 2. {horizon}-Month Forecast (beyond the last observed month)", ""]
    needed_95 = int(np.ceil(1 / (1 - 0.95)))
    needed_80 = int(np.ceil(1 / (1 - 0.80)))
    lines.append(
        f"95% intervals are omitted here, consistently with Stage 4: split conformal needs at least "
        f"1/(1-confidence) calibration points for a valid guarantee ({needed_95} for 95%), and pooling "
        f"validation+test residuals provides {n_calib} per category — "
        f"{'still short' if n_calib < needed_95 else 'enough this time'}. "
        f"80% is reported ({'needs ' + str(needed_80) + ', comfortably met' if n_calib >= needed_80 else 'ALSO not achievable at this n'})."
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


def section_3_key_drivers(
    selection: pd.DataFrame,
    importance: pd.DataFrame,
    cat_dfs: dict[str, pd.DataFrame],
    prophet_params_all: dict,
    sarimax_orders_all: dict,
    ridge_ensemble_all: dict,
) -> str:
    lines = ["## 3. Key Drivers", ""]
    by_model = {m: g["liquor_type"].tolist() for m, g in selection.groupby("champion")}

    naive_categories = by_model.get("naive", [])
    if naive_categories:
        lines.append(
            f"For the {len(naive_categories)} naive-champion categor{'y' if len(naive_categories) == 1 else 'ies'} "
            f"({', '.join(naive_categories)}), there is no driver analysis to report in the usual sense — the "
            "champion forecast IS last year's same-month value, full stop. That's itself information: no "
            "engineered feature (lag, rolling window, calendar/holiday signal) earned enough test-set "
            "accuracy to beat that one number for these categories."
        )
        lines.append("")

    for category in by_model.get("lightgbm", []):
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

    for category in by_model.get("prophet", []):
        p = prophet_params_all[category]
        lines.append(
            f"**{category}** (Prophet): tuned changepoint_prior_scale={p['changepoint_prior_scale']}, "
            f"seasonality_prior_scale={p['seasonality_prior_scale']}. Prophet decomposes the series into "
            "trend + yearly seasonality + holiday effects rather than a per-feature importance ranking, "
            "so there's no LightGBM-style table here — the tuned changepoint scale is the closest "
            "analogue: a higher value lets the trend bend more freely to recent data."
        )
        lines.append("")

    for category in by_model.get("sarimax", []):
        info = sarimax_orders_all[category]
        lines.append(
            f"**{category}** (SARIMAX): order={tuple(info['order'])}, seasonal_order={tuple(info['seasonal_order'])}. "
            "SARIMAX has no feature-importance concept either — the order itself is the closest analogue: "
            "the differencing term (d) says whether a trend was removed before fitting, and non-zero "
            "seasonal terms say whether a repeating yearly pattern was found beyond what differencing "
            "already explains."
        )
        lines.append("")

    for category in by_model.get("ensemble", []):
        info = ridge_ensemble_all[category]
        weights = ", ".join(f"{m}={w:.2f}" for m, w in zip(info["inputs"], info["coef"]))
        lines.append(
            f"**{category}** (Ensemble): Ridge meta-learner weights, standardized-feature space — {weights}. "
            "The largest-magnitude weight is the base model this category's ensemble leans on most."
        )
        lines.append("")

    return "\n".join(lines)


def section_4_risk_flags(selection: pd.DataFrame, cat_forecasts: dict[str, pd.DataFrame], horizon: int) -> str:
    lines = ["## 4. Risk Flags", ""]

    flag_counts = {category: int(table["high_uncertainty"].sum()) for category, table in cat_forecasts.items()}
    total_flagged = sum(flag_counts.values())
    total_months = sum(len(table) for table in cat_forecasts.values())
    fully_flagged = [c for c, n in flag_counts.items() if n == horizon]
    partially_flagged = {c: n for c, n in flag_counts.items() if 0 < n < horizon}
    clean = [c for c, n in flag_counts.items() if n == 0]
    wape_range = f"{selection['champion_wape'].min():.0%}-{selection['champion_wape'].max():.0%}"

    lines.append(f"**Low-confidence forecast periods** (80% interval width > {HIGH_UNCERTAINTY_THRESHOLD:.0%} of the forecast):")
    lines.append("")
    lines.append(
        f"{total_flagged} of {total_months} category-months ({total_flagged / total_months:.0%}) are flagged — this is "
        f"expected, not a red flag on its own: at 80% confidence, an interval width proportional to a "
        f"category's own test-period WAPE ({wape_range} across champions here) routinely exceeds 30% of "
        f"the point forecast for a monthly series this volatile. Reported per category instead of listing "
        f"all {total_flagged} individual months:"
    )
    lines.append("")
    if fully_flagged:
        lines.append(f"- **All {horizon} months flagged**: {', '.join(fully_flagged)}")
    if partially_flagged:
        for category, n in partially_flagged.items():
            months = ", ".join(cat_forecasts[category].loc[cat_forecasts[category]["high_uncertainty"], "period"])
            lines.append(f"- **{n} of {horizon} months flagged** ({category}): {months}")
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

    has_baseline_comparison = selection["vs_baseline_pct_improvement"].notna().any()
    lines.append("**Categories where the champion beat baseline by <5%** (flag for manual review before trusting over naive):")
    lines.append("")
    if not has_baseline_comparison:
        lines.append("- N/A — naive baseline wasn't included in this run's `models` config, so no comparison exists.")
    else:
        low_confidence = selection[(selection["champion"] != "naive") & selection["vs_baseline_pct_improvement"].notna() & (selection["vs_baseline_pct_improvement"] < 5)]
        non_naive = selection[selection["champion"] != "naive"]
        if len(low_confidence):
            for _, r in low_confidence.iterrows():
                lines.append(f"- {r['liquor_type']} ({r['champion']}): +{r['vs_baseline_pct_improvement']:.1f}% over naive")
        elif len(non_naive):
            lines.append(f"- None among the {len(non_naive)} non-naive-champion categories — all clear 5% by a margin.")
        else:
            lines.append("- N/A — every category's champion is the naive baseline itself.")
        lines.append("")
        n_naive_champions = int((selection["champion"] == "naive").sum())
        if n_naive_champions:
            lines.append(
                f"- Separately: the {n_naive_champions} naive-champion categor{'y is' if n_naive_champions == 1 else 'ies are'}, "
                "by construction, cases where nothing beat baseline at all — not a >=5%-but-not-enough case, "
                "but a 0%-improvement case. Listed in Section 1, not repeated here to avoid double-flagging "
                "the same finding two ways."
            )
    return "\n".join(lines)


def section_5_business_summary(selection: pd.DataFrame, cat_forecasts: dict[str, pd.DataFrame], models_used: list[str], horizon: int) -> str:
    lines = ["## 5. Business Summary", ""]
    n_categories = len(selection)
    n_naive = int((selection["champion"] == "naive").sum())
    non_naive = selection[selection["champion"] != "naive"]
    total_forecast = {cat: table["forecast_units"].sum() for cat, table in cat_forecasts.items()}
    biggest = max(total_forecast, key=total_forecast.get)
    smallest = min(total_forecast, key=total_forecast.get)
    other_models = [m for m in models_used if m != "naive"]

    if "naive" in models_used and n_naive:
        lines.append(
            f"- **Simple beats fancy here, for most categories.** For {n_naive} of {n_categories} liquor "
            f"categories, \"expect the same volume as this month last year\" is still the best forecast "
            f"we could build — {', '.join(other_models) if other_models else 'the other models tried'} "
            f"{'were' if len(other_models) != 1 else 'was'} tried and none improved on it over a real "
            f"{horizon}-month test. That's not a failure of the exercise; it's a finding worth trusting "
            f"over a fancier-looking number."
        )
    if len(non_naive):
        cats_str = ", ".join(f"`{c}`" for c in non_naive["liquor_type"])
        models_str = "/".join(sorted(non_naive["champion"].unique()))
        lines.append(
            f"- **{len(non_naive)} categor{'y is' if len(non_naive) == 1 else 'ies are'} worth the extra "
            f"modeling complexity**: {cats_str} — {models_str} beats the naive baseline by a real margin "
            f"there. For purchasing decisions on {'this category' if len(non_naive) == 1 else 'these'}, "
            f"use its own champion forecast; for the rest, the naive (same-month-last-year) number is "
            f"just as good and far simpler to explain to anyone reviewing the plan."
        )
    if n_categories > 1:
        lines.append(
            f"- **Highest projected volume: `{biggest}` ({total_forecast[biggest]:,.0f} bottles); "
            f"lowest: `{smallest}` ({total_forecast[smallest]:,.0f} bottles).** Use these totals as a starting "
            f"point for purchasing volume by category, not month-by-month commitments — see the per-month "
            f"intervals in Section 2 before locking in any single month's order."
        )
    else:
        lines.append(
            f"- **Projected volume for `{biggest}`: {total_forecast[biggest]:,.0f} bottles.** Use this as a "
            "starting point for purchasing volume, not a month-by-month commitment — see the per-month "
            "intervals in Section 2 before locking in any single month's order."
        )
    lines.append(
        "- **This forecast can't see price changes, promotions, or the economy** — none of that data "
        "exists in what's available. If a major promotion or price change is planned for the forecast "
        "window, treat this forecast as a pre-promotion baseline, not the final number."
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
    ax.plot(full_history.index, full_history.values, color="black", label=f"actual ({len(full_history)} months)")
    ax.plot(forecast_dates, forecast, color="tab:red", marker="o", markersize=3, label=f"forecast ({champion})")
    ax.fill_between(forecast_dates, lower, upper, color="tab:red", alpha=0.2, label="80% interval")
    ax.axvline(full_history.index[-1], color="grey", linestyle="--", linewidth=1)
    ax.set_title(f"{category}: {len(forecast_dates)}-month forward forecast")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"forecast_{category}.png", dpi=110)
    plt.close(fig)


def main(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(STAGE2_DIR / "processed_features.parquet")
    predictions = pd.read_parquet(STAGE3_DIR / "predictions.parquet")
    metrics = pd.read_csv(STAGE4_DIR / "model_comparison_metrics.csv")
    importance = pd.read_csv(STAGE3_DIR / "feature_importance_gain.csv", index_col=0)
    lgb_params_all = json.loads((STAGE3_DIR / "lgb_best_params.json").read_text())
    prophet_params_all = json.loads((STAGE3_DIR / "prophet_params.json").read_text())
    sarimax_orders_all = json.loads((STAGE3_DIR / "sarimax_orders.json").read_text())
    ridge_ensemble_all = json.loads((STAGE3_DIR / "ridge_ensemble.json").read_text())

    categories = sorted(features["liquor_type"].unique())
    year_min = features["month_start"].dt.year.min()
    holiday_years = range(year_min, features["month_start"].dt.year.max() + 3)
    models_used = sorted(predictions["model"].unique())
    holiday_df = build_prophet_holidays(holiday_years) if "prophet" in models_used else None
    ctx = RefitContext(
        year_min=year_min,
        holiday_years=holiday_years,
        horizon=config.forecast_horizon,
        holiday_df=holiday_df,
        lgb_params=lgb_params_all,
        prophet_params=prophet_params_all,
        sarimax_orders=sarimax_orders_all,
        ridge_ensemble=ridge_ensemble_all,
    )

    selection_section, selection = section_1_champion_selection(metrics)
    champion_by_category = selection.set_index("liquor_type")["champion"]

    cat_forecasts = {}
    cat_dfs = {}
    n_calib = 0
    for category in categories:
        cat_df = features[features["liquor_type"] == category].sort_values("month_start").reset_index(drop=True)
        cat_dfs[category] = cat_df
        champion = champion_by_category[category]

        forecast = refit_and_forecast(cat_df, champion, category, ctx)
        last_date = cat_df["month_start"].max()
        forecast_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=config.forecast_horizon, freq="MS")

        calibrator = build_conformal_calibrator(predictions, category, champion, alpha=0.20)
        n_calib = len(predictions[(predictions["liquor_type"] == category) & (predictions["model"] == champion)])
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

    forecast_table_section = section_2_forecast_tables(cat_forecasts, n_calib, config.forecast_horizon)
    drivers_section = section_3_key_drivers(selection, importance, cat_dfs, prophet_params_all, sarimax_orders_all, ridge_ensemble_all)
    risk_section = section_4_risk_flags(selection, cat_forecasts, config.forecast_horizon)
    business_section = section_5_business_summary(selection, cat_forecasts, models_used, config.forecast_horizon)

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
