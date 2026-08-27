"""
Stage 4 — Model evaluation for the monthly, category-level forecasting
exploration. Consumes exploration/outputs/stage3_modeling/predictions.parquet
and evaluates every model on the 12-month TEST holdout only — the
validation-period numbers already served their purpose (tuning, picking
the champion's hyperparameters) in Stage 3 and would be optimistic here.

Everything below is still done per category: the champion model differs
by category (Stage 3 found naive wins 6 of 8, LightGBM wins 2), so
"the champion" in Sections 3 and 5 means that CATEGORY's own champion,
not one model picked for the whole dataset.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import OUTPUT_ROOT, holidays_per_month
from mapie.regression import SplitConformalRegressor
from pipeline_config import PipelineConfig
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin

from demand_forecasting.evaluation.eval_suite import rmse, wape

warnings.filterwarnings("ignore", message="Estimator does not appear fitted")

IN_DIR = OUTPUT_ROOT / "stage3_modeling"
OUT = OUTPUT_ROOT / "stage4_evaluation"
CANONICAL_MODEL_ORDER = ["naive", "prophet", "lightgbm", "sarimax", "ensemble"]
# 95% is requested by the original spec but not statistically achievable here: split
# conformal needs at least 1/(1-confidence_level) calibration points, and each category
# has only 6 (the validation split). 1/(1-0.95)=20 > 6, so MAPIE correctly refuses it;
# 1/(1-0.80)=5 <= 6, so 80% is the highest level this calibration set can actually support.
CONFIDENCE_LEVELS = [0.80]


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Excludes zero-actual months (undefined denominator) rather than
    inflating the average with a division-by-zero artifact."""
    nonzero = y_true != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


class FrozenForecaster(BaseEstimator, RegressorMixin):
    """A precomputed forecast dressed up as a fitted sklearn regressor, so
    MAPIE's SplitConformalRegressor (prefit=True) can wrap it. `.predict`
    just looks up the ordinal index encoded in X — there's no real feature
    -based model here, only Stage 3's already-computed point forecasts."""

    def __init__(self, lookup: dict):
        self.lookup = lookup

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self.lookup[int(row[0])] for row in X])


def section_1_model_comparison(test: pd.DataFrame, model_order: list[str]) -> tuple[str, pd.DataFrame]:
    rows = []
    for (category, model), group in test.groupby(["liquor_type", "model"]):
        y_true, y_pred = group["actual"].to_numpy(), group["predicted"].to_numpy()
        rows.append(
            {
                "liquor_type": category,
                "model": model,
                "WAPE": wape(y_true, y_pred),
                "MAPE": mape(y_true, y_pred),
                "RMSE": rmse(y_true, y_pred),
                "MAE": mae(y_true, y_pred),
                "Bias": bias(y_true, y_pred),
            }
        )
    metrics = pd.DataFrame(rows)
    wape_pivot = metrics.pivot(index="liquor_type", columns="model", values="WAPE")[model_order]
    champion = wape_pivot.idxmin(axis=1)
    champion_wape = wape_pivot.min(axis=1)

    lines = [
        "SECTION 1 — MODEL COMPARISON (test holdout, all metrics x all models x all categories)",
        "",
        "  WAPE (primary metric):",
        _indent(wape_pivot.round(3).to_string()),
        "",
        "  Champion (lowest test WAPE) per category:",
        _indent(pd.DataFrame({"champion": champion, "wape": champion_wape.round(3)}).to_string()),
        "",
        "  Full metric table (WAPE, MAPE%, RMSE, MAE, Bias) — Bias>0 means the model",
        "  over-forecasts on average, Bias<0 means it under-forecasts:",
        _indent(metrics.set_index(["liquor_type", "model"]).round(2).to_string()),
    ]
    return "\n".join(lines), metrics.assign(is_champion=lambda d: d.apply(lambda r: champion[r["liquor_type"]] == r["model"], axis=1))


def section_2_forecast_plots(test: pd.DataFrame, model_order: list[str]) -> str:
    categories = sorted(test["liquor_type"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=False)
    for ax, category in zip(axes.flat, categories):
        cat_test = test[test["liquor_type"] == category].sort_values("month_start")
        actual = cat_test[["month_start", "actual"]].drop_duplicates()  # same regardless of which model's rows they came from
        ax.fill_between(actual["month_start"], actual["actual"] * 0.8, actual["actual"] * 1.2, color="grey", alpha=0.15, label="+/-20%")
        ax.fill_between(actual["month_start"], actual["actual"] * 0.9, actual["actual"] * 1.1, color="grey", alpha=0.25, label="+/-10%")
        ax.plot(actual["month_start"], actual["actual"], color="black", linewidth=2, label="actual")
        for model in model_order:
            sub = cat_test[cat_test["model"] == model].sort_values("month_start")
            ax.plot(sub["month_start"], sub["predicted"], marker="o", markersize=3, label=model, alpha=0.8)
        ax.set_title(category)
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Test-period forecast vs. actual, all models, +/-10%/20% bands around actual")
    fig.tight_layout()
    fig.savefig(OUT / "forecast_vs_actual.png", dpi=110)
    plt.close(fig)
    return (
        "SECTION 2 — FORECAST VS ACTUAL PLOTS\n\n"
        "  Saved forecast_vs_actual.png: one panel per category (all models overlaid on the same\n"
        "  axes, rather than 40 separate model x category plots — far more useful for comparing\n"
        "  models against each other, not just against actual)."
    )


def section_3_residual_analysis(test: pd.DataFrame, champion_by_category: pd.Series, holiday_months: set) -> tuple[str, pd.DataFrame]:
    categories = sorted(test["liquor_type"].unique())
    residual_rows = []
    for category in categories:
        model = champion_by_category[category]
        sub = test[(test["liquor_type"] == category) & (test["model"] == model)].sort_values("month_start")
        residual = sub["predicted"].to_numpy() - sub["actual"].to_numpy()
        is_holiday = sub["month_start"].dt.to_period("M").isin(holiday_months).to_numpy()
        for date, r, h in zip(sub["month_start"], residual, is_holiday):
            residual_rows.append({"liquor_type": category, "champion": model, "month_start": date, "residual": r, "is_holiday_month": h})

    residuals = pd.DataFrame(residual_rows)

    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=False)
    for ax, category in zip(axes.flat, categories):
        sub = residuals[residuals["liquor_type"] == category]
        colors = np.where(sub["is_holiday_month"], "tab:red", "tab:blue")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.scatter(sub["month_start"], sub["residual"], c=colors, s=25)
        ax.plot(sub["month_start"], sub["residual"], alpha=0.3, color="grey")
        ax.set_title(f"{category} (champion: {champion_by_category[category]})")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Champion model residuals over time (red = holiday month)")
    fig.tight_layout()
    fig.savefig(OUT / "residuals_over_time.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    normality_rows = []
    for ax, category in zip(axes.flat, categories):
        sub = residuals[residuals["liquor_type"] == category]["residual"].to_numpy()
        ax.hist(sub, bins=8, color="tab:blue", alpha=0.8)
        ax.set_title(category)
        if len(sub) >= 3:
            stat, p = stats.shapiro(sub)
            normality_rows.append({"liquor_type": category, "shapiro_stat": stat, "shapiro_p": p, "normal_at_5pct": p >= 0.05})
    fig.suptitle("Champion model residual distributions (test period)")
    fig.tight_layout()
    fig.savefig(OUT / "residual_distributions.png", dpi=110)
    plt.close(fig)

    holiday_comparison = (
        residuals.groupby(["liquor_type", "is_holiday_month"])["residual"]
        .apply(lambda s: np.mean(np.abs(s)))
        .unstack()
        .rename(columns={False: "mean_abs_resid_non_holiday", True: "mean_abs_resid_holiday"})
    )
    pooled_holiday = residuals.groupby("is_holiday_month")["residual"].apply(lambda s: np.mean(np.abs(s)))

    normality = pd.DataFrame(normality_rows)
    lines = [
        "SECTION 3 — RESIDUAL ANALYSIS (champion model per category, test period)",
        "",
        "  Saved residuals_over_time.png and residual_distributions.png.",
        "",
        "  Shapiro-Wilk normality test on residuals (H0: residuals are normal; n=12/category, low power):",
        _indent(normality.round(4).to_string(index=False)),
        "",
        "  Mean |residual| in holiday vs. non-holiday test months, per category:",
        _indent(holiday_comparison.round(1).to_string()),
        "",
        (
            f"  Pooled across all categories: non-holiday={pooled_holiday.get(False, float('nan')):.1f}, "
            f"holiday={pooled_holiday.get(True, float('nan')):.1f}."
        ),
    ]
    return "\n".join(lines), residuals


def _horizon_bucket_edges(n_months: int) -> tuple[list[int], list[str]]:
    """3 near/mid/long-term buckets, sized to whatever the test window
    actually is. Preserves the exact original months_1-3/4-6/7-12 labels
    at the default 12-month window; generalizes (as evenly as an integer
    split allows) for any other configured test_months."""
    if n_months == 12:
        return [0, 3, 6, 12], ["months_1-3", "months_4-6", "months_7-12"]
    if n_months < 3:
        return [0, n_months], [f"months_1-{n_months}"]
    third = n_months // 3
    edges = [0, third, 2 * third, n_months]
    labels = [f"months_1-{third}", f"months_{third + 1}-{2 * third}", f"months_{2 * third + 1}-{n_months}"]
    return edges, labels


def section_4_error_by_horizon(test: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    df = test.copy()
    df["horizon_month"] = df.groupby(["liquor_type", "model"])["month_start"].rank(method="first").astype(int)
    n_months = int(df["horizon_month"].max())
    edges, labels = _horizon_bucket_edges(n_months)
    df["horizon_bucket"] = pd.cut(df["horizon_month"], bins=edges, labels=labels)

    rows = []
    for (category, model, bucket), group in df.groupby(["liquor_type", "model", "horizon_bucket"], observed=True):
        rows.append({"liquor_type": category, "model": model, "horizon_bucket": bucket, "wape": wape(group["actual"].to_numpy(), group["predicted"].to_numpy())})
    horizon_wape = pd.DataFrame(rows)
    pivot = horizon_wape.pivot_table(index=["liquor_type", "model"], columns="horizon_bucket", values="wape", observed=True)[labels]

    avg_by_bucket_model = horizon_wape.groupby(["model", "horizon_bucket"], observed=True)["wape"].mean().unstack()[labels]

    lines = [
        f"SECTION 4 — ERROR BY FORECAST HORIZON ({' / '.join(labels)} into the {n_months}-month test window)",
        "",
        f"  Average WAPE by model, across all {test['liquor_type'].nunique()} categories:",
        _indent(avg_by_bucket_model.round(3).to_string()),
        "",
        "  Full breakdown (every model x category):",
        _indent(pivot.round(3).to_string()),
    ]
    return "\n".join(lines), pivot


def section_5_mapie_intervals(predictions: pd.DataFrame, champion_by_category: pd.Series) -> tuple[str, pd.DataFrame]:
    categories = sorted(champion_by_category.index)
    coverage_rows = []
    interval_rows = []

    for category in categories:
        model = champion_by_category[category]
        cat_preds = predictions[(predictions["liquor_type"] == category) & (predictions["model"] == model)].sort_values(["split", "month_start"])
        val = cat_preds[cat_preds["split"] == "validation"].reset_index(drop=True)
        test = cat_preds[cat_preds["split"] == "test"].reset_index(drop=True)

        n_val = len(val)
        needed = int(np.ceil(1 / (1 - CONFIDENCE_LEVELS[0])))
        if n_val < needed:
            coverage_rows.append(
                {
                    "liquor_type": category,
                    "champion": model,
                    "nominal_coverage": CONFIDENCE_LEVELS[0],
                    "empirical_coverage": float("nan"),
                    "mean_interval_width": float("nan"),
                }
            )
            continue  # val_months configured smaller than split conformal needs (1/(1-confidence)) for this level

        lookup = {i: pred for i, pred in enumerate(pd.concat([val["predicted"], test["predicted"]]).to_numpy())}
        X_calib = np.arange(n_val).reshape(-1, 1)
        X_test = np.arange(n_val, n_val + len(test)).reshape(-1, 1)

        mapie_reg = SplitConformalRegressor(estimator=FrozenForecaster(lookup), confidence_level=CONFIDENCE_LEVELS, prefit=True)
        mapie_reg.conformalize(X_calib, val["actual"].to_numpy())
        _, y_pis = mapie_reg.predict_interval(X_test)

        for level_idx, level in enumerate(CONFIDENCE_LEVELS):
            lower = y_pis[:, 0, level_idx]
            upper = y_pis[:, 1, level_idx]
            covered = (test["actual"].to_numpy() >= lower) & (test["actual"].to_numpy() <= upper)
            coverage_rows.append(
                {
                    "liquor_type": category,
                    "champion": model,
                    "nominal_coverage": level,
                    "empirical_coverage": float(np.mean(covered)),
                    "mean_interval_width": float(np.mean(upper - lower)),
                }
            )
            if level == 0.80:
                for date, lo, hi, pred in zip(test["month_start"], lower, upper, test["predicted"]):
                    interval_rows.append({"liquor_type": category, "month_start": date, "predicted": pred, "lower_80": lo, "upper_80": hi})

    coverage = pd.DataFrame(coverage_rows)
    coverage_pivot = coverage.pivot(index="liquor_type", columns="nominal_coverage", values="empirical_coverage")
    intervals = pd.DataFrame(interval_rows)

    n_val = len(predictions[(predictions["liquor_type"] == categories[0]) & (predictions["split"] == "validation") & (predictions["model"] == champion_by_category[categories[0]])])
    n_test = len(predictions[(predictions["liquor_type"] == categories[0]) & (predictions["split"] == "test") & (predictions["model"] == champion_by_category[categories[0]])])
    max_confidence_at_n = 1 - 1 / (n_val + 1) if n_val else 0.0
    n_full_coverage = int((coverage["empirical_coverage"] == 1.0).sum())
    level = 0.8 if n_val >= int(np.ceil(1 / 0.2)) else float("nan")
    quantile_level = np.ceil((n_val + 1) * level) / n_val if n_val and not np.isnan(level) else float("nan")

    lines = [
        "SECTION 5 — MAPIE CONFORMAL PREDICTION INTERVALS (champion model per category)",
        "",
        (
            f"  95% requested by spec but DROPPED — not statistically achievable: split conformal needs "
            f"at least 1/(1-confidence_level) calibration points, and each category has {n_val} (the "
            f"validation split, val_months={n_val} in this run). 1/(1-0.95)=20 "
            f"{'>' if n_val < 20 else '<='} {n_val}"
            f"{' (MAPIE itself refuses this)' if n_val < 20 else ''}; 1/(1-0.80)=5 "
            f"{'<=' if n_val >= 5 else '>'} {n_val}, so 80% is "
            f"{'the highest level this calibration set can actually support' if n_val >= 5 else 'ALSO not achievable — every category is skipped below'}"
            " — reported below instead of forcing an invalid guarantee."
        ),
        "",
        (
            f"  Calibrated on that category's {n_val}-month validation residuals, applied to the "
            f"{n_test}-month test forecasts. n={n_val} calibration / n={n_test} test points per category "
            f"is small — read exact coverage numbers as noisy, not precise."
        ),
        "",
        (
            f"  All categories skipped above — 80% isn't achievable at n={n_val} calibration points "
            f"(needs 5), so there's nothing to report on the coverage/width tradeoff at this level."
            if np.isnan(level)
            else (
                f"  {n_full_coverage} of {len(categories)} categories show 100% empirical coverage at the 80% "
                "nominal level — expected, not a bug, when it happens: split conformal's finite-sample "
                "correction (same formula as the production pipeline's conformal.py) needs "
                f"ceil((n+1)*confidence_level)/n of the calibration residuals; at n={n_val} that's "
                f"ceil({n_val + 1}*0.8)/{n_val} = {quantile_level:.2f}"
                + (
                    " — the interval width is forced to the SINGLE LARGEST calibration residual, wide enough "
                    "to cover nearly anything, which is why coverage runs high but mean_interval_width is so "
                    f"large relative to the point forecast. A genuinely informative interval at this level "
                    f"would need more than {n_val} calibration points (max achievable confidence at this n is "
                    f"~{max_confidence_at_n:.1%})."
                    if quantile_level >= 1.0
                    else "."
                )
            )
        ),
        "",
        "  Empirical vs. nominal (80%) coverage:",
        _indent(coverage.round(3).to_string(index=False)),
        "",
        "  Empirical coverage pivot:",
        _indent(coverage_pivot.round(3).to_string()),
    ]
    return "\n".join(lines), intervals


def main(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(IN_DIR / "predictions.parquet")
    test = predictions[predictions["split"] == "test"].copy()
    test["month_start"] = pd.to_datetime(test["month_start"])
    model_order = [m for m in CANONICAL_MODEL_ORDER if m in test["model"].unique()]

    years = range(test["month_start"].dt.year.min(), test["month_start"].dt.year.max() + 1)
    holiday_counts = holidays_per_month(years)
    holiday_months = set(holiday_counts[holiday_counts > 0].index)

    report = [f"STAGE 4 EVALUATION REPORT — {datetime.now(tz=timezone.utc).date().isoformat()}", "=" * 78]
    report.append(f"Models present: {model_order}. Categories: {sorted(test['liquor_type'].unique())}.")
    report.append("=" * 78)

    comparison_section, metrics = section_1_model_comparison(test, model_order)
    report.append(comparison_section)
    report.append("=" * 78)

    plot_section = section_2_forecast_plots(test, model_order)
    report.append(plot_section)
    report.append("=" * 78)

    wape_pivot = metrics[metrics["is_champion"]][["liquor_type", "model"]].set_index("liquor_type")["model"]
    residual_section, residuals = section_3_residual_analysis(test, wape_pivot, holiday_months)
    report.append(residual_section)
    report.append("=" * 78)

    horizon_section, horizon_table = section_4_error_by_horizon(test)
    report.append(horizon_section)
    report.append("=" * 78)

    mapie_section, intervals = section_5_mapie_intervals(predictions, wape_pivot)
    report.append(mapie_section)

    full_report = "\n\n".join(report)
    print(full_report)

    (OUT / "stage4_evaluation_report.txt").write_text(full_report)
    metrics.to_csv(OUT / "model_comparison_metrics.csv", index=False)
    residuals.to_csv(OUT / "champion_residuals.csv", index=False)
    horizon_table.to_csv(OUT / "error_by_horizon.csv")
    intervals.to_csv(OUT / "champion_intervals_80pct.csv", index=False)
    print(f"\nWrote report, metrics, and plots to {OUT}")


if __name__ == "__main__":
    main()
