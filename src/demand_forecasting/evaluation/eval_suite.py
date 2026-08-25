"""
Rolling-origin backtesting for the whole tiered stack.

A single train/test split answers "did this model get lucky on one
snapshot of the calendar." Rolling-origin backtesting answers "does this
model hold up across many snapshots" — which is the question that
actually matters for a forecaster meant to run every week. This suite
checks three genuinely different things:

1. Point-forecast accuracy (WAPE, RMSE) for every tier, per fold,
   benchmarked against the naive baselines. If the global model doesn't
   beat seasonal-naive on a given fold, that's reported as-is — it's a
   real finding about this dataset, not something to quietly drop.
2. Interval coverage calibration — does the conformal-wrapped global
   tier's claimed (1 - alpha) coverage match its empirical coverage on
   held-out folds? An interval that claims 90% and delivers 60% is worse
   than useless: it's confidently wrong.
3. Regime-shift robustness — folds whose forecast horizon overlaps the
   2020 COVID window are scored separately from folds that don't, so
   degradation during a real demand shock is visible instead of averaged
   away into a single aggregate number.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecasting.conformal import ConformalCalibrator, empirical_coverage
from demand_forecasting.models.baselines import naive_forecast, seasonal_naive_forecast
from demand_forecasting.models.global_model import GlobalDemandModel
from demand_forecasting.models.statistical import exponential_smoothing_forecast
from demand_forecasting.schemas import DemandSeries, EvalResult, ForecastPoint, ForecastResult


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error: sum(|error|) / sum(|actual|).
    Scale-independent (unlike RMSE), so it's the metric used to compare
    tiers across series of very different volume. NaN if actuals sum to
    zero — reported as-is rather than silently swallowed."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error — reported alongside WAPE since it
    penalizes large individual misses more than WAPE does."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass
class FoldSpec:
    """One rolling-origin fold: train on data through train_end, score the
    forecast for horizon_dates. spans_covid flags whether any date in
    horizon_dates falls inside the configured COVID stress-test window."""

    fold_id: int
    train_end: date
    horizon_dates: list[date]
    spans_covid: bool


@dataclass
class EvalReport:
    """Flat list of EvalResult produced by one run_backtest call."""

    results: list[EvalResult] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join(f"[{r.dimension}] {r.detail}" for r in self.results)

    def to_records(self) -> list[dict]:
        """JSON-serializable records — what write_report writes to disk."""
        return [r.model_dump(mode="json") for r in self.results]


def generate_folds(
    series_list: list[DemandSeries],
    horizon_weeks: int,
    n_folds: int,
    step_weeks: int,
    covid_start: date,
    covid_end: date,
) -> list[FoldSpec]:
    """Walk backward from the most recent observed week in n_folds folds,
    each offset by step_weeks, each scoring a horizon_weeks-long horizon.
    Returned in chronological order (oldest fold first)."""
    all_dates = sorted({o.week_start for s in series_list for o in s.observations})
    if not all_dates:
        raise ValueError("cannot generate backtest folds from an empty set of series")
    last_date = pd.Timestamp(all_dates[-1])

    folds = []
    for i in range(n_folds):
        horizon_end = last_date - pd.Timedelta(weeks=step_weeks * i)
        horizon_start = horizon_end - pd.Timedelta(weeks=horizon_weeks - 1)
        train_end = horizon_start - pd.Timedelta(weeks=1)
        horizon_dates = [(horizon_start + pd.Timedelta(weeks=w)).date() for w in range(horizon_weeks)]
        spans_covid = any(covid_start <= d <= covid_end for d in horizon_dates)
        folds.append(FoldSpec(fold_id=i, train_end=train_end.date(), horizon_dates=horizon_dates, spans_covid=spans_covid))

    folds.reverse()
    for new_id, fold in enumerate(folds):
        fold.fold_id = new_id
    return folds


def _to_forecast_result(series: DemandSeries, model_name: str, dates: list[date], values: np.ndarray) -> ForecastResult:
    """Package a tier's raw forecast array into the typed ForecastResult."""
    points = [ForecastPoint(week_start=d, point_forecast=float(v)) for d, v in zip(dates, values)]
    return ForecastResult(series_id=series.series_id, model_name=model_name, points=points)


def _actual_lookup(series_list: list[DemandSeries]) -> dict[tuple[str, str], float]:
    """(series_id, isoformat week_start) -> observed bottles_sold, built
    once per backtest so every fold/tier can look up ground truth in O(1)."""
    return {(s.series_id, o.week_start.isoformat()): o.bottles_sold for s in series_list for o in s.observations}


def _score_tier(
    forecasts: list[ForecastResult],
    model_name: str,
    fold: FoldSpec,
    actual_lookup: dict[tuple[str, str], float],
) -> list[EvalResult]:
    """WAPE + RMSE for one tier's forecasts in one fold, pooled across
    every series and every horizon step. Returns [] if none of the
    forecasted points have a known actual (shouldn't happen in practice,
    but is safer than dividing by zero)."""
    y_true, y_pred = [], []
    for fc in forecasts:
        for p in fc.points:
            key = (fc.series_id, p.week_start.isoformat())
            if key in actual_lookup:
                y_true.append(actual_lookup[key])
                y_pred.append(p.point_forecast)
    if not y_true:
        return []

    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    regime = "COVID-spanning" if fold.spans_covid else "stable"
    w, r = wape(y_true_arr, y_pred_arr), rmse(y_true_arr, y_pred_arr)
    return [
        EvalResult(
            dimension="point_accuracy_wape",
            model_name=model_name,
            fold_id=fold.fold_id,
            spans_covid=fold.spans_covid,
            value=w,
            detail=f"fold {fold.fold_id} ({regime}): {model_name} WAPE={w:.4f} over {len(y_true)} obs",
        ),
        EvalResult(
            dimension="point_accuracy_rmse",
            model_name=model_name,
            fold_id=fold.fold_id,
            spans_covid=fold.spans_covid,
            value=r,
            detail=f"fold {fold.fold_id} ({regime}): {model_name} RMSE={r:.4f} over {len(y_true)} obs",
        ),
    ]


def run_backtest(
    series_list: list[DemandSeries],
    lag_weeks: list[int],
    rolling_windows: list[int],
    horizon_weeks: int,
    n_folds: int,
    step_weeks: int,
    conformal_alpha: float,
    covid_start: date,
    covid_end: date,
    min_series_length_weeks: int,
    seasonal_period_weeks: int = 52,
    lgb_params: dict | None = None,
) -> EvalReport:
    """Run the full rolling-origin backtest: generate folds, forecast every
    tier for every fold, score point accuracy per tier and conformal
    interval coverage for the global tier, and return the flat report. See
    module docstring for what each dimension checks and why."""
    folds = generate_folds(series_list, horizon_weeks, n_folds, step_weeks, covid_start, covid_end)
    actual_lookup = _actual_lookup(series_list)
    results: list[EvalResult] = []

    for fold in folds:
        horizon = len(fold.horizon_dates)
        train_series = [s.truncate(fold.train_end) for s in series_list]
        train_series = [s for s in train_series if len(s.observations) >= min_series_length_weeks]
        if not train_series:
            continue

        tier_forecasts: dict[str, list[ForecastResult]] = {"naive": [], "seasonal_naive": [], "statistical": []}
        for s in train_series:
            hist = s.as_series()
            tier_forecasts["naive"].append(
                _to_forecast_result(s, "naive", fold.horizon_dates, naive_forecast(hist, horizon))
            )
            tier_forecasts["seasonal_naive"].append(
                _to_forecast_result(
                    s,
                    "seasonal_naive",
                    fold.horizon_dates,
                    seasonal_naive_forecast(hist, horizon, season_length=seasonal_period_weeks),
                )
            )
            tier_forecasts["statistical"].append(
                _to_forecast_result(
                    s,
                    "statistical",
                    fold.horizon_dates,
                    exponential_smoothing_forecast(hist, horizon, season_length=seasonal_period_weeks),
                )
            )

        global_model = GlobalDemandModel(lag_weeks, rolling_windows, lgb_params)
        global_model.fit(train_series)
        tier_forecasts["global"] = [
            _to_forecast_result(s, "global", fold.horizon_dates, global_model.predict(s, horizon))
            for s in train_series
        ]

        for model_name, forecasts in tier_forecasts.items():
            results.extend(_score_tier(forecasts, model_name, fold, actual_lookup))

        results.extend(
            _score_conformal_coverage(
                train_series=train_series,
                fold=fold,
                global_forecasts=tier_forecasts["global"],
                lag_weeks=lag_weeks,
                rolling_windows=rolling_windows,
                conformal_alpha=conformal_alpha,
                min_series_length_weeks=min_series_length_weeks,
                actual_lookup=actual_lookup,
                lgb_params=lgb_params,
            )
        )

    return EvalReport(results=results)


def _score_conformal_coverage(
    train_series: list[DemandSeries],
    fold: FoldSpec,
    global_forecasts: list[ForecastResult],
    lag_weeks: list[int],
    rolling_windows: list[int],
    conformal_alpha: float,
    min_series_length_weeks: int,
    actual_lookup: dict[tuple[str, str], float],
    lgb_params: dict | None,
) -> list[EvalResult]:
    """Split conformal, done properly: fit a calibration model on train
    data held further back, forecast the withheld calibration horizon
    (which has known actuals), calibrate a quantile from those residuals,
    then apply it to the real fold's forecasts — the calibration data
    never overlaps the data the reported coverage is measured on."""
    horizon = len(fold.horizon_dates)
    calib_train_end = pd.Timestamp(fold.train_end) - pd.Timedelta(weeks=horizon)
    calib_series = [s.truncate(calib_train_end.date()) for s in train_series]
    calib_series = [s for s in calib_series if len(s.observations) >= min_series_length_weeks]
    if not calib_series:
        return []

    calib_model = GlobalDemandModel(lag_weeks, rolling_windows, lgb_params)
    calib_model.fit(calib_series)

    calib_dates = [(calib_train_end + pd.Timedelta(weeks=w + 1)).date() for w in range(horizon)]
    calib_rows = []
    for s in calib_series:
        preds = calib_model.predict(s, horizon)
        for d, pred in zip(calib_dates, preds):
            key = (s.series_id, d.isoformat())
            if key in actual_lookup:
                calib_rows.append({"actual": actual_lookup[key], "point_forecast": pred, "liquor_type": s.liquor_type})

    if not calib_rows:
        return []

    calibrator = ConformalCalibrator(alpha=conformal_alpha)
    calibrator.calibrate(pd.DataFrame(calib_rows))

    intervals = []
    for fc in global_forecasts:
        liquor_type = fc.series_id.split("::")[1]
        intervals.extend(calibrator.wrap(fc, liquor_type=liquor_type))
    if not intervals:
        return []

    coverage = empirical_coverage(intervals, actual_lookup)
    regime = "COVID-spanning" if fold.spans_covid else "stable"
    target = 1 - conformal_alpha
    return [
        EvalResult(
            dimension="interval_coverage",
            model_name="global+conformal",
            fold_id=fold.fold_id,
            spans_covid=fold.spans_covid,
            value=coverage,
            detail=(
                f"fold {fold.fold_id} ({regime}): empirical coverage={coverage:.3f} "
                f"target={target:.2f} over {len(intervals)} intervals"
            ),
        )
    ]


def write_report(report: EvalReport, out_dir: str | Path) -> Path:
    """Write report as timestamped JSON under out_dir and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"eval_run_{timestamp}.json"
    path.write_text(json.dumps(report.to_records(), indent=2, default=str))
    return path
