"""
Stage 3 — Modeling for the monthly, category-level forecasting
exploration. Five tiers, fit and evaluated SEPARATELY for each of the 8
categories (per the explicit instruction after Stage 1/2) — nothing here
pools categories, including the LightGBM tier, which is deliberately
per-category here (unlike the production pipeline's pooled global model).

Consumes exploration/outputs/stage2_data_prep/processed_features.parquet
(the cleaned, feature-engineered, split-labeled output of Stage 2).

Two methodological choices worth stating up front rather than leaving
implicit:

1. LightGBM and SARIMAX are trained on the RAW engineered features, not
   Stage 2's RobustScaler output. LightGBM is a tree model — split
   decisions depend only on feature order, not magnitude, so a monotonic
   rescaling changes nothing about what it learns. SARIMAX's exogenous
   regressors here are small binary/count flags with no scale disparity
   worth correcting. The Ridge ensemble in Model 5 DOES need scaling,
   though for a different reason than usual: its three inputs are all the
   same unit (bottles/month) so there's no cross-feature mismatch, but
   Ridge's penalty (alpha * sum(w^2)) is only meaningful relative to the
   squared-error loss it's competing against — with predictions in the
   tens of thousands, that loss is ~1e8-1e10, so alpha values anywhere
   near sklearn's default (1.0) or even a few thousand are numerically
   inert. Model 5's inputs are standardized before fitting specifically so
   a leave-one-out alpha search actually has a regularization effect to
   find.

2. Every model is evaluated with genuine multi-step-ahead forecasts, not
   one-step predictions computed from real historical lags. Concretely:
   fit on TRAIN only, forecast the entire validation horizon by feeding
   each step's own prior forecasts into the next step's lag features
   (recursive rollout) — mirroring exactly how the production pipeline's
   GlobalDemandModel.predict works, and how a real deployment would have
   to forecast, since real future actuals aren't available yet. The one
   exception is LightGBM's Optuna tuning objective, which uses the
   validation SPLIT'S real lag features (built from true history, since
   they're already known at tuning time) only for early-stopping
   efficiency — the actual objective Optuna minimizes is still the
   genuine recursive-rollout WAPE.

Two-phase fitting per model, per category:
  - Phase A (validation): fit on TRAIN only, forecast the 6-month
    validation horizon. Used for hyperparameter tuning and for training
    the Model 5 Ridge ensemble.
  - Phase B (test): refit on TRAIN+VALIDATION, forecast the 12-month test
    holdout. This is the number Stage 4 evaluates as the honest one.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import warnings
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pmdarima as pm
from common import (
    OUTPUT_ROOT,
    PANDEMIC_END,
    PANDEMIC_START,
    holiday_month_starts,
    holidays_per_month,
    major_holidays,
)
from pmdarima.arima import ARIMA as PmdARIMA
from prophet import Prophet
from run_pipeline import ENSEMBLE_INPUTS, PipelineConfig
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from demand_forecasting.evaluation.eval_suite import rmse, wape
from demand_forecasting.models.baselines import seasonal_naive_forecast

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

MODEL_ORDER = ["naive", "prophet", "lightgbm", "sarimax", "ensemble"]

IN_DIR = OUTPUT_ROOT / "stage2_data_prep"
OUT = OUTPUT_ROOT / "stage3_modeling"

LAG_MONTHS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]
EXOG_COLS = ["is_holiday_month", "n_holidays_in_month", "is_pandemic"]
LGB_FEATURE_COLS = (
    ["month_sin", "month_cos", "quarter", "time_index", "n_holidays_in_month", "is_holiday_month"]
    + ["months_since_last_holiday", "months_until_next_holiday", "is_pandemic", "is_imputed", "was_clipped"]
    + [f"lag_{lag}" for lag in LAG_MONTHS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + ["rolling_std_3"]
)
FIXED_LGB_PARAMS = {"n_estimators": 500, "bagging_freq": 1, "random_state": 7, "verbosity": -1}
N_OPTUNA_TRIALS = 50


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# --------------------------------------------------------------------------
# Model 1 — naive seasonal baseline (reuses the production tier directly)
# --------------------------------------------------------------------------
def naive_seasonal(history: pd.Series, horizon: int) -> np.ndarray:
    return seasonal_naive_forecast(history, horizon, season_length=12)


# --------------------------------------------------------------------------
# Model 2 — Prophet
# --------------------------------------------------------------------------
def build_prophet_holidays(years: range) -> pd.DataFrame:
    holidays = major_holidays(years)
    df = holidays.rename(columns={"holiday_name": "holiday"})[["holiday", "date"]]
    df["ds"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df[["ds", "holiday"]]


def fit_prophet_forecast(train_df: pd.DataFrame, horizon: int, cps: float, sps: float, holiday_df: pd.DataFrame) -> np.ndarray:
    p_df = train_df[["month_start", "units_sold", "is_pandemic"]].rename(columns={"month_start": "ds", "units_sold": "y"})
    with _quiet():
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  # N/A at monthly grain
            daily_seasonality=False,
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            holidays=holiday_df,
        )
        model.add_regressor("is_pandemic")
        model.fit(p_df)
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        future["is_pandemic"] = future["ds"].apply(lambda d: int(PANDEMIC_START <= d.date() <= PANDEMIC_END))
        forecast = model.predict(future)
    return np.clip(forecast["yhat"].tail(horizon).to_numpy(), 0, None)


def tune_prophet(train_df: pd.DataFrame, val_actual: np.ndarray, holiday_df: pd.DataFrame) -> tuple[float, float, np.ndarray, float]:
    best = None
    for cps in (0.01, 0.05, 0.1, 0.5):
        for sps in (0.1, 1.0, 10.0):
            forecast = fit_prophet_forecast(train_df, len(val_actual), cps, sps, holiday_df)
            score = wape(val_actual, forecast)
            if best is None or score < best[3]:
                best = (cps, sps, forecast, score)
    return best


# --------------------------------------------------------------------------
# Model 3 — LightGBM, per category, Optuna-tuned
# --------------------------------------------------------------------------
def calendar_features_for_date(month_start: pd.Timestamp, holiday_idx: pd.DatetimeIndex, holiday_counts: pd.Series, year_min: int) -> dict:
    period = month_start.to_period("M")
    n_hol = int(holiday_counts.get(period, 0))
    past = holiday_idx[holiday_idx <= month_start]
    future = holiday_idx[holiday_idx >= month_start]
    months_since = int((period - past[-1].to_period("M")).n) if len(past) else np.nan
    months_until = int((future[0].to_period("M") - period).n) if len(future) else np.nan
    return {
        "month_sin": np.sin(2 * np.pi * month_start.month / 12),
        "month_cos": np.cos(2 * np.pi * month_start.month / 12),
        "quarter": month_start.quarter,
        "time_index": (month_start.year - year_min) * 12 + month_start.month,
        "n_holidays_in_month": n_hol,
        "is_holiday_month": int(n_hol > 0),
        "months_since_last_holiday": months_since,
        "months_until_next_holiday": months_until,
        "is_pandemic": int(PANDEMIC_START <= month_start.date() <= PANDEMIC_END),
        "is_imputed": 0,
        "was_clipped": 0,
    }


class LightGBMCategoryModel:
    def __init__(self, params: dict):
        self.params = params
        self.model: lgb.LGBMRegressor | None = None
        self.best_iteration: int | None = None

    def fit(self, train_rows: pd.DataFrame, eval_rows: pd.DataFrame | None = None) -> LightGBMCategoryModel:
        X, y = train_rows[LGB_FEATURE_COLS], train_rows["units_sold"]
        fit_kwargs = {}
        if eval_rows is not None and len(eval_rows):
            fit_kwargs["eval_set"] = [(eval_rows[LGB_FEATURE_COLS], eval_rows["units_sold"])]
            fit_kwargs["callbacks"] = [lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)]
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y, **fit_kwargs)
        self.best_iteration = getattr(self.model, "best_iteration_", None) or self.params["n_estimators"]
        return self

    def recursive_forecast(self, history: pd.Series, horizon: int, year_min: int, holiday_years: range) -> np.ndarray:
        holiday_idx = holiday_month_starts(holiday_years)
        holiday_counts = holidays_per_month(holiday_years)
        extended = history.copy()
        preds = np.empty(horizon)
        for h in range(horizon):
            target = extended.index[-1] + pd.DateOffset(months=1)
            vals = extended.to_numpy(dtype=float)
            feat = calendar_features_for_date(target, holiday_idx, holiday_counts, year_min)
            for lag in LAG_MONTHS:
                feat[f"lag_{lag}"] = vals[-lag] if len(vals) >= lag else np.nan
            for w in ROLLING_WINDOWS:
                feat[f"rolling_mean_{w}"] = float(np.mean(vals[-w:])) if len(vals) >= w else np.nan
            feat["rolling_std_3"] = float(np.std(vals[-3:], ddof=1)) if len(vals) >= 3 else np.nan
            row = pd.DataFrame([feat])[LGB_FEATURE_COLS]
            pred = max(float(self.model.predict(row)[0]), 0.0)
            preds[h] = pred
            extended.loc[target] = pred
        return preds


def tune_lightgbm(category_df: pd.DataFrame, val_actual: np.ndarray, year_min: int, holiday_years: range) -> tuple[dict, np.ndarray, int, pd.Series]:
    train_rows = category_df[(category_df["split"] == "train") & category_df["lag_1"].notna()]
    val_rows = category_df[category_df["split"] == "validation"]
    history = category_df[category_df["split"] == "train"].set_index("month_start")["units_sold"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            **FIXED_LGB_PARAMS,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 30),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = LightGBMCategoryModel(params).fit(train_rows, eval_rows=val_rows)
        forecast = model.recursive_forecast(history, len(val_actual), year_min, holiday_years)
        return wape(val_actual, forecast)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params = {**FIXED_LGB_PARAMS, **study.best_params}
    final_model = LightGBMCategoryModel(best_params).fit(train_rows, eval_rows=val_rows)
    val_forecast = final_model.recursive_forecast(history, len(val_actual), year_min, holiday_years)
    gain = final_model.model.booster_.feature_importance(importance_type="gain")
    importance = pd.Series(gain, index=LGB_FEATURE_COLS).sort_values(ascending=False)
    return best_params, val_forecast, final_model.best_iteration, importance


# --------------------------------------------------------------------------
# Model 4 — SARIMAX via auto_arima
# --------------------------------------------------------------------------
def fit_sarimax_forecast(train_df: pd.DataFrame, exog_future: np.ndarray, horizon: int):
    y = train_df["units_sold"].to_numpy(dtype=float)
    X = train_df[EXOG_COLS].to_numpy(dtype=float)
    model = pm.auto_arima(y, X=X, seasonal=True, m=12, information_criterion="aic", stepwise=True, suppress_warnings=True, error_action="ignore")
    forecast = np.clip(np.asarray(model.predict(n_periods=horizon, X=exog_future)), 0, None)
    return forecast, model.order, model.seasonal_order


def refit_sarimax_forecast(order, seasonal_order, trainval_df: pd.DataFrame, exog_future: np.ndarray, horizon: int) -> np.ndarray:
    """Reuse the order auto_arima selected on TRAIN, refit coefficients on
    TRAIN+VALIDATION for the test forecast — avoids re-running the AIC
    search twice per category while still using all pre-test data."""
    y = trainval_df["units_sold"].to_numpy(dtype=float)
    X = trainval_df[EXOG_COLS].to_numpy(dtype=float)
    model = PmdARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
    model.fit(y, X=X)
    return np.clip(np.asarray(model.predict(n_periods=horizon, X=exog_future)), 0, None)


# --------------------------------------------------------------------------
# Model 5 — Ridge ensemble, alpha chosen by leave-one-out (not left at an
# untuned default): fitting 3 weights + an intercept on only 6 validation
# points, from 3 base-model predictions that are themselves highly
# correlated (all forecasting the same series), is a classic small-N,
# near-collinear regression — prone to wild, unstable coefficients that
# fit the 6 validation points closely but generalize badly. Leave-one-out
# over the 6 points picks the regularization strength that actually
# minimizes out-of-sample error on the only real evidence available,
# rather than trusting alpha=1.0 by convention.
# --------------------------------------------------------------------------
RIDGE_ALPHAS = (0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def fit_ridge_ensemble_loo(X_val: np.ndarray, y_val: np.ndarray) -> tuple[Ridge, StandardScaler, float]:
    """Standardize X first (see module docstring for why Ridge's alpha is
    meaningless unscaled here), then pick alpha by leave-one-out over the
    (small) validation set. Returns the fitted scaler alongside the model
    since callers must apply it to any future X before predicting."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_val)
    n = len(y_val)
    best_alpha, best_error = RIDGE_ALPHAS[0], np.inf
    for alpha in RIDGE_ALPHAS:
        errors = []
        for i in range(n):
            mask = np.arange(n) != i
            model = Ridge(alpha=alpha, positive=True)
            model.fit(X_scaled[mask], y_val[mask])
            errors.append(abs(model.predict(X_scaled[[i]])[0] - y_val[i]))
        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_alpha, best_error = alpha, mean_error
    final_model = Ridge(alpha=best_alpha, positive=True).fit(X_scaled, y_val)
    return final_model, scaler, best_alpha


def main(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    enabled = set(config.models)
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(IN_DIR / "processed_features.parquet")

    available_categories = set(features["liquor_type"].unique())
    if config.categories is not None:
        unknown = set(config.categories) - available_categories
        if unknown:
            raise ValueError(f"unknown categor{'y' if len(unknown) == 1 else 'ies'} in config: {sorted(unknown)} — available: {sorted(available_categories)}")
        categories = sorted(config.categories)
    else:
        categories = sorted(available_categories)

    year_min = features["month_start"].dt.year.min()
    holiday_years = range(year_min, features["month_start"].dt.year.max() + 2)
    holiday_df = build_prophet_holidays(holiday_years) if "prophet" in enabled else None

    report = [f"STAGE 3 MODELING REPORT — {datetime.now(tz=timezone.utc).date().isoformat()}", "=" * 78]
    report.append(f"Models enabled: {sorted(enabled)}. Categories: {categories}.")
    report.append("=" * 78)

    all_predictions = []
    wape_rows = []
    prophet_params_rows = []
    sarimax_order_rows = []
    ridge_weight_rows = []
    importance_frames = []
    lgb_final_params = {}
    prophet_final_params = {}
    sarimax_final_orders = {}
    ridge_final = {}

    for category in categories:
        cat_df = features[features["liquor_type"] == category].sort_values("month_start").reset_index(drop=True)
        train_df = cat_df[cat_df["split"] == "train"]
        val_df = cat_df[cat_df["split"] == "validation"]
        test_df = cat_df[cat_df["split"] == "test"]
        trainval_df = cat_df[cat_df["split"].isin(["train", "validation"])]

        val_actual = val_df["units_sold"].to_numpy()
        test_actual = test_df["units_sold"].to_numpy()
        history_train = train_df.set_index("month_start")["units_sold"]
        history_trainval = trainval_df.set_index("month_start")["units_sold"]

        val_forecasts: dict[str, np.ndarray] = {}
        test_forecasts: dict[str, np.ndarray] = {}

        # --- Model 1: naive seasonal ---
        if "naive" in enabled:
            val_forecasts["naive"] = naive_seasonal(history_train, len(val_df))
            test_forecasts["naive"] = naive_seasonal(history_trainval, len(test_df))

        # --- Model 2: Prophet ---
        if "prophet" in enabled:
            cps, sps, prophet_val, prophet_val_wape = tune_prophet(train_df, val_actual, holiday_df)
            prophet_test = fit_prophet_forecast(trainval_df, len(test_df), cps, sps, holiday_df)
            val_forecasts["prophet"] = prophet_val
            test_forecasts["prophet"] = prophet_test
            prophet_params_rows.append({"liquor_type": category, "changepoint_prior_scale": cps, "seasonality_prior_scale": sps, "val_wape": prophet_val_wape})
            prophet_final_params[category] = {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps}

        # --- Model 3: LightGBM ---
        if "lightgbm" in enabled:
            best_lgb_params, lgb_val, best_iter, importance = tune_lightgbm(cat_df, val_actual, year_min, holiday_years)
            trainval_rows = cat_df[cat_df["split"].isin(["train", "validation"]) & cat_df["lag_1"].notna()]
            test_model_params = {**best_lgb_params, "n_estimators": best_iter}
            test_model = LightGBMCategoryModel(test_model_params).fit(trainval_rows)
            lgb_test = test_model.recursive_forecast(history_trainval, len(test_df), year_min, holiday_years)
            val_forecasts["lightgbm"] = lgb_val
            test_forecasts["lightgbm"] = lgb_test
            importance_frames.append(importance.rename(category))
            lgb_final_params[category] = test_model_params

        # --- Model 4: SARIMAX ---
        if "sarimax" in enabled:
            sarimax_val, order, seasonal_order = fit_sarimax_forecast(train_df, val_df[EXOG_COLS].to_numpy(dtype=float), len(val_df))
            sarimax_test = refit_sarimax_forecast(order, seasonal_order, trainval_df, test_df[EXOG_COLS].to_numpy(dtype=float), len(test_df))
            val_forecasts["sarimax"] = sarimax_val
            test_forecasts["sarimax"] = sarimax_test
            sarimax_order_rows.append({"liquor_type": category, "order": str(order), "seasonal_order": str(seasonal_order)})
            sarimax_final_orders[category] = {"order": list(order), "seasonal_order": list(seasonal_order)}

        # --- Model 5: Ridge ensemble, stacking whichever of prophet/lightgbm/sarimax are enabled ---
        if "ensemble" in enabled:
            ensemble_inputs = [m for m in ENSEMBLE_INPUTS if m in enabled]  # config validation guarantees len>=2
            X_val = np.column_stack([val_forecasts[m] for m in ensemble_inputs])
            X_test = np.column_stack([test_forecasts[m] for m in ensemble_inputs])
            ridge, ridge_scaler, ridge_alpha = fit_ridge_ensemble_loo(X_val, val_actual)
            ensemble_test = np.clip(ridge.predict(ridge_scaler.transform(X_test)), 0, None)
            ensemble_val_fitted = np.clip(ridge.predict(ridge_scaler.transform(X_val)), 0, None)  # in-sample, context only
            val_forecasts["ensemble"] = ensemble_val_fitted
            test_forecasts["ensemble"] = ensemble_test
            # Ridge was fit on standardized inputs (see fit_ridge_ensemble_loo); convert
            # coef_/intercept_ back to original-scale weights so they're interpretable
            # as "contribution per bottle of that model's raw prediction."
            raw_weights = ridge.coef_ / ridge_scaler.scale_
            raw_intercept = ridge.intercept_ - np.sum(ridge.coef_ * ridge_scaler.mean_ / ridge_scaler.scale_)
            ridge_weight_rows.append(
                {
                    "liquor_type": category,
                    "alpha": ridge_alpha,
                    "inputs": "+".join(ensemble_inputs),
                    **{f"w_{m}": w for m, w in zip(ensemble_inputs, raw_weights)},
                    "intercept": raw_intercept,
                }
            )
            ridge_final[category] = {
                "inputs": ensemble_inputs,
                "scaler_mean": ridge_scaler.mean_.tolist(),
                "scaler_scale": ridge_scaler.scale_.tolist(),
                "coef": ridge.coef_.tolist(),
                "intercept": float(ridge.intercept_),
            }

        # --- collect predictions ---
        for split_name, dates, actual, forecasts in (
            ("validation", val_df["month_start"], val_actual, val_forecasts),
            ("test", test_df["month_start"], test_actual, test_forecasts),
        ):
            for model_name, preds in forecasts.items():
                for date, act, pred in zip(dates, actual, preds):
                    all_predictions.append({"liquor_type": category, "month_start": date, "split": split_name, "model": model_name, "actual": act, "predicted": pred})
                w = wape(actual, np.asarray(preds))
                r = rmse(actual, np.asarray(preds))
                wape_rows.append({"liquor_type": category, "split": split_name, "model": model_name, "wape": w, "rmse": r})

    predictions = pd.DataFrame(all_predictions)
    wape_table = pd.DataFrame(wape_rows)
    prophet_params = pd.DataFrame(prophet_params_rows)
    sarimax_orders = pd.DataFrame(sarimax_order_rows)
    ridge_weights = pd.DataFrame(ridge_weight_rows)
    importance_table = pd.concat(importance_frames, axis=1) if importance_frames else pd.DataFrame()

    model_order = [m for m in MODEL_ORDER if m in enabled]
    val_pivot = wape_table[wape_table["split"] == "validation"].pivot(index="liquor_type", columns="model", values="wape")[model_order]
    test_pivot = wape_table[wape_table["split"] == "test"].pivot(index="liquor_type", columns="model", values="wape")[model_order]

    report.append("SECTION 1 — VALIDATION WAPE BY MODEL x CATEGORY (recursive rollout)")
    report.append("")
    if "ensemble" in enabled:
        report.append("  NOTE: 'ensemble' here is the Ridge meta-learner's IN-SAMPLE fit to this same")
        report.append("  validation data (it was trained on exactly these predictions/actuals) — expect it to")
        report.append("  look artificially strong. The honest ensemble number is the TEST WAPE in Section 2.")
        report.append("")
    report.append(_indent(val_pivot.round(3).to_string()))
    report.append("=" * 78)

    report.append("SECTION 2 — TEST WAPE BY MODEL x CATEGORY (holdout, genuinely out-of-sample)")
    report.append("")
    report.append(_indent(test_pivot.round(3).to_string()))
    report.append("")
    champion = test_pivot.idxmin(axis=1)
    report.append("  Champion (lowest test WAPE) per category:")
    report.append(_indent(champion.to_string()))
    report.append("=" * 78)

    report.append("SECTION 3 — PROPHET: TUNED HYPERPARAMETERS (grid search on validation WAPE)")
    report.append("")
    report.append(_indent(prophet_params.round(4).to_string(index=False)) if "prophet" in enabled else "  Skipped — 'prophet' not in config.models.")
    report.append("=" * 78)

    report.append("SECTION 4 — SARIMAX: AUTO_ARIMA-SELECTED ORDERS")
    report.append("")
    report.append(_indent(sarimax_orders.to_string(index=False)) if "sarimax" in enabled else "  Skipped — 'sarimax' not in config.models.")
    report.append("=" * 78)

    report.append("SECTION 5 — LIGHTGBM: TOP-5 FEATURE IMPORTANCE (gain-based) BY CATEGORY")
    report.append("")
    if "lightgbm" in enabled:
        report.append("  Full gain-based importance for every feature x category saved to feature_importance_gain.csv.")
        top5 = pd.DataFrame({cat: importance_table[cat].sort_values(ascending=False).head(5).index for cat in categories})
        report.append(_indent(top5.to_string(index=False)))
    else:
        report.append("  Skipped — 'lightgbm' not in config.models.")
    report.append("=" * 78)

    report.append("SECTION 6 — RIDGE ENSEMBLE WEIGHTS (alpha chosen by leave-one-out over the validation points)")
    report.append("")
    report.append(_indent(ridge_weights.round(3).to_string(index=False)) if "ensemble" in enabled else "  Skipped — 'ensemble' not in config.models.")

    full_report = "\n\n".join(report)
    print(full_report)

    (OUT / "stage3_modeling_report.txt").write_text(full_report)
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    wape_table.to_csv(OUT / "wape_table.csv", index=False)
    importance_table.to_csv(OUT / "feature_importance_gain.csv")
    (OUT / "lgb_best_params.json").write_text(json.dumps(lgb_final_params, indent=2))
    (OUT / "prophet_params.json").write_text(json.dumps(prophet_final_params, indent=2))
    (OUT / "sarimax_orders.json").write_text(json.dumps(sarimax_final_orders, indent=2))
    (OUT / "ridge_ensemble.json").write_text(json.dumps(ridge_final, indent=2))
    print(f"\nWrote report, predictions.parquet, wape_table.csv, and per-model param JSON files to {OUT}")


if __name__ == "__main__":
    main()
