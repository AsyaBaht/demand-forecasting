"""
The global tier: one LightGBM regressor trained jointly across every
(store, liquor_type) series, with lag features, rolling-window features,
and store_number / liquor_type as categorical features.

Why one joint model instead of ~200 independent per-series models: most of
these series are short and individually noisy (a mid-volume store's
brandy sales, say), and a per-series model has nothing to generalize
from — it has to learn the whole demand pattern (weekly seasonality,
holiday spikes, trend) from that one series alone. Pooling series lets the
model borrow statistical strength across stores and categories: whiskey's
December spike shows up in dozens of series, so the model can learn "this
is a December effect" instead of re-discovering it from scratch, noisily,
per series. The trade-off is that a genuinely idiosyncratic series (one
store with an unusual local pattern) can get smoothed toward the pooled
average — which is exactly what the statistical tier exists to catch when
it doesn't.

Feature leakage note: every lag and rolling-window feature is built from
`shift(1)` before the rolling window is applied, so the feature for week t
only ever sees data through week t-1. Recursive multi-step prediction
mirrors this exactly — each forecasted step is appended to the series
before computing features for the next step.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand_forecasting.schemas import DemandSeries

DEFAULT_LGB_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 7,
    "verbosity": -1,
}


class GlobalDemandModel:
    """One LightGBM regressor fit jointly across every series passed to
    `fit`. `lag_weeks` and `rolling_windows` control the feature set (see
    module docstring); `lgb_params` overrides DEFAULT_LGB_PARAMS."""

    def __init__(
        self,
        lag_weeks: list[int],
        rolling_windows: list[int],
        lgb_params: dict | None = None,
    ):
        self.lag_weeks = sorted(lag_weeks)
        self.rolling_windows = sorted(rolling_windows)
        self.lgb_params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
        self.model: lgb.LGBMRegressor | None = None
        self.feature_columns: list[str] | None = None
        self._store_categories: list[int] | None = None
        self._liquor_categories: list[str] | None = None

    @property
    def categorical_columns(self) -> list[str]:
        return ["store_number", "liquor_type"]

    def _feature_frame_for_training(self, series_list: list[DemandSeries]) -> pd.DataFrame:
        """Build the long-format (series x week) training frame: one row
        per observed week per series, with lag/rolling/calendar features
        and the raw bottles_sold target."""
        frames = []
        for s in series_list:
            sr = s.as_series()
            if len(sr) < 2:
                continue
            df = pd.DataFrame(index=sr.index)
            df["bottles_sold"] = sr
            for lag in self.lag_weeks:
                df[f"lag_{lag}"] = sr.shift(lag)
            for w in self.rolling_windows:
                shifted = sr.shift(1)
                df[f"roll_mean_{w}"] = shifted.rolling(w).mean()
                df[f"roll_std_{w}"] = shifted.rolling(w).std()
            df["store_number"] = s.store_number
            df["liquor_type"] = s.liquor_type
            df["month"] = df.index.month
            df["weekofyear"] = df.index.isocalendar().week.astype(int)
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        full = pd.concat(frames)
        # Rows where even the shortest lag is unavailable carry no signal
        # beyond the calendar/categorical features — drop rather than let
        # LightGBM learn from an all-NaN-lag row.
        min_lag_col = f"lag_{self.lag_weeks[0]}"
        return full[full[min_lag_col].notna()].reset_index(drop=True)

    def fit(self, series_list: list[DemandSeries]) -> None:
        """Fit the joint LightGBM model on every series in series_list.
        Records the store_number / liquor_type categories seen here so
        `predict` can encode new series consistently."""
        frame = self._feature_frame_for_training(series_list)
        if frame.empty:
            raise ValueError("no training rows after feature construction — series too short for configured lags")

        self._store_categories = sorted({s.store_number for s in series_list})
        self._liquor_categories = sorted({s.liquor_type for s in series_list})
        frame["store_number"] = pd.Categorical(frame["store_number"], categories=self._store_categories)
        frame["liquor_type"] = pd.Categorical(frame["liquor_type"], categories=self._liquor_categories)

        self.feature_columns = [
            *[f"lag_{lag}" for lag in self.lag_weeks],
            *[col for w in self.rolling_windows for col in (f"roll_mean_{w}", f"roll_std_{w}")],
            "month",
            "weekofyear",
            "store_number",
            "liquor_type",
        ]
        X = frame[self.feature_columns]
        y = frame["bottles_sold"]
        self.model = lgb.LGBMRegressor(**self.lgb_params)
        self.model.fit(X, y, categorical_feature=self.categorical_columns)

    def _next_step_feature_row(
        self, extended: pd.Series, store_number: int, liquor_type: str, target_week: pd.Timestamp
    ) -> pd.DataFrame:
        """Build the single feature row for forecasting target_week, given
        `extended` (history plus any already-forecasted steps). Mirrors the
        shift(1)-then-rolling logic in `_feature_frame_for_training` exactly."""
        vals = extended.to_numpy(dtype=float)
        feat: dict = {}
        for lag in self.lag_weeks:
            feat[f"lag_{lag}"] = vals[-lag] if len(vals) >= lag else np.nan
        for w in self.rolling_windows:
            if len(vals) >= w:
                window = vals[-w:]
                feat[f"roll_mean_{w}"] = float(np.mean(window))
                feat[f"roll_std_{w}"] = float(np.std(window, ddof=1))
            else:
                feat[f"roll_mean_{w}"] = np.nan
                feat[f"roll_std_{w}"] = np.nan
        feat["month"] = target_week.month
        feat["weekofyear"] = int(target_week.isocalendar()[1])

        row = pd.DataFrame([feat])
        row["store_number"] = pd.Categorical([store_number], categories=self._store_categories)
        row["liquor_type"] = pd.Categorical([liquor_type], categories=self._liquor_categories)
        return row[self.feature_columns]

    def predict(self, series: DemandSeries, horizon: int) -> np.ndarray:
        """Recursive multi-step forecast: predict one week, append the
        prediction to the series, recompute features, repeat. Negative
        predictions are clipped to zero (bottle counts can't be negative)."""
        if self.model is None:
            raise RuntimeError("GlobalDemandModel must be fit before predict")

        extended = series.as_series().copy()
        preds = np.empty(horizon)
        step = pd.Timedelta(weeks=1)
        for h in range(horizon):
            target_week = extended.index[-1] + step
            row = self._next_step_feature_row(extended, series.store_number, series.liquor_type, target_week)
            pred = max(float(self.model.predict(row)[0]), 0.0)
            preds[h] = pred
            extended.loc[target_week] = pred
        return preds
