# Demand Forecasting

Weekly retail demand forecasting on real Iowa liquor wholesale data
(`bigquery-public-data.iowa_liquor_sales.sales`) — a tiered model stack
(naive baselines, per-series statistical models, one global LightGBM model
pooled across series) wrapped in split conformal prediction intervals, and
backtested with rolling-origin evaluation instead of a single train/test
split.

## Why this exists

A point forecast without an honest interval isn't very useful for a
planning decision — "we'll sell about 40 bottles" doesn't tell a store
manager how much safety stock to hold. And a model that beats a naive
baseline on one lucky snapshot of the calendar isn't demonstrating
skill, it's demonstrating variance. This project exists to build both
things properly: calibrated intervals (conformal prediction, which makes
no distributional assumption about residuals) and a backtest that scores
every tier across multiple rolling folds, including folds that span a
real demand shock (2020), instead of one split that happens to avoid it.

## Scope decisions, stated plainly

- **Top 25 stores by volume, not all ~3,000.** Iowa's dataset has a long
  tail of stores with a handful of transactions a year — not enough
  history to forecast anything. Capping at the top 25 keeps every series
  in scope actually worth forecasting, and keeps the joint model's
  training set (~25 stores × up to 8 liquor types ≈ 200 series) small
  enough to fit and reason about on a laptop.
- **Weekly granularity, not daily.** Daily bottle counts per
  store/liquor_type are dominated by zero-sale days and day-of-week
  ordering effects most categories don't sell every day — that's noise a
  forecaster would have to learn around for no benefit, since this is a
  stocking/planning problem, not same-day fulfillment. Weekly aggregation
  trades resolution nobody needs for a much better signal-to-noise ratio.
- **Category grouped into 7 liquor types + "other", not left at ~90 raw
  category codes.** Iowa's category codes are far more granular than
  useful here (multiple whiskey sub-codes, etc.); grouping by the first
  three digits collapses them into buckets with enough volume per store
  to model, instead of 90+ mostly-empty per-SKU series.
- **2019-onward, COVID included, not excluded.** Early years (2012-2018)
  are sparse and inconsistently reported. 2020 is left in deliberately —
  see "Regime-shift evaluation" below.

## Architecture: the tiered stack

1. **Naive baselines** (`models/baselines.py`) — plain naive (repeat the
   last value) and seasonal-naive (same week last year). These aren't
   scaffolding to delete later; they're the reference frame every fancier
   tier has to actually beat.
2. **Statistical tier** (`models/statistical.py`) — per-series Holt-Winters
   exponential smoothing (SARIMAX also available), fit independently for
   each (store, liquor_type) series. Exists to catch idiosyncratic,
   per-series patterns the pooled global model might smooth away.
3. **Global tier** (`models/global_model.py`) — one LightGBM regressor
   trained jointly across *every* series, with lag features (1, 2, 3, 4,
   8, 52 weeks), rolling-window mean/std features, and store_number /
   liquor_type as categorical features. A global model generalizes across
   sparse series better than ~200 independent per-series models — whiskey's
   December spike shows up in dozens of series, so the model learns "this
   is a December effect" instead of re-discovering it noisily per series.
4. **Conformal prediction** (`conformal.py`) — split conformal prediction
   wraps the global tier's point forecasts in calibrated intervals,
   grouped by liquor_type (residual scale varies enormously between, say,
   whiskey and cordial liqueur — a single pooled quantile would be wrong
   for both). No distributional assumption on residuals required; the
   interval has close to its target marginal coverage by construction.

## Evaluation: rolling-origin backtesting

`evaluation/eval_suite.py` walks backward from the most recent observed
week in several folds, each scoring a multi-week horizon, and checks
three genuinely different things:

1. **Point-forecast accuracy** (WAPE, RMSE) for every tier, per fold,
   against the naive baselines. If the global model doesn't beat
   seasonal-naive on a given fold, that's reported as a real finding, not
   hidden.
2. **Interval coverage calibration** — does the conformal-wrapped global
   tier's claimed (1 − α) coverage match its *empirical* coverage on
   held-out folds? An interval that claims 90% and delivers 60% is worse
   than useless — it's confidently wrong.
3. **Regime-shift robustness** — folds whose forecast horizon overlaps the
   2020 COVID window are scored and reported separately from folds that
   don't.

Each run writes a timestamped JSON report to `reports/eval_runs/`, not
just stdout.

## A real finding from the synthetic fixture

`tests/conftest.py` builds a synthetic multi-series dataset with a known,
fixed data-generating process (strong exact seasonality + linear trend +
small noise + a synthetic COVID-style 1.6× demand shock over
2020-03–2020-12), used so the whole suite runs with zero BigQuery access.
Running the backtest against it surfaces something worth stating plainly
rather than averaging away: **seasonal-naive's "same week last year"
reference stays contaminated by the shock for a full year after it ends**,
not just on folds whose forecast horizon overlaps 2020 — a fold scored in
2022 can still pull a shocked reference from 2021. Plain naive doesn't
have this failure mode, because its most recent observation already
reflects whatever regime the series is currently in. This is exactly the
kind of thing a single train/test split would never surface, and exactly
why the eval suite scores COVID-spanning folds separately instead of
folding them into one aggregate number.

## Project layout

```
config/settings.py       # every tunable knob — externalized, not hardcoded
sql/extract_demand.sql   # the BigQuery extraction query
src/demand_forecasting/
  schemas.py              # typed models passed between every stage
  ingestion/               # BigQuery extraction + raw -> DemandSeries aggregation
  models/                  # baselines, statistical tier, global LightGBM tier
  conformal.py             # split conformal prediction
  evaluation/eval_suite.py # rolling-origin backtest + report writer
  cli.py                   # extract / aggregate / backtest subcommands
tests/                    # everything runs against tests/conftest.py's
                            synthetic fixture — zero BigQuery access required
data/, reports/           # gitignored — regenerated by the pipeline
```

## Running it

```bash
pip install -e ".[dev,bigquery]"
pytest                                        # offline, no GCP credentials needed

# requires GCP application-default credentials with BigQuery read access
demand-forecasting extract --project YOUR_GCP_PROJECT
demand-forecasting aggregate
demand-forecasting backtest
```

`extract` and `aggregate` are not covered by the test suite — they need
real credentials and a real query. Everything else runs against the
synthetic fixture in `tests/conftest.py`.
