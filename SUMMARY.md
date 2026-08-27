# Demand Forecasting — Project Summary

Demand forecasting on real Iowa liquor wholesale data — a tiered
naive/Holt-Winters/global-LightGBM stack wrapped in split-conformal
prediction intervals, plus a second Prophet/LightGBM/SARIMAX/Ridge-ensemble
bake-off at monthly category granularity, all benchmarked against naive
and seasonal-naive baselines — validated with rolling-origin backtesting
and a synthetic fixture with known seasonal and regime-shift ground truth
before being applied to 73,482 real weekly transactions across Iowa's top
25 liquor retailers (`bigquery-public-data.iowa_liquor_sales.sales`).

## PROBLEM

Retail demand forecasting exists to answer a planning question — "how
much should we stock next month" — not to produce an impressive-looking
number. That framing sets real constraints most forecasting write-ups
skip past:

- **A point forecast alone isn't actionable.** "About 40 bottles" doesn't
  tell a buyer how much safety stock to hold. Any credible answer needs a
  calibrated interval, not a bare number.
- **A model that wins on one lucky train/test split isn't demonstrating
  skill.** It's demonstrating variance. The only honest test is scoring
  across many rolling backtest folds, including folds that span a real
  demand shock (2020).
- **The real dataset has real gaps.** No price, promotion, temperature,
  payday, unemployment, consumer-confidence, or CPI data exists anywhere
  in this data. A forecasting exercise that quietly fabricates those
  inputs to look more sophisticated is worse than one that says plainly
  what it doesn't have.
- **"Beat the baseline" is not guaranteed.** Naive and seasonal-naive
  baselines exist because they're hard to beat in practice, not as a
  formality every fancier model is assumed to clear.

## APPROACH

Two complementary pipelines, held to the same validation standard:

**Production pipeline** (`src/demand_forecasting/`) — weekly granularity,
per (store × liquor-type) series, 200 series across the top 25 stores by
volume:

```
naive → seasonal-naive → Holt-Winters (per series) → global LightGBM (pooled) → split-conformal intervals
```

Backtested with rolling-origin evaluation (6 folds), scoring point
accuracy (WAPE/RMSE), interval coverage against the claimed 90% target,
and — separately — folds whose horizon overlaps the 2020 COVID window, so
regime-shift degradation is visible instead of averaged away. Every test
runs against a synthetic fixture with a known, fixed seasonal pattern and
a deliberate synthetic demand shock, so the full suite (24 tests) never
touches BigQuery.

**Exploration track** (`exploration/`) — monthly granularity, aggregated
to 8 liquor categories, a much heavier 5-stage bake-off:

```
1. EDA  →  2. data prep  →  3. model + tune  →  4. evaluate  →  5. champion report
```

Five models per category (naive, Prophet, LightGBM, SARIMAX, and a Ridge
ensemble of the three), each walk-forward validated: fit on train, tune
on a validation window, then refit on train+validation and score once on
a genuinely unseen 12-month holdout. Config-driven end to end
(`run_pipeline.py`) — categories, models, split sizes, and forecast
horizon are all adjustable from one place, and every stage still runs
correctly when only a subset is selected.

Validation techniques used throughout: rolling-origin and walk-forward
backtesting, split-conformal prediction (with its finite-sample
calibration-size floor respected, not glossed over), ADF/KPSS
stationarity tests, Shapiro-Wilk residual normality, leave-one-out
cross-validation for regularization strength, and data-driven (elbow-cut)
clustering in place of a hand-picked threshold.

## OUTCOME

- **Naive seasonal wins 6 of 8 categories** on the exploration track's
  12-month test holdout — Prophet, LightGBM, SARIMAX, and their ensemble
  were all fairly tried and lost. Reported as the headline finding, not
  hidden under whichever model looks most sophisticated.
- **Granularity changes which baseline is worth beating.** The production
  pipeline's weekly, per-store backtest tells the opposite story:
  Holt-Winters averages 0.47 WAPE against naive's 0.59 across 6 real
  folds — a clear, consistent win once there's enough series and history
  for the extra structure to pay off.
- **Complexity earned its place exactly twice.** `other` and `vodka` both
  beat naive by a real, double-digit margin under LightGBM (−18% and
  −26% WAPE) — the honest bar for "worth deploying something more
  complex than a lookup table."
- **Interval calibration held up, and its limits were respected.** The
  production pipeline's conformal intervals average 90.5% empirical
  coverage against a 90% target across 6 folds. The exploration track's
  95% intervals were dropped outright rather than faked — split conformal
  needs ≥20 calibration points for that guarantee, and only 6–18 existed.
- **Zero fabricated inputs.** Every requested-but-unavailable feature
  (price, promotions, temperature, payday timing, macro indicators) is
  explicitly marked *not available* in the EDA output rather than
  approximated.
- **Six real bugs were caught by checking the numbers, not by the code
  running without errors** — a BigQuery date-type crash, real returns
  data rejected by an overly strict schema, Ridge regularization that
  silently had zero effect until inputs were standardized, a pooled
  correlation that masked every real per-category signal, a significance
  test that stopped discriminating anything at scale, and a hardcoded
  assumption that broke the moment the pipeline became configurable.

Full technical detail, model-comparison tables, and validation
methodology are in [`README.md`](README.md) and the `exploration/`
reports; a designed retrospective covering the same material is published
separately as an artifact.
