"""
Stage 1 — Exploratory Data Analysis for the monthly, category-level
forecasting exploration (see common.py for why this is a separate track
from the production weekly/per-store pipeline).

Data reality check, stated once here rather than buried in comments: this
dataset (Iowa liquor wholesale purchases, real BigQuery public data) has
no price, promotion, temperature, payday, unemployment, consumer-
confidence, or CPI columns. Every sub-section below that asks for one of
those either substitutes the closest real thing available (e.g. sale
timing relative to real US holidays, computed from the calendar — not
fabricated) or is explicitly marked NOT AVAILABLE in the EDA summary
table rather than silently skipped or faked.

Usage:
    python exploration/stage1_eda.py

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (
    OUTPUT_ROOT,
    PANDEMIC_END,
    PANDEMIC_START,
    load_monthly_category_frame,
    load_weekly_frame,
    major_holidays,
)
from pipeline_config import PipelineConfig
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from scipy.stats import t as t_dist
from sklearn.ensemble import IsolationForest
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

OUT = OUTPUT_ROOT / "stage1_eda"
SIGNIFICANT_LAGS_ALPHA = 1.96  # ~95% white-noise confidence band, for the ACF/PACF order heuristic


def section_1_shape_and_missingness(weekly: pd.DataFrame, monthly: pd.DataFrame) -> str:
    lines = ["SECTION 1 — SHAPE, DTYPES, MISSINGNESS", ""]
    for label, df in (("weekly (per store x liquor_type)", weekly), ("monthly (per liquor_type, summed across stores)", monthly)):
        lines.append(f"[{label}]")
        lines.append(f"  shape: {df.shape}")
        lines.append(f"  dtypes:\n{_indent(df.dtypes.to_string())}")
        missing_count = df.isna().sum()
        missing_pct = (missing_count / len(df) * 100).round(3)
        missing = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
        lines.append(f"  missingness:\n{_indent(missing.to_string())}")
        lines.append("")
    return "\n".join(lines)


def section_2_plot_series_with_holidays(monthly: pd.DataFrame) -> str:
    holidays = major_holidays(range(monthly["month_start"].dt.year.min(), monthly["month_start"].dt.year.max() + 1))
    holiday_months = set(holidays["date"].apply(lambda d: pd.Period(d, freq="M")))

    categories = sorted(monthly["liquor_type"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    for ax, category in zip(axes.flat, categories):
        sub = monthly[monthly["liquor_type"] == category].set_index("month_start")["units_sold"]
        ax.plot(sub.index, sub.values, marker="o", markersize=2)
        for m in sub.index:
            if pd.Period(m, freq="M") in holiday_months:
                ax.axvline(m, color="red", alpha=0.15, linewidth=6)
        ax.axvspan(PANDEMIC_START, PANDEMIC_END, color="orange", alpha=0.1)
        ax.set_title(category)
    fig.suptitle("Monthly units sold by category (red = month containing a major holiday, orange = pandemic window)")
    fig.tight_layout()
    fig.savefig(OUT / "monthly_series_by_category.png", dpi=110)
    plt.close(fig)

    return (
        "SECTION 2 — TIME SERIES PLOTS\n\n"
        f"  Saved monthly_series_by_category.png ({len(categories)} categories).\n"
        "  Promotion-period annotation requested but NOT AVAILABLE — this dataset has no promotion\n"
        "  flags or discount data. Holiday-month shading uses real US calendar dates (see\n"
        "  common.major_holidays), not fabricated data."
    )


def _stl_strength(result) -> tuple[float, float]:
    """Hyndman & Athanasopoulos (2021) trend/seasonal strength: 1 minus the
    ratio of residual variance to (component + residual) variance, floored
    at 0 since the ratio can slightly exceed 1 with noisy STL fits."""
    resid_var = np.var(result.resid)
    trend_strength = max(0.0, 1 - resid_var / np.var(result.trend + result.resid))
    seasonal_strength = max(0.0, 1 - resid_var / np.var(result.seasonal + result.resid))
    return trend_strength, seasonal_strength


def _fit_stl_per_category(monthly: pd.DataFrame) -> dict:
    """Fit STL once per category and reuse the result everywhere it's
    needed (Section 3's decomposition plots/strengths, Section 9's
    seasonal-profile similarity) instead of re-fitting it twice."""
    results = {}
    for category in sorted(monthly["liquor_type"].unique()):
        series = monthly[monthly["liquor_type"] == category].set_index("month_start")["units_sold"].asfreq("MS")
        results[category] = STL(series, period=12, robust=True).fit()
    return results


def section_3_stl_decomposition(stl_results: dict) -> tuple[str, pd.DataFrame]:
    rows = []
    for category, result in stl_results.items():
        trend_strength, seasonal_strength = _stl_strength(result)
        rows.append({"liquor_type": category, "trend_strength": trend_strength, "seasonal_strength": seasonal_strength})

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(result.trend)
        axes[0].set_title(f"{category}: trend")
        axes[1].plot(result.seasonal)
        axes[1].set_title("seasonal")
        axes[2].plot(result.resid)
        axes[2].set_title("residual")
        axes[2].axhline(0, color="black", linewidth=0.8)
        fig.suptitle(f"STL decomposition — {category} (trend strength={trend_strength:.2f}, seasonal strength={seasonal_strength:.2f})")
        fig.tight_layout()
        fig.savefig(OUT / f"stl_{category}.png", dpi=110)
        plt.close(fig)

    strengths = pd.DataFrame(rows).sort_values("seasonal_strength", ascending=False)
    lines = [
        "SECTION 3 — STL DECOMPOSITION (period=12 months)",
        "",
        _indent(strengths.round(3).to_string(index=False)),
        "",
        "  Interpretation: strength close to 1 means that component dominates the series;",
        "  close to 0 means it contributes little beyond noise.",
    ]
    return "\n".join(lines), strengths


def _suggest_order(values: np.ndarray, n: int) -> int:
    """Heuristic AR/MA order suggestion: last lag (>=1) before the
    autocorrelation function first drops inside the ~95% white-noise band
    and stays there. Genuinely a heuristic, not a substitute for
    likelihood-based order selection (see auto_arima in a later stage)."""
    bound = SIGNIFICANT_LAGS_ALPHA / np.sqrt(n)
    last_significant = 0
    for lag in range(1, len(values)):
        if abs(values[lag]) > bound:
            last_significant = lag
    return last_significant


def section_4_acf_pacf(monthly: pd.DataFrame, n_lags: int = 24) -> tuple[str, pd.DataFrame]:
    rows = []
    for category in sorted(monthly["liquor_type"].unique()):
        series = monthly[monthly["liquor_type"] == category].set_index("month_start")["units_sold"].asfreq("MS")
        max_lags = min(n_lags, len(series) // 2 - 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(series, lags=max_lags, ax=axes[0], title=f"{category}: ACF")
        plot_pacf(series, lags=max_lags, ax=axes[1], method="ywm", title=f"{category}: PACF")
        fig.tight_layout()
        fig.savefig(OUT / f"acf_pacf_{category}.png", dpi=110)
        plt.close(fig)

        acf_vals = acf(series, nlags=max_lags, fft=True)
        pacf_vals = pacf(series, nlags=max_lags, method="ywm")
        suggested_ma = _suggest_order(acf_vals, len(series))
        suggested_ar = _suggest_order(pacf_vals, len(series))
        rows.append({"liquor_type": category, "lags_plotted": max_lags, "suggested_AR(p)": suggested_ar, "suggested_MA(q)": suggested_ma})

    orders = pd.DataFrame(rows)
    lines = [
        f"SECTION 4 — ACF/PACF (up to {n_lags} monthly lags, capped by series length)",
        "",
        _indent(orders.to_string(index=False)),
        "",
        "  AR/MA order suggestions are a heuristic (last lag outside the ~95% white-noise band)",
        "  for a quick read, not a substitute for auto_arima's AIC-based search (Stage 3).",
    ]
    return "\n".join(lines), orders


def section_5_stationarity_tests(monthly: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for category in sorted(monthly["liquor_type"].unique()):
        series = monthly[monthly["liquor_type"] == category].set_index("month_start")["units_sold"].asfreq("MS")
        _adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
        _kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")

        adf_stationary = adf_p < 0.05
        kpss_stationary = kpss_p >= 0.05
        if adf_stationary and kpss_stationary:
            conclusion = "stationary"
        elif not adf_stationary and not kpss_stationary:
            conclusion = "non-stationary"
        elif adf_stationary and not kpss_stationary:
            conclusion = "trend-stationary (conflicting tests, trend likely present)"
        else:
            conclusion = "difference-stationary (conflicting tests)"

        rows.append(
            {
                "liquor_type": category,
                "adf_p": round(adf_p, 4),
                "kpss_p": round(kpss_p, 4),
                "conclusion": conclusion,
            }
        )

    results = pd.DataFrame(rows)
    lines = [
        "SECTION 5 — STATIONARITY (ADF null=unit root, KPSS null=stationary)",
        "",
        _indent(results.to_string(index=False)),
    ]
    return "\n".join(lines), results


def section_6_outliers(monthly: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    categories = sorted(monthly["liquor_type"].unique())
    all_flags = []
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    for ax, category in zip(axes.flat, categories):
        sub = monthly[monthly["liquor_type"] == category].sort_values("month_start").copy()
        q1, q3 = sub["units_sold"].quantile([0.25, 0.75])
        iqr = q3 - q1
        sub["iqr_outlier"] = (sub["units_sold"] < q1 - 1.5 * iqr) | (sub["units_sold"] > q3 + 1.5 * iqr)

        forest = IsolationForest(contamination=0.08, random_state=7)
        sub["iso_outlier"] = forest.fit_predict(sub[["units_sold"]]) == -1
        sub["in_pandemic"] = (sub["month_start"].dt.date >= PANDEMIC_START) & (sub["month_start"].dt.date <= PANDEMIC_END)
        all_flags.append(sub)

        ax.plot(sub["month_start"], sub["units_sold"], color="grey", linewidth=1, zorder=1)
        ax.scatter(sub.loc[sub["iqr_outlier"], "month_start"], sub.loc[sub["iqr_outlier"], "units_sold"], color="blue", label="IQR", zorder=2)
        ax.scatter(sub.loc[sub["iso_outlier"], "month_start"], sub.loc[sub["iso_outlier"], "units_sold"], facecolors="none", edgecolors="red", s=80, label="IsoForest", zorder=3)
        ax.axvspan(PANDEMIC_START, PANDEMIC_END, color="orange", alpha=0.1)
        ax.set_title(category)
        ax.legend(fontsize=7)
    fig.suptitle("Outliers by category: IQR (blue dot) vs. IsolationForest (red circle), pandemic window shaded")
    fig.tight_layout()
    fig.savefig(OUT / "outliers_by_category.png", dpi=110)
    plt.close(fig)

    flagged = pd.concat(all_flags, ignore_index=True)
    summary = (
        flagged.groupby("liquor_type")
        .agg(
            iqr_outliers=("iqr_outlier", "sum"),
            iso_outliers=("iso_outlier", "sum"),
            iqr_outliers_in_pandemic=("iqr_outlier", lambda s: (s & flagged.loc[s.index, "in_pandemic"]).sum()),
        )
        .reset_index()
    )
    lines = [
        "SECTION 6 — OUTLIERS (IQR 1.5x, IsolationForest contamination=0.08, per category)",
        "",
        _indent(summary.to_string(index=False)),
        "",
        f"  Pandemic window flagged separately: {PANDEMIC_START} to {PANDEMIC_END}.",
    ]
    return "\n".join(lines), flagged


def add_calendar_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Real, computable calendar features — the substitute for the
    requested-but-unavailable price/promotion/temperature/macro inputs
    (see module docstring). month_sin/month_cos encode month-of-year on a
    circle (Dec and Jan a distance apart of ~1 month, not 11) so
    correlating against it isn't distorted by the raw 1-12 numbering."""
    df = monthly.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month_start"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_start"].dt.month / 12)
    df["quarter"] = df["month_start"].dt.quarter
    df["time_index"] = (df["month_start"].dt.year - df["month_start"].dt.year.min()) * 12 + df["month_start"].dt.month
    df["is_pandemic"] = ((df["month_start"].dt.date >= PANDEMIC_START) & (df["month_start"].dt.date <= PANDEMIC_END)).astype(int)

    holidays = major_holidays(range(df["month_start"].dt.year.min(), df["month_start"].dt.year.max() + 1))
    holidays_per_month = holidays.groupby(pd.PeriodIndex(holidays["date"], freq="M")).size()
    df["n_holidays_in_month"] = df["month_start"].dt.to_period("M").map(holidays_per_month).fillna(0)
    df["is_holiday_month"] = (df["n_holidays_in_month"] > 0).astype(int)
    return df


CALENDAR_FEATURE_COLS = ["month_sin", "month_cos", "quarter", "is_holiday_month", "n_holidays_in_month", "time_index", "is_pandemic"]


def section_7_correlation_by_category(monthly: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Correlation of units_sold vs. calendar features, computed
    SEPARATELY per category — pooling categories before correlating (the
    first pass of this analysis) mixes series of very different scale and
    washes out real per-category relationships, so every r and p-value
    here comes from that one category's ~89 monthly observations only."""
    df = add_calendar_features(monthly)
    categories = sorted(df["liquor_type"].unique())

    r_rows, p_rows = [], []
    for category in categories:
        sub = df[df["liquor_type"] == category]
        r_row, p_row = {"liquor_type": category}, {"liquor_type": category}
        for feature in CALENDAR_FEATURE_COLS:
            if sub[feature].nunique() < 2:
                r_row[feature], p_row[feature] = np.nan, np.nan
                continue
            r, p = pearsonr(sub["units_sold"], sub[feature])
            r_row[feature], p_row[feature] = r, p
        r_rows.append(r_row)
        p_rows.append(p_row)

    r_table = pd.DataFrame(r_rows).set_index("liquor_type")
    p_table = pd.DataFrame(p_rows).set_index("liquor_type")
    significant = p_table < 0.05

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(r_table.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(CALENDAR_FEATURE_COLS)))
    ax.set_xticklabels(CALENDAR_FEATURE_COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    for i in range(len(categories)):
        for j in range(len(CALENDAR_FEATURE_COLS)):
            marker = "*" if significant.values[i, j] else ""
            ax.text(j, i, f"{r_table.values[i, j]:.2f}{marker}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation with units_sold, per category (* = p < 0.05)")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_by_category.png", dpi=110)
    plt.close(fig)

    n_significant = int(significant.sum().sum())
    lines = [
        "SECTION 7 — CORRELATION BY CATEGORY (not pooled)",
        "",
        "  Requested: sales vs. price, promotions, holidays, temperature, payday weeks, unemployment.",
        "  Available: none of price/promotions/temperature/payday/unemployment — see Section 8's",
        "  explicit NOT AVAILABLE list. Substituted real calendar features instead. Each row below",
        "  is that category's OWN correlation (Pearson r), computed independently, n=len(category series).",
        "",
        "  Pearson r (per category):",
        _indent(r_table.round(3).to_string()),
        "",
        "  p-values:",
        _indent(p_table.round(4).to_string()),
        "",
        f"  {n_significant} of {r_table.size} (category, feature) correlations are significant at p<0.05.",
        "  With ~89 monthly points per category, only |r| greater than ~0.21 reaches p<0.05 — treat",
        "  sub-0.2 correlations as noise, not signal, regardless of category.",
    ]
    return "\n".join(lines), r_table


def _critical_r(n: int, alpha: float = 0.05) -> float:
    """The Pearson |r| threshold that reaches p<alpha for a two-sided test
    with n observations — used below as a statistical (not arbitrary)
    cutoff for deciding which category pairs are similar enough to be a
    joint-modeling candidate."""
    df = n - 2
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    return float(t_crit / np.sqrt(df + t_crit**2))


def _pairwise_corr_with_p(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """frame: columns = categories, rows = aligned observations (a 12-point
    seasonal profile or a T-point differenced series). Returns (r matrix, p
    matrix) for every category pair, each cell from its own pearsonr call
    (not derived from a single global covariance matrix) so the p-values
    are the actual per-pair test, not a plug-in approximation."""
    categories = list(frame.columns)
    r = pd.DataFrame(np.eye(len(categories)), index=categories, columns=categories)
    p = pd.DataFrame(np.zeros((len(categories), len(categories))), index=categories, columns=categories)
    for i, a in enumerate(categories):
        for j, b in enumerate(categories):
            if i == j:
                continue
            r_val, p_val = pearsonr(frame[a], frame[b])
            r.loc[a, b] = r_val
            p.loc[a, b] = p_val
    return r, p


def _plot_corr_matrix(r: pd.DataFrame, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(r.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(r.columns)))
    ax.set_xticklabels(r.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(r.index)))
    ax.set_yticklabels(r.index)
    for i in range(len(r.index)):
        for j in range(len(r.columns)):
            ax.text(j, i, f"{r.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def section_9_cross_category_similarity(monthly: pd.DataFrame, stl_results: dict) -> str:
    """Statistically tests whether any categories are similar enough to be
    worth modeling jointly, on three measures:

    1. Seasonal-PROFILE similarity: correlate each category's own 12-point,
       z-scored (shape, not amplitude) seasonal cycle from STL. n=12 per
       pair — under-powered, context only.
    2. RAW co-movement: correlate categories' month-over-month differenced
       units_sold directly. n~88 per pair, but this is confounded by any
       market-wide effect shared by every category (a shared holiday
       calendar, a shared statewide trend) — reported to show that
       confound explicitly, not as the deciding measure.
    3. RESIDUAL co-movement (the deciding measure): correlate categories'
       STL RESIDUALS — the idiosyncratic month-to-month variation left
       after removing each category's own trend and seasonality. Two
       categories sharing residual shocks (a demand spike neither
       category's own trend/season explains) is the real statistical case
       for pooling them, since a per-category model already captures
       shared trend/seasonality on its own.

    Categories are clustered on residual co-movement (average-linkage
    hierarchical clustering, distance = 1 - r) using a cut threshold set to
    the actual critical r for significance at alpha=0.05 given the
    residual series length — not a round-number guess.
    """
    categories = sorted(monthly["liquor_type"].unique())

    # --- seasonal-profile similarity (shape, z-scored, n=12 per pair) ---
    profiles = {}
    for category, result in stl_results.items():
        seasonal = result.seasonal
        by_month = seasonal.groupby(seasonal.index.month).mean()
        profiles[category] = (by_month - by_month.mean()) / by_month.std()
    profile_frame = pd.DataFrame(profiles)[categories]
    profile_r, profile_p = _pairwise_corr_with_p(profile_frame)
    _plot_corr_matrix(profile_r, "Seasonal-profile correlation (shape only, n=12/pair)", OUT / "similarity_seasonal_profile.png")

    # --- raw co-movement (month-over-month diff, n~T per pair) — confounded, shown for contrast ---
    wide = monthly.pivot_table(index="month_start", columns="liquor_type", values="units_sold")[categories]
    diffs = wide.diff().dropna()
    raw_r, raw_p = _pairwise_corr_with_p(diffs)
    _plot_corr_matrix(raw_r, f"RAW co-movement (MoM diff, n={len(diffs)}/pair) — confounded by shared calendar effects", OUT / "similarity_comovement_raw.png")

    # --- residual co-movement (STL residuals, n~T per pair) — the deciding measure ---
    resid_wide = pd.DataFrame({category: result.resid for category, result in stl_results.items()})[categories].dropna()
    resid_r, resid_p = _pairwise_corr_with_p(resid_wide)
    _plot_corr_matrix(resid_r, f"RESIDUAL co-movement (STL resid, n={len(resid_wide)}/pair) — deciding measure", OUT / "similarity_comovement_residual.png")

    # --- clustering on residual co-movement ---
    # A plain significance cut (r_crit) turns out not to discriminate at all here:
    # every one of the 28 pairs clears it, because this dataset has a strong
    # UNIVERSAL cross-category residual correlation (min pairwise r=0.32, well
    # above the ~0.21 needed for p<0.05 at n=89) — almost certainly a shared
    # month-level effect common to every category (active store count, business
    # days per month, statewide demand shocks), not category-specific similarity.
    # A significance test can't tell "more similar than the universal baseline"
    # from "just as similar as everything else," so the cut used below instead
    # is the elbow in the linkage merge heights — the largest jump in the
    # dendrogram's merge distances, a standard data-driven way to choose a
    # cluster cut without picking a threshold by hand.
    r_crit = _critical_r(len(resid_wide))
    distance = 1 - resid_r.values
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2, 0, 2)  # symmetrize against float rounding
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    merge_heights = np.sort(linkage_matrix[:, 2])
    gaps = np.diff(merge_heights)
    elbow_idx = int(np.argmax(gaps))
    elbow_cut = float((merge_heights[elbow_idx] + merge_heights[elbow_idx + 1]) / 2)

    fig, ax = plt.subplots(figsize=(9, 5))
    dendrogram(linkage_matrix, labels=categories, ax=ax, color_threshold=elbow_cut)
    ax.axhline(elbow_cut, color="grey", linestyle="--", linewidth=1, label=f"elbow cut (distance={elbow_cut:.2f})")
    ax.set_title("Category clustering by RESIDUAL co-movement (average linkage, elbow-cut)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "similarity_dendrogram.png", dpi=110)
    plt.close(fig)

    cluster_ids = fcluster(linkage_matrix, t=elbow_cut, criterion="distance")
    clusters: dict[int, list[str]] = {}
    for category, cid in zip(categories, cluster_ids):
        clusters.setdefault(cid, []).append(category)
    joint_candidates = [members for members in clusters.values() if len(members) > 1]

    # --- rank all pairs on the deciding measure, and specifically evaluate vodka/rum ---
    pairs = []
    for i, a in enumerate(categories):
        for b in categories[i + 1 :]:
            pairs.append(
                {
                    "pair": f"{a} & {b}",
                    "residual_r": resid_r.loc[a, b],
                    "residual_p": resid_p.loc[a, b],
                    "raw_comovement_r": raw_r.loc[a, b],
                    "seasonal_profile_r": profile_r.loc[a, b],
                    "seasonal_profile_p": profile_p.loc[a, b],
                }
            )
    pairs_df = pd.DataFrame(pairs).sort_values("residual_r", ascending=False).reset_index(drop=True)
    n_raw_significant = int((raw_p.where(~np.eye(len(categories), dtype=bool)) < 0.05).sum().sum() / 2)
    n_resid_significant = int((resid_p.where(~np.eye(len(categories), dtype=bool)) < 0.05).sum().sum() / 2)

    lines = [
        "SECTION 9 — CROSS-CATEGORY SIMILARITY (can any categories be modeled jointly?)",
        "",
        f"  RAW co-movement: {n_raw_significant} of 28 pairs significant at p<0.05 (mean r={raw_r.values[np.triu_indices(len(categories), 1)].mean():.2f}).",
        "  Every category's total demand moves with every other category's almost as strongly as with",
        "  its closest match — a shared market-wide/calendar effect (partly captured by the",
        "  is_holiday_month/month features in Section 7), NOT category-specific similarity. Reported",
        "  for contrast, not used for clustering below.",
        "",
        f"  RESIDUAL co-movement: {n_resid_significant} of 28 pairs STILL significant at p<0.05 even",
        f"  after removing each category's own trend and seasonality (naive significance cut |r| > {r_crit:.3f} at",
        "  p<0.05, n={} residual months) — the weakest of all 28 pairs (r={:.2f}) still clears it.".format(
            len(resid_wide), pairs_df["residual_r"].min()
        ),
        "  A plain significance test cannot discriminate 'genuinely more similar' from 'baseline similar'",
        "  when EVERYTHING is significant — this points to a real, universal shared driver behind every",
        "  category's month-to-month demand shocks (plausibly active-store-count or reporting effects",
        "  common to the whole panel), not a bug in the test. Clustering below instead uses the elbow",
        f"  (largest jump) in the dendrogram's own merge distances as the cut, at distance={elbow_cut:.3f}.",
        "",
        "  All 28 pairs, ranked by residual co-movement:",
        _indent(pairs_df.round(3).to_string(index=True)),
        "",
    ]

    if joint_candidates and len(joint_candidates) < len(categories):
        lines.append(f"  ELBOW-CUT CLUSTERS (distance={elbow_cut:.3f}):")
        for group in clusters.values():
            if len(group) > 1:
                members = set(group)
                group_mask = pairs_df["pair"].apply(lambda p, m=members: set(p.split(" & ")).issubset(m))
                group_pairs = pairs_df[group_mask]
                lines.append(f"    {', '.join(group)}  ->")
                lines.append(_indent(group_pairs.round(3).to_string(index=False), "      "))
            else:
                lines.append(f"    {group[0]}  (singleton — no strong partner even at the elbow cut)")
        lines.append("")
        lines.append(
            "  Recommendation: the categories inside a multi-member cluster above share statistically"
        )
        lines.append(
            "  distinguishable residual co-movement (relative to the rest) and are a reasonable candidate"
        )
        lines.append(
            "  for a pooled/joint model; singleton categories show no partner similar enough even at the"
        )
        lines.append("  data-driven elbow cut and should stay modeled separately.")
    else:
        if joint_candidates:
            lines.append(
                "  The elbow cut still merges all 8 categories into one cluster — even the data-driven cut"
            )
            lines.append(
                "  can't separate a meaningful subgroup from the rest. Combined with the RAW/RESIDUAL results"
            )
            lines.append(
                "  above, the honest conclusion is: this dataset's category-level series share a strong common"
            )
            lines.append(
                "  driver, but there's no statistically distinguishable subset that's more similar to each"
            )
            lines.append("  other than to the rest — so grouping any specific pair isn't better-justified than")
            lines.append("  grouping any other pair, or not grouping at all.")
        else:
            lines.append("  No category pair clears the elbow cut — every category's demand shocks are")
            lines.append("  independent enough to model separately.")
        lines.append("  Recommendation: model every category separately (consistent with what was asked).")

    vodka_rum = pairs_df[pairs_df["pair"].isin(["rum & vodka", "vodka & rum"])]
    if not vodka_rum.empty:
        row = vodka_rum.iloc[0]
        rank = pairs_df.index[pairs_df["pair"] == row["pair"]][0] + 1
        n_pairs = len(pairs_df)
        percentile = 100 * (n_pairs - rank) / (n_pairs - 1)
        vodka_cluster = next(group for group in clusters.values() if "vodka" in group)
        rum_cluster = next(group for group in clusters.values() if "rum" in group)
        same_cluster = vodka_cluster is rum_cluster and len(vodka_cluster) > 1

        lines.append("")
        lines.append(
            f"  Vodka & rum specifically: residual r={row['residual_r']:.3f} (p={row['residual_p']:.4f}), "
            f"raw co-movement r={row['raw_comovement_r']:.3f} (inflated by the shared market effect), "
            f"seasonal-profile r={row['seasonal_profile_r']:.3f} (p={row['seasonal_profile_p']:.4f}, n=12 — "
            f"under-powered, context only). Ranked #{rank} of {n_pairs} pairs on residual co-movement "
            f"({percentile:.0f}th percentile)."
        )
        if same_cluster:
            others = [c for c in vodka_cluster if c not in ("vodka", "rum")]
            lines.append(
                f"  Statistical verdict: SUPPORTED, but not uniquely — the elbow-cut clustering places vodka "
                f"and rum in the same {len(vodka_cluster)}-category cluster together with "
                f"{', '.join(others)}. The data supports pooling this whole group, not vodka+rum alone; "
                f"treating them as a special pair while modeling {', '.join(others)} separately would not "
                f"be justified by this analysis."
            )
        else:
            lines.append(
                "  Statistical verdict: NOT supported as a distinguishable pair — vodka and rum land in "
                "different elbow-cut clusters, so despite ranking high in raw similarity, the data doesn't "
                "separate them from the general background correlation shared by every category."
            )

    return "\n".join(lines)


def section_8_summary_table(stl_strengths: pd.DataFrame, per_category_r: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """corr_range reports [min, max] of that feature's per-category Pearson
    r (from Section 7) rather than a single pooled number — a feature can
    be strong for one category and noise for another, and a single pooled
    figure would hide exactly that."""

    def corr_range(feature: str) -> str:
        col = per_category_r[feature]
        return f"[{col.min():.2f}, {col.max():.2f}]"

    rows = [
        {"feature": "units_sold (target)", "correlation_with_target_range": "1.00", "missing_rate": 0.0, "action": "keep (target)"},
        {"feature": "month_sin / month_cos", "correlation_with_target_range": f"{corr_range('month_sin')} / {corr_range('month_cos')}", "missing_rate": 0.0, "action": "keep (cyclical encoding, not raw 1-12)"},
        {"feature": "quarter", "correlation_with_target_range": corr_range("quarter"), "missing_rate": 0.0, "action": "keep"},
        {"feature": "is_holiday_month", "correlation_with_target_range": corr_range("is_holiday_month"), "missing_rate": 0.0, "action": "keep"},
        {"feature": "n_holidays_in_month", "correlation_with_target_range": corr_range("n_holidays_in_month"), "missing_rate": 0.0, "action": "keep"},
        {"feature": "time_index (trend proxy)", "correlation_with_target_range": corr_range("time_index"), "missing_rate": 0.0, "action": "keep"},
        {"feature": "is_pandemic", "correlation_with_target_range": corr_range("is_pandemic"), "missing_rate": 0.0, "action": "keep (regime-shift flag)"},
        {"feature": "price / price_per_unit", "correlation_with_target_range": "n/a", "missing_rate": 1.0, "action": "NOT AVAILABLE — would require re-extracting sale_dollars from BigQuery"},
        {"feature": "promotions", "correlation_with_target_range": "n/a", "missing_rate": 1.0, "action": "NOT AVAILABLE — no promotion data exists in this dataset"},
        {"feature": "temperature", "correlation_with_target_range": "n/a", "missing_rate": 1.0, "action": "NOT AVAILABLE — would require an external weather data source"},
        {"feature": "payday_weeks", "correlation_with_target_range": "n/a", "missing_rate": 1.0, "action": "NOT AVAILABLE — computable calendar-only, but not built in this pass"},
        {"feature": "unemployment_rate", "correlation_with_target_range": "n/a", "missing_rate": 1.0, "action": "NOT AVAILABLE — would require an external macro data source (e.g. FRED)"},
        {"feature": "consumer_confidence_index", "correlation_with_target_range": "n/a", "missing_rate": 1.0, "action": "NOT AVAILABLE — would require an external macro data source"},
        {"feature": "cpi_beverages", "correlation_with_target_range": "n/a", "missing_rate": 1.0, "action": "NOT AVAILABLE — would require an external macro data source"},
    ]
    table = pd.DataFrame(rows)
    avg_seasonal_strength = stl_strengths["seasonal_strength"].mean()
    avg_trend_strength = stl_strengths["trend_strength"].mean()
    lines = [
        "SECTION 8 — EDA SUMMARY TABLE",
        "",
        "  correlation_with_target_range is [min, max] of that feature's per-category Pearson r",
        "  (Section 7) — see that section for the full per-category breakdown with p-values.",
        "",
        _indent(table.to_string(index=False)),
        "",
        f"  Average seasonal strength across categories: {avg_seasonal_strength:.2f}",
        f"  Average trend strength across categories: {avg_trend_strength:.2f}",
    ]
    return "\n".join(lines), table


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def main(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    weekly = load_weekly_frame()
    monthly = load_monthly_category_frame()
    if config.categories is not None:
        available = set(monthly["liquor_type"].unique())
        unknown = set(config.categories) - available
        if unknown:
            raise ValueError(f"unknown categor{'y' if len(unknown) == 1 else 'ies'} in config: {sorted(unknown)} — available: {sorted(available)}")
        weekly = weekly[weekly["liquor_type"].isin(config.categories)].reset_index(drop=True)
        monthly = monthly[monthly["liquor_type"].isin(config.categories)].reset_index(drop=True)

    report = [f"STAGE 1 EDA REPORT — {datetime.now(tz=timezone.utc).date().isoformat()}", "=" * 78]
    report.append(section_1_shape_and_missingness(weekly, monthly))
    report.append("=" * 78)
    report.append(section_2_plot_series_with_holidays(monthly))
    report.append("=" * 78)
    stl_results = _fit_stl_per_category(monthly)
    stl_section, stl_strengths = section_3_stl_decomposition(stl_results)
    report.append(stl_section)
    report.append("=" * 78)
    acf_section, _orders = section_4_acf_pacf(monthly)
    report.append(acf_section)
    report.append("=" * 78)
    stationarity_section, _stationarity = section_5_stationarity_tests(monthly)
    report.append(stationarity_section)
    report.append("=" * 78)
    outlier_section, _flags = section_6_outliers(monthly)
    report.append(outlier_section)
    report.append("=" * 78)
    corr_section, per_category_r = section_7_correlation_by_category(monthly)
    report.append(corr_section)
    report.append("=" * 78)
    if monthly["liquor_type"].nunique() >= 2:
        similarity_section = section_9_cross_category_similarity(monthly, stl_results)
        report.append(similarity_section)
        report.append("=" * 78)
    else:
        report.append("SECTION 9 — CROSS-CATEGORY SIMILARITY\n\n  Skipped: needs >=2 categories, config selected only 1.")
        report.append("=" * 78)

    summary_section, _table = section_8_summary_table(stl_strengths, per_category_r)
    report.append(summary_section)

    full_report = "\n\n".join(report)
    print(full_report)
    (OUT / "stage1_eda_report.txt").write_text(full_report)
    print(f"\nWrote report and plots to {OUT}")


if __name__ == "__main__":
    main()
