"""
Forward-return distribution builder (point-in-time safe).

This module provides the bridge between historical OHLCV data and the
:mod:`engine.ev_engine` — it converts an OHLCV time series into an empirical
distribution of N-day-ahead log-returns that callers can pass as the
``forward_log_returns`` argument to :meth:`engine.ev_engine.EVEngine.evaluate`.

The whole point of this module is PIT (point-in-time) safety. The EV engine
cannot see into the future and must not be fed a distribution built from the
future either. Every function here accepts an explicit ``as_of`` date and
strictly filters history to ``date <= as_of`` before doing any math.

Three distribution sources are supported, in increasing order of sophistication:

1. :func:`empirical_forward_log_returns` — stationary, non-overlapping N-day
   log-returns from the trailing window (default 3 years of history). This
   is the most conservative and what the EV engine ranker uses by default.

2. :func:`block_bootstrap_log_returns` — stationary block bootstrap that
   preserves autocorrelation structure for multi-day horizons. Use when you
   need a lot of scenarios for tail estimation without over-sampling the
   same window.

3. :func:`har_rv_conditional_distribution` — HAR-RV (Corsi 2009) conditional
   forecast. The module fits a simple HAR model on past realised vol, uses
   it to project a point forecast, and generates synthetic log-returns with
   that vol and realistic skew/kurtosis.

None of these functions load data themselves — they all accept an OHLCV
DataFrame indexed by date, so you stay in control of how history is fetched
and caching happens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Calendar->trading-day horizon conversion (D21). An option DTE is a
# CALENDAR-day quantity, but the samplers below index *trading-day* price bars.
# ~252 trading days span ~365 calendar days, so a 35-DTE option evolves over
# ~24 trading bars, not 35. Indexing calendar DTE directly over-states the
# horizon (~45%) and over-disperses the terminal distribution feeding EV. The
# orchestrator converts once; the low-level samplers take bars.
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365


def calendar_days_to_trading_bars(calendar_days: int) -> int:
    """Convert a calendar-day horizon (e.g. an option DTE) to trading-day bars.

    Returns at least 1 bar. ``round(35 * 252 / 365) == 24``.

    DEFERRED (DECISIONS D21): this is the dimensionally-correct conversion for the
    forward-distribution horizon, but ``best_available_forward_distribution`` does
    NOT apply it yet — doing so shifts every EV/prob_profit value and would
    de-calibrate the published prob_profit matrix and all backtest snapshots. It
    is kept here, validated by a unit test, ready to wire in during a coordinated
    re-baseline.
    """
    bars = round(int(calendar_days) * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR)
    return max(1, int(bars))


# ----------------------------------------------------------------------
# 1. Empirical non-overlapping forward log-returns
# ----------------------------------------------------------------------
def empirical_forward_log_returns(
    ohlcv: pd.DataFrame,
    horizon_days: int,
    as_of: pd.Timestamp | str | None = None,
    lookback_years: float = 5.0,
    min_samples: int = 20,
    price_col: str = "close",
    non_overlapping: bool = True,
) -> np.ndarray:
    """Return a 1-D array of historical N-day-ahead log-returns.

    Strictly filters to ``date <= as_of`` before computing returns so there
    is no look-ahead. When ``non_overlapping=True`` (default) the returns are
    sampled every ``horizon_days`` days to avoid autocorrelation inflation
    of the effective sample size.

    Args:
        ohlcv: DataFrame indexed by date with at least a ``close`` column.
        horizon_days: Forward horizon in **trading-day bars** (this is a
                      bar-indexed sampler). Callers holding a calendar DTE
                      should go through :func:`best_available_forward_distribution`,
                      which converts via :func:`calendar_days_to_trading_bars`
                      (D21), or convert themselves.
        as_of: PIT cutoff — history strictly before or equal to this date is
               used. When ``None`` we use the last date in the frame.
        lookback_years: Maximum years of history to sample from.
        min_samples: Minimum number of returns required. If the filtered
                     history has fewer samples, the function returns an
                     empty array and the caller should fall back to a
                     model-based distribution.
        price_col: Name of the column holding close prices.
        non_overlapping: When True, down-sample to avoid overlap.

    Returns:
        1-D numpy array of log-returns (dimensionless). Empty array if the
        filter produced fewer than ``min_samples`` observations.
    """
    if ohlcv is None or ohlcv.empty or price_col not in ohlcv.columns:
        return np.asarray([], dtype=float)

    df = ohlcv.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        df = df.loc[df.index <= cutoff]

    if lookback_years is not None and lookback_years > 0 and not df.empty:
        start = df.index.max() - pd.Timedelta(days=int(lookback_years * 365))
        df = df.loc[df.index >= start]

    prices = df[price_col].dropna().astype(float).values
    n = len(prices)
    if n <= horizon_days:
        return np.asarray([], dtype=float)

    # Forward log-returns at every trading-day index. np.log is DELIBERATELY
    # unguarded: NaN closes are removed by the dropna above; a zero/negative
    # close yields non-finite returns that the isfinite filter below removes
    # (pinned: test_forward_distribution_invariants.py::TestEmpiricalBoundary).
    # Do NOT clamp/ffill/eps-floor here — it changes shipped EV (the dropna
    # splice sets the non-overlapping sampling phase).
    log_prices = np.log(prices)
    log_rets = log_prices[horizon_days:] - log_prices[:-horizon_days]

    if non_overlapping:
        log_rets = log_rets[::horizon_days]

    log_rets = log_rets[np.isfinite(log_rets)]

    if len(log_rets) < min_samples:
        return np.asarray([], dtype=float)

    return log_rets.astype(float)


# ----------------------------------------------------------------------
# 2. Stationary block bootstrap (Politis-Romano 1994)
# ----------------------------------------------------------------------
def block_bootstrap_log_returns(
    ohlcv: pd.DataFrame,
    horizon_days: int,
    n_scenarios: int = 2000,
    block_size: int = 5,
    as_of: pd.Timestamp | str | None = None,
    lookback_years: float = 5.0,
    price_col: str = "close",
    seed: int | None = 42,
) -> np.ndarray:
    """Generate forward log-returns via stationary block bootstrap.

    Preserves short-horizon autocorrelation (important for vol clustering).
    Uses the Politis-Romano stationary bootstrap with geometric block-length
    distribution to avoid the periodicity artefacts of fixed-block bootstrap.

    Returns an array of length ``n_scenarios`` giving the total log-return
    over the horizon for each synthetic path.
    """
    if ohlcv is None or ohlcv.empty or price_col not in ohlcv.columns:
        return np.asarray([], dtype=float)

    df = ohlcv.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        df = df.loc[df.index <= cutoff]

    if lookback_years is not None and lookback_years > 0 and not df.empty:
        start = df.index.max() - pd.Timedelta(days=int(lookback_years * 365))
        df = df.loc[df.index >= start]

    prices = df[price_col].dropna().astype(float).values
    if len(prices) < max(horizon_days * 2, 100):
        return np.asarray([], dtype=float)

    daily_rets = np.diff(np.log(prices))
    n_rets = len(daily_rets)
    if n_rets < horizon_days + block_size:
        return np.asarray([], dtype=float)

    rng = np.random.default_rng(seed)
    p = 1.0 / max(block_size, 1)
    scenarios = np.empty(n_scenarios, dtype=float)

    for s in range(n_scenarios):
        total = 0.0
        i = 0
        # Start each path at a random index
        idx = rng.integers(0, n_rets)
        while i < horizon_days:
            total += daily_rets[idx]
            i += 1
            # Start a new block with probability p
            if rng.random() < p:
                idx = rng.integers(0, n_rets)
            else:
                idx = (idx + 1) % n_rets
        scenarios[s] = total

    return scenarios


# ----------------------------------------------------------------------
# 3. HAR-RV conditional distribution (Corsi 2009)
# ----------------------------------------------------------------------
def har_rv_conditional_distribution(
    ohlcv: pd.DataFrame,
    horizon_days: int,
    n_scenarios: int = 5000,
    as_of: pd.Timestamp | str | None = None,
    lookback_years: float = 3.0,
    price_col: str = "close",
    seed: int | None = 42,
) -> np.ndarray:
    """HAR-RV-conditioned synthetic forward log-return distribution.

    Fits the HAR-RV(1,5,22) model:

        RV_t+1 = β0 + βD·RV_t + βW·RV_t^(5) + βM·RV_t^(22) + ε

    on 5-minute... (we only have daily data here, so we use daily squared
    log-returns as the RV proxy, which is still a usable point forecast
    — Corsi's original result holds on daily data too, just with larger
    measurement noise).

    The projected RV is then used as the per-day variance and Monte Carlo
    paths are drawn from a t(6) distribution for realistic fat tails.
    """
    if ohlcv is None or ohlcv.empty or price_col not in ohlcv.columns:
        return np.asarray([], dtype=float)

    df = ohlcv.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        df = df.loc[df.index <= cutoff]

    if lookback_years is not None and lookback_years > 0 and not df.empty:
        start = df.index.max() - pd.Timedelta(days=int(lookback_years * 365))
        df = df.loc[df.index >= start]

    prices = df[price_col].dropna().astype(float).values
    if len(prices) < 60:
        return np.asarray([], dtype=float)

    log_rets = np.diff(np.log(prices))
    rv_daily = log_rets**2  # noisy daily RV proxy
    n = len(rv_daily)
    if n < 30:
        return np.asarray([], dtype=float)

    # Build HAR regressors
    rv_w = pd.Series(rv_daily).rolling(5).mean().values
    rv_m = pd.Series(rv_daily).rolling(22).mean().values

    # Drop initial NaNs
    start_idx = 22
    y = rv_daily[start_idx + 1 :]
    X = np.column_stack(
        [
            np.ones(n - start_idx - 1),
            rv_daily[start_idx : n - 1],
            rv_w[start_idx : n - 1],
            rv_m[start_idx : n - 1],
        ]
    )
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]
    if len(y) < 30:
        return np.asarray([], dtype=float)

    # OLS fit
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.asarray([], dtype=float)

    # Next-step forecast
    rv_d = rv_daily[-1]
    rv_w_now = float(np.nanmean(rv_daily[-5:])) if len(rv_daily) >= 5 else rv_d
    rv_m_now = float(np.nanmean(rv_daily[-22:])) if len(rv_daily) >= 22 else rv_d
    rv_fc = float(beta[0] + beta[1] * rv_d + beta[2] * rv_w_now + beta[3] * rv_m_now)
    rv_fc = max(rv_fc, 1e-6)  # floor — no negative variance
    sigma_daily_fc = np.sqrt(rv_fc)

    # Fat-tailed forward paths using Student-t(df=6) — calibrated to
    # equity-index excess kurtosis of ~5 in daily log-returns.
    rng = np.random.default_rng(seed)
    df_t = 6.0
    z = rng.standard_t(df_t, size=(n_scenarios, horizon_days))
    # Rescale to unit variance for the t distribution
    z = z * np.sqrt((df_t - 2) / df_t)
    daily_sim = sigma_daily_fc * z
    horizon_log_rets = daily_sim.sum(axis=1)
    return horizon_log_rets.astype(float)


# ----------------------------------------------------------------------
# 4. One-stop-shop: best-available distribution
# ----------------------------------------------------------------------
def best_available_forward_distribution(
    ohlcv: pd.DataFrame,
    horizon_days: int,
    as_of: pd.Timestamp | str | None = None,
    min_empirical_samples: int = 20,
    n_scenarios: int = 5000,
    price_col: str = "close",
    seed: int | None = 42,
) -> tuple[np.ndarray, str]:
    """Return the best forward distribution available for this history.

    Strategy (cascading fall-back so the EV engine always gets *something*):
      1. **Non-overlapping empirical** — most statistically honest, used
         when history is deep enough.
      2. **Overlapping empirical** — more samples but inflated effective N
         due to autocorrelation; acceptable when NOS returns too few.
      3. **Stationary block bootstrap** — preserves autocorrelation and
         generates as many scenarios as we need.
      4. **HAR-RV conditional Monte Carlo** — vol-forecast-driven
         synthetic distribution for very short history.

    Returns ``(log_returns, method_name)``. When nothing works, the returned
    array is empty and ``method_name`` is ``"none"``.

    KNOWN DISCREPANCY (DEFERRED — see DECISIONS D21): callers pass the option's
    *calendar* DTE as ``horizon_days``, but the samplers below index *trading-day*
    bars, so the effective horizon is ~46% too long. The dimensionally-correct
    conversion is available as :func:`calendar_days_to_trading_bars`, but it is
    intentionally NOT applied here yet: doing so shifts every EV/prob_profit value
    and would de-calibrate the published prob_profit matrix and all backtest
    snapshots. Applying it is a coordinated re-baseline change, not a point fix.
    """
    rets = empirical_forward_log_returns(
        ohlcv,
        horizon_days=horizon_days,
        as_of=as_of,
        min_samples=min_empirical_samples,
        price_col=price_col,
        non_overlapping=True,
    )
    if len(rets) >= min_empirical_samples:
        return rets, "empirical_non_overlapping"

    rets = empirical_forward_log_returns(
        ohlcv,
        horizon_days=horizon_days,
        as_of=as_of,
        min_samples=max(min_empirical_samples * 3, 60),
        price_col=price_col,
        non_overlapping=False,
    )
    if len(rets) > 0:
        return rets, "empirical_overlapping"

    rets = block_bootstrap_log_returns(
        ohlcv,
        horizon_days=horizon_days,
        n_scenarios=n_scenarios,
        as_of=as_of,
        price_col=price_col,
        seed=seed,
    )
    if len(rets) > 0:
        return rets, "block_bootstrap"

    rets = har_rv_conditional_distribution(
        ohlcv,
        horizon_days=horizon_days,
        n_scenarios=n_scenarios,
        as_of=as_of,
        price_col=price_col,
        seed=seed,
    )
    if len(rets) > 0:
        return rets, "har_rv"

    return np.asarray([], dtype=float), "none"


# ----------------------------------------------------------------------
# Sampling-honesty predicate for prob_profit's Wilson CI
# ----------------------------------------------------------------------
#
# prob_profit is a k/N binomial frequency over the forward-scenario set, and
# ev_engine surfaces a Wilson 95% CI for it (engine.ev_engine._wilson_score_interval).
# A Wilson binomial interval is only an honest *sampling* spread when N is a
# count of INDEPENDENT Bernoulli trials. That holds for exactly one tier:
#
#   * empirical_non_overlapping — disjoint, ~IID forward windows (N ~ 30-35).
#     The Wilson CI is honest here.
#
# It does NOT hold for the other tiers, all of which report an ``n_scenarios``
# that is not an independent-trial count, so a Wilson CI over it is deceptively
# TIGHT (false precision — the opposite of the honesty goal):
#
#   * empirical_overlapping — autocorrelated windows; effective N << count.
#   * block_bootstrap / har_rv — large synthetic resample counts (n ~ 5000).
#   * lognormal_fallback — parametric draws (n ~ 20000; ev_engine's own label).
#   * none — no scenarios evaluated.
#
# Callers (the wheel_runner rankers) gate CI emission on this predicate so the
# only interval a trader ever sees is a genuine sampling spread. This is the
# Python source of truth mirrored by the dashboard's ``samplingCiHonest``
# (dashboard/src/lib/cockpit-trust.ts).
_IID_FORWARD_SOURCES: frozenset[str] = frozenset({"empirical_non_overlapping"})


def is_iid_forward_source(source: str | None) -> bool:
    """True iff ``source`` is a forward tier whose ``n_scenarios`` is a count
    of INDEPENDENT trials — i.e. prob_profit's Wilson 95% CI is statistically
    honest for it.

    Only ``"empirical_non_overlapping"`` qualifies (see ``_IID_FORWARD_SOURCES``).
    Every other tier label (``empirical_overlapping``, ``block_bootstrap``,
    ``har_rv``, ``lognormal_fallback``, ``none``, or ``None``) reports an N that
    is not an independent-trial count, so a binomial CI over it is false
    precision and must be suppressed by the caller.
    """
    return source in _IID_FORWARD_SOURCES


# ----------------------------------------------------------------------
# 5. F4 follow-up — realized-vol-ratio widening
# ----------------------------------------------------------------------
#
# Background (`docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10): the rolled-back
# Fix B1 used HMM cold-tail posterior as the widening signal. The
# post-rollback signal probe showed HMM posterior is a weak
# discriminator of tail-realized dates (~1.3x lift; ~50% non-tail
# false-positive rate at the threshold that catches 60% of tails).
# Aggressive widening on this signal inverts S27 Spearman ρ.
#
# Realized-vol ratio (rv30 / rv252) is a stronger signal: 2.07x lift
# at threshold 1.30, recall 27% on tail dates, false-positive 13% on
# non-tail dates. Captures vol-clustering — when 30d realized vol is
# materially elevated vs the 1y baseline, the next 35d is empirically
# more likely to also be elevated. Independent of HMM (price-derived,
# different math, different cause).
#
# This widening is SCOPED CONSERVATIVELY:
#   - Gate threshold 1.30 (only ~14% of 2022-2024 dates fire)
#   - Max factor 1.15 (vs rolled-back Fix B1's 1.50)
#   - Mean-preserving (same as Fix B1 — only widens spread)
#   - Sign-preserving (factor always >= 1.0; downgrade-only)
#
# Does NOT close named F4 cases (COST 2022-04 had rv ratio 0.96,
# below threshold — calm pre-drawdown). The R10 single-name cap
# remains the damage-bounding mechanism for those.
def realized_vol_ratio(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
    price_col: str = "close",
    short_window: int = 30,
    long_window: int = 252,
) -> float:
    """Return the ratio of ``short_window`` realized vol to
    ``long_window`` realized vol at the as_of date.

    PIT-safe: strictly filters to ``date <= as_of`` before computing
    log-returns. Returns ``1.0`` (no-fire default) when history is
    insufficient — gates the widening to no-op rather than firing
    on noise.

    Args:
        ohlcv: DataFrame indexed by date with a ``close`` column.
        as_of: PIT cutoff. ``None`` means latest available.
        price_col: Column name for close prices.
        short_window: Trading days for the numerator vol (default 30).
        long_window: Trading days for the denominator baseline
            (default 252 — 1 trading year).

    Returns:
        ``rv_short / rv_long`` as a float. Returns 1.0 when:
        - OHLCV is empty
        - Less than ``long_window + 1`` log-returns available after PIT filter
        - Long-window vol is ~zero (degenerate constant-price history)
        - The ratio is non-finite (a non-positive close inside the window
          poisons the log-returns — absent evidence never widens)
    """
    if ohlcv is None or ohlcv.empty or price_col not in ohlcv.columns:
        return 1.0
    df = ohlcv.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        df = df.loc[df.index <= cutoff]
    closes = df[price_col].dropna().astype(float).values
    if len(closes) < long_window + 1:
        return 1.0
    log_rets = np.diff(np.log(closes))
    if len(log_rets) < long_window:
        return 1.0
    rv_short = float(np.std(log_rets[-short_window:]))
    rv_long = float(np.std(log_rets[-long_window:]))
    if rv_long <= 1e-9:
        return 1.0
    ratio = rv_short / rv_long
    if not np.isfinite(ratio):
        # A non-positive close (corrupt input — the data layer pins these
        # away, see test_ohlcv_prices_positive) survives the dropna above and
        # poisons np.log -> non-finite log-returns -> nan std -> nan ratio.
        # A nan returned here would skip every downstream threshold guard
        # (nan comparisons are False) and silently hit max_widening via
        # min(max_widening, nan). Same no-fire 1.0 as the other degenerate
        # routes: absent evidence never widens.
        return 1.0
    return ratio


def realized_vol_widening_factor(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
    *,
    threshold: float = 1.30,
    slope: float = 0.20,
    max_widening: float = 1.15,
    price_col: str = "close",
    short_window: int = 30,
    long_window: int = 252,
) -> float:
    """Compute the realized-vol-ratio widening factor at the as_of date.

    Returns 1.0 (no widening) when ``rv30 / rv252 < threshold``. Above
    the threshold, ramps linearly with slope ``slope`` and caps at
    ``max_widening``.

    Concretely::

        ratio = rv30 / rv252
        if ratio < threshold:
            return 1.0
        return min(max_widening, 1.0 + slope * (ratio - threshold))

    Default calibration (threshold 1.30, slope 0.20, max 1.15):
        - ratio = 1.30 → factor 1.00 (no widening at threshold)
        - ratio = 1.50 → factor 1.04
        - ratio = 1.80 → factor 1.10
        - ratio = 2.05 → factor 1.15 (cap)

    Sign- and mean-preserving. Per the F4 signal probe
    (2026-05-27, branch `claude/fix-f4-threshold-gated-widening`):
        - Fires on 14% of 2022-2024 (ticker, date) probes
        - Catches 27% of tail-realized dates (lift 2.07x vs random)
        - UNH 2024-11-11 (the cleanest F4 case): ratio 1.36, factor 1.012
        - COST 2022-04-04: ratio 0.96, no fire (calm pre-drawdown)
        - AAPL 2026-02-13 (control): ratio 0.85, no fire

    Calibration choice rationale: the rolled-back HMM-based Fix B1
    used max widening 1.50 on a signal that fired on 98% of dates and
    inverted S27 ρ. This calibration is intentionally much gentler
    (max 1.15 on a signal that fires on 14% of dates) so the S27 ρ
    stays positive while the engine becomes meaningfully more
    cautious during vol-cluster regimes.
    """
    ratio = realized_vol_ratio(
        ohlcv,
        as_of=as_of,
        price_col=price_col,
        short_window=short_window,
        long_window=long_window,
    )
    if ratio < threshold:
        return 1.0
    return min(max_widening, 1.0 + slope * (ratio - threshold))


def realized_vol_widened_log_returns(
    log_returns: np.ndarray,
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
    *,
    threshold: float = 1.30,
    slope: float = 0.20,
    max_widening: float = 1.15,
    price_col: str = "close",
) -> np.ndarray:
    """Widen the std of an empirical log-return array by the
    realized-vol-ratio widening factor.

    Mean-preserving (only widens spread around the mean). Sign-
    preserving (factor always >= 1.0 — downgrade-only). No-op when
    rv ratio < threshold (the cheap fast path; 86% of the 2022-2024
    sample).
    """
    if log_returns is None or len(log_returns) == 0:
        return log_returns
    factor = realized_vol_widening_factor(
        ohlcv,
        as_of=as_of,
        threshold=threshold,
        slope=slope,
        max_widening=max_widening,
        price_col=price_col,
    )
    if factor <= 1.0 + 1e-9:
        return log_returns
    arr = np.asarray(log_returns, dtype=float)
    mu = float(arr.mean())
    return mu + factor * (arr - mu)
