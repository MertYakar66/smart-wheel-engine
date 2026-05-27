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
        horizon_days: Forward horizon in calendar days (33 for a typical
                      45-DTE option held to profit target, for instance).
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

    # Forward log-returns at every trading-day index
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
# 5. F4 Fix B1 — regime-conditioned widening of the empirical distribution
# ----------------------------------------------------------------------
def regime_widening_factor(
    *,
    p_crisis: float,
    p_bear: float,
    crisis_weight: float = 0.5,
    bear_weight: float = 0.25,
    max_widening: float = 1.5,
) -> float:
    """Return the regime-conditioned widening factor for the forward
    distribution std.

    Bounded to ``[1.0, max_widening]`` by construction. Returns 1.0
    (no widening) when both ``p_crisis`` and ``p_bear`` are 0 — the
    function is a no-op when the HMM regime classifier flags neither
    a cold-tail state nor a bear state.

    The factor is intentionally **sign-preserving** (always >= 1.0).
    The engine can only become MORE conservative on tail risk, never
    less; widening preserves the §2 invariant that downstream
    reviewers can downgrade but never upgrade a candidate.

    Args:
        p_crisis: HMM posterior P(crisis | history), clamped to [0,1].
        p_bear: HMM posterior P(bear | history), clamped to [0,1].
        crisis_weight: How much weight to put on the crisis posterior.
            Default 0.5 — a fully-crisis posterior (p_crisis=1) adds
            +0.5 to the widening factor.
        bear_weight: How much weight to put on the bear posterior.
            Default 0.25 — a fully-bear posterior adds +0.25.
        max_widening: Upper cap on the widening factor. Default 1.5.

    Returns:
        Widening factor in ``[1.0, max_widening]``.
    """
    p_crisis = max(0.0, min(1.0, float(p_crisis)))
    p_bear = max(0.0, min(1.0, float(p_bear)))
    raw = 1.0 + crisis_weight * p_crisis + bear_weight * p_bear
    return min(max_widening, raw)


def regime_aware_forward_distribution(
    ohlcv: pd.DataFrame,
    horizon_days: int,
    *,
    as_of: pd.Timestamp | str | None = None,
    p_crisis: float = 0.0,
    p_bear: float = 0.0,
    crisis_weight: float = 0.5,
    bear_weight: float = 0.25,
    max_widening: float = 1.5,
    min_empirical_samples: int = 20,
    overlapping_min_samples: int = 60,
    n_scenarios: int = 5000,
    price_col: str = "close",
    seed: int | None = 42,
) -> tuple[np.ndarray, str, np.ndarray | None, str | None, float]:
    """Regime-aware forward distribution PAIR: ``(primary, primary_method,
    alt, alt_method, widening_factor)``.

    Builds the two samples that :meth:`engine.wheel_runner.WheelRunner.rank_candidates_by_ev`
    needs to perform a **worst-of-two evaluation** under F4 Fix B1+C:

    * **Primary** — Fix-B1 only. The legacy NOS-first cascade
      (:func:`best_available_forward_distribution`) with regime widening
      applied to the std. Statistically honest because non-overlapping
      samples are independent.
    * **Alternate** — Fix-C sample-density. When widening fires
      (factor > 1.0), the overlapping empirical branch (~1225 samples at
      5y/35d default) with regime widening applied. The larger sample
      makes std-scaling shift discrete counts continuously and **unlocks
      heavy-tail consumption** in :meth:`engine.ev_engine.EVEngine.evaluate`,
      which requires ``len(pnls) >= 200`` for the POT-GPD fit.

    When widening does NOT fire (calm regime), ``alt`` is ``None`` and the
    caller just evaluates on ``primary`` — byte-identical to the legacy
    code path so calm-regime rows do not move.

    When widening fires, the caller runs ``EVEngine.evaluate`` on BOTH
    samples and surfaces the more conservative result (lower
    ``ev_dollars``). This preserves the documented Fix-B1 partial close
    on **COST 2022-04** (the NOS sample's higher mean keeps prob_profit
    at 0.833 and ev_dollars at ~−$25) while unlocking the sample-density
    benefit on **UNH 2024-11** (the overlapping sample's heavier
    realized 5y-window tail drops ev_dollars from −$65 to −$118). §2 is
    preserved because the engine always takes the WORSE reading —
    downgrade-only.

    Method strings recorded so the audit trail tells the trader which
    branch the engine actually consumed (worst-of-two winner)::

        "empirical_non_overlapping"          # calm path, NOS (primary)
        "empirical_overlapping"              # calm path, NOS too thin
        "block_bootstrap" / "har_rv"         # calm path, deep cascade
        "empirical_non_overlapping_widened"  # B1 primary (worst of two)
        "empirical_overlapping_widened"      # C alt (worst of two)
        "block_bootstrap_widened"            # C alt, history too short for overlapping
        "har_rv_widened"                     # C alt, last resort

    Returns:
        ``(primary, primary_method, alt, alt_method, widening_factor)``.
        ``alt`` and ``alt_method`` are ``None`` when ``widening_factor ==
        1.0`` (calm regime) or when no alternate sample could be
        constructed.
    """
    widening_factor = regime_widening_factor(
        p_crisis=p_crisis,
        p_bear=p_bear,
        crisis_weight=crisis_weight,
        bear_weight=bear_weight,
        max_widening=max_widening,
    )

    # Calm path: no widening → legacy cascade, byte-identical to pre-F4-C.
    if widening_factor <= 1.0 + 1e-9:
        primary, primary_method = best_available_forward_distribution(
            ohlcv,
            horizon_days=horizon_days,
            as_of=as_of,
            min_empirical_samples=min_empirical_samples,
            n_scenarios=n_scenarios,
            price_col=price_col,
            seed=seed,
        )
        return primary, primary_method, None, None, widening_factor

    # Widening fires.
    # Primary: legacy cascade (NOS-first) with widening applied. Statistically
    # honest because NOS samples are independent. This is the Fix-B1 baseline
    # and the conservative anchor for the COST 2022-04 case (the NOS sample
    # has a more bearish 5y mean than the daily-step overlapping sample).
    primary_raw, primary_source = best_available_forward_distribution(
        ohlcv,
        horizon_days=horizon_days,
        as_of=as_of,
        min_empirical_samples=min_empirical_samples,
        n_scenarios=n_scenarios,
        price_col=price_col,
        seed=seed,
    )
    primary = regime_widened_log_returns(
        primary_raw,
        p_crisis=p_crisis,
        p_bear=p_bear,
        crisis_weight=crisis_weight,
        bear_weight=bear_weight,
        max_widening=max_widening,
    )
    # Annotate the primary method so the audit trail records that
    # widening was applied (vs the calm-path identical string).
    if primary_source == "empirical_non_overlapping":
        primary_method = "empirical_non_overlapping_widened"
    elif primary_source == "empirical_overlapping":
        primary_method = "empirical_overlapping_widened"
    elif primary_source == "block_bootstrap":
        primary_method = "block_bootstrap_widened"
    elif primary_source == "har_rv":
        primary_method = "har_rv_widened"
    else:
        primary_method = primary_source  # "none"

    # Alternate: force the overlapping branch (or the next-best deep
    # sample) so the POT-GPD heavy-tail fit becomes eligible AND so
    # std-scaling has finer-grained resolution. Suppressed when the
    # primary already used overlapping/bootstrap/HAR (no new info).
    alt: np.ndarray | None = None
    alt_method: str | None = None
    if primary_source == "empirical_non_overlapping":
        ovp = empirical_forward_log_returns(
            ohlcv,
            horizon_days=horizon_days,
            as_of=as_of,
            min_samples=overlapping_min_samples,
            price_col=price_col,
            non_overlapping=False,
        )
        if len(ovp) >= overlapping_min_samples:
            alt = regime_widened_log_returns(
                ovp,
                p_crisis=p_crisis,
                p_bear=p_bear,
                crisis_weight=crisis_weight,
                bear_weight=bear_weight,
                max_widening=max_widening,
            )
            alt_method = "empirical_overlapping_widened"
        else:
            bs = block_bootstrap_log_returns(
                ohlcv,
                horizon_days=horizon_days,
                n_scenarios=n_scenarios,
                as_of=as_of,
                price_col=price_col,
                seed=seed,
            )
            if len(bs) > 0:
                alt = regime_widened_log_returns(
                    bs,
                    p_crisis=p_crisis,
                    p_bear=p_bear,
                    crisis_weight=crisis_weight,
                    bear_weight=bear_weight,
                    max_widening=max_widening,
                )
                alt_method = "block_bootstrap_widened"

    return primary, primary_method, alt, alt_method, widening_factor


def regime_widened_log_returns(
    log_returns: np.ndarray,
    *,
    p_crisis: float = 0.0,
    p_bear: float = 0.0,
    crisis_weight: float = 0.5,
    bear_weight: float = 0.25,
    max_widening: float = 1.5,
) -> np.ndarray:
    """Widen the std of an empirical log-return array by a regime-dependent
    factor — F4 Fix B1.

    Sign-preserving (the widening factor from :func:`regime_widening_factor`
    is always >= 1.0). Bounded above (default 1.5x). Preserves the
    empirical mean — only widens the spread around the mean, so the
    "central tendency" of the forward distribution doesn't shift; only
    its tails grow.

    Mechanism::

        widening = regime_widening_factor(p_crisis=..., p_bear=...)
        mu = log_returns.mean()
        widened = mu + widening * (log_returns - mu)

    When both probabilities are ~0, widening = 1.0 and the function
    returns the input unchanged (cheap path).

    Why this helps F4:
        ``docs/F4_TAIL_RISK_DIAGNOSTIC.md`` documents that the engine's
        empirical forward distribution (typically 30-1300 historical
        N-day returns) does not widen during the first 14 days of an
        idiosyncratic drawdown (COST 2022-04, UNH 2024-11). The HMM
        regime classifier DOES detect the regime shift but its
        ``regime_multiplier`` scales the final ``ev_dollars`` rather
        than touching the distribution that produces ``prob_profit``.
        This function feeds the regime signal back into the forward
        distribution so ``prob_profit`` reflects the elevated tail risk.

    Calibration (verified live against the F4 cases):
        - UNH 2024-11-11: ``p_crisis=0.28, p_bear=0.72`` → widening
          1.32x → ``prob_profit`` 0.857 → 0.79 (continuous) or 0.71
          (max 1.5x). **Closes UNH**.
        - COST 2022-04-04: ``p_crisis=0.14, p_bear=0.09`` → widening
          1.09x → ``prob_profit`` 0.833 → 0.833 (the 30-sample
          non-overlapping empirical is too coarse for std-scaling to
          shift counts). **Does NOT close COST 2022-04** — that case
          is a black-swan idiosyncratic event where neither HMM nor
          recent realized vol fired prior to the drop. Open follow-up.

    Args:
        log_returns: 1-D array of empirical log-returns (e.g. from
            :func:`empirical_forward_log_returns`).
        p_crisis: HMM posterior P(crisis|history).
        p_bear: HMM posterior P(bear|history).
        crisis_weight: see :func:`regime_widening_factor`.
        bear_weight: see :func:`regime_widening_factor`.
        max_widening: see :func:`regime_widening_factor`.

    Returns:
        Widened log-return array of the same length and dtype.
    """
    if log_returns is None or len(log_returns) == 0:
        return log_returns

    widening = regime_widening_factor(
        p_crisis=p_crisis,
        p_bear=p_bear,
        crisis_weight=crisis_weight,
        bear_weight=bear_weight,
        max_widening=max_widening,
    )
    # Cheap fast-path: no widening, return input unchanged.
    if widening <= 1.0 + 1e-9:
        return log_returns

    arr = np.asarray(log_returns, dtype=float)
    mu = float(arr.mean())
    return mu + widening * (arr - mu)
