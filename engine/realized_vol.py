"""
Realised-volatility estimators beyond close-to-close.

Standard close-to-close vol throws away 75% of the information in each
trading day (open, high, low). These estimators use the full OHLC bar
and are 4-7x more statistically efficient:

- Parkinson (1980): uses only (H, L). Efficiency ~5.2x vs close-to-close.
- Garman-Klass (1980): uses (O, H, L, C). Efficiency ~7.4x.
- Rogers-Satchell (1991): drift-adjusted (O, H, L, C). Better for
  trending markets.
- Yang-Zhang (2000): combines overnight, open-to-close, and RS terms.
  Minimum-variance unbiased; the industry default for drift-adjusted
  OHLC vol.

All functions return ANNUALISED volatility as a decimal (e.g. 0.2617 =
26.17%). Input: pandas DataFrame with columns ``open``, ``high``,
``low``, ``close``. Assumes 252 trading days per year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def _log(x):
    return np.log(np.asarray(x, dtype=float))


def close_to_close_vol(df: pd.DataFrame, window: int = 20) -> float:
    """Classic close-to-close volatility over the last `window` bars."""
    if df is None or df.empty or len(df) < window + 1:
        return float("nan")
    close = df["close"].tail(window + 1).values
    r = np.diff(_log(close))
    return float(np.std(r, ddof=1) * np.sqrt(_TRADING_DAYS))


def parkinson_vol(df: pd.DataFrame, window: int = 20) -> float:
    """Parkinson estimator — uses high/low range only.

    σ_P^2 = (1 / (4 N ln 2)) * Σ (ln(H/L))^2 * (252 / window)
    """
    if df is None or df.empty or len(df) < window:
        return float("nan")
    tail = df.tail(window)
    hl = _log(tail["high"].values / tail["low"].values) ** 2
    var = (1.0 / (4.0 * np.log(2.0))) * np.mean(hl)
    return float(np.sqrt(var * _TRADING_DAYS))


def garman_klass_vol(df: pd.DataFrame, window: int = 20) -> float:
    """Garman-Klass estimator — uses OHLC, assumes zero drift.

    σ_GK^2 = mean( 0.5 * (ln(H/L))^2 - (2 ln 2 - 1) * (ln(C/O))^2 )
    """
    if df is None or df.empty or len(df) < window:
        return float("nan")
    tail = df.tail(window)
    o, h, l, c = (tail[col].values for col in ("open", "high", "low", "close"))
    hl = _log(h / l) ** 2
    co = _log(c / o) ** 2
    var = np.mean(0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co)
    return float(np.sqrt(max(var, 0.0) * _TRADING_DAYS))


def rogers_satchell_vol(df: pd.DataFrame, window: int = 20) -> float:
    """Rogers-Satchell estimator — drift-independent OHLC vol.

    σ_RS^2 = mean( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
    """
    if df is None or df.empty or len(df) < window:
        return float("nan")
    tail = df.tail(window)
    o, h, l, c = (tail[col].values for col in ("open", "high", "low", "close"))
    term = _log(h / c) * _log(h / o) + _log(l / c) * _log(l / o)
    var = np.mean(term)
    return float(np.sqrt(max(var, 0.0) * _TRADING_DAYS))


def yang_zhang_vol(df: pd.DataFrame, window: int = 20, k: float | None = None) -> float:
    """Yang-Zhang estimator — unbiased, drift-robust, minimum-variance.

    σ_YZ^2 = σ_overnight^2 + k * σ_open_to_close^2 + (1-k) * σ_RS^2

    where k = 0.34 / (1.34 + (N+1)/(N-1)) is the optimal weight.
    """
    if df is None or df.empty or len(df) < window + 1:
        return float("nan")
    tail = df.tail(window + 1)
    o, h, l, c = (tail[col].values for col in ("open", "high", "low", "close"))

    # Overnight returns: ln(O_t / C_{t-1})
    over = _log(o[1:] / c[:-1])
    # Open-to-close: ln(C_t / O_t)
    otc = _log(c[1:] / o[1:])
    # RS component on the same window
    rs = _log(h[1:] / c[1:]) * _log(h[1:] / o[1:]) + _log(l[1:] / c[1:]) * _log(l[1:] / o[1:])

    var_over = np.var(over, ddof=1)
    var_otc = np.var(otc, ddof=1)
    var_rs = np.mean(rs)

    n = len(over)
    if k is None:
        k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
    var_yz = var_over + k * var_otc + (1.0 - k) * var_rs
    return float(np.sqrt(max(var_yz, 0.0) * _TRADING_DAYS))


def realised_vol_bundle(df: pd.DataFrame, window: int = 20) -> dict:
    """Return all estimators as a dict — consume whichever you trust.

    Keys: close_to_close, parkinson, garman_klass, rogers_satchell, yang_zhang.
    """
    return {
        "close_to_close": close_to_close_vol(df, window),
        "parkinson": parkinson_vol(df, window),
        "garman_klass": garman_klass_vol(df, window),
        "rogers_satchell": rogers_satchell_vol(df, window),
        "yang_zhang": yang_zhang_vol(df, window),
    }


def vol_risk_premium_bundle(df: pd.DataFrame, iv_atm: float, window: int = 20) -> dict:
    """IV minus each realised-vol estimate. Positive = rich premium.

    Returns dict with per-estimator VRP and a ``consensus`` mean of the
    three best (GK, RS, YZ).
    """
    rv = realised_vol_bundle(df, window)
    iv = float(iv_atm)
    vrp = {f"vrp_{k}": iv - v if v == v else float("nan") for k, v in rv.items()}
    robust = [rv["garman_klass"], rv["rogers_satchell"], rv["yang_zhang"]]
    robust = [x for x in robust if x == x]
    vrp["consensus_rv"] = float(np.mean(robust)) if robust else float("nan")
    vrp["consensus_vrp"] = iv - vrp["consensus_rv"] if robust else float("nan")
    vrp["iv_atm"] = iv
    return {**rv, **vrp}
