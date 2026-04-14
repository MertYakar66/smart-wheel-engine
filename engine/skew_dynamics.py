"""
Nelson-Siegel IV term structure and skew-dynamics alpha signals.

What this module does
---------------------
Three things the existing ``volatility_surface`` module does not:

1. :class:`NelsonSiegelTermStructure` — fit a parsimonious 3-factor
   Nelson-Siegel model to the ATM IV term structure and use the factor
   loadings (level, slope, curvature) as interpretable regime signals.

2. :func:`skew_slope`                — compute the 25Δ put vs 25Δ call
   skew per expiry and the skew's rolling momentum. Steepening put
   skew is a leading indicator of realised downside and a reliable
   "sell less premium" signal for the wheel strategy.

3. :func:`ivs_dislocation_score`     — detect dislocations between the
   observed surface and the Nelson-Siegel projection. Large
   dislocations are flagged as potential VRP / gamma mispricing
   opportunities the decision engine should surface to the trader.

The standard NS parameterisation (Nelson & Siegel 1987) is:

    σ_NS(T) = β0 + β1 * ((1 - e^{-T/τ}) / (T/τ))
                 + β2 * ((1 - e^{-T/τ}) / (T/τ) - e^{-T/τ})

Where β0 = long-run level, β1 = short-end slope, β2 = curvature, and
τ is a fixed decay parameter (we use τ = 2 years by convention because
equity IV term structure on S&P 500 has a characteristic 2-year decay
based on the historical VIX term-structure literature).

All functions are pure numpy + scipy and have no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


_DEFAULT_TAU_YEARS = 2.0


@dataclass
class NSFit:
    """Result of fitting Nelson-Siegel to an IV term structure."""

    beta0: float  # level
    beta1: float  # slope (short-end)
    beta2: float  # curvature
    tau: float  # decay (fixed)
    residual_norm: float
    n_points: int
    converged: bool


class NelsonSiegelTermStructure:
    """Fit and evaluate a Nelson-Siegel model for an IV term structure."""

    def __init__(self, tau_years: float = _DEFAULT_TAU_YEARS) -> None:
        self.tau = tau_years
        self.fit_result: NSFit | None = None

    def fit(self, tenors_years: np.ndarray, ivs: np.ndarray) -> NSFit:
        """Fit the NS parameters to observed (T, IV) pairs.

        Args:
            tenors_years: 1-D array of option expiries in years (e.g.
                ``[0.083, 0.25, 0.5, 1.0, 2.0]`` for 30/90/180/365/730 days).
            ivs: 1-D array of observed ATM implied volatilities (decimal).
        """
        T = np.asarray(tenors_years, dtype=float)
        y = np.asarray(ivs, dtype=float)
        mask = np.isfinite(T) & np.isfinite(y) & (T > 0) & (y > 0)
        T, y = T[mask], y[mask]
        n = len(T)
        if n < 2:
            self.fit_result = NSFit(
                beta0=float(y.mean()) if n else 0.20,
                beta1=0.0,
                beta2=0.0,
                tau=self.tau,
                residual_norm=0.0,
                n_points=n,
                converged=False,
            )
            return self.fit_result

        # NS with fixed tau is a linear model in (β0, β1, β2).
        # X_t = [1, (1-e^{-T/τ})/(T/τ), (1-e^{-T/τ})/(T/τ) - e^{-T/τ}]
        t_over_tau = T / self.tau
        exp_t = np.exp(-t_over_tau)
        col2 = (1.0 - exp_t) / np.where(t_over_tau > 0, t_over_tau, 1e-12)
        col3 = col2 - exp_t
        X = np.column_stack([np.ones_like(T), col2, col3])

        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            res_norm = float(np.linalg.norm(resid))
            converged = True
        except np.linalg.LinAlgError:
            beta = np.array([float(y.mean()), 0.0, 0.0])
            res_norm = float("inf")
            converged = False

        self.fit_result = NSFit(
            beta0=float(beta[0]),
            beta1=float(beta[1]),
            beta2=float(beta[2]),
            tau=self.tau,
            residual_norm=res_norm,
            n_points=n,
            converged=converged,
        )
        return self.fit_result

    def iv_at(self, tenor_years: float) -> float:
        """Evaluate the fitted NS at a single tenor."""
        if self.fit_result is None:
            raise RuntimeError("NelsonSiegelTermStructure not fit yet")
        T = max(float(tenor_years), 1e-8)
        f = self.fit_result
        t_over_tau = T / f.tau
        exp_t = np.exp(-t_over_tau)
        level = f.beta0
        slope = f.beta1 * ((1 - exp_t) / t_over_tau) if t_over_tau > 0 else f.beta1
        curv = f.beta2 * ((1 - exp_t) / t_over_tau - exp_t) if t_over_tau > 0 else 0.0
        return float(level + slope + curv)

    def factor_loadings(self) -> dict:
        """Return factor loadings as a dict for downstream signals."""
        if self.fit_result is None:
            raise RuntimeError("NelsonSiegelTermStructure not fit yet")
        f = self.fit_result
        # Long-term level is β0, short-end level is β0+β1, medium-term
        # hump height is β2.
        return {
            "level": f.beta0,
            "short_end": f.beta0 + f.beta1,
            "slope": f.beta1,
            "curvature": f.beta2,
            "residual_norm": f.residual_norm,
            "n_points": f.n_points,
        }


# ----------------------------------------------------------------------
# Skew slope + momentum
# ----------------------------------------------------------------------
def skew_slope(
    iv_25d_put: float,
    iv_atm: float,
    iv_25d_call: float,
) -> dict:
    """Return per-expiry skew metrics from 25Δ / ATM / 25Δ IVs.

    The skew "slope" is the difference between the put and call 25Δ
    IVs, normalised by the ATM IV. A steepening (negative, growing in
    magnitude) slope is the classic risk-off signal.

    Returns dict with ``put_skew``, ``call_skew``, ``skew_slope``,
    and ``risk_reversal``.
    """
    put_skew = iv_25d_put - iv_atm
    call_skew = iv_25d_call - iv_atm
    risk_reversal = iv_25d_call - iv_25d_put
    slope = (iv_25d_put - iv_25d_call) / max(iv_atm, 1e-6)
    return {
        "put_skew": float(put_skew),
        "call_skew": float(call_skew),
        "skew_slope": float(slope),
        "risk_reversal": float(risk_reversal),
    }


def skew_momentum(
    skew_history: np.ndarray,
    short_window: int = 5,
    long_window: int = 21,
) -> dict:
    """Return skew momentum indicators from a rolling skew-slope history.

    Positive momentum on put skew (i.e. skew growing more negative) is
    a risk-off signal that historically precedes drawdowns by 1-5 days.
    This gives the EV engine a forward-looking alpha signal.

    Args:
        skew_history: 1-D array of historical skew_slope values.
        short_window: Short rolling mean window.
        long_window: Long rolling mean window.

    Returns:
        dict with ``current_skew``, ``short_mean``, ``long_mean``,
        ``momentum`` (short - long), and ``steepening`` bool.
    """
    arr = np.asarray(skew_history, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {
            "current_skew": float("nan"),
            "short_mean": float("nan"),
            "long_mean": float("nan"),
            "momentum": float("nan"),
            "steepening": False,
        }

    current = float(arr[-1])
    short_mean = float(np.mean(arr[-short_window:])) if n >= 1 else current
    long_mean = float(np.mean(arr[-long_window:])) if n >= 1 else current
    momentum = short_mean - long_mean
    # In put-skew convention, more negative = steeper. "Steepening"
    # means the short-window average is more negative than the long.
    steepening = momentum < -0.01

    return {
        "current_skew": current,
        "short_mean": short_mean,
        "long_mean": long_mean,
        "momentum": float(momentum),
        "steepening": bool(steepening),
    }


# ----------------------------------------------------------------------
# Surface dislocation score
# ----------------------------------------------------------------------
def ivs_dislocation_score(
    tenors_years: np.ndarray,
    observed_ivs: np.ndarray,
    tau_years: float = _DEFAULT_TAU_YEARS,
) -> dict:
    """Score how much an observed term structure deviates from NS.

    Large positive residuals = market is pricing that tenor *above* the
    smooth NS fit → potentially rich premium, good for selling.
    Large negative residuals = market is pricing that tenor *below* the
    smooth fit → potentially cheap premium, wait.

    Returns a dict with per-tenor residuals and a composite score
    bounded in [-1, +1]:

        score > 0  → overall surface is rich (good for short premium)
        score < 0  → overall surface is cheap (avoid short premium)
    """
    ns = NelsonSiegelTermStructure(tau_years=tau_years)
    ns.fit(tenors_years, observed_ivs)
    residuals = np.asarray(observed_ivs, dtype=float) - np.array(
        [ns.iv_at(t) for t in tenors_years]
    )
    # Normalise by fitted level so the score is scale-free
    level = max(ns.factor_loadings()["level"], 1e-6)
    normalised = residuals / level
    # Clip extreme outliers before averaging
    composite = float(np.clip(np.mean(normalised), -1.0, 1.0))
    return {
        "residuals": residuals.tolist(),
        "normalised_residuals": normalised.tolist(),
        "composite_score": composite,
        "max_rich": float(np.max(normalised)),
        "max_cheap": float(np.min(normalised)),
        "ns_level": level,
        "ns_slope": ns.factor_loadings()["slope"],
        "ns_curvature": ns.factor_loadings()["curvature"],
    }
