"""
Extreme-value tail-risk estimation via Peaks-over-Threshold + Generalised Pareto.

Why this module exists
----------------------
Historical-simulation CVaR has a fundamental weakness: it cannot see any
loss bigger than the worst observation in the sample. When the sample is
a few hundred daily returns, the 0.5% or 1% CVaR is effectively driven by
a single data point and has enormous estimation noise.

The institutional fix is extreme-value theory (EVT): instead of relying
on the raw empirical tail, fit a Generalised Pareto distribution to all
losses that exceed a threshold ``u`` and then use the fitted GPD to
compute quantiles beyond the empirical support.

This module implements the standard POT-GPD estimator (Pickands 1975,
Balkema-de Haan 1974, McNeil & Frey 2000) on top of ``scipy.stats``:

* :func:`select_threshold`     — data-driven threshold via the 95th
  percentile of absolute losses (standard default for equity returns).
* :func:`fit_gpd_tail`         — MLE fit of the GPD shape & scale
  parameters to exceedances.
* :func:`gpd_var_cvar`         — VaR and CVaR at an arbitrary confidence
  level, computed from the fitted GPD.
* :func:`tail_regime_flag`     — returns True when the fitted shape
  parameter xi > 0.3, indicating heavy-tailed regime where the EV engine
  should down-size positions.

All functions accept plain numpy arrays of returns (or losses) and return
dictionaries so they integrate cleanly with :class:`engine.ev_engine.EVEngine`
and :mod:`engine.risk_manager`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class GPDTailFit:
    """Result of a Peaks-over-Threshold GPD tail fit."""

    threshold: float  # u — excess threshold (negative for losses)
    n_exceedances: int  # number of observations beyond u
    n_total: int  # full sample size
    shape_xi: float  # shape parameter ξ (heavy tail when > 0)
    scale_beta: float  # scale parameter β > 0
    tail_fraction: float  # n_exceedances / n_total
    converged: bool  # did scipy MLE converge?
    log_likelihood: float  # log-likelihood at the fit


def select_threshold(
    losses: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Return the POT threshold as the p-th percentile of ``losses``.

    ``losses`` must be a numpy array of **positive** loss values
    (i.e. negative of log-returns for long positions). The threshold is
    chosen so that roughly ``100 - percentile`` percent of observations
    are classified as tail events — 5% for the default. This is the
    standard "rule of thumb" from McNeil, Frey & Embrechts 2005.

    Args:
        losses: 1-D array of positive losses.
        percentile: Threshold percentile in (0, 100).

    Returns:
        The threshold ``u`` such that ``loss > u`` for roughly 5% of rows.
    """
    arr = np.asarray(losses, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, percentile))


def fit_gpd_tail(
    losses: np.ndarray,
    threshold: float | None = None,
    min_exceedances: int = 15,
) -> GPDTailFit:
    """Fit a Generalised Pareto distribution to the tail of ``losses``.

    The standard POT recipe:
      1. Select a threshold ``u`` (95th percentile by default).
      2. Take all exceedances ``y_i = loss_i - u`` for ``loss_i > u``.
      3. MLE-fit a GPD(ξ, β) to the exceedances.

    Args:
        losses: 1-D array of positive losses (e.g. ``-log_returns``).
        threshold: Optional explicit threshold. When ``None`` the 95th
            percentile is used.
        min_exceedances: Minimum number of exceedances required for a
            valid fit. When fewer, the function returns a sentinel
            ``GPDTailFit`` with ``converged=False``.

    Returns:
        A :class:`GPDTailFit` dataclass.
    """
    arr = np.asarray(losses, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_total = len(arr)
    if n_total == 0:
        return GPDTailFit(0.0, 0, 0, 0.0, 1e-6, 0.0, False, 0.0)

    if threshold is None:
        threshold = select_threshold(arr)

    exceedances = arr[arr > threshold] - threshold
    n_exc = len(exceedances)

    if n_exc < min_exceedances:
        return GPDTailFit(
            threshold=threshold,
            n_exceedances=n_exc,
            n_total=n_total,
            shape_xi=0.0,
            scale_beta=max(float(np.std(arr)), 1e-6),
            tail_fraction=n_exc / n_total,
            converged=False,
            log_likelihood=float("nan"),
        )

    # scipy's genpareto has a "floc" location parameter which we fix at 0
    # because we are fitting *excesses over threshold*.
    try:
        xi, loc, beta = stats.genpareto.fit(exceedances, floc=0)
        # Sanity: beta must be positive
        if beta <= 0 or not np.isfinite(xi) or not np.isfinite(beta):
            raise ValueError("invalid GPD params")
        ll = float(np.sum(stats.genpareto.logpdf(exceedances, xi, loc=0, scale=beta)))
        converged = True
    except Exception:
        # Method-of-moments fall-back: E[Y]=β/(1-ξ), Var[Y]=β²/((1-ξ)²(1-2ξ))
        mean_e = float(np.mean(exceedances))
        var_e = float(np.var(exceedances))
        if var_e > 0 and mean_e > 0:
            ratio = mean_e**2 / var_e
            xi = 0.5 * (1 - ratio)
            beta = 0.5 * mean_e * (1 + ratio)
        else:
            xi, beta = 0.0, max(mean_e, 1e-6)
        ll = float("nan")
        converged = False

    return GPDTailFit(
        threshold=threshold,
        n_exceedances=n_exc,
        n_total=n_total,
        shape_xi=float(xi),
        scale_beta=float(beta),
        tail_fraction=n_exc / n_total,
        converged=converged,
        log_likelihood=ll,
    )


def gpd_var_cvar(
    fit: GPDTailFit,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Compute VaR and CVaR at ``confidence`` using the fitted GPD tail.

    Formulas (McNeil, Frey & Embrechts 2005, Chapter 7):

      VaR_α(L) = u + (β/ξ) * [ ((n/n_u) * (1-α))^(-ξ) - 1 ]    ξ ≠ 0
      VaR_α(L) = u + β * log(n/(n_u*(1-α)))                    ξ = 0

      CVaR_α(L) = (VaR_α + β - ξ*u) / (1 - ξ)                  ξ < 1

    Returns positive numbers — losses, not returns.

    When ``ξ >= 1`` the GPD has no finite mean and CVaR is formally
    infinite; we return a sentinel of ``VaR * 3`` (a practical upper
    bound often used in institutional risk systems) and flag this via
    a warning is emitted by the caller if desired.

    Args:
        fit: GPDTailFit from :func:`fit_gpd_tail`.
        confidence: VaR confidence level in (0, 1). 0.99 = 1% tail.

    Returns:
        (var, cvar) as positive-loss dollar magnitudes (or whatever unit
        ``losses`` was in when you fit the GPD).
    """
    if fit.n_total == 0 or fit.n_exceedances == 0:
        return 0.0, 0.0

    alpha = confidence
    n = fit.n_total
    n_u = fit.n_exceedances
    u = fit.threshold
    xi = fit.shape_xi
    beta = fit.scale_beta

    tail_prob = n_u / n  # P(L > u)

    # Need (1-α) < tail_prob for POT formula to apply.
    if (1 - alpha) > tail_prob:
        # Confidence level not deep enough — return empirical quantile.
        return u, u + beta

    ratio = (n / n_u) * (1 - alpha)

    if abs(xi) < 1e-8:
        # ξ → 0 (exponential tail)
        var = u + beta * (-np.log(ratio))
        cvar = var + beta
    else:
        var = u + (beta / xi) * (ratio ** (-xi) - 1)
        if xi < 1:
            cvar = (var + beta - xi * u) / (1 - xi)
        else:
            # ξ ≥ 1: infinite mean; return practical upper bound.
            cvar = var * 3.0

    return float(var), float(cvar)


def tail_regime_flag(fit: GPDTailFit, heavy_tail_threshold: float = 0.3) -> bool:
    """Return True when the fitted tail is heavy (ξ > threshold).

    Used by the EV engine to down-size positions during heavy-tail
    regimes. A ξ of 0.3 corresponds to roughly finite-but-volatile mean,
    and ξ > 0.5 is truly fat-tailed (e.g. crisis periods).
    """
    return fit.converged and fit.shape_xi > heavy_tail_threshold


def pot_gpd_cvar(
    losses: np.ndarray,
    confidence: float = 0.99,
    threshold_percentile: float = 95.0,
) -> dict:
    """One-shot convenience wrapper: losses → POT-GPD CVaR dict.

    Returns a dict with all the diagnostics a risk system wants:
    ``var``, ``cvar``, ``xi``, ``beta``, ``threshold``, ``heavy_tail``,
    and ``fit_quality``.
    """
    threshold = select_threshold(losses, percentile=threshold_percentile)
    fit = fit_gpd_tail(losses, threshold=threshold)
    var, cvar = gpd_var_cvar(fit, confidence=confidence)

    if not fit.converged:
        quality = "insufficient_data"
    elif fit.shape_xi > 0.5:
        quality = "extreme_heavy_tail"
    elif fit.shape_xi > 0.3:
        quality = "heavy_tail"
    elif fit.shape_xi > 0.0:
        quality = "finite_variance"
    else:
        quality = "thin_tail"

    return {
        "var": var,
        "cvar": cvar,
        "xi": fit.shape_xi,
        "beta": fit.scale_beta,
        "threshold": fit.threshold,
        "n_exceedances": fit.n_exceedances,
        "n_total": fit.n_total,
        "tail_fraction": fit.tail_fraction,
        "heavy_tail": tail_regime_flag(fit),
        "fit_quality": quality,
        "converged": fit.converged,
    }
