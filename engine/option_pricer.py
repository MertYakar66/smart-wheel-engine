"""
Black-Scholes Option Pricing with Full Greeks

Supports European option pricing with continuous dividend yield (Merton model).
Includes vectorized operations for efficient batch processing.

Implementation follows Hull (11th Edition) with extensions for:
- Second-order Greeks (Vanna, Charm, Volga)
- Implied volatility solver (Newton-Raphson with bisection fallback)
- Consistent edge case handling across all functions
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Literal, Union, Tuple, Optional
import pandas as pd


# =============================================================================
# Input Validation
# =============================================================================

_VALID_OPTION_TYPES = ('call', 'put')


def _validate_inputs(
    S: float,
    K: float,
    sigma: float,
    option_type: Optional[str] = None,
    validate_positive_only: bool = False
) -> None:
    """
    Validate option pricing inputs.

    Args:
        S: Spot price (must be > 0)
        K: Strike price (must be > 0)
        sigma: Volatility (must be >= 0)
        option_type: If provided, must be 'call' or 'put'
        validate_positive_only: If True, only check positivity (for vectorized)

    Raises:
        ValueError: If any input is invalid
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if not validate_positive_only and sigma < 0:
        raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")
    if option_type is not None and option_type not in _VALID_OPTION_TYPES:
        raise ValueError(
            f"option_type must be 'call' or 'put', got '{option_type}'"
        )


def _handle_deterministic_case(
    S: float, K: float, T: float, r: float, q: float,
    option_type: Literal['call', 'put']
) -> dict:
    """
    Handle deterministic case (T=0 or sigma=0) consistently.

    When there's no randomness, option value is deterministic forward value.
    Greeks are boundary values based on moneyness.
    """
    exp_qT = np.exp(-q * T) if T > 0 else 1.0
    exp_rT = np.exp(-r * T) if T > 0 else 1.0

    forward = S * exp_qT
    discounted_strike = K * exp_rT

    if option_type == 'call':
        price = max(0.0, forward - discounted_strike)
        # Delta is e^(-qT) for ITM, 0 for OTM, undefined at ATM (use 0.5)
        if forward > discounted_strike:
            delta = exp_qT
        elif forward < discounted_strike:
            delta = 0.0
        else:
            delta = 0.5 * exp_qT  # ATM boundary
    else:
        price = max(0.0, discounted_strike - forward)
        if forward < discounted_strike:
            delta = -exp_qT
        elif forward > discounted_strike:
            delta = 0.0
        else:
            delta = -0.5 * exp_qT  # ATM boundary

    return {
        'price': price,
        'delta': delta,
        'gamma': 0.0,
        'theta': 0.0,
        'vega': 0.0,
        'rho': 0.0,
        'vanna': 0.0,
        'charm': 0.0,
        'volga': 0.0,
        'speed': 0.0,
        'color': 0.0,
        'ultima': 0.0,
    }


# =============================================================================
# Core Black-Scholes Functions
# =============================================================================

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0
) -> float:
    """
    Black-Scholes closed-form solution for European options.

    Args:
        S: Current stock price (must be > 0)
        K: Strike price (must be > 0)
        T: Time to expiration (years)
        r: Risk-free rate (annualized)
        sigma: Implied volatility (annualized, must be >= 0)
        option_type: 'call' or 'put'
        q: Continuous dividend yield (annualized, default 0.0)

    Returns:
        Option price

    Raises:
        ValueError: If inputs are invalid
    """
    _validate_inputs(S, K, sigma, option_type)

    if T <= 0:
        # At expiration, return intrinsic value
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)

    if sigma <= 0:
        # Zero volatility edge case
        if option_type == 'call':
            return max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            return max(0, K * np.exp(-r * T) - S * np.exp(-q * T))

    # Standard Black-Scholes with continuous dividend yield (Merton)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price


def black_scholes_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0
) -> float:
    """
    Calculate option delta (sensitivity to underlying price).

    Args:
        S, K, T, r, sigma, option_type, q: Same as black_scholes_price

    Returns:
        Delta value: [0, 1] for calls, [-1, 0] for puts
    """
    _validate_inputs(S, K, sigma, option_type)

    if T <= 0 or sigma <= 0:
        result = _handle_deterministic_case(S, K, T, r, q, option_type)
        return result['delta']

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def black_scholes_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Calculate option gamma (sensitivity of delta to underlying price).
    Same for calls and puts.

    Args:
        S, K, T, r, sigma, q: Same as black_scholes_price

    Returns:
        Gamma value (always positive)
    """
    _validate_inputs(S, K, sigma)

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def black_scholes_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0
) -> float:
    """
    Calculate option theta (time decay per year).
    Divide by 365 for daily theta.

    Args:
        S, K, T, r, sigma, option_type, q: Same as black_scholes_price

    Returns:
        Theta value (typically negative for long options)
    """
    _validate_inputs(S, K, sigma, option_type)

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # First term is always negative (time decay)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

    if option_type == 'call':
        theta = term1 + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = term1 - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)

    return theta


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Calculate option vega (sensitivity to volatility).
    Returns vega per 1% change in volatility (divide by 100 for per 1 vol point).
    Same for calls and puts.

    Args:
        S, K, T, r, sigma, q: Same as black_scholes_price

    Returns:
        Vega value (always positive)
    """
    _validate_inputs(S, K, sigma)

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100


def black_scholes_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0
) -> float:
    """
    Calculate option rho (sensitivity to interest rate).
    Returns rho per 1% change in rate.

    Args:
        S, K, T, r, sigma, option_type, q: Same as black_scholes_price

    Returns:
        Rho value
    """
    _validate_inputs(S, K, sigma, option_type)

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def black_scholes_speed(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Calculate option speed (∂Gamma/∂S = ∂³V/∂S³).

    Speed measures how gamma changes as the underlying moves.
    Important for dynamic hedging of large positions.

    Args:
        S, K, T, r, sigma, q: Same as black_scholes_price

    Returns:
        Speed value
    """
    _validate_inputs(S, K, sigma)

    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    gamma = black_scholes_gamma(S, K, T, r, sigma, q)

    return -gamma * (d1 / (sigma * sqrt_T) + 1) / S


def black_scholes_color(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Calculate option color (∂Gamma/∂T), also known as gamma decay or gamma bleed.

    Color measures how gamma changes over time.
    Critical for managing gamma exposure near expiration.

    Args:
        S, K, T, r, sigma, q: Same as black_scholes_price

    Returns:
        Color value (per year; divide by 365 for daily)
    """
    _validate_inputs(S, K, sigma)

    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    gamma = black_scholes_gamma(S, K, T, r, sigma, q)

    color_factor = 2 * (r - q) * T - d2 * sigma * sqrt_T
    return -gamma / (2 * T) * (2 * q * T + 1 + d1 * color_factor / (sigma * sqrt_T))


def black_scholes_ultima(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Calculate option ultima (∂Volga/∂σ = ∂³V/∂σ³).

    Ultima measures how volga changes with volatility.
    Used for managing vega convexity risk in vol trading.

    Args:
        S, K, T, r, sigma, q: Same as black_scholes_price

    Returns:
        Ultima value
    """
    _validate_inputs(S, K, sigma)

    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    vega = black_scholes_vega(S, K, T, r, sigma, q)

    return -vega * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2) / (sigma**2)


def black_scholes_all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0,
    include_second_order: bool = True
) -> dict:
    """
    Calculate price and all Greeks in one call (most efficient).

    Args:
        S, K, T, r, sigma, option_type, q: Same as black_scholes_price
        include_second_order: Include higher-order Greeks (default True)

    Returns:
        Dictionary with price, delta, gamma, theta, vega, rho
        If include_second_order: also vanna, charm, volga, speed, color, ultima

    Second-order Greeks:
        vanna: ∂Delta/∂σ - Vol-delta cross sensitivity
        charm: ∂Delta/∂t (calendar time) - Delta decay as time passes
        volga: ∂Vega/∂σ - Vega convexity

    Third-order Greeks:
        speed: ∂Gamma/∂S - Gamma sensitivity to spot
        color: ∂Gamma/∂T - Gamma decay over time
        ultima: ∂Volga/∂σ - Volga sensitivity to vol
    """
    _validate_inputs(S, K, sigma, option_type)

    if T <= 0 or sigma <= 0:
        result = _handle_deterministic_case(S, K, T, r, q, option_type)
        if not include_second_order:
            # Remove second-order Greeks if not requested
            for key in ['vanna', 'charm', 'volga']:
                result.pop(key, None)
        return result

    # Calculate d1, d2 once
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Common terms (computed once for efficiency)
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)

    # First-order Greeks
    if option_type == 'call':
        price = S * exp_qT * Nd1 - K * exp_rT * Nd2
        delta = exp_qT * Nd1
        theta = (-S * exp_qT * nd1 * sigma / (2 * sqrt_T)
                 + q * S * exp_qT * Nd1
                 - r * K * exp_rT * Nd2)
        rho = K * T * exp_rT * Nd2 / 100
    else:
        Nm_d1 = 1.0 - Nd1  # norm.cdf(-d1) = 1 - norm.cdf(d1)
        Nm_d2 = 1.0 - Nd2
        price = K * exp_rT * Nm_d2 - S * exp_qT * Nm_d1
        delta = exp_qT * (Nd1 - 1.0)
        theta = (-S * exp_qT * nd1 * sigma / (2 * sqrt_T)
                 - q * S * exp_qT * Nm_d1
                 + r * K * exp_rT * Nm_d2)
        rho = -K * T * exp_rT * Nm_d2 / 100

    gamma = exp_qT * nd1 / (S * sigma * sqrt_T)
    vega = S * exp_qT * nd1 * sqrt_T / 100

    result = {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

    # Second-order Greeks (optional but recommended for risk management)
    if include_second_order:
        # Vanna: ∂Delta/∂σ = ∂Vega/∂S = -e^(-qT) * d2 * n(d1) / σ
        # Measures how delta changes with volatility
        vanna = -exp_qT * nd1 * d2 / sigma

        # Charm (Delta Decay): ∂Delta/∂t (calendar time convention)
        # NOTE: This is -∂Delta/∂τ where τ = time-to-expiry.
        # Positive charm means delta increases as time passes.
        # Industry standard: charm measures delta sensitivity to passage of time.
        charm_common = exp_qT * nd1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
        if option_type == 'call':
            charm = q * exp_qT * Nd1 - charm_common
        else:
            charm = -q * exp_qT * (1 - Nd1) - charm_common

        # Volga (Vomma): ∂²Price/∂σ² = Vega * d1 * d2 / σ
        # Measures convexity of vega
        volga = vega * d1 * d2 / sigma

        # Third-order Greeks
        # Speed: ∂Gamma/∂S = -Gamma * (d1 / (σ√T) + 1) / S
        # Measures how gamma changes with spot price movement
        speed = -gamma * (d1 / (sigma * sqrt_T) + 1) / S

        # Color: ∂Gamma/∂T (gamma bleed)
        # Measures how gamma decays over time
        color_factor = 2 * (r - q) * T - d2 * sigma * sqrt_T
        color = -gamma / (2 * T) * (2 * q * T + 1 + d1 * color_factor / (sigma * sqrt_T))

        # Ultima: ∂Volga/∂σ = ∂³V/∂σ³
        # Measures how volga changes with volatility
        # Formula: -vega * (d1*d2*(1 - d1*d2) + d1² + d2²) / σ²
        ultima = -vega * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2) / (sigma**2)

        result['vanna'] = vanna
        result['charm'] = charm
        result['volga'] = volga
        result['speed'] = speed
        result['color'] = color
        result['ultima'] = ultima

    return result


# =============================================================================
# Implied Volatility Solver
# =============================================================================

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0,
    precision: float = 1e-6,
    max_iterations: int = 100,
) -> Optional[float]:
    """
    Calculate implied volatility from market price using Newton-Raphson.

    Uses Newton-Raphson with Brent's method as fallback for robustness.
    This is the standard industry approach for IV calculation.

    Args:
        market_price: Observed market price of the option
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        option_type: 'call' or 'put'
        q: Continuous dividend yield
        precision: Convergence tolerance (default 1e-6)
        max_iterations: Maximum Newton-Raphson iterations

    Returns:
        Implied volatility (annualized), or None if no solution found

    Raises:
        ValueError: If inputs are invalid
    """
    if market_price <= 0:
        return None

    if T <= 0:
        return None

    # Validate basic arbitrage bounds
    forward = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    if option_type == 'call':
        intrinsic = max(0, forward - discounted_strike)
        upper_bound = forward  # Call can't be worth more than forward
    else:
        intrinsic = max(0, discounted_strike - forward)
        upper_bound = discounted_strike  # Put can't be worth more than PV of strike

    if market_price < intrinsic - precision:
        return None  # Below intrinsic (arbitrage)
    if market_price > upper_bound + precision:
        return None  # Above theoretical maximum

    # Newton-Raphson with initial guess
    # Use Brenner-Subrahmanyam approximation for initial guess
    sigma = np.sqrt(2 * np.pi / T) * market_price / S

    # Clamp initial guess to reasonable range
    sigma = max(0.01, min(sigma, 5.0))

    for _ in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type, q)
        vega_raw = black_scholes_vega(S, K, T, r, sigma, q) * 100  # Undo the /100 scaling

        diff = price - market_price

        if abs(diff) < precision:
            return sigma

        if vega_raw < 1e-10:
            # Vega too small, Newton-Raphson won't converge
            break

        # Newton-Raphson step
        sigma_new = sigma - diff / vega_raw

        # Ensure sigma stays positive and reasonable
        sigma_new = max(0.001, min(sigma_new, 10.0))

        if abs(sigma_new - sigma) < precision:
            return sigma_new

        sigma = sigma_new

    # Fallback to Brent's method if Newton-Raphson fails
    try:
        def objective(vol):
            return black_scholes_price(S, K, T, r, vol, option_type, q) - market_price

        sigma = brentq(objective, 0.001, 10.0, xtol=precision)
        return sigma
    except ValueError:
        return None


def estimate_option_price_from_iv(
    underlying_price: float,
    strike: float,
    dte: int,
    iv: float,
    risk_free_rate: float,
    option_type: Literal['call', 'put'],
    dividend_yield: float = 0.0
) -> float:
    """
    Convenience wrapper: estimate option price given IV.

    LIMITATION: This uses European option pricing (no early exercise).
    For deep ITM American options, this underestimates value slightly.

    Args:
        underlying_price: Current stock price
        strike: Strike price
        dte: Days to expiration
        iv: Implied volatility (annualized, e.g., 0.25 = 25%)
        risk_free_rate: Risk-free rate (annualized, e.g., 0.04 = 4%)
        option_type: 'call' or 'put'
        dividend_yield: Continuous dividend yield (default 0.0)

    Returns:
        Estimated option price per share
    """
    T = dte / 365.0
    return black_scholes_price(underlying_price, strike, T, risk_free_rate, iv, option_type, q=dividend_yield)


# =============================================================================
# American Option Pricing (Barone-Adesi-Whaley Approximation)
# =============================================================================

def _baw_critical_price_call(
    S: float, K: float, T: float, r: float, sigma: float, q: float,
    tolerance: float = 1e-6, max_iter: int = 100
) -> Tuple[float, float, float]:
    """
    Find critical stock price S* for American call using Newton-Raphson.

    Returns:
        Tuple of (S_star, q2, A2)
    """
    sqrt_T = np.sqrt(T)
    sigma_sq = sigma ** 2

    # Calculate M, N, K coefficients for the quadratic equation
    M = 2 * r / sigma_sq
    N = 2 * (r - q) / sigma_sq
    k = 1 - np.exp(-r * T)

    # q2 is the positive root of the characteristic equation
    # q2 = (-(N-1) + sqrt((N-1)^2 + 4M/k)) / 2
    discriminant = (N - 1) ** 2 + 4 * M / k
    q2 = (-(N - 1) + np.sqrt(discriminant)) / 2

    # Initial guess for S* (critical price)
    # Start at a reasonable multiple of strike
    S_star = K + (S - K) * 0.5 if S > K else K * 1.1

    # Newton-Raphson to find S* where:
    # S* - K = C_euro(S*) + (1 - e^(-qT) * N(d1(S*))) * S* / q2
    for _ in range(max_iter):
        d1_star = (np.log(S_star / K) + (r - q + 0.5 * sigma_sq) * T) / (sigma * sqrt_T)
        Nd1_star = norm.cdf(d1_star)
        nd1_star = norm.pdf(d1_star)
        exp_qT = np.exp(-q * T)

        # European call at S*
        d2_star = d1_star - sigma * sqrt_T
        C_euro = S_star * exp_qT * Nd1_star - K * np.exp(-r * T) * norm.cdf(d2_star)

        # Equation: f(S*) = S* - K - C_euro - (1 - exp(-qT)*N(d1)) * S* / q2 = 0
        # Rearranged: early exercise premium = intrinsic - european value
        lhs = S_star - K
        rhs = C_euro + (1 - exp_qT * Nd1_star) * S_star / q2

        f = lhs - rhs
        if abs(f) < tolerance:
            break

        # Derivative for Newton-Raphson
        delta_euro = exp_qT * Nd1_star
        # df/dS* = 1 - delta_euro - (1/q2) * (1 - exp(-qT)*N(d1) - S* * exp(-qT) * n(d1) / (S*σ√T))
        f_prime = 1 - delta_euro - (1 / q2) * (
            1 - exp_qT * Nd1_star + S_star * exp_qT * nd1_star / (S_star * sigma * sqrt_T)
        )

        # Safeguarded Newton step
        if abs(f_prime) > 1e-10:
            S_star_new = S_star - f / f_prime
            S_star_new = max(K * 1.001, S_star_new)  # S* must be > K for calls
            S_star = S_star_new
        else:
            # Bisection fallback
            S_star = (S_star + K) / 2 if f > 0 else S_star * 1.1

    # Calculate A2
    d1_star = (np.log(S_star / K) + (r - q + 0.5 * sigma_sq) * T) / (sigma * sqrt_T)
    exp_qT = np.exp(-q * T)
    A2 = (S_star / q2) * (1 - exp_qT * norm.cdf(d1_star))

    return S_star, q2, A2


def _baw_critical_price_put(
    S: float, K: float, T: float, r: float, sigma: float, q: float,
    tolerance: float = 1e-6, max_iter: int = 100
) -> Tuple[float, float, float]:
    """
    Find critical stock price S* for American put using Newton-Raphson.

    Returns:
        Tuple of (S_star, q1, A1)
    """
    sqrt_T = np.sqrt(T)
    sigma_sq = sigma ** 2

    M = 2 * r / sigma_sq
    N = 2 * (r - q) / sigma_sq
    k = 1 - np.exp(-r * T)

    # q1 is the negative root
    discriminant = (N - 1) ** 2 + 4 * M / k
    q1 = (-(N - 1) - np.sqrt(discriminant)) / 2

    # Initial guess for S* (critical price for put)
    S_star = K * 0.9 if S < K else K - (K - S) * 0.5

    for _ in range(max_iter):
        d1_star = (np.log(S_star / K) + (r - q + 0.5 * sigma_sq) * T) / (sigma * sqrt_T)
        Nm_d1_star = norm.cdf(-d1_star)  # N(-d1)
        nm_d1_star = norm.pdf(d1_star)  # n(d1) = n(-d1)
        exp_qT = np.exp(-q * T)

        # European put at S*
        d2_star = d1_star - sigma * sqrt_T
        P_euro = K * np.exp(-r * T) * norm.cdf(-d2_star) - S_star * exp_qT * Nm_d1_star

        # Equation: K - S* = P_euro + (1 - exp(-qT)*N(-d1)) * (-S*/q1)
        # Note: q1 < 0, so -S*/q1 > 0
        lhs = K - S_star
        rhs = P_euro - (1 - exp_qT * Nm_d1_star) * S_star / q1

        f = lhs - rhs
        if abs(f) < tolerance:
            break

        # Derivative: df/dS* = -1 - delta_euro + (1/q1) * [d((1-exp_qT*N(-d1))*S*)/dS*]
        # d((1-exp_qT*N(-d1))*S*)/dS* = (1-exp_qT*N(-d1)) + exp_qT*n(d1)/(sigma*sqrt_T)
        delta_euro = -exp_qT * Nm_d1_star
        f_prime = -1 - delta_euro + (1 / q1) * (
            1 - exp_qT * Nm_d1_star + exp_qT * nm_d1_star / (sigma * sqrt_T)
        )

        if abs(f_prime) > 1e-10:
            S_star_new = S_star - f / f_prime
            S_star_new = max(0.001, min(K * 0.999, S_star_new))  # S* must be < K for puts
            S_star = S_star_new
        else:
            S_star = (S_star + K) / 2 if f > 0 else S_star * 0.9

    # Calculate A1
    d1_star = (np.log(S_star / K) + (r - q + 0.5 * sigma_sq) * T) / (sigma * sqrt_T)
    exp_qT = np.exp(-q * T)
    A1 = -(S_star / q1) * (1 - exp_qT * norm.cdf(-d1_star))

    return S_star, q1, A1


def american_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0
) -> float:
    """
    American option pricing using Barone-Adesi-Whaley (1987) approximation.

    This is the most widely-used analytical approximation for American options.
    Accuracy is typically within 0.1% of binomial tree prices.

    The BAW model approximates the early exercise premium as a function of the
    critical stock price S* where immediate exercise becomes optimal.

    Args:
        S: Current stock price (must be > 0)
        K: Strike price (must be > 0)
        T: Time to expiration (years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized, must be > 0)
        option_type: 'call' or 'put'
        q: Continuous dividend yield (annualized, default 0.0)

    Returns:
        American option price

    References:
        Barone-Adesi, G. and Whaley, R.E. (1987). "Efficient Analytic
        Approximation of American Option Values." Journal of Finance, 42(2).
    """
    _validate_inputs(S, K, sigma, option_type)

    # Edge cases
    if T <= 0:
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)

    if sigma <= 0:
        # Zero volatility: American = intrinsic value (exercise immediately if ITM)
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)

    # European price as baseline
    european = black_scholes_price(S, K, T, r, sigma, option_type, q)

    # For American calls with no dividends, early exercise is never optimal
    if option_type == 'call' and q <= 0:
        return european

    # For American puts with zero interest rate, early exercise is never optimal
    if option_type == 'put' and r <= 0:
        return european

    if option_type == 'call':
        # American call with dividends
        S_star, q2, A2 = _baw_critical_price_call(S, K, T, r, sigma, q)

        if S >= S_star:
            # Optimal to exercise immediately
            return S - K
        else:
            # Early exercise premium
            return european + A2 * (S / S_star) ** q2

    else:
        # American put
        S_star, q1, A1 = _baw_critical_price_put(S, K, T, r, sigma, q)

        if S <= S_star:
            # Optimal to exercise immediately
            return K - S
        else:
            # Early exercise premium
            return european + A1 * (S_star / S) ** (-q1)


def american_option_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0,
    dS: float = 0.01,
    dT: float = 1/365,
    d_sigma: float = 0.01,
    dr: float = 0.01
) -> dict:
    """
    Calculate American option Greeks using finite differences.

    Since BAW doesn't have closed-form Greeks, we compute them numerically.

    Args:
        S, K, T, r, sigma, option_type, q: Same as american_option_price
        dS: Spot bump for delta/gamma (as fraction of S)
        dT: Time bump for theta (in years, default 1 day)
        d_sigma: Vol bump for vega (absolute, default 1%)
        dr: Rate bump for rho (absolute, default 1%)

    Returns:
        Dictionary with price, delta, gamma, theta, vega, rho
    """
    price = american_option_price(S, K, T, r, sigma, option_type, q)

    # Delta and Gamma via central differences
    bump = S * dS
    price_up = american_option_price(S + bump, K, T, r, sigma, option_type, q)
    price_down = american_option_price(S - bump, K, T, r, sigma, option_type, q)
    delta = (price_up - price_down) / (2 * bump)
    gamma = (price_up - 2 * price + price_down) / (bump ** 2)

    # Theta (forward difference to avoid T < 0)
    if T > dT:
        price_T_minus = american_option_price(S, K, T - dT, r, sigma, option_type, q)
        theta = (price_T_minus - price) / dT  # Negative = time decay
    else:
        theta = 0.0

    # Vega
    price_vol_up = american_option_price(S, K, T, r, sigma + d_sigma, option_type, q)
    vega = (price_vol_up - price) / d_sigma / 100  # Per 1% vol move

    # Rho
    price_r_up = american_option_price(S, K, T, r + dr, sigma, option_type, q)
    rho = (price_r_up - price) / dr / 100  # Per 1% rate move

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho,
    }


# =============================================================================
# Vectorized Operations (Batch Processing)
# =============================================================================

def _vectorized_intrinsic(
    S: np.ndarray, K: np.ndarray, is_call: np.ndarray,
    exp_qT: np.ndarray, exp_rT: np.ndarray
) -> np.ndarray:
    """Compute intrinsic/deterministic value for edge cases."""
    forward = S * exp_qT
    discounted_K = K * exp_rT
    call_intrinsic = np.maximum(0.0, forward - discounted_K)
    put_intrinsic = np.maximum(0.0, discounted_K - forward)
    return np.where(is_call, call_intrinsic, put_intrinsic)


def vectorized_bs_price(
    S: Union[np.ndarray, pd.Series],
    K: Union[np.ndarray, pd.Series],
    T: Union[np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[np.ndarray, pd.Series],
    is_call: Union[np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0
) -> np.ndarray:
    """
    Fully vectorized Black-Scholes pricing.

    Handles edge cases (T<=0, sigma<=0) consistently with scalar API
    by returning intrinsic/deterministic values.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T_raw = np.asarray(T, dtype=float)
    sigma_raw = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    if isinstance(r, (int, float)):
        r = np.full_like(S, r)
    else:
        r = np.asarray(r, dtype=float)
    if isinstance(q, (int, float)):
        q = np.full_like(S, q)
    else:
        q = np.asarray(q, dtype=float)

    # Identify edge cases
    is_deterministic = (T_raw <= 0) | (sigma_raw <= 0)

    # Safe values for BS calculation (will be masked out for edge cases)
    T = np.where(is_deterministic, 1.0, T_raw)
    sigma = np.where(is_deterministic, 0.2, sigma_raw)

    # Standard BS calculation
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    exp_qT = np.exp(-q * T_raw)  # Use raw T for discounting
    exp_rT = np.exp(-r * T_raw)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    call_price = S * exp_qT * Nd1 - K * exp_rT * Nd2
    put_price = K * exp_rT * (1.0 - Nd2) - S * exp_qT * (1.0 - Nd1)
    bs_price = np.where(is_call, call_price, put_price)

    # Replace edge cases with intrinsic value
    intrinsic = _vectorized_intrinsic(S, K, is_call, exp_qT, exp_rT)
    return np.where(is_deterministic, intrinsic, bs_price)


def vectorized_bs_delta(
    S: Union[np.ndarray, pd.Series],
    K: Union[np.ndarray, pd.Series],
    T: Union[np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[np.ndarray, pd.Series],
    is_call: Union[np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0
) -> np.ndarray:
    """Vectorized delta with proper edge case handling."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T_raw = np.asarray(T, dtype=float)
    sigma_raw = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    if isinstance(r, (int, float)):
        r = np.full_like(S, r)
    else:
        r = np.asarray(r, dtype=float)
    if isinstance(q, (int, float)):
        q = np.full_like(S, q)
    else:
        q = np.asarray(q, dtype=float)

    is_deterministic = (T_raw <= 0) | (sigma_raw <= 0)
    T = np.where(is_deterministic, 1.0, T_raw)
    sigma = np.where(is_deterministic, 0.2, sigma_raw)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    exp_qT = np.exp(-q * T_raw)

    call_delta = exp_qT * norm.cdf(d1)
    put_delta = exp_qT * (norm.cdf(d1) - 1.0)
    bs_delta = np.where(is_call, call_delta, put_delta)

    # Edge case: delta is e^(-qT) for ITM, 0 for OTM, 0.5*e^(-qT) for ATM
    forward = S * exp_qT
    discounted_K = K * np.exp(-r * T_raw)
    call_edge = np.where(forward > discounted_K, exp_qT,
                         np.where(forward < discounted_K, 0.0, 0.5 * exp_qT))
    put_edge = np.where(forward < discounted_K, -exp_qT,
                        np.where(forward > discounted_K, 0.0, -0.5 * exp_qT))
    edge_delta = np.where(is_call, call_edge, put_edge)

    return np.where(is_deterministic, edge_delta, bs_delta)


def vectorized_bs_all_greeks(
    S: Union[np.ndarray, pd.Series],
    K: Union[np.ndarray, pd.Series],
    T: Union[np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[np.ndarray, pd.Series],
    is_call: Union[np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0
) -> pd.DataFrame:
    """
    Vectorized calculation of all Greeks with proper edge case handling.

    Returns:
        DataFrame with columns: price, delta, gamma, theta, vega, rho
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T_raw = np.asarray(T, dtype=float)
    sigma_raw = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    if isinstance(r, (int, float)):
        r = np.full_like(S, r)
    else:
        r = np.asarray(r, dtype=float)
    if isinstance(q, (int, float)):
        q = np.full_like(S, q)
    else:
        q = np.asarray(q, dtype=float)

    is_deterministic = (T_raw <= 0) | (sigma_raw <= 0)
    T = np.where(is_deterministic, 1.0, T_raw)
    sigma = np.where(is_deterministic, 0.2, sigma_raw)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    exp_qT = np.exp(-q * T_raw)
    exp_rT = np.exp(-r * T_raw)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    Nm_d1 = 1.0 - Nd1
    Nm_d2 = 1.0 - Nd2

    # Standard BS Greeks
    call_price = S * exp_qT * Nd1 - K * exp_rT * Nd2
    put_price = K * exp_rT * Nm_d2 - S * exp_qT * Nm_d1
    price = np.where(is_call, call_price, put_price)

    call_delta = exp_qT * Nd1
    put_delta = exp_qT * (Nd1 - 1.0)
    delta = np.where(is_call, call_delta, put_delta)

    gamma = exp_qT * nd1 / (S * sigma * sqrt_T)

    common_theta = -S * exp_qT * nd1 * sigma / (2 * sqrt_T)
    call_theta = common_theta + q * S * exp_qT * Nd1 - r * K * exp_rT * Nd2
    put_theta = common_theta - q * S * exp_qT * Nm_d1 + r * K * exp_rT * Nm_d2
    theta = np.where(is_call, call_theta, put_theta)

    vega = S * exp_qT * nd1 * sqrt_T / 100

    call_rho = K * T * exp_rT * Nd2 / 100
    put_rho = -K * T * exp_rT * Nm_d2 / 100
    rho = np.where(is_call, call_rho, put_rho)

    # Edge case values
    intrinsic = _vectorized_intrinsic(S, K, is_call, exp_qT, exp_rT)
    forward = S * exp_qT
    discounted_K = K * exp_rT
    call_edge_delta = np.where(forward > discounted_K, exp_qT,
                               np.where(forward < discounted_K, 0.0, 0.5 * exp_qT))
    put_edge_delta = np.where(forward < discounted_K, -exp_qT,
                              np.where(forward > discounted_K, 0.0, -0.5 * exp_qT))
    edge_delta = np.where(is_call, call_edge_delta, put_edge_delta)

    return pd.DataFrame({
        'price': np.where(is_deterministic, intrinsic, price),
        'delta': np.where(is_deterministic, edge_delta, delta),
        'gamma': np.where(is_deterministic, 0.0, gamma),
        'theta': np.where(is_deterministic, 0.0, theta),
        'vega': np.where(is_deterministic, 0.0, vega),
        'rho': np.where(is_deterministic, 0.0, rho),
    })
