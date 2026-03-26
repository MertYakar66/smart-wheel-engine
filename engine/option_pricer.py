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
        include_second_order: Include Vanna, Charm, Volga (default True)

    Returns:
        Dictionary with price, delta, gamma, theta, vega, rho
        If include_second_order: also vanna, charm, volga
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

        # Charm (Delta Decay): ∂Delta/∂T
        # Measures how delta changes over time
        charm_common = exp_qT * nd1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
        if option_type == 'call':
            charm = q * exp_qT * Nd1 - charm_common
        else:
            charm = -q * exp_qT * (1 - Nd1) - charm_common

        # Volga (Vomma): ∂²Price/∂σ² = Vega * d1 * d2 / σ
        # Measures convexity of vega
        volga = vega * d1 * d2 / sigma

        result['vanna'] = vanna
        result['charm'] = charm
        result['volga'] = volga

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
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Compute common terms ONCE (optimization: avoid redundant CDF calls)
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)

    # Use N(-x) = 1 - N(x) for puts
    Nm_d1 = 1.0 - Nd1
    Nm_d2 = 1.0 - Nd2

    # Price
    call_price = S * exp_qT * Nd1 - K * exp_rT * Nd2
    put_price = K * exp_rT * Nm_d2 - S * exp_qT * Nm_d1
    price = np.where(is_call, call_price, put_price)

    # Delta
    call_delta = exp_qT * Nd1
    put_delta = exp_qT * (Nd1 - 1.0)
    delta = np.where(is_call, call_delta, put_delta)

    # Gamma (same for calls and puts)
    gamma = exp_qT * nd1 / (S * sigma * sqrt_T)

    # Theta
    common_theta = -S * exp_qT * nd1 * sigma / (2 * sqrt_T)
    call_theta = common_theta + q * S * exp_qT * Nd1 - r * K * exp_rT * Nd2
    put_theta = common_theta - q * S * exp_qT * Nm_d1 + r * K * exp_rT * Nm_d2
    theta = np.where(is_call, call_theta, put_theta)

    # Vega (same for calls and puts)
    vega = S * exp_qT * nd1 * sqrt_T / 100

    # Rho
    call_rho = K * T * exp_rT * Nd2 / 100
    put_rho = -K * T * exp_rT * Nm_d2 / 100
    rho = np.where(is_call, call_rho, put_rho)

    return pd.DataFrame({
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    })
