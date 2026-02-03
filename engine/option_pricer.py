"""
Black-Scholes Option Pricing with Full Greeks

Supports European option pricing with continuous dividend yield (Merton model).
Includes vectorized operations for efficient batch processing.
"""

import numpy as np
from scipy.stats import norm
from typing import Literal, Union, Tuple
import pandas as pd


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
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annualized)
        sigma: Implied volatility (annualized)
        option_type: 'call' or 'put'
        q: Continuous dividend yield (annualized, default 0.0)

    Returns:
        Option price
    """
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
        Delta value (-1 to 1)
    """
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    if sigma <= 0:
        sigma = 1e-10

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
    q: float = 0.0
) -> dict:
    """
    Calculate price and all Greeks in one call (more efficient).

    Args:
        S, K, T, r, sigma, option_type, q: Same as black_scholes_price

    Returns:
        Dictionary with price, delta, gamma, theta, vega, rho
    """
    if T <= 0:
        intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
        delta = 1.0 if option_type == 'call' and S > K else (-1.0 if option_type == 'put' and S < K else 0.0)
        return {
            'price': intrinsic,
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    if sigma <= 0:
        sigma = 1e-10

    # Calculate d1, d2 once
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Common terms
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)

    if option_type == 'call':
        price = S * exp_qT * Nd1 - K * exp_rT * Nd2
        delta = exp_qT * Nd1
        theta = (-S * exp_qT * nd1 * sigma / (2 * sqrt_T)
                 + q * S * exp_qT * Nd1
                 - r * K * exp_rT * Nd2)
        rho = K * T * exp_rT * Nd2 / 100
    else:
        price = K * exp_rT * norm.cdf(-d2) - S * exp_qT * norm.cdf(-d1)
        delta = exp_qT * (Nd1 - 1.0)
        theta = (-S * exp_qT * nd1 * sigma / (2 * sqrt_T)
                 - q * S * exp_qT * norm.cdf(-d1)
                 + r * K * exp_rT * norm.cdf(-d2))
        rho = -K * T * exp_rT * norm.cdf(-d2) / 100

    gamma = exp_qT * nd1 / (S * sigma * sqrt_T)
    vega = S * exp_qT * nd1 * sqrt_T / 100

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


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


# Vectorized operations for batch processing
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
    Fully vectorized Black-Scholes pricing for numpy arrays.

    Args:
        S: Underlying prices (array)
        K: Strike prices (array)
        T: Time to expiration in years (array)
        r: Risk-free rate (scalar or array)
        sigma: Implied volatility (array)
        is_call: Boolean array (True for calls, False for puts)
        q: Dividend yield (scalar or array)

    Returns:
        Array of option prices
    """
    # Convert to numpy arrays
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    if isinstance(r, (int, float)):
        r = np.full_like(S, r)
    if isinstance(q, (int, float)):
        q = np.full_like(S, q)

    # Handle edge cases
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)

    # Calculate d1, d2
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Calculate prices
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)

    call_price = S * exp_qT * norm.cdf(d1) - K * exp_rT * norm.cdf(d2)
    put_price = K * exp_rT * norm.cdf(-d2) - S * exp_qT * norm.cdf(-d1)

    return np.where(is_call, call_price, put_price)


def vectorized_bs_delta(
    S: Union[np.ndarray, pd.Series],
    K: Union[np.ndarray, pd.Series],
    T: Union[np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[np.ndarray, pd.Series],
    is_call: Union[np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0
) -> np.ndarray:
    """Vectorized delta calculation."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-10)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-10)
    is_call = np.asarray(is_call, dtype=bool)

    if isinstance(r, (int, float)):
        r = np.full_like(S, r)
    if isinstance(q, (int, float)):
        q = np.full_like(S, q)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    exp_qT = np.exp(-q * T)

    call_delta = exp_qT * norm.cdf(d1)
    put_delta = exp_qT * (norm.cdf(d1) - 1.0)

    return np.where(is_call, call_delta, put_delta)


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
    Vectorized calculation of all Greeks.

    Returns:
        DataFrame with columns: price, delta, gamma, theta, vega, rho
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-10)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-10)
    is_call = np.asarray(is_call, dtype=bool)

    if isinstance(r, (int, float)):
        r = np.full_like(S, r)
    if isinstance(q, (int, float)):
        q = np.full_like(S, q)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)

    # Price
    call_price = S * exp_qT * Nd1 - K * exp_rT * Nd2
    put_price = K * exp_rT * norm.cdf(-d2) - S * exp_qT * norm.cdf(-d1)
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
    put_theta = common_theta - q * S * exp_qT * norm.cdf(-d1) + r * K * exp_rT * norm.cdf(-d2)
    theta = np.where(is_call, call_theta, put_theta)

    # Vega (same for calls and puts)
    vega = S * exp_qT * nd1 * sqrt_T / 100

    # Rho
    call_rho = K * T * exp_rT * Nd2 / 100
    put_rho = -K * T * exp_rT * norm.cdf(-d2) / 100
    rho = np.where(is_call, call_rho, put_rho)

    return pd.DataFrame({
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    })
