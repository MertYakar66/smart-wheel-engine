"""
Black-Scholes Option Pricing
Handles both European and American-style options (approximation).
"""

import numpy as np
from scipy.stats import norm
from typing import Literal


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
    """Calculate option delta"""
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


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


# For vectorized operations (pandas DataFrame)
def vectorized_bs_price(df, S_col='underlying_price', K_col='strike', 
                        T_col='dte', r_col='risk_free_rate', 
                        sigma_col='implied_volatility', type_col='option_type'):
    """
    Apply Black-Scholes pricing to entire DataFrame.
    
    Args:
        df: DataFrame with required columns
        *_col: Column names for each input
        
    Returns:
        Series of option prices
    """
    prices = []
    
    for _, row in df.iterrows():
        S = row[S_col]
        K = row[K_col]
        T = row[T_col] / 365.0
        r = row[r_col]
        sigma = row[sigma_col]
        opt_type = 'call' if row[type_col].upper().startswith('C') else 'put'
        
        price = black_scholes_price(S, K, T, r, sigma, opt_type)
        prices.append(price)
    
    return prices