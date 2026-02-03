"""
Tests for Black-Scholes option pricing and Greeks.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.option_pricer import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_theta,
    black_scholes_vega,
    black_scholes_all_greeks,
    estimate_option_price_from_iv,
    vectorized_bs_price,
    vectorized_bs_all_greeks
)


class TestBlackScholesPrice:
    """Tests for Black-Scholes pricing."""

    def test_atm_call_price(self):
        """ATM call should have positive value."""
        price = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
        assert price > 0
        assert price < 100  # Should be less than underlying

    def test_atm_put_price(self):
        """ATM put should have positive value."""
        price = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
        assert price > 0
        assert price < 100

    def test_put_call_parity(self):
        """Put-call parity should hold."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        call = black_scholes_price(S, K, T, r, sigma, 'call')
        put = black_scholes_price(S, K, T, r, sigma, 'put')
        # C - P = S - K*e^(-rT)
        expected_diff = S - K * np.exp(-r * T)
        assert abs(call - put - expected_diff) < 0.001

    def test_expired_call_otm(self):
        """Expired OTM call should be worthless."""
        price = black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type='call')
        assert price == 0

    def test_expired_call_itm(self):
        """Expired ITM call should equal intrinsic value."""
        price = black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type='call')
        assert price == 10

    def test_expired_put_otm(self):
        """Expired OTM put should be worthless."""
        price = black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type='put')
        assert price == 0

    def test_expired_put_itm(self):
        """Expired ITM put should equal intrinsic value."""
        price = black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type='put')
        assert price == 10

    def test_zero_volatility_call(self):
        """Zero vol call should equal PV of intrinsic."""
        S, K, T, r = 110, 100, 1.0, 0.05
        price = black_scholes_price(S, K, T, r, sigma=0, option_type='call')
        expected = max(0, S - K * np.exp(-r * T))
        assert abs(price - expected) < 0.001

    def test_zero_volatility_put(self):
        """Zero vol put should equal PV of intrinsic."""
        S, K, T, r = 90, 100, 1.0, 0.05
        price = black_scholes_price(S, K, T, r, sigma=0, option_type='put')
        expected = max(0, K * np.exp(-r * T) - S)
        assert abs(price - expected) < 0.001

    def test_dividend_yield_reduces_call_price(self):
        """Dividend yield should reduce call price."""
        no_div = black_scholes_price(100, 100, 0.5, 0.05, 0.20, 'call', q=0)
        with_div = black_scholes_price(100, 100, 0.5, 0.05, 0.20, 'call', q=0.03)
        assert with_div < no_div

    def test_dividend_yield_increases_put_price(self):
        """Dividend yield should increase put price."""
        no_div = black_scholes_price(100, 100, 0.5, 0.05, 0.20, 'put', q=0)
        with_div = black_scholes_price(100, 100, 0.5, 0.05, 0.20, 'put', q=0.03)
        assert with_div > no_div


class TestBlackScholesGreeks:
    """Tests for Greeks calculations."""

    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        delta = black_scholes_delta(100, 100, 0.25, 0.05, 0.20, 'call')
        assert 0 <= delta <= 1

    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        delta = black_scholes_delta(100, 100, 0.25, 0.05, 0.20, 'put')
        assert -1 <= delta <= 0

    def test_atm_call_delta_near_half(self):
        """ATM call delta should be near 0.5."""
        delta = black_scholes_delta(100, 100, 0.25, 0.05, 0.20, 'call')
        assert 0.45 <= delta <= 0.65

    def test_gamma_always_positive(self):
        """Gamma should always be positive."""
        gamma = black_scholes_gamma(100, 100, 0.25, 0.05, 0.20)
        assert gamma > 0

    def test_gamma_highest_atm(self):
        """Gamma should be highest ATM."""
        gamma_atm = black_scholes_gamma(100, 100, 0.25, 0.05, 0.20)
        gamma_itm = black_scholes_gamma(110, 100, 0.25, 0.05, 0.20)
        gamma_otm = black_scholes_gamma(90, 100, 0.25, 0.05, 0.20)
        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm

    def test_vega_always_positive(self):
        """Vega should always be positive."""
        vega = black_scholes_vega(100, 100, 0.25, 0.05, 0.20)
        assert vega > 0

    def test_theta_call_negative(self):
        """Long call theta should typically be negative."""
        theta = black_scholes_theta(100, 100, 0.25, 0.05, 0.20, 'call')
        # Note: theta can be positive for deep ITM calls with high rates
        # For typical ATM options, theta is negative
        assert theta < 0

    def test_all_greeks_consistency(self):
        """All Greeks function should match individual functions."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

        greeks = black_scholes_all_greeks(S, K, T, r, sigma, 'call')

        price = black_scholes_price(S, K, T, r, sigma, 'call')
        delta = black_scholes_delta(S, K, T, r, sigma, 'call')
        gamma = black_scholes_gamma(S, K, T, r, sigma)
        theta = black_scholes_theta(S, K, T, r, sigma, 'call')
        vega = black_scholes_vega(S, K, T, r, sigma)

        assert abs(greeks['price'] - price) < 0.0001
        assert abs(greeks['delta'] - delta) < 0.0001
        assert abs(greeks['gamma'] - gamma) < 0.0001
        assert abs(greeks['theta'] - theta) < 0.0001
        assert abs(greeks['vega'] - vega) < 0.0001


class TestVectorizedPricing:
    """Tests for vectorized operations."""

    def test_vectorized_price_matches_scalar(self):
        """Vectorized pricing should match scalar pricing."""
        S = np.array([100, 110, 90])
        K = np.array([100, 100, 100])
        T = np.array([0.25, 0.25, 0.25])
        sigma = np.array([0.20, 0.20, 0.20])
        is_call = np.array([True, True, True])

        vec_prices = vectorized_bs_price(S, K, T, 0.05, sigma, is_call)

        for i in range(len(S)):
            scalar_price = black_scholes_price(S[i], K[i], T[i], 0.05, sigma[i], 'call')
            assert abs(vec_prices[i] - scalar_price) < 0.0001

    def test_vectorized_mixed_types(self):
        """Vectorized pricing with mixed calls and puts."""
        S = np.array([100, 100])
        K = np.array([100, 100])
        T = np.array([0.25, 0.25])
        sigma = np.array([0.20, 0.20])
        is_call = np.array([True, False])

        vec_prices = vectorized_bs_price(S, K, T, 0.05, sigma, is_call)

        call_price = black_scholes_price(100, 100, 0.25, 0.05, 0.20, 'call')
        put_price = black_scholes_price(100, 100, 0.25, 0.05, 0.20, 'put')

        assert abs(vec_prices[0] - call_price) < 0.0001
        assert abs(vec_prices[1] - put_price) < 0.0001

    def test_vectorized_greeks_dataframe(self):
        """Vectorized Greeks should return proper DataFrame."""
        S = np.array([100, 110, 90])
        K = np.array([100, 100, 100])
        T = np.array([0.25, 0.25, 0.25])
        sigma = np.array([0.20, 0.20, 0.20])
        is_call = np.array([True, True, True])

        greeks_df = vectorized_bs_all_greeks(S, K, T, 0.05, sigma, is_call)

        assert isinstance(greeks_df, pd.DataFrame)
        assert len(greeks_df) == 3
        assert set(greeks_df.columns) == {'price', 'delta', 'gamma', 'theta', 'vega', 'rho'}


class TestEstimateOptionPrice:
    """Tests for convenience wrapper function."""

    def test_dte_conversion(self):
        """DTE should convert to years correctly."""
        # 365 DTE = 1 year
        price_1y = estimate_option_price_from_iv(
            underlying_price=100, strike=100, dte=365,
            iv=0.20, risk_free_rate=0.05, option_type='call'
        )

        price_direct = black_scholes_price(100, 100, 1.0, 0.05, 0.20, 'call')
        assert abs(price_1y - price_direct) < 0.0001

    def test_zero_dte(self):
        """Zero DTE should return intrinsic value."""
        call_itm = estimate_option_price_from_iv(110, 100, 0, 0.20, 0.05, 'call')
        assert call_itm == 10

        put_otm = estimate_option_price_from_iv(110, 100, 0, 0.20, 0.05, 'put')
        assert put_otm == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
