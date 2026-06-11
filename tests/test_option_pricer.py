"""
Tests for Black-Scholes option pricing and Greeks.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.option_pricer import (
    black_scholes_all_greeks,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_theta,
    black_scholes_vega,
    estimate_option_price_from_iv,
    vectorized_bs_all_greeks,
    vectorized_bs_delta,
    vectorized_bs_price,
)


class TestBlackScholesPrice:
    """Tests for Black-Scholes pricing."""

    def test_atm_call_price(self):
        """ATM call should have positive value."""
        price = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert price > 0
        assert price < 100  # Should be less than underlying

    def test_atm_put_price(self):
        """ATM put should have positive value."""
        price = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put")
        assert price > 0
        assert price < 100

    def test_put_call_parity(self):
        """Put-call parity should hold."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        call = black_scholes_price(S, K, T, r, sigma, "call")
        put = black_scholes_price(S, K, T, r, sigma, "put")
        # C - P = S - K*e^(-rT)
        expected_diff = S - K * np.exp(-r * T)
        assert abs(call - put - expected_diff) < 0.001

    def test_expired_call_otm(self):
        """Expired OTM call should be worthless."""
        price = black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type="call")
        assert price == 0

    def test_expired_call_itm(self):
        """Expired ITM call should equal intrinsic value."""
        price = black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type="call")
        assert price == 10

    def test_expired_put_otm(self):
        """Expired OTM put should be worthless."""
        price = black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type="put")
        assert price == 0

    def test_expired_put_itm(self):
        """Expired ITM put should equal intrinsic value."""
        price = black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type="put")
        assert price == 10

    def test_zero_volatility_call(self):
        """Zero vol call should equal PV of intrinsic."""
        S, K, T, r = 110, 100, 1.0, 0.05
        price = black_scholes_price(S, K, T, r, sigma=0, option_type="call")
        expected = max(0, S - K * np.exp(-r * T))
        assert abs(price - expected) < 0.001

    def test_zero_volatility_put(self):
        """Zero vol put should equal PV of intrinsic."""
        S, K, T, r = 90, 100, 1.0, 0.05
        price = black_scholes_price(S, K, T, r, sigma=0, option_type="put")
        expected = max(0, K * np.exp(-r * T) - S)
        assert abs(price - expected) < 0.001

    def test_dividend_yield_reduces_call_price(self):
        """Dividend yield should reduce call price."""
        no_div = black_scholes_price(100, 100, 0.5, 0.05, 0.20, "call", q=0)
        with_div = black_scholes_price(100, 100, 0.5, 0.05, 0.20, "call", q=0.03)
        assert with_div < no_div

    def test_dividend_yield_increases_put_price(self):
        """Dividend yield should increase put price."""
        no_div = black_scholes_price(100, 100, 0.5, 0.05, 0.20, "put", q=0)
        with_div = black_scholes_price(100, 100, 0.5, 0.05, 0.20, "put", q=0.03)
        assert with_div > no_div


class TestBlackScholesGreeks:
    """Tests for Greeks calculations."""

    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        delta = black_scholes_delta(100, 100, 0.25, 0.05, 0.20, "call")
        assert 0 <= delta <= 1

    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        delta = black_scholes_delta(100, 100, 0.25, 0.05, 0.20, "put")
        assert -1 <= delta <= 0

    def test_atm_call_delta_near_half(self):
        """ATM call delta should be near 0.5."""
        delta = black_scholes_delta(100, 100, 0.25, 0.05, 0.20, "call")
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
        theta = black_scholes_theta(100, 100, 0.25, 0.05, 0.20, "call")
        # Note: theta can be positive for deep ITM calls with high rates
        # For typical ATM options, theta is negative
        assert theta < 0

    def test_all_greeks_consistency(self):
        """All Greeks function should match individual functions."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

        greeks = black_scholes_all_greeks(S, K, T, r, sigma, "call")

        price = black_scholes_price(S, K, T, r, sigma, "call")
        delta = black_scholes_delta(S, K, T, r, sigma, "call")
        gamma = black_scholes_gamma(S, K, T, r, sigma)
        theta = black_scholes_theta(S, K, T, r, sigma, "call")
        vega = black_scholes_vega(S, K, T, r, sigma)

        assert abs(greeks["price"] - price) < 0.0001
        assert abs(greeks["delta"] - delta) < 0.0001
        assert abs(greeks["gamma"] - gamma) < 0.0001
        assert abs(greeks["theta"] - theta) < 0.0001
        assert abs(greeks["vega"] - vega) < 0.0001


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
            scalar_price = black_scholes_price(S[i], K[i], T[i], 0.05, sigma[i], "call")
            assert abs(vec_prices[i] - scalar_price) < 0.0001

    def test_vectorized_mixed_types(self):
        """Vectorized pricing with mixed calls and puts."""
        S = np.array([100, 100])
        K = np.array([100, 100])
        T = np.array([0.25, 0.25])
        sigma = np.array([0.20, 0.20])
        is_call = np.array([True, False])

        vec_prices = vectorized_bs_price(S, K, T, 0.05, sigma, is_call)

        call_price = black_scholes_price(100, 100, 0.25, 0.05, 0.20, "call")
        put_price = black_scholes_price(100, 100, 0.25, 0.05, 0.20, "put")

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
        assert set(greeks_df.columns) == {"price", "delta", "gamma", "theta", "vega", "rho"}


class TestVectorizedFailLoud:
    """R8: vectorized pricers must fail loud on S<=0 / K<=0 (no silent NaN).

    A single bad strike or spot from a data gap would otherwise flow through as
    log(<=0) -> NaN into batch Greek/exposure computation. The guard mirrors the
    scalar _validate_inputs positivity contract: raise ValueError if ANY element
    is non-positive. These tests also prove the guard changes NOTHING for valid
    inputs by equality-checking against the scalar pricer.
    """

    # Valid base arrays (no element violates positivity).
    S = np.array([100.0, 110.0, 90.0])
    K = np.array([100.0, 105.0, 95.0])
    T = np.array([0.25, 0.50, 0.10])
    sigma = np.array([0.20, 0.25, 0.18])
    is_call = np.array([True, False, True])
    r = 0.05

    def test_price_raises_on_negative_strike(self):
        K_bad = np.array([100.0, -5.0, 95.0])
        with pytest.raises(ValueError, match="Strike price K must be positive"):
            vectorized_bs_price(self.S, K_bad, self.T, self.r, self.sigma, self.is_call)

    def test_price_raises_on_zero_spot(self):
        S_bad = np.array([100.0, 0.0, 90.0])
        with pytest.raises(ValueError, match="Spot price S must be positive"):
            vectorized_bs_price(S_bad, self.K, self.T, self.r, self.sigma, self.is_call)

    def test_delta_raises_on_negative_strike(self):
        K_bad = np.array([100.0, 105.0, -1.0])
        with pytest.raises(ValueError, match="Strike price K must be positive"):
            vectorized_bs_delta(self.S, K_bad, self.T, self.r, self.sigma, self.is_call)

    def test_delta_raises_on_zero_spot(self):
        S_bad = np.array([0.0, 110.0, 90.0])
        with pytest.raises(ValueError, match="Spot price S must be positive"):
            vectorized_bs_delta(S_bad, self.K, self.T, self.r, self.sigma, self.is_call)

    def test_all_greeks_raises_on_negative_strike(self):
        K_bad = np.array([-100.0, 105.0, 95.0])
        with pytest.raises(ValueError, match="Strike price K must be positive"):
            vectorized_bs_all_greeks(self.S, K_bad, self.T, self.r, self.sigma, self.is_call)

    def test_all_greeks_raises_on_zero_spot(self):
        S_bad = np.array([100.0, 110.0, 0.0])
        with pytest.raises(ValueError, match="Spot price S must be positive"):
            vectorized_bs_all_greeks(S_bad, self.K, self.T, self.r, self.sigma, self.is_call)

    def test_valid_arrays_unchanged_price(self):
        """Guard must not alter valid-input behavior: match scalar pricer exactly."""
        prices = vectorized_bs_price(self.S, self.K, self.T, self.r, self.sigma, self.is_call)
        assert np.all(np.isfinite(prices))
        for i in range(len(self.S)):
            opt = "call" if self.is_call[i] else "put"
            scalar = black_scholes_price(
                self.S[i], self.K[i], self.T[i], self.r, self.sigma[i], opt
            )
            assert abs(prices[i] - scalar) < 1e-9

    def test_valid_arrays_unchanged_delta(self):
        """Guard must not alter valid-input behavior: match scalar delta exactly."""
        deltas = vectorized_bs_delta(self.S, self.K, self.T, self.r, self.sigma, self.is_call)
        assert np.all(np.isfinite(deltas))
        for i in range(len(self.S)):
            opt = "call" if self.is_call[i] else "put"
            scalar = black_scholes_delta(
                self.S[i], self.K[i], self.T[i], self.r, self.sigma[i], opt
            )
            assert abs(deltas[i] - scalar) < 1e-9

    def test_valid_arrays_unchanged_all_greeks(self):
        """Guard must not alter valid-input behavior: match scalar all-greeks exactly."""
        df = vectorized_bs_all_greeks(self.S, self.K, self.T, self.r, self.sigma, self.is_call)
        assert df.to_numpy().shape[0] == len(self.S)
        assert np.all(np.isfinite(df.to_numpy()))
        for i in range(len(self.S)):
            opt = "call" if self.is_call[i] else "put"
            scalar = black_scholes_all_greeks(
                self.S[i], self.K[i], self.T[i], self.r, self.sigma[i], opt
            )
            for col in ["price", "delta", "gamma", "theta", "vega", "rho"]:
                assert abs(df.iloc[i][col] - scalar[col]) < 1e-9, f"mismatch in {col} at row {i}"


class TestEstimateOptionPrice:
    """Tests for convenience wrapper function."""

    def test_dte_conversion(self):
        """DTE should convert to years correctly."""
        # 365 DTE = 1 year
        price_1y = estimate_option_price_from_iv(
            underlying_price=100,
            strike=100,
            dte=365,
            iv=0.20,
            risk_free_rate=0.05,
            option_type="call",
        )

        price_direct = black_scholes_price(100, 100, 1.0, 0.05, 0.20, "call")
        assert abs(price_1y - price_direct) < 0.0001

    def test_zero_dte(self):
        """Zero DTE should return intrinsic value."""
        call_itm = estimate_option_price_from_iv(110, 100, 0, 0.20, 0.05, "call")
        assert call_itm == 10

        put_otm = estimate_option_price_from_iv(110, 100, 0, 0.20, 0.05, "put")
        assert put_otm == 0


class TestNonFiniteSigmaTContract:
    """Pins the DELIBERATE non-finite pass-through (heavy-verify 2026-06-09
    Site C): non-finite sigma/T pass _validate_inputs by design and price to
    an honest NaN without raising; the NaN EV is hard-blocked downstream by
    R1a (verdict_reason="ev_non_finite", PR #204). A raise here would abort
    whole rank runs (no per-candidate try/except at the decision-layer call
    sites). Pins behavior, not signature."""

    # Full mixed matrix {nan, +inf, finite} x {nan, +inf, finite} minus the
    # finite-finite cell. The production-relevant cell is non-finite sigma x
    # FINITE T (T comes from an int dte; the realistic corruption is the IV).
    NONFINITE_CASES = [
        (float("nan"), 30 / 365),
        (float("inf"), 30 / 365),
        (0.25, float("nan")),
        (0.25, float("inf")),
        (float("nan"), float("nan")),
        (float("nan"), float("inf")),
        (float("inf"), float("nan")),
        (float("inf"), float("inf")),
    ]

    @pytest.mark.parametrize("sigma,T", NONFINITE_CASES)
    def test_scalar_entry_points_nan_no_raise(self, sigma, T):
        price = black_scholes_price(S=100.0, K=95.0, T=T, r=0.04, sigma=sigma, option_type="put")
        delta = black_scholes_delta(S=100.0, K=95.0, T=T, r=0.04, sigma=sigma, option_type="put")
        vega = black_scholes_vega(S=100.0, K=95.0, T=T, r=0.04, sigma=sigma)
        assert np.isnan(price) and np.isnan(delta) and np.isnan(vega)

    def test_all_greeks_all_nan_no_raise(self):
        greeks = black_scholes_all_greeks(
            S=100.0, K=95.0, T=float("nan"), r=0.04, sigma=0.25, option_type="put"
        )
        assert len(greeks) == 12
        assert all(np.isnan(v) for v in greeks.values()), greeks

    @pytest.mark.parametrize("sigma", [-0.5, float("-inf")])
    def test_negative_sigma_still_raises(self, sigma):
        # The finite-impossible fail-loud contract is untouched; -inf is caught
        # by the existing sigma < 0 check.
        with pytest.raises(ValueError, match="non-negative"):
            black_scholes_price(S=100.0, K=95.0, T=0.1, r=0.04, sigma=sigma, option_type="put")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
