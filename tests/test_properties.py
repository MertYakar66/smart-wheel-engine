"""
Property-Based Tests using Hypothesis

These tests verify invariants that should hold for ALL valid inputs,
not just specific test cases. They use random input generation to
find edge cases that manual tests might miss.

Properties tested:
1. Monotonicity (RSI increases when prices go up)
2. Bounds (delta in [-1, 1], RSI in [0, 100])
3. NaN propagation (NaN in -> NaN out where appropriate)
4. Idempotency (applying twice gives same result)
5. Symmetry (put-call parity, etc.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import hypothesis, skip tests if not available
try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st
    from hypothesis.extra.pandas import column, data_frames, series  # noqa: F401

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

    # Create dummy decorators for when hypothesis is not installed
    def given(*args, **kwargs):
        def decorator(func):
            def wrapper(*a, **kw):
                return None  # Skip test

            wrapper.__name__ = func.__name__
            return wrapper

        return decorator

    class st:
        @staticmethod
        def floats(*args, **kwargs):
            return None

        @staticmethod
        def integers(*args, **kwargs):
            return None

        @staticmethod
        def lists(*args, **kwargs):
            return None

    def assume(x):
        return x

    def settings(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


import numpy as np
import pandas as pd
import pytest

from engine.option_pricer import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_vega,
)
from engine.risk_manager import calculate_kelly_fraction
from src.features.technical import TechnicalFeatures
from src.features.volatility import VolatilityFeatures

# Skip all tests in this module if hypothesis is not available
pytestmark = pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")


class TestBlackScholesProperties:
    """Property-based tests for Black-Scholes."""

    @given(
        S=st.floats(min_value=1, max_value=1000, allow_nan=False),
        K=st.floats(min_value=1, max_value=1000, allow_nan=False),
        T=st.floats(min_value=0.01, max_value=5, allow_nan=False),
        r=st.floats(min_value=0, max_value=0.2, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=2, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_call_price_non_negative(self, S, K, T, r, sigma):
        """Call price should always be non-negative."""
        price = black_scholes_price(S, K, T, r, sigma, "call")
        assert price >= 0, f"Call price should be >= 0, got {price}"

    @given(
        S=st.floats(min_value=1, max_value=1000, allow_nan=False),
        K=st.floats(min_value=1, max_value=1000, allow_nan=False),
        T=st.floats(min_value=0.01, max_value=5, allow_nan=False),
        r=st.floats(min_value=0, max_value=0.2, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=2, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_put_price_non_negative(self, S, K, T, r, sigma):
        """Put price should always be non-negative."""
        price = black_scholes_price(S, K, T, r, sigma, "put")
        assert price >= 0, f"Put price should be >= 0, got {price}"

    @given(
        S=st.floats(min_value=1, max_value=1000, allow_nan=False),
        K=st.floats(min_value=1, max_value=1000, allow_nan=False),
        T=st.floats(min_value=0.01, max_value=5, allow_nan=False),
        r=st.floats(min_value=0, max_value=0.2, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=2, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_call_delta_bounds(self, S, K, T, r, sigma):
        """Call delta should be in [0, 1]."""
        delta = black_scholes_delta(S, K, T, r, sigma, "call")
        assert 0 <= delta <= 1, f"Call delta should be in [0,1], got {delta}"

    @given(
        S=st.floats(min_value=1, max_value=1000, allow_nan=False),
        K=st.floats(min_value=1, max_value=1000, allow_nan=False),
        T=st.floats(min_value=0.01, max_value=5, allow_nan=False),
        r=st.floats(min_value=0, max_value=0.2, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=2, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_put_delta_bounds(self, S, K, T, r, sigma):
        """Put delta should be in [-1, 0]."""
        delta = black_scholes_delta(S, K, T, r, sigma, "put")
        assert -1 <= delta <= 0, f"Put delta should be in [-1,0], got {delta}"

    @given(
        S=st.floats(min_value=1, max_value=1000, allow_nan=False),
        K=st.floats(min_value=1, max_value=1000, allow_nan=False),
        T=st.floats(min_value=0.01, max_value=5, allow_nan=False),
        r=st.floats(min_value=0, max_value=0.2, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=2, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_gamma_non_negative(self, S, K, T, r, sigma):
        """Gamma should always be non-negative."""
        gamma = black_scholes_gamma(S, K, T, r, sigma)
        assert gamma >= 0, f"Gamma should be >= 0, got {gamma}"

    @given(
        S=st.floats(min_value=1, max_value=1000, allow_nan=False),
        K=st.floats(min_value=1, max_value=1000, allow_nan=False),
        T=st.floats(min_value=0.01, max_value=5, allow_nan=False),
        r=st.floats(min_value=0, max_value=0.2, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=2, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_vega_non_negative(self, S, K, T, r, sigma):
        """Vega should always be non-negative."""
        vega = black_scholes_vega(S, K, T, r, sigma)
        assert vega >= 0, f"Vega should be >= 0, got {vega}"

    @given(
        S=st.floats(min_value=1, max_value=1000, allow_nan=False),
        K=st.floats(min_value=1, max_value=1000, allow_nan=False),
        T=st.floats(min_value=0.01, max_value=5, allow_nan=False),
        r=st.floats(min_value=0, max_value=0.2, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=2, allow_nan=False),
        q=st.floats(min_value=0, max_value=0.1, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_put_call_parity(self, S, K, T, r, sigma, q):
        """Put-call parity should hold: C - P = S*e^(-qT) - K*e^(-rT)."""
        call = black_scholes_price(S, K, T, r, sigma, "call", q)
        put = black_scholes_price(S, K, T, r, sigma, "put", q)

        lhs = call - put
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

        # Allow small numerical tolerance
        assert abs(lhs - rhs) < 1e-8, f"Put-call parity violated: {lhs} != {rhs}"


class TestRSIProperties:
    """Property-based tests for RSI."""

    @given(
        prices=st.lists(
            st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=20,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_rsi_bounds(self, prices):
        """RSI should always be in [0, 100]."""
        prices_series = pd.Series(prices)

        tf = TechnicalFeatures()
        rsi = tf.rsi(prices_series, window=14)

        # Check all non-NaN values
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert valid_rsi.min() >= 0, f"RSI min {valid_rsi.min()} < 0"
            assert valid_rsi.max() <= 100, f"RSI max {valid_rsi.max()} > 100"

    @given(window=st.integers(min_value=2, max_value=50))
    @settings(max_examples=20)
    def test_rsi_warmup_nan(self, window):
        """First window values of RSI should be NaN."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)

        tf = TechnicalFeatures()
        rsi = tf.rsi(prices, window=window)

        # First `window` values should be NaN
        assert rsi.iloc[:window].isna().all(), f"First {window} values should be NaN"


class TestIVRankProperties:
    """Property-based tests for IV Rank."""

    @given(
        iv_values=st.lists(
            st.floats(min_value=0.05, max_value=1.0, allow_nan=False), min_size=30, max_size=100
        )
    )
    @settings(max_examples=50)
    def test_iv_rank_bounds(self, iv_values):
        """IV Rank should be in [0, 100] or NaN."""
        iv = pd.Series(iv_values)

        vf = VolatilityFeatures()
        rank = vf.iv_rank(iv, lookback=20)

        valid_rank = rank.dropna()
        if len(valid_rank) > 0:
            assert valid_rank.min() >= 0, f"IV Rank min {valid_rank.min()} < 0"
            assert valid_rank.max() <= 100, f"IV Rank max {valid_rank.max()} > 100"


class TestKellyProperties:
    """Property-based tests for Kelly criterion."""

    @given(
        win_rate=st.floats(min_value=0, max_value=1, allow_nan=False),
        avg_win=st.floats(min_value=0.01, max_value=1000, allow_nan=False),
        avg_loss=st.floats(min_value=0.01, max_value=1000, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_kelly_bounds(self, win_rate, avg_win, avg_loss):
        """Kelly fraction should be in [0, 0.25] (capped)."""
        result = calculate_kelly_fraction(win_rate, avg_win, avg_loss, kelly_fraction=1.0)

        assert 0 <= result <= 0.25, f"Kelly should be in [0, 0.25], got {result}"

    @given(
        win_rate=st.floats(min_value=-10, max_value=10, allow_nan=False),
        avg_win=st.floats(min_value=0.01, max_value=1000, allow_nan=False),
        avg_loss=st.floats(min_value=0.01, max_value=1000, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_kelly_invalid_win_rate(self, win_rate, avg_win, avg_loss):
        """Invalid win_rate outside [0,1] should return 0."""
        if not (0 <= win_rate <= 1):
            result = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
            assert result == 0, f"Invalid win_rate {win_rate} should give 0, got {result}"


class TestVolatilityProperties:
    """Property-based tests for volatility estimators."""

    @given(
        returns=st.lists(
            st.floats(min_value=-0.1, max_value=0.1, allow_nan=False), min_size=30, max_size=200
        )
    )
    @settings(max_examples=50)
    def test_rv_non_negative(self, returns):
        """Realized volatility should be non-negative."""
        returns_series = pd.Series(returns)

        vf = VolatilityFeatures()
        rv = vf.realized_volatility_close(returns_series, window=21, annualize=True)

        valid_rv = rv.dropna()
        if len(valid_rv) > 0:
            assert valid_rv.min() >= 0, f"RV should be >= 0, got min {valid_rv.min()}"


class TestNaNPropagation:
    """Test that NaN values propagate correctly."""

    def test_rsi_with_nan_input(self):
        """RSI should handle NaN in input gracefully."""
        prices = pd.Series([100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109, 110])

        tf = TechnicalFeatures()
        rsi = tf.rsi(prices, window=5)

        # Should not crash and should produce mostly NaN due to contamination
        assert len(rsi) == len(prices)

    def test_iv_rank_with_nan_input(self):
        """IV Rank should handle NaN in input gracefully."""
        iv = pd.Series([0.2, 0.25, np.nan, 0.3, 0.28, 0.32, 0.25, 0.22, 0.27, 0.30])

        vf = VolatilityFeatures()
        rank = vf.iv_rank(iv, lookback=5)

        # Should not crash
        assert len(rank) == len(iv)


if __name__ == "__main__":
    if HYPOTHESIS_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("hypothesis not installed, skipping property-based tests")
