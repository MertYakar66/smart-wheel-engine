"""
Tests for option pricer, risk manager, and stress tester under extreme market conditions.

Validates numerical stability for edge cases: near-expiry, near-zero vol,
deep ITM/OTM, negative carry, and extreme stress scenarios.
"""

import math

import numpy as np
import pytest

from engine.option_pricer import black_scholes_all_greeks, black_scholes_price
from engine.stress_testing import Scenario, ScenarioType, StressTester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_greeks_finite(greeks: dict) -> None:
    """Assert every value in the greeks dict is finite (not NaN, not inf)."""
    for key, val in greeks.items():
        if isinstance(val, (int, float)):
            assert np.isfinite(val), f"Greek '{key}' is not finite: {val}"


# ---------------------------------------------------------------------------
# 1. Near-expiry numerics
# ---------------------------------------------------------------------------

class TestNearExpiryNumerics:
    """Edge cases around T -> 0."""

    def test_very_small_T(self):
        """T=0.001 (a few hours) should not error; prices should be >= 0."""
        for opt in ("call", "put"):
            price = black_scholes_price(S=100, K=100, T=0.001, r=0.05, sigma=0.20, option_type=opt)
            assert np.isfinite(price)
            assert price >= 0

    def test_T_zero(self):
        """T=0 should return intrinsic value."""
        call_price = black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type="call")
        put_price = black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type="put")

        assert call_price == pytest.approx(10.0, abs=1e-9)
        assert put_price == pytest.approx(10.0, abs=1e-9)

        # OTM at expiry => 0
        assert black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type="call") == 0.0
        assert black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type="put") == 0.0

    def test_T_negative(self):
        """T=-0.1 should handle gracefully (return intrinsic or 0)."""
        for opt in ("call", "put"):
            price = black_scholes_price(S=100, K=100, T=-0.1, r=0.05, sigma=0.20, option_type=opt)
            assert np.isfinite(price)
            assert price >= 0


# ---------------------------------------------------------------------------
# 2. Near-zero volatility
# ---------------------------------------------------------------------------

class TestNearZeroVolNumerics:
    """Edge cases around sigma -> 0."""

    def test_sigma_zero(self):
        """sigma=0 should return deterministic (PV of intrinsic) value."""
        call_price = black_scholes_price(S=110, K=100, T=0.5, r=0.05, sigma=0, option_type="call")
        put_price = black_scholes_price(S=90, K=100, T=0.5, r=0.05, sigma=0, option_type="put")

        assert np.isfinite(call_price)
        assert np.isfinite(put_price)
        assert call_price > 0
        assert put_price > 0

    def test_very_small_sigma(self):
        """sigma=0.001 should not error; greeks should be finite."""
        greeks = black_scholes_all_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.001, option_type="call"
        )
        _all_greeks_finite(greeks)
        assert greeks["price"] >= 0

    def test_sigma_zero_all_greeks(self):
        """All greeks should be finite when sigma=0."""
        for opt in ("call", "put"):
            greeks = black_scholes_all_greeks(
                S=100, K=100, T=0.25, r=0.05, sigma=0, option_type=opt
            )
            _all_greeks_finite(greeks)


# ---------------------------------------------------------------------------
# 3. Deep ITM / OTM
# ---------------------------------------------------------------------------

class TestDeepITMOTMNumerics:
    """Extreme moneyness edge cases."""

    def test_deep_itm_put(self):
        """S=50, K=150: deep ITM put should have delta near -1."""
        greeks = black_scholes_all_greeks(
            S=50, K=150, T=0.25, r=0.05, sigma=0.20, option_type="put"
        )
        _all_greeks_finite(greeks)
        assert greeks["delta"] == pytest.approx(-1.0, abs=0.01)

    def test_deep_otm_put(self):
        """S=200, K=50: deep OTM put should have delta near 0."""
        greeks = black_scholes_all_greeks(
            S=200, K=50, T=0.25, r=0.05, sigma=0.20, option_type="put"
        )
        _all_greeks_finite(greeks)
        assert greeks["delta"] == pytest.approx(0.0, abs=0.01)

    def test_deep_itm_call(self):
        """S=200, K=50: deep ITM call should have delta near 1."""
        greeks = black_scholes_all_greeks(
            S=200, K=50, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        _all_greeks_finite(greeks)
        assert greeks["delta"] == pytest.approx(1.0, abs=0.01)

    def test_deep_otm_call(self):
        """S=50, K=200: deep OTM call should have delta near 0."""
        greeks = black_scholes_all_greeks(
            S=50, K=200, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        _all_greeks_finite(greeks)
        assert greeks["delta"] == pytest.approx(0.0, abs=0.01)

    def test_extreme_moneyness_greeks_finite(self):
        """All greeks should be finite for extreme moneyness, no NaN/inf."""
        extreme_cases = [
            (10, 500, "call"),
            (500, 10, "call"),
            (10, 500, "put"),
            (500, 10, "put"),
        ]
        for S, K, opt in extreme_cases:
            greeks = black_scholes_all_greeks(
                S=S, K=K, T=0.25, r=0.05, sigma=0.30, option_type=opt
            )
            _all_greeks_finite(greeks)


# ---------------------------------------------------------------------------
# 4. Negative carry edge cases
# ---------------------------------------------------------------------------

class TestNegativeCarryEdgeCases:
    """High dividend yield and negative interest rates."""

    def test_high_dividend_yield(self):
        """q=0.10, r=0.02 (negative carry) should produce valid prices/greeks."""
        for opt in ("call", "put"):
            price = black_scholes_price(
                S=100, K=100, T=0.5, r=0.02, sigma=0.25, option_type=opt, q=0.10
            )
            assert np.isfinite(price)
            assert price >= 0

            greeks = black_scholes_all_greeks(
                S=100, K=100, T=0.5, r=0.02, sigma=0.25, option_type=opt, q=0.10
            )
            _all_greeks_finite(greeks)

    def test_negative_rate(self):
        """r=-0.01 should not error and produce valid results."""
        for opt in ("call", "put"):
            price = black_scholes_price(
                S=100, K=100, T=0.5, r=-0.01, sigma=0.25, option_type=opt
            )
            assert np.isfinite(price)
            assert price >= 0

            greeks = black_scholes_all_greeks(
                S=100, K=100, T=0.5, r=-0.01, sigma=0.25, option_type=opt
            )
            _all_greeks_finite(greeks)


# ---------------------------------------------------------------------------
# 5. Stress tester extremes
# ---------------------------------------------------------------------------

class TestStressTesterExtremes:
    """Stress tester under extreme position parameters."""

    def _make_position(self, dte: int, iv: float) -> list[dict]:
        return [
            {
                "symbol": "TEST",
                "option_type": "put",
                "strike": 100,
                "dte": dte,
                "iv": iv,
                "contracts": 1,
                "is_short": True,
            }
        ]

    def _crash_scenario(self) -> Scenario:
        return Scenario(
            name="test_crash",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="Test crash scenario",
            spot_change_pct=-0.20,
            iv_change_abs=0.15,
            rate_change_bps=0,
            time_decay_days=0,
        )

    def test_stress_with_near_expiry(self):
        """Positions with dte=1 should not cause numerical errors."""
        tester = StressTester(risk_free_rate=0.05)
        positions = self._make_position(dte=1, iv=0.30)
        spot_prices = {"TEST": 100.0}

        result = tester.run_scenario(
            scenario=self._crash_scenario(),
            positions=positions,
            spot_prices=spot_prices,
            portfolio_value=10_000.0,
        )

        assert np.isfinite(result.portfolio_pnl)
        assert np.isfinite(result.portfolio_pnl_pct)
        assert np.isfinite(result.new_portfolio_value)

    def test_stress_with_very_high_iv(self):
        """iv=2.0 (200%) should not cause numerical errors."""
        tester = StressTester(risk_free_rate=0.05)
        positions = self._make_position(dte=30, iv=2.0)
        spot_prices = {"TEST": 100.0}

        result = tester.run_scenario(
            scenario=self._crash_scenario(),
            positions=positions,
            spot_prices=spot_prices,
            portfolio_value=10_000.0,
        )

        assert np.isfinite(result.portfolio_pnl)
        assert np.isfinite(result.portfolio_pnl_pct)
        assert np.isfinite(result.new_portfolio_value)
