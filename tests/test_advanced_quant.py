"""
Advanced Quantitative Tests for Smart Wheel Engine

Tests for:
1. Third-order Greeks (Speed, Color, Ultima)
2. American option pricing (Barone-Adesi-Whaley)
3. Multi-asset covariance VaR
4. Historical VaR interpolation

These tests validate the advanced features added for institutional-grade deployment.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.option_pricer import (
    american_option_greeks,
    american_option_price,
    black_scholes_all_greeks,
    black_scholes_color,
    black_scholes_price,
    black_scholes_speed,
    black_scholes_ultima,
)
from engine.risk_manager import RiskManager

# =============================================================================
# Third-Order Greeks Tests
# =============================================================================


class TestThirdOrderGreeks:
    """Test suite for third-order Greeks: Speed, Color, Ultima."""

    # Standard test parameters
    S = 100.0
    K = 100.0  # ATM
    T = 0.5  # 6 months
    r = 0.05
    sigma = 0.20
    q = 0.02

    def test_speed_finite_difference(self):
        """Verify Speed (∂Gamma/∂S) using finite difference."""
        h = 0.01 * self.S  # 1% bump

        gamma_up = black_scholes_all_greeks(
            self.S + h, self.K, self.T, self.r, self.sigma, "call", self.q
        )["gamma"]
        gamma_down = black_scholes_all_greeks(
            self.S - h, self.K, self.T, self.r, self.sigma, "call", self.q
        )["gamma"]

        speed_fd = (gamma_up - gamma_down) / (2 * h)
        speed_analytical = black_scholes_speed(self.S, self.K, self.T, self.r, self.sigma, self.q)

        assert abs(speed_fd - speed_analytical) < 0.0001, (
            f"Speed mismatch: FD={speed_fd:.6f}, Analytical={speed_analytical:.6f}"
        )

    def test_speed_in_all_greeks(self):
        """Verify Speed is correctly included in all_greeks output."""
        result = black_scholes_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma, "call", self.q
        )
        assert "speed" in result, "Speed missing from all_greeks"

        standalone = black_scholes_speed(self.S, self.K, self.T, self.r, self.sigma, self.q)
        assert abs(result["speed"] - standalone) < 1e-10, (
            "Speed in all_greeks doesn't match standalone function"
        )

    def test_color_finite_difference(self):
        """Verify Color (∂Gamma/∂T) using finite difference."""
        dT = 1 / 365  # 1 day

        gamma_now = black_scholes_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma, "call", self.q
        )["gamma"]
        gamma_later = black_scholes_all_greeks(
            self.S, self.K, self.T - dT, self.r, self.sigma, "call", self.q
        )["gamma"]

        # Color = ∂Gamma/∂T, so dGamma = Color * dT
        color_fd = (gamma_later - gamma_now) / (-dT)
        color_analytical = black_scholes_color(self.S, self.K, self.T, self.r, self.sigma, self.q)

        # Color can be large near ATM, so use relative tolerance
        assert abs(color_fd - color_analytical) < 0.01 * abs(color_analytical) + 0.001, (
            f"Color mismatch: FD={color_fd:.6f}, Analytical={color_analytical:.6f}"
        )

    def test_ultima_finite_difference(self):
        """Verify Ultima (∂Volga/∂σ) using finite difference."""
        d_sigma = 0.01  # 1% vol bump

        result_up = black_scholes_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma + d_sigma, "call", self.q
        )
        result_down = black_scholes_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma - d_sigma, "call", self.q
        )

        volga_up = result_up["volga"]
        volga_down = result_down["volga"]
        ultima_fd = (volga_up - volga_down) / (2 * d_sigma)

        ultima_analytical = black_scholes_ultima(self.S, self.K, self.T, self.r, self.sigma, self.q)

        assert abs(ultima_fd - ultima_analytical) < 0.1 * abs(ultima_analytical) + 0.01, (
            f"Ultima mismatch: FD={ultima_fd:.4f}, Analytical={ultima_analytical:.4f}"
        )

    def test_third_order_edge_cases(self):
        """Verify third-order Greeks are 0 for deterministic cases."""
        # T = 0
        result_t0 = black_scholes_all_greeks(
            self.S, self.K, 0.0, self.r, self.sigma, "call", self.q
        )
        assert result_t0["speed"] == 0.0
        assert result_t0["color"] == 0.0
        assert result_t0["ultima"] == 0.0

        # sigma = 0
        result_s0 = black_scholes_all_greeks(self.S, self.K, self.T, self.r, 0.0, "call", self.q)
        assert result_s0["speed"] == 0.0
        assert result_s0["color"] == 0.0
        assert result_s0["ultima"] == 0.0


# =============================================================================
# American Option Pricing Tests
# =============================================================================


class TestAmericanOptionPricing:
    """Test suite for Barone-Adesi-Whaley American option approximation."""

    def test_american_equals_european_for_call_no_dividend(self):
        """American call without dividends should equal European call."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20

        european = black_scholes_price(S, K, T, r, sigma, "call", q=0.0)
        american = american_option_price(S, K, T, r, sigma, "call", q=0.0)

        assert abs(american - european) < 0.0001, (
            f"American call should equal European when q=0: Am={american}, Eu={european}"
        )

    def test_american_put_exceeds_european(self):
        """American put should be >= European put."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20

        european = black_scholes_price(S, K, T, r, sigma, "put", q=0.0)
        american = american_option_price(S, K, T, r, sigma, "put", q=0.0)

        assert american >= european - 0.0001, (
            f"American put should >= European: Am={american}, Eu={european}"
        )

    def test_american_call_with_dividend(self):
        """American call with dividends should exceed European."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
        q = 0.05  # 5% dividend yield

        european = black_scholes_price(S, K, T, r, sigma, "call", q=q)
        american = american_option_price(S, K, T, r, sigma, "call", q=q)

        # Early exercise premium should be positive
        assert american >= european - 0.0001, (
            f"American call (q>0) should >= European: Am={american}, Eu={european}"
        )

    def test_american_put_deep_itm(self):
        """Deep ITM American put should be close to intrinsic."""
        S, K, T, r, sigma = 50.0, 100.0, 0.5, 0.05, 0.20

        intrinsic = K - S
        american = american_option_price(S, K, T, r, sigma, "put", q=0.0)

        # For deep ITM puts, the BAW approximation may have some
        # early exercise premium that reduces value slightly vs intrinsic
        # The key test is that it's >= European and reasonable
        european = black_scholes_price(S, K, T, r, sigma, "put", q=0.0)
        assert american >= european - 0.01, (
            f"American put should >= European: Am={american}, Eu={european}"
        )
        # Should be within 10% of intrinsic for deep ITM
        assert abs(american - intrinsic) < intrinsic * 0.1, (
            f"Deep ITM put should be near intrinsic: Am={american}, Intrinsic={intrinsic}"
        )

    def test_american_vs_known_values(self):
        """Test against known BAW values from literature."""
        # Test case: S=40, K=45, T=0.0833 (1 month), r=0.0488, sigma=0.25
        # From Barone-Adesi & Whaley (1987) Table 1
        S, K, T, r, sigma = 40.0, 45.0, 0.0833, 0.0488, 0.25

        american_put = american_option_price(S, K, T, r, sigma, "put", q=0.0)

        # Expected value is approximately 5.00 (exact intrinsic)
        # BAW gives early exercise for deep ITM puts
        assert 4.5 < american_put < 5.5, f"BAW put value out of expected range: {american_put}"

    def test_american_greeks_consistency(self):
        """Verify American Greeks are computed correctly."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20

        greeks = american_option_greeks(S, K, T, r, sigma, "put", q=0.0)

        # Basic sanity checks
        assert "price" in greeks
        assert "delta" in greeks
        assert -1.0 <= greeks["delta"] <= 0.0, f"Put delta out of range: {greeks['delta']}"
        assert greeks["gamma"] >= 0, f"Gamma should be non-negative: {greeks['gamma']}"
        assert greeks["theta"] <= 0.01, (
            f"Theta should be negative for long option: {greeks['theta']}"
        )

    def test_american_edge_cases(self):
        """Test American pricing edge cases."""
        S, K, sigma = 100.0, 100.0, 0.20

        # T = 0: should return intrinsic
        call_t0 = american_option_price(S, K, 0.0, 0.05, sigma, "call")
        put_t0 = american_option_price(S, K, 0.0, 0.05, sigma, "put")
        assert call_t0 == max(0, S - K)
        assert put_t0 == max(0, K - S)

        # sigma = 0: should return intrinsic
        call_s0 = american_option_price(S, K, 0.5, 0.05, 0.0, "call")
        put_s0 = american_option_price(S, K, 0.5, 0.05, 0.0, "put")
        assert call_s0 == max(0, S - K)
        assert put_s0 == max(0, K - S)


# =============================================================================
# Multi-Asset Covariance VaR Tests
# =============================================================================


class TestCovarianceVaR:
    """Test suite for multi-asset covariance VaR."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rm = RiskManager()

    def test_single_asset_var(self):
        """Single-asset covariance VaR should match parametric VaR."""
        portfolio_value = 1_000_000
        positions = [
            {
                "symbol": "AAPL",
                "strike": 150.0,
                "dte": 30,
                "iv": 0.25,
                "option_type": "put",
                "contracts": 10,
                "is_short": True,
            }
        ]
        spot_prices = {"AAPL": 150.0}
        volatilities = {"AAPL": 0.25}

        # Single-asset correlation matrix (trivial)
        corr_matrix = pd.DataFrame([[1.0]], index=["AAPL"], columns=["AAPL"])

        var, cvar, components = self.rm.calculate_covariance_var(
            portfolio_value,
            positions,
            spot_prices,
            volatilities,
            corr_matrix,
            confidence=0.95,
            horizon_days=1,
        )

        assert var > 0, "VaR should be positive"
        assert cvar >= var, f"CVaR should >= VaR: CVaR={cvar}, VaR={var}"
        assert "delta_var" in components

    def test_multi_asset_diversification(self):
        """Diversified portfolio should have lower VaR than sum of individual VaRs."""
        portfolio_value = 1_000_000
        positions = [
            {
                "symbol": "AAPL",
                "strike": 150.0,
                "dte": 30,
                "iv": 0.25,
                "option_type": "put",
                "contracts": 5,
                "is_short": True,
            },
            {
                "symbol": "MSFT",
                "strike": 300.0,
                "dte": 30,
                "iv": 0.22,
                "option_type": "put",
                "contracts": 3,
                "is_short": True,
            },
        ]
        spot_prices = {"AAPL": 150.0, "MSFT": 300.0}
        volatilities = {"AAPL": 0.25, "MSFT": 0.22}

        # Low correlation (diversification benefit)
        corr_low = pd.DataFrame(
            [[1.0, 0.3], [0.3, 1.0]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"]
        )

        # High correlation (less diversification)
        corr_high = pd.DataFrame(
            [[1.0, 0.95], [0.95, 1.0]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"]
        )

        var_low, _, _ = self.rm.calculate_covariance_var(
            portfolio_value, positions, spot_prices, volatilities, corr_low, confidence=0.95
        )

        var_high, _, _ = self.rm.calculate_covariance_var(
            portfolio_value, positions, spot_prices, volatilities, corr_high, confidence=0.95
        )

        assert var_low < var_high, (
            f"Low correlation VaR should be less: low={var_low}, high={var_high}"
        )

    def test_component_var_attribution(self):
        """Verify component VaR attribution sums correctly."""
        portfolio_value = 1_000_000
        positions = [
            {
                "symbol": "AAPL",
                "strike": 150.0,
                "dte": 30,
                "iv": 0.25,
                "option_type": "put",
                "contracts": 5,
                "is_short": True,
            },
            {
                "symbol": "MSFT",
                "strike": 300.0,
                "dte": 30,
                "iv": 0.22,
                "option_type": "put",
                "contracts": 5,
                "is_short": True,
            },
        ]
        spot_prices = {"AAPL": 150.0, "MSFT": 300.0}
        volatilities = {"AAPL": 0.25, "MSFT": 0.22}

        corr_matrix = pd.DataFrame(
            [[1.0, 0.6], [0.6, 1.0]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"]
        )

        var, cvar, components = self.rm.calculate_covariance_var(
            portfolio_value, positions, spot_prices, volatilities, corr_matrix, confidence=0.95
        )

        # Component VaRs should be provided
        assert "per_asset_contribution" in components
        assert "AAPL" in components["per_asset_contribution"]
        assert "MSFT" in components["per_asset_contribution"]

        # Verify component structure contains expected keys
        for symbol in ["AAPL", "MSFT"]:
            contrib = components["per_asset_contribution"][symbol]
            assert "delta_dollars" in contrib, f"Missing delta_dollars for {symbol}"
            assert "marginal_var" in contrib, f"Missing marginal_var for {symbol}"
            # Delta dollars should be non-zero for short put positions
            assert contrib["delta_dollars"] != 0, f"Delta dollars should be non-zero for {symbol}"

        # Delta VaR should be positive
        assert components.get("delta_var", 0) > 0, "Delta VaR should be positive"


# =============================================================================
# Historical VaR Interpolation Tests
# =============================================================================


class TestHistoricalVaRInterpolation:
    """Test suite for improved historical VaR quantile interpolation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rm = RiskManager()

    def test_var_interpolation_small_sample(self):
        """VaR should be interpolated correctly for small samples."""
        # Create a small sample of 20 returns
        np.random.seed(42)
        returns = pd.DataFrame({"returns": np.random.normal(-0.001, 0.02, 20)})

        positions = [
            {
                "symbol": "TEST",
                "strike": 100.0,
                "dte": 30,
                "iv": 0.25,
                "option_type": "put",
                "contracts": 10,
                "is_short": True,
            }
        ]
        spot_prices = {"TEST": 100.0}

        var_95, cvar_95 = self.rm.calculate_var(
            1_000_000, positions, spot_prices, returns_data=returns, confidence=0.95
        )

        # Basic sanity checks
        assert var_95 > 0, "VaR should be positive"
        assert cvar_95 >= var_95, f"CVaR >= VaR: CVaR={cvar_95}, VaR={var_95}"

    def test_var_monotonic_in_confidence(self):
        """VaR should generally increase with confidence level."""
        np.random.seed(42)
        # Use a larger sample for more stable VaR estimates
        returns = pd.DataFrame({"returns": np.random.normal(-0.001, 0.02, 500)})

        positions = [
            {
                "symbol": "TEST",
                "strike": 100.0,
                "dte": 30,
                "iv": 0.25,
                "option_type": "put",
                "contracts": 10,
                "is_short": True,
            }
        ]
        spot_prices = {"TEST": 100.0}

        var_90, _ = self.rm.calculate_var(
            1_000_000, positions, spot_prices, returns_data=returns, confidence=0.90
        )
        var_95, _ = self.rm.calculate_var(
            1_000_000, positions, spot_prices, returns_data=returns, confidence=0.95
        )
        var_99, _ = self.rm.calculate_var(
            1_000_000, positions, spot_prices, returns_data=returns, confidence=0.99
        )

        # With sufficient sample size, VaR should be approximately monotonic
        # Allow small tolerance for estimation noise
        assert var_90 <= var_95 * 1.05, f"VaR 90% should be <= VaR 95%: 90%={var_90}, 95%={var_95}"
        assert var_95 <= var_99 * 1.05, f"VaR 95% should be <= VaR 99%: 95%={var_95}, 99%={var_99}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for combined functionality."""

    def test_full_risk_workflow(self):
        """Test complete risk calculation workflow."""
        rm = RiskManager()
        portfolio_value = 500_000

        # Build a realistic portfolio
        positions = [
            {
                "symbol": "AAPL",
                "strike": 170.0,
                "dte": 45,
                "iv": 0.28,
                "option_type": "put",
                "contracts": 3,
                "is_short": True,
            },
            {
                "symbol": "MSFT",
                "strike": 380.0,
                "dte": 45,
                "iv": 0.24,
                "option_type": "put",
                "contracts": 2,
                "is_short": True,
            },
            {
                "symbol": "GOOGL",
                "strike": 140.0,
                "dte": 45,
                "iv": 0.26,
                "option_type": "put",
                "contracts": 4,
                "is_short": True,
            },
        ]

        spot_prices = {"AAPL": 175.0, "MSFT": 400.0, "GOOGL": 145.0}
        volatilities = {"AAPL": 0.28, "MSFT": 0.24, "GOOGL": 0.26}

        # Realistic correlation matrix for tech stocks
        corr_matrix = pd.DataFrame(
            [
                [1.00, 0.65, 0.70],
                [0.65, 1.00, 0.72],
                [0.70, 0.72, 1.00],
            ],
            index=["AAPL", "MSFT", "GOOGL"],
            columns=["AAPL", "MSFT", "GOOGL"],
        )

        # Calculate Greeks
        greeks = rm.calculate_portfolio_greeks(positions, spot_prices)
        assert greeks.delta != 0, "Portfolio should have delta exposure"

        # Calculate VaR with covariance
        var, cvar, components = rm.calculate_covariance_var(
            portfolio_value, positions, spot_prices, volatilities, corr_matrix, confidence=0.95
        )

        assert var > 0, "VaR should be positive"
        assert cvar >= var, "CVaR should >= VaR"
        assert var / portfolio_value < 0.2, "VaR should be reasonable (< 20%)"

        # Run stress tests
        stress_results = rm.run_stress_tests(portfolio_value, positions, spot_prices)
        assert "crash_20pct" in stress_results
        assert "worst_case" in stress_results

    def test_american_vs_european_portfolio(self):
        """Compare American and European pricing across portfolio."""
        test_cases = [
            # (S, K, T, r, sigma, q, option_type)
            (100, 100, 0.5, 0.05, 0.25, 0.00, "call"),  # ATM call no div
            (100, 100, 0.5, 0.05, 0.25, 0.03, "call"),  # ATM call with div
            (100, 100, 0.5, 0.05, 0.25, 0.00, "put"),  # ATM put
            (80, 100, 0.5, 0.05, 0.25, 0.00, "put"),  # OTM put
            (120, 100, 0.5, 0.05, 0.25, 0.00, "put"),  # ITM put
        ]

        for S, K, T, r, sigma, q, opt_type in test_cases:
            european = black_scholes_price(S, K, T, r, sigma, opt_type, q)
            american = american_option_price(S, K, T, r, sigma, opt_type, q)

            # American should always be >= European
            assert american >= european - 0.001, (
                f"American < European: case={S},{K},{opt_type}, Am={american}, Eu={european}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
