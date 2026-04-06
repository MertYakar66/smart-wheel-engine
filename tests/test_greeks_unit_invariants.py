"""
Greeks Unit Consistency Invariant Tests

These tests verify that Greeks calculations follow the canonical unit contract
defined in docs/GREEKS_UNIT_CONTRACT.md.

Key invariants tested:
1. Vega finite-difference check
2. Theta annual-to-daily conversion
3. Cross-module consistency
4. P&L decomposition accuracy
"""

import pytest
import numpy as np

from engine.option_pricer import (
    black_scholes_price,
    black_scholes_all_greeks,
    black_scholes_vega,
    black_scholes_theta,
)


class TestVegaUnitConsistency:
    """Test that vega is correctly scaled per 1% (1 vol point) change."""

    def test_vega_finite_difference_matches(self):
        """Vega should match finite difference: (P(sigma+0.01) - P(sigma)) / 0.01."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20

        # Get analytic vega (per 1 vol point)
        analytic_vega = black_scholes_vega(S, K, T, r, sigma, "call")

        # Finite difference approximation
        d_sigma = 0.01  # 1 vol point
        price_up = black_scholes_price(S, K, T, r, sigma + d_sigma, "call")
        price_base = black_scholes_price(S, K, T, r, sigma, "call")
        fd_vega = price_up - price_base  # Per 1% change

        # Should match within 1% (finite difference vs analytic)
        assert abs(analytic_vega - fd_vega) / abs(analytic_vega) < 0.01, (
            f"Vega mismatch: analytic={analytic_vega:.6f}, fd={fd_vega:.6f}"
        )

    def test_vega_pnl_formula_correctness(self):
        """Test vega P&L calculation with IV change in decimal and vol points."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25

        vega = black_scholes_vega(S, K, T, r, sigma, "call")

        # IV increases from 25% to 30% (5 vol points, or 0.05 in decimal)
        iv_change_decimal = 0.05
        iv_change_vol_points = 5.0

        # Method 1: vol_change in decimal -> multiply by 100
        pnl_method1 = vega * iv_change_decimal * 100

        # Method 2: vol_change in vol points -> direct multiply
        pnl_method2 = vega * iv_change_vol_points

        assert abs(pnl_method1 - pnl_method2) < 1e-10, (
            f"Vega P&L methods should match: {pnl_method1} vs {pnl_method2}"
        )

        # Verify against actual price change
        price_base = black_scholes_price(S, K, T, r, sigma, "call")
        price_new = black_scholes_price(S, K, T, r, sigma + iv_change_decimal, "call")
        actual_pnl = price_new - price_base

        # Should be within 5% (linear approximation vs actual)
        assert abs(pnl_method1 - actual_pnl) / abs(actual_pnl) < 0.05, (
            f"Vega P&L should approximate actual: {pnl_method1:.4f} vs {actual_pnl:.4f}"
        )


class TestThetaUnitConsistency:
    """Test that theta conversion from annual to daily is correct."""

    def test_theta_annual_vs_daily(self):
        """Annual theta / 365 should equal daily theta."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20

        annual_theta = black_scholes_theta(S, K, T, r, sigma, "call")
        daily_theta = annual_theta / 365

        # Verify by computing price decay over 1 day
        T_minus_1day = T - 1/365
        price_base = black_scholes_price(S, K, T, r, sigma, "call")
        price_1day = black_scholes_price(S, K, T_minus_1day, r, sigma, "call")
        actual_daily_decay = price_1day - price_base  # Negative for long positions

        # Daily theta should approximate actual decay within 5%
        # Note: theta is negative for long options, so use abs for comparison
        if abs(actual_daily_decay) > 0.001:
            rel_error = abs(daily_theta - actual_daily_decay) / abs(actual_daily_decay)
            assert rel_error < 0.05, (
                f"Daily theta={daily_theta:.6f} should match decay={actual_daily_decay:.6f}"
            )

    def test_theta_pnl_over_multiple_days(self):
        """Theta P&L over N days should be: (annual_theta / 365) * N."""
        S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.05, 0.25
        days = 7

        greeks = black_scholes_all_greeks(S, K, T, r, sigma, "call")
        annual_theta = greeks["theta"]

        # Correct formula per GREEKS_UNIT_CONTRACT.md
        theta_pnl = (annual_theta / 365) * days

        # Verify against actual price change
        T_new = T - days/365
        price_base = greeks["price"]
        price_new = black_scholes_price(S, K, T_new, r, sigma, "call")
        actual_pnl = price_new - price_base

        # Linear approximation should be within 10% for 7 days
        if abs(actual_pnl) > 0.01:
            rel_error = abs(theta_pnl - actual_pnl) / abs(actual_pnl)
            assert rel_error < 0.10, (
                f"Theta P&L={theta_pnl:.4f} should approximate actual={actual_pnl:.4f}"
            )


class TestRhoUnitConsistency:
    """Test that rho is correctly scaled per 1% rate change."""

    def test_rho_finite_difference_matches(self):
        """Rho should match: (P(r+0.01) - P(r)) / 0.01 for 1% rate change."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

        greeks = black_scholes_all_greeks(S, K, T, r, sigma, "call")
        analytic_rho = greeks["rho"]

        # Finite difference: 1% rate increase
        d_r = 0.01
        price_base = greeks["price"]
        price_up = black_scholes_price(S, K, T, r + d_r, sigma, "call")
        fd_rho = price_up - price_base  # Per 1% rate change

        # Should match within 1%
        if abs(analytic_rho) > 0.01:
            rel_error = abs(analytic_rho - fd_rho) / abs(analytic_rho)
            assert rel_error < 0.01, (
                f"Rho mismatch: analytic={analytic_rho:.6f}, fd={fd_rho:.6f}"
            )


class TestCrossModuleConsistency:
    """Test that different modules produce consistent Greek values."""

    def test_all_greeks_matches_individual(self):
        """black_scholes_all_greeks should match individual Greek functions."""
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.25

        all_greeks = black_scholes_all_greeks(S, K, T, r, sigma, "call")

        individual_vega = black_scholes_vega(S, K, T, r, sigma, "call")
        individual_theta = black_scholes_theta(S, K, T, r, sigma, "call")

        assert abs(all_greeks["vega"] - individual_vega) < 1e-10
        assert abs(all_greeks["theta"] - individual_theta) < 1e-10


class TestPnLDecompositionAccuracy:
    """Test that sum of Greek P&Ls approximates full repricing."""

    def test_greek_pnl_sum_approximates_repricing(self):
        """
        For small moves, Greek decomposition should approximate full repricing.

        delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl ~ actual_pnl
        """
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25

        # Small shocks
        dS = 2.0           # $2 spot move (2%)
        d_sigma = 0.02     # 2 vol point IV increase
        d_r = 0.005        # 50 bps rate increase
        days = 5           # 5 days decay

        greeks = black_scholes_all_greeks(S, K, T, r, sigma, "put")

        # Greek P&L decomposition (per GREEKS_UNIT_CONTRACT.md)
        delta_pnl = greeks["delta"] * dS
        gamma_pnl = 0.5 * greeks["gamma"] * dS**2
        theta_pnl = (greeks["theta"] / 365) * days
        vega_pnl = greeks["vega"] * d_sigma * 100  # Convert decimal to vol points
        rho_pnl = greeks["rho"] * d_r * 100        # Convert decimal to percentage

        greek_sum_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl

        # Full repricing
        T_new = T - days/365
        S_new = S + dS
        sigma_new = sigma + d_sigma
        r_new = r + d_r

        new_price = black_scholes_price(S_new, K, T_new, r_new, sigma_new, "put")
        old_price = greeks["price"]
        actual_pnl = new_price - old_price

        # Should be within 5% for small moves
        if abs(actual_pnl) > 0.01:
            rel_error = abs(greek_sum_pnl - actual_pnl) / abs(actual_pnl)
            assert rel_error < 0.05, (
                f"Greek sum P&L={greek_sum_pnl:.4f} should approximate "
                f"actual={actual_pnl:.4f} (error={rel_error:.1%})"
            )


class TestStressTestingUnitConsistency:
    """Test that stress_testing module uses correct unit conversions."""

    def test_stress_test_theta_uses_daily_conversion(self):
        """
        Verify stress_testing.py converts annual theta to daily.

        This is a regression test for the P0 bug where theta was multiplied
        by days without dividing by 365 first.
        """
        # Import here to avoid import errors if module has issues
        from engine.stress_testing import greeks_stress_ladder, black_scholes_all_greeks

        positions = [{
            "symbol": "AAPL",
            "strike": 150.0,
            "dte": 30,
            "iv": 0.25,
            "contracts": 1,
            "option_type": "put",
            "is_short": True,
        }]
        spot_prices = {"AAPL": 150.0}

        results = greeks_stress_ladder(positions, spot_prices, 100000.0)

        # The "theta_only" scenario applies 30 days of decay
        theta_scenario = results.get("theta_only", {})

        if theta_scenario:
            theta_pnl = theta_scenario.get("greek_attribution", {}).get("theta_pnl", 0)

            # Calculate expected theta P&L correctly
            greeks = black_scholes_all_greeks(
                S=150.0, K=150.0, T=30/365, r=0.05,
                sigma=0.25, option_type="put"
            )
            # Short position: direction = -1, multiplier = 1 * 100 * -1 = -100
            # (annual_theta / 365) * 30_days * -100
            expected_theta_pnl = (greeks["theta"] / 365) * 30 * (-100)

            # Allow 1% tolerance
            if abs(expected_theta_pnl) > 0.01:
                rel_error = abs(theta_pnl - expected_theta_pnl) / abs(expected_theta_pnl)
                assert rel_error < 0.01, (
                    f"Stress test theta P&L={theta_pnl:.2f} should match "
                    f"expected={expected_theta_pnl:.2f}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
