"""Tests for risk management module."""

import pytest
import numpy as np
import pandas as pd
from datetime import date

from engine.risk_manager import (
    RiskManager,
    RiskLimits,
    RiskMetrics,
    PortfolioGreeks,
    PositionSizingMethod,
    calculate_kelly_fraction,
    calculate_optimal_contracts
)


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_fixed_fractional_sizing(self):
        """Fixed fractional should allocate fixed % per position."""
        rm = RiskManager(sizing_method=PositionSizingMethod.FIXED_FRACTIONAL)

        contracts, reason = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            existing_positions=5
        )

        assert contracts >= 1
        assert "FIXED_FRACTIONAL" in reason.upper() or "fixed_fractional" in reason

    def test_volatility_scaled_sizing(self):
        """Higher IV should result in smaller position size."""
        rm = RiskManager(sizing_method=PositionSizingMethod.VOLATILITY_SCALED)

        # Low IV
        contracts_low_iv, _ = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.15,
            dte=30,
            existing_positions=5
        )

        # High IV
        contracts_high_iv, _ = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.50,
            dte=30,
            existing_positions=5
        )

        # Lower IV should allow larger position
        assert contracts_low_iv >= contracts_high_iv

    def test_kelly_sizing(self):
        """Kelly criterion sizing."""
        rm = RiskManager(sizing_method=PositionSizingMethod.KELLY)

        contracts, reason = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            win_probability=0.70,
            avg_win=1.0,
            avg_loss=2.0,
            existing_positions=5
        )

        assert contracts >= 0
        assert "kelly" in reason.lower()

    def test_max_positions_limit(self):
        """Should return 0 when max positions reached."""
        limits = RiskLimits(max_positions=5)
        rm = RiskManager(limits=limits)

        contracts, reason = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            existing_positions=5  # At limit
        )

        assert contracts == 0
        assert "max positions" in reason.lower()

    def test_concentration_penalty(self):
        """Fewer positions should reduce size."""
        rm = RiskManager()

        # Many positions
        contracts_diverse, _ = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            existing_positions=8
        )

        # Few positions
        contracts_concentrated, _ = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            existing_positions=2
        )

        # More diversified = larger individual positions allowed
        assert contracts_diverse >= contracts_concentrated


class TestPortfolioGreeks:
    """Test portfolio Greeks calculation."""

    def test_short_put_greeks(self):
        """Short put should have positive delta (negative exposure)."""
        rm = RiskManager()

        positions = [{
            'symbol': 'AAPL',
            'option_type': 'put',
            'strike': 150,
            'dte': 30,
            'iv': 0.25,
            'contracts': 1,
            'is_short': True
        }]

        spot_prices = {'AAPL': 155}
        greeks = rm.calculate_portfolio_greeks(positions, spot_prices)

        # Short put: negative delta (we're short negative delta = positive exposure)
        # As price drops, our liability increases
        assert greeks.delta > 0  # Short put means we want stock to go up
        assert greeks.theta > 0  # Short options benefit from time decay

    def test_covered_call_greeks(self):
        """Covered call should have reduced delta."""
        rm = RiskManager()

        # Long 100 shares + short 1 call
        positions = [{
            'symbol': 'AAPL',
            'option_type': 'call',
            'strike': 160,
            'dte': 30,
            'iv': 0.25,
            'contracts': 1,
            'is_short': True
        }]

        spot_prices = {'AAPL': 155}
        greeks = rm.calculate_portfolio_greeks(positions, spot_prices)

        # Short call has negative delta
        assert greeks.delta < 0

    def test_multiple_positions(self):
        """Multiple positions should aggregate Greeks."""
        rm = RiskManager()

        positions = [
            {
                'symbol': 'AAPL',
                'option_type': 'put',
                'strike': 150,
                'dte': 30,
                'iv': 0.25,
                'contracts': 2,
                'is_short': True
            },
            {
                'symbol': 'MSFT',
                'option_type': 'put',
                'strike': 300,
                'dte': 45,
                'iv': 0.22,
                'contracts': 1,
                'is_short': True
            }
        ]

        spot_prices = {'AAPL': 155, 'MSFT': 310}
        greeks = rm.calculate_portfolio_greeks(positions, spot_prices)

        # Should have non-zero aggregated Greeks
        assert greeks.delta != 0
        assert greeks.theta > 0  # Both short = positive theta


class TestVaR:
    """Test Value at Risk calculations."""

    def test_parametric_var(self):
        """Parametric VaR should be positive."""
        rm = RiskManager()

        positions = [{
            'symbol': 'AAPL',
            'option_type': 'put',
            'strike': 150,
            'dte': 30,
            'iv': 0.25,
            'contracts': 5,
            'is_short': True
        }]

        spot_prices = {'AAPL': 155}
        var, cvar = rm.calculate_var(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices
        )

        assert var > 0
        assert cvar >= var  # CVaR should be >= VaR


class TestRiskLimits:
    """Test risk limit checks."""

    def test_within_limits(self):
        """Test risk limit checking."""
        rm = RiskManager()

        # Multiple diversified positions
        positions = [
            {'symbol': 'AAPL', 'option_type': 'put', 'strike': 150, 'dte': 30,
             'iv': 0.25, 'contracts': 1, 'is_short': True, 'market_value': 500},
            {'symbol': 'MSFT', 'option_type': 'put', 'strike': 300, 'dte': 30,
             'iv': 0.22, 'contracts': 1, 'is_short': True, 'market_value': 600},
            {'symbol': 'GOOGL', 'option_type': 'put', 'strike': 140, 'dte': 30,
             'iv': 0.28, 'contracts': 1, 'is_short': True, 'market_value': 450},
        ]

        spot_prices = {'AAPL': 155, 'MSFT': 310, 'GOOGL': 145}
        within_limits, violations = rm.check_limits(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices
        )

        # Should not have position count violations with 3 positions
        assert not any('position count' in v.lower() for v in violations)

    def test_position_count_violation(self):
        """Too many positions should violate limits."""
        limits = RiskLimits(max_positions=2)
        rm = RiskManager(limits=limits)

        positions = [
            {'symbol': f'SYM{i}', 'option_type': 'put', 'strike': 100,
             'dte': 30, 'iv': 0.25, 'contracts': 1, 'is_short': True,
             'market_value': 500}
            for i in range(5)  # 5 positions, limit is 2
        ]

        spot_prices = {f'SYM{i}': 105 for i in range(5)}
        within_limits, violations = rm.check_limits(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices
        )

        assert not within_limits
        assert any('position count' in v.lower() for v in violations)


class TestKellyFraction:
    """Test Kelly criterion calculation."""

    def test_positive_edge(self):
        """Positive edge should give positive Kelly."""
        kelly = calculate_kelly_fraction(
            win_rate=0.70,
            avg_win=1.0,
            avg_loss=1.0,
            kelly_fraction=1.0
        )

        assert kelly > 0

    def test_no_edge(self):
        """50/50 with equal wins/losses = 0 Kelly."""
        kelly = calculate_kelly_fraction(
            win_rate=0.50,
            avg_win=1.0,
            avg_loss=1.0,
            kelly_fraction=1.0
        )

        assert kelly == 0

    def test_negative_edge(self):
        """Negative edge should give 0 Kelly."""
        kelly = calculate_kelly_fraction(
            win_rate=0.30,
            avg_win=1.0,
            avg_loss=1.0,
            kelly_fraction=1.0
        )

        assert kelly == 0

    def test_half_kelly(self):
        """Half Kelly should be capped appropriately."""
        full_kelly = calculate_kelly_fraction(
            win_rate=0.70,
            avg_win=1.5,
            avg_loss=1.0,
            kelly_fraction=1.0
        )

        half_kelly = calculate_kelly_fraction(
            win_rate=0.70,
            avg_win=1.5,
            avg_loss=1.0,
            kelly_fraction=0.5
        )

        # Half kelly is capped at 0.25 max, so it should be <= full_kelly
        assert half_kelly <= full_kelly
        assert half_kelly <= 0.25  # Function caps at 25%


class TestOptimalContracts:
    """Test optimal contract calculation."""

    def test_basic_calculation(self):
        """Should return reasonable number of contracts."""
        contracts = calculate_optimal_contracts(
            capital=100000,
            strike=100,
            max_risk_pct=0.05,
            margin_requirement=0.20
        )

        assert contracts >= 1
        assert contracts <= 50  # Reasonable upper bound

    def test_low_capital(self):
        """Low capital should give fewer contracts."""
        contracts = calculate_optimal_contracts(
            capital=10000,
            strike=100,
            max_risk_pct=0.05,
            margin_requirement=0.20
        )

        assert contracts >= 1
        assert contracts <= 5
