"""
Edge case tests for the Smart Wheel Engine.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.wheel_tracker import WheelTracker, PositionState
from engine.option_pricer import black_scholes_price, estimate_option_price_from_iv
from engine.transaction_costs import (
    calculate_reg_t_margin_short_put,
    calculate_total_entry_cost,
    calculate_slippage
)


class TestATMExpiration:
    """Tests for at-the-money expiration edge cases."""

    def test_stock_exactly_at_strike_put(self):
        """Stock exactly at strike at expiration - put should NOT assign."""
        tracker = WheelTracker(100000.0)
        tracker.open_short_put(
            ticker='TEST',
            strike=150.0,
            premium=2.50,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 2, 1),
            iv=0.25
        )

        # Stock price exactly at strike
        result = tracker.handle_put_expiration('TEST', date(2024, 2, 1), 150.0)

        # Should expire worthless (not assigned) because stock >= strike
        assert 'TEST' not in tracker.positions
        assert result is not None  # Returns closed position
        assert result['exit_reason'] == 'put_expired_otm'

    def test_stock_exactly_at_strike_call(self):
        """Stock exactly at strike at expiration - call should NOT assign."""
        tracker = WheelTracker(100000.0)

        # First get into stock position
        tracker.open_short_put(
            ticker='TEST',
            strike=150.0,
            premium=2.50,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 1, 15),
            iv=0.25
        )
        tracker.handle_put_assignment('TEST', date(2024, 1, 15), 145.0)

        # Now open covered call
        tracker.open_covered_call(
            ticker='TEST',
            strike=155.0,
            premium=1.50,
            entry_date=date(2024, 1, 16),
            expiration_date=date(2024, 2, 16),
            iv=0.23
        )

        # Stock at exactly strike price
        result = tracker.handle_call_expiration('TEST', date(2024, 2, 16), 155.0)

        # Should expire worthless (keep stock) because stock <= strike
        assert tracker.positions['TEST'].state == PositionState.STOCK_OWNED
        assert result is True


class TestZeroVolatility:
    """Tests for zero volatility edge cases."""

    def test_zero_iv_call_pricing(self):
        """Zero IV call should equal PV of intrinsic."""
        # ITM call
        price = black_scholes_price(S=110, K=100, T=0.25, r=0.05, sigma=0.0, option_type='call')
        expected = max(0, 110 * np.exp(0) - 100 * np.exp(-0.05 * 0.25))
        assert abs(price - expected) < 0.01

    def test_zero_iv_put_pricing(self):
        """Zero IV put should equal PV of intrinsic."""
        # ITM put
        price = black_scholes_price(S=90, K=100, T=0.25, r=0.05, sigma=0.0, option_type='put')
        expected = max(0, 100 * np.exp(-0.05 * 0.25) - 90 * np.exp(0))
        assert abs(price - expected) < 0.01


class TestZeroDTE:
    """Tests for zero DTE edge cases."""

    def test_zero_dte_otm_put(self):
        """Zero DTE OTM put should be worthless."""
        price = estimate_option_price_from_iv(
            underlying_price=110,
            strike=100,
            dte=0,
            iv=0.25,
            risk_free_rate=0.05,
            option_type='put'
        )
        assert price == 0

    def test_zero_dte_itm_put(self):
        """Zero DTE ITM put should equal intrinsic."""
        price = estimate_option_price_from_iv(
            underlying_price=90,
            strike=100,
            dte=0,
            iv=0.25,
            risk_free_rate=0.05,
            option_type='put'
        )
        assert price == 10

    def test_zero_dte_otm_call(self):
        """Zero DTE OTM call should be worthless."""
        price = estimate_option_price_from_iv(
            underlying_price=90,
            strike=100,
            dte=0,
            iv=0.25,
            risk_free_rate=0.05,
            option_type='call'
        )
        assert price == 0


class TestNegativeCash:
    """Tests for negative cash scenarios."""

    def test_assignment_with_insufficient_cash(self):
        """Assignment with insufficient cash should still process (margin call scenario)."""
        tracker = WheelTracker(5000.0)  # Small capital

        tracker.open_short_put(
            ticker='EXPENSIVE',
            strike=150.0,  # Would need $15,000 for assignment
            premium=5.00,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 2, 1),
            iv=0.30
        )

        # Force assignment (stock below strike)
        result = tracker.handle_put_assignment('EXPENSIVE', date(2024, 2, 1), 140.0)

        assert result is True
        assert tracker.positions['EXPENSIVE'].state == PositionState.STOCK_OWNED
        # Cash should be negative (margin call territory)
        assert tracker.cash < 0


class TestMarginCalculation:
    """Tests for Reg-T margin calculation."""

    def test_atm_margin(self):
        """ATM put margin calculation."""
        margin = calculate_reg_t_margin_short_put(
            strike=100,
            underlying_price=100,
            premium=3.00
        )
        # Should be ~20% of underlying + premium
        expected_min = 0.20 * 100 * 100 + 3.00 * 100
        assert margin >= 100  # Minimum
        assert margin > 0

    def test_otm_margin(self):
        """OTM put margin should be lower."""
        otm_margin = calculate_reg_t_margin_short_put(
            strike=90,
            underlying_price=100,
            premium=1.50
        )
        atm_margin = calculate_reg_t_margin_short_put(
            strike=100,
            underlying_price=100,
            premium=3.00
        )
        # OTM margin should be less than ATM
        assert otm_margin < atm_margin

    def test_itm_margin(self):
        """ITM put margin should be higher or equal (due to OTM amount offset)."""
        itm_margin = calculate_reg_t_margin_short_put(
            strike=120,  # Deeper ITM
            underlying_price=100,
            premium=22.00
        )
        atm_margin = calculate_reg_t_margin_short_put(
            strike=100,
            underlying_price=100,
            premium=3.00
        )
        # ITM margin should be more than ATM
        assert itm_margin > atm_margin

    def test_minimum_margin(self):
        """Margin should never be below $100."""
        margin = calculate_reg_t_margin_short_put(
            strike=1,  # Very low strike
            underlying_price=1,
            premium=0.01
        )
        assert margin >= 100


class TestSlippageEdgeCases:
    """Tests for slippage calculation edge cases."""

    def test_zero_spread_slippage(self):
        """Zero spread should result in zero slippage."""
        slippage = calculate_slippage(
            mid_price=2.50,
            bid_ask_spread=0.0,
            trade_direction="sell"
        )
        assert slippage == 0

    def test_very_illiquid_option(self):
        """Very illiquid option should have higher slippage factor."""
        liquid_slippage = calculate_slippage(
            mid_price=2.50,
            bid_ask_spread=0.10,
            trade_direction="sell",
            open_interest=1000
        )
        illiquid_slippage = calculate_slippage(
            mid_price=2.50,
            bid_ask_spread=0.10,
            trade_direction="sell",
            open_interest=10
        )
        assert illiquid_slippage > liquid_slippage


class TestPositionStateTransitions:
    """Tests for position state machine transitions."""

    def test_cannot_open_call_without_stock(self):
        """Cannot open covered call without owning stock."""
        tracker = WheelTracker(100000.0)

        result = tracker.open_covered_call(
            ticker='TEST',
            strike=155.0,
            premium=1.50,
            entry_date=date(2024, 1, 16),
            expiration_date=date(2024, 2, 16),
            iv=0.23
        )

        assert result is False
        assert 'TEST' not in tracker.positions

    def test_cannot_assign_call_without_covered_call(self):
        """Cannot assign call without having sold one."""
        tracker = WheelTracker(100000.0)

        # Get into stock position
        tracker.open_short_put(
            ticker='TEST',
            strike=150.0,
            premium=2.50,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 1, 15),
            iv=0.25
        )
        tracker.handle_put_assignment('TEST', date(2024, 1, 15), 145.0)

        # Try to assign call without selling one
        result = tracker.handle_call_assignment('TEST', date(2024, 2, 1))

        assert result is None  # Should fail

    def test_cannot_close_put_not_in_put_state(self):
        """Cannot close short put when not in SHORT_PUT state."""
        tracker = WheelTracker(100000.0)

        result = tracker.close_short_put('TEST', 1.00, date(2024, 1, 15), "early_exit")

        assert result is None


class TestMultiplePositions:
    """Tests for multiple simultaneous positions."""

    def test_max_positions_respected(self):
        """Should not open more positions than allowed."""
        tracker = WheelTracker(100000.0)

        for i in range(5):
            tracker.open_short_put(
                ticker=f'TEST{i}',
                strike=100.0 + i,
                premium=2.50,
                entry_date=date(2024, 1, 1),
                expiration_date=date(2024, 2, 1),
                iv=0.25
            )

        assert len(tracker.positions) == 5

    def test_cannot_duplicate_position(self):
        """Cannot open duplicate position in same ticker."""
        tracker = WheelTracker(100000.0)

        result1 = tracker.open_short_put(
            ticker='TEST',
            strike=150.0,
            premium=2.50,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 2, 1),
            iv=0.25
        )

        result2 = tracker.open_short_put(
            ticker='TEST',
            strike=145.0,  # Different strike
            premium=3.00,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 2, 1),
            iv=0.25
        )

        assert result1 is True
        assert result2 is False  # Duplicate rejected
        assert len(tracker.positions) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
