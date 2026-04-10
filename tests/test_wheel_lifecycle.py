"""
Tests for wheel tracker partial assignment and roll mechanics.
"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.wheel_tracker import PositionState, WheelTracker

# =============================================================================
# HELPERS
# =============================================================================


def _make_tracker_with_short_put(
    capital: float = 100_000.0,
    ticker: str = "TEST",
    strike: float = 150.0,
    premium: float = 2.50,
    entry_date: date = date(2024, 1, 1),
    expiration_date: date = date(2024, 2, 1),
    iv: float = 0.25,
) -> WheelTracker:
    """Create a tracker with an open short put position."""
    tracker = WheelTracker(capital)
    tracker.open_short_put(
        ticker=ticker,
        strike=strike,
        premium=premium,
        entry_date=entry_date,
        expiration_date=expiration_date,
        iv=iv,
    )
    return tracker


def _make_tracker_with_covered_call(
    capital: float = 100_000.0,
    ticker: str = "TEST",
    put_strike: float = 150.0,
    put_premium: float = 2.50,
    call_strike: float = 155.0,
    call_premium: float = 1.50,
) -> WheelTracker:
    """Create a tracker with a covered call position (put assigned, then call sold)."""
    tracker = _make_tracker_with_short_put(
        capital=capital, ticker=ticker, strike=put_strike, premium=put_premium
    )
    tracker.handle_put_assignment(ticker, date(2024, 1, 15), stock_price=145.0)
    tracker.open_covered_call(
        ticker=ticker,
        strike=call_strike,
        premium=call_premium,
        entry_date=date(2024, 1, 16),
        expiration_date=date(2024, 2, 16),
        iv=0.23,
    )
    return tracker


# =============================================================================
# PARTIAL ASSIGNMENT TESTS
# =============================================================================


class TestPartialAssignment:
    """Tests for partial put and call assignment mechanics."""

    def test_partial_put_assignment_50_shares(self):
        """Partial put assignment of 50 shares transitions to STOCK_OWNED with correct count."""
        tracker = _make_tracker_with_short_put()

        result = tracker.handle_partial_put_assignment(
            ticker="TEST",
            assignment_date=date(2024, 2, 1),
            stock_price=145.0,
            shares_assigned=50,
        )

        assert result is True
        pos = tracker.positions["TEST"]
        assert pos.state == PositionState.STOCK_OWNED
        assert pos.stock_shares == 50
        assert pos.stock_basis is not None

    def test_partial_put_assignment_validates_bounds(self):
        """Shares outside 1-100 range should be rejected."""
        tracker_zero = _make_tracker_with_short_put(ticker="T0")
        tracker_over = _make_tracker_with_short_put(ticker="T1")

        result_zero = tracker_zero.handle_partial_put_assignment(
            ticker="T0",
            assignment_date=date(2024, 2, 1),
            stock_price=145.0,
            shares_assigned=0,
        )
        assert result_zero is False

        result_over = tracker_over.handle_partial_put_assignment(
            ticker="T1",
            assignment_date=date(2024, 2, 1),
            stock_price=145.0,
            shares_assigned=101,
        )
        assert result_over is False

        # Negative should also fail
        tracker_neg = _make_tracker_with_short_put(ticker="T2")
        result_neg = tracker_neg.handle_partial_put_assignment(
            ticker="T2",
            assignment_date=date(2024, 2, 1),
            stock_price=145.0,
            shares_assigned=-5,
        )
        assert result_neg is False

    def test_partial_call_assignment_keeps_remaining_shares(self):
        """Partial call assignment should leave remaining shares in STOCK_OWNED state."""
        tracker = _make_tracker_with_covered_call()

        result = tracker.handle_partial_call_assignment(
            ticker="TEST",
            assignment_date=date(2024, 2, 16),
            shares_called=60,
        )

        assert result is not None
        assert result["shares_called"] == 60
        assert result["remaining_shares"] == 40
        assert result["state"] == PositionState.STOCK_OWNED.value

        pos = tracker.positions["TEST"]
        assert pos.state == PositionState.STOCK_OWNED
        assert pos.stock_shares == 40
        # Call fields should be cleared
        assert pos.call_strike is None
        assert pos.call_premium is None

    def test_partial_call_assignment_all_shares_finalizes(self):
        """Calling away all shares should close the position entirely."""
        tracker = _make_tracker_with_covered_call()

        result = tracker.handle_partial_call_assignment(
            ticker="TEST",
            assignment_date=date(2024, 2, 16),
            shares_called=100,
        )

        assert result is not None
        assert "TEST" not in tracker.positions
        assert result["exit_reason"] == "call_assigned"
        assert len(tracker.closed_positions) == 1


# =============================================================================
# ROLL MECHANICS TESTS
# =============================================================================


class TestRollMechanics:
    """Tests for rolling puts and calls."""

    def test_roll_put_basic(self):
        """Roll a short put to a new strike and expiration."""
        tracker = _make_tracker_with_short_put(
            strike=150.0,
            premium=2.50,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 2, 1),
        )

        result = tracker.roll_put(
            ticker="TEST",
            roll_date=date(2024, 1, 28),
            new_strike=145.0,
            new_premium=3.00,
            new_expiration=date(2024, 3, 1),
            new_iv=0.28,
            buyback_price=0.50,
        )

        assert result is not None
        assert result["old_strike"] == 150.0
        assert result["new_strike"] == 145.0
        assert result["new_expiration"] == date(2024, 3, 1)

        pos = tracker.positions["TEST"]
        assert pos.state == PositionState.SHORT_PUT
        assert pos.put_strike == 145.0
        assert pos.put_premium == 3.00
        assert pos.put_expiration_date == date(2024, 3, 1)

    def test_roll_put_returns_net_credit(self):
        """Rolling a put should return a dict with net_credit_debit field."""
        tracker = _make_tracker_with_short_put(
            strike=150.0,
            premium=2.50,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 2, 1),
        )

        result = tracker.roll_put(
            ticker="TEST",
            roll_date=date(2024, 1, 28),
            new_strike=145.0,
            new_premium=3.00,
            new_expiration=date(2024, 3, 1),
            new_iv=0.28,
            buyback_price=0.20,  # Cheap buyback -> should be net credit
        )

        assert result is not None
        assert "net_credit_debit" in result
        # New premium (3.00) is much larger than buyback (0.20), so net should be positive
        assert result["net_credit_debit"] > 0

    def test_roll_call_basic(self):
        """Roll a covered call to a new strike and expiration."""
        tracker = _make_tracker_with_covered_call(
            call_strike=155.0,
            call_premium=1.50,
        )

        result = tracker.roll_call(
            ticker="TEST",
            roll_date=date(2024, 2, 14),
            new_strike=160.0,
            new_premium=2.00,
            new_expiration=date(2024, 3, 16),
            new_iv=0.24,
            buyback_price=0.30,
        )

        assert result is not None
        assert result["old_strike"] == 155.0
        assert result["new_strike"] == 160.0
        assert result["new_expiration"] == date(2024, 3, 16)

        pos = tracker.positions["TEST"]
        assert pos.state == PositionState.COVERED_CALL
        assert pos.call_strike == 160.0
        assert pos.call_premium == 2.00
        assert pos.call_expiration_date == date(2024, 3, 16)

    def test_roll_requires_correct_state(self):
        """roll_put should require SHORT_PUT state; wrong states return None."""
        tracker = _make_tracker_with_covered_call()

        # Position is in COVERED_CALL state, not SHORT_PUT
        result = tracker.roll_put(
            ticker="TEST",
            roll_date=date(2024, 2, 14),
            new_strike=145.0,
            new_premium=3.00,
            new_expiration=date(2024, 3, 1),
            new_iv=0.28,
            buyback_price=0.50,
        )

        assert result is None

        # Also fails for non-existent ticker
        result_missing = tracker.roll_put(
            ticker="MISSING",
            roll_date=date(2024, 2, 14),
            new_strike=145.0,
            new_premium=3.00,
            new_expiration=date(2024, 3, 1),
            new_iv=0.28,
            buyback_price=0.50,
        )
        assert result_missing is None

    def test_roll_call_requires_covered_call_state(self):
        """roll_call should require COVERED_CALL state; SHORT_PUT returns None."""
        tracker = _make_tracker_with_short_put()

        # Position is in SHORT_PUT state, not COVERED_CALL
        result = tracker.roll_call(
            ticker="TEST",
            roll_date=date(2024, 1, 28),
            new_strike=160.0,
            new_premium=2.00,
            new_expiration=date(2024, 3, 16),
            new_iv=0.24,
            buyback_price=0.30,
        )

        assert result is None

        # Also fails for non-existent ticker
        result_missing = tracker.roll_call(
            ticker="MISSING",
            roll_date=date(2024, 1, 28),
            new_strike=160.0,
            new_premium=2.00,
            new_expiration=date(2024, 3, 16),
            new_iv=0.24,
            buyback_price=0.30,
        )
        assert result_missing is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
