"""Tests for :meth:`engine.wheel_tracker.WheelTracker.available_buying_power`.

Issue #118 P4. S2 / S4 / S8 logged three times that ``tracker.cash``
overstates deployable capital — ``open_short_put`` credits the premium to
cash but never reserves the ``strike × 100`` collateral a cash-secured
put ties up. ``available_buying_power()`` is the query that nets that
reserved collateral out: ``cash − Σ(put_strike × 100)`` over open
``SHORT_PUT`` positions only.

Pinned here:
  * empty book → cash;
  * one / several open CSPs → cash − Σ(strike × 100);
  * a SHORT_PUT → STOCK_OWNED transition (assignment) releases the
    reservation; STOCK_OWNED and COVERED_CALL reserve nothing;
  * the over-committed case returns a raw negative number, not a
    clamped zero.
"""

from __future__ import annotations

from datetime import date

from engine.wheel_tracker import PositionState, WheelTracker

_ENTRY = date(2026, 1, 5)
_EXPIRY = date(2026, 2, 20)


# ----------------------------------------------------------------------
# Helpers — drive positions into each lifecycle state.
# ----------------------------------------------------------------------
def _open_csp(t: WheelTracker, ticker: str, strike: float, premium: float = 2.5) -> None:
    """Open a short cash-secured put — leaves the position in SHORT_PUT."""
    ok = t.open_short_put(
        ticker=ticker,
        strike=strike,
        premium=premium,
        entry_date=_ENTRY,
        expiration_date=_EXPIRY,
        iv=0.25,
    )
    assert ok, f"open_short_put({ticker}) failed — check capital / margin"
    assert t.positions[ticker].state == PositionState.SHORT_PUT


def _drive_to_stock_owned(t: WheelTracker, ticker: str, strike: float) -> None:
    """SHORT_PUT → assigned → STOCK_OWNED."""
    _open_csp(t, ticker, strike)
    ok = t.handle_put_assignment(ticker, _EXPIRY, strike)
    assert ok, f"handle_put_assignment({ticker}) failed"
    assert t.positions[ticker].state == PositionState.STOCK_OWNED


def _drive_to_covered_call(t: WheelTracker, ticker: str, put_strike: float) -> None:
    """SHORT_PUT → assigned → STOCK_OWNED → COVERED_CALL."""
    _drive_to_stock_owned(t, ticker, put_strike)
    ok = t.open_covered_call(
        ticker=ticker,
        strike=put_strike * 1.1,
        premium=2.0,
        entry_date=_EXPIRY,
        expiration_date=date(2026, 4, 17),
        iv=0.25,
    )
    assert ok, f"open_covered_call({ticker}) failed"
    assert t.positions[ticker].state == PositionState.COVERED_CALL


# ======================================================================
# 1. Empty book
# ======================================================================
class TestEmptyBook:
    def test_empty_book_equals_cash(self):
        t = WheelTracker(initial_capital=100_000.0)
        assert t.available_buying_power() == 100_000.0
        assert t.available_buying_power() == t.cash


# ======================================================================
# 2. Open cash-secured puts reserve strike × 100
# ======================================================================
class TestCspReservations:
    def test_one_csp_reserves_strike_times_100(self):
        t = WheelTracker(initial_capital=100_000.0)
        _open_csp(t, "AAA", strike=200.0)
        # one contract = 100 shares → $20,000 collateral reserved
        assert t.available_buying_power() == t.cash - 200.0 * 100

    def test_multiple_csps_sum_their_reservations(self):
        t = WheelTracker(initial_capital=200_000.0)
        _open_csp(t, "AAA", strike=200.0)
        _open_csp(t, "BBB", strike=150.0)
        _open_csp(t, "CCC", strike=90.0)
        reserved = (200.0 + 150.0 + 90.0) * 100
        assert t.available_buying_power() == t.cash - reserved

    def test_reservation_uses_strike_not_premium_or_spot(self):
        """Collateral is strike × 100 — independent of the premium
        collected. Two CSPs at the same strike but different premiums
        reserve the same amount."""
        t1 = WheelTracker(initial_capital=100_000.0)
        _open_csp(t1, "AAA", strike=180.0, premium=1.0)
        t2 = WheelTracker(initial_capital=100_000.0)
        _open_csp(t2, "AAA", strike=180.0, premium=9.0)
        assert (t1.cash - t1.available_buying_power()) == 180.0 * 100
        assert (t2.cash - t2.available_buying_power()) == 180.0 * 100

    def test_closing_a_put_releases_its_reservation(self):
        t = WheelTracker(initial_capital=100_000.0)
        _open_csp(t, "AAA", strike=200.0)
        _open_csp(t, "BBB", strike=150.0)
        t.close_short_put("AAA", buyback_price=0.5, exit_date=date(2026, 2, 1))
        # AAA is closed (deleted) → only BBB's $15,000 still reserved
        assert t.available_buying_power() == t.cash - 150.0 * 100


# ======================================================================
# 3. Only SHORT_PUT reserves — assignment releases it
# ======================================================================
class TestNonCspStatesReserveNothing:
    def test_assignment_releases_the_reservation(self):
        """SHORT_PUT → STOCK_OWNED: the cash-secured collateral is freed
        (the cash was actually spent acquiring the shares)."""
        t = WheelTracker(initial_capital=100_000.0)
        _open_csp(t, "AAA", strike=200.0)
        assert t.available_buying_power() == t.cash - 200.0 * 100  # reserved
        t.handle_put_assignment("AAA", _EXPIRY, 200.0)
        assert t.positions["AAA"].state == PositionState.STOCK_OWNED
        assert t.available_buying_power() == t.cash  # released

    def test_stock_owned_position_reserves_nothing(self):
        t = WheelTracker(initial_capital=100_000.0)
        _drive_to_stock_owned(t, "AAA", strike=180.0)
        assert t.available_buying_power() == t.cash

    def test_covered_call_position_reserves_nothing(self):
        """A short call is covered by the held shares, not by cash."""
        t = WheelTracker(initial_capital=100_000.0)
        _drive_to_covered_call(t, "AAA", put_strike=180.0)
        assert t.available_buying_power() == t.cash

    def test_mixed_book_only_short_puts_reserve(self):
        t = WheelTracker(initial_capital=300_000.0)
        _open_csp(t, "AAA", strike=200.0)  # SHORT_PUT  → reserves 20,000
        _drive_to_stock_owned(t, "BBB", strike=150.0)  # STOCK_OWNED → 0
        _drive_to_covered_call(t, "CCC", put_strike=90.0)  # COVERED_CALL → 0
        # only AAA reserves
        assert t.available_buying_power() == t.cash - 200.0 * 100


# ======================================================================
# 4. The over-committed (negative) case — returned raw, not clamped
# ======================================================================
class TestOverCommittedBook:
    def test_over_committed_book_returns_negative(self):
        """Three CSPs reserving $75,000 against a $50,000 account: the
        book is over-committed and available_buying_power is negative."""
        t = WheelTracker(initial_capital=50_000.0)
        _open_csp(t, "AAA", strike=250.0)
        _open_csp(t, "BBB", strike=250.0)
        _open_csp(t, "CCC", strike=250.0)
        abp = t.available_buying_power()
        assert abp < 0.0, "over-committed book must report negative buying power"
        # raw, not clamped to zero — exact cash − Σ collateral
        assert abp == t.cash - 3 * 250.0 * 100

    def test_negative_result_is_not_clamped_to_zero(self):
        t = WheelTracker(initial_capital=20_000.0)
        _open_csp(t, "AAA", strike=300.0)  # reserves $30,000 vs ~$20k cash
        abp = t.available_buying_power()
        assert abp != 0.0
        assert abp < 0.0
        assert abp == t.cash - 300.0 * 100
