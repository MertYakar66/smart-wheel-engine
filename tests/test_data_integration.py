"""
Tests for Bloomberg data integration modules.

Tests data_integration.py (event calendar loading, risk-free rates, dividends)
and wheel_runner.py (the main orchestrator).
"""

from datetime import date
from pathlib import Path

import pytest

from engine.data_integration import (
    build_calendar_from_bloomberg,
    get_current_risk_free_rate,
    get_discrete_dividends_for_option,
    load_dividends_from_bloomberg,
    load_earnings_from_bloomberg,
)

DATA_DIR = Path("data/bloomberg")
HAS_BLOOMBERG_DATA = (DATA_DIR / "sp500_earnings.csv").exists()


@pytest.mark.skipif(not HAS_BLOOMBERG_DATA, reason="Bloomberg data not available")
class TestEarningsLoading:
    """Test loading earnings from Bloomberg CSV."""

    def test_load_all_earnings(self):
        events = load_earnings_from_bloomberg()
        assert len(events) > 0
        assert all(e.event_type.value == "earnings" for e in events)

    def test_load_earnings_filtered_by_ticker(self):
        events = load_earnings_from_bloomberg(tickers=["A"])
        assert len(events) > 0
        assert all(e.symbol == "A" for e in events)

    def test_load_earnings_filtered_by_date(self):
        events = load_earnings_from_bloomberg(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        assert len(events) > 0
        assert all(e.event_date.year == 2025 for e in events)

    def test_earnings_have_time_of_day(self):
        events = load_earnings_from_bloomberg(tickers=["AAPL"])
        assert len(events) > 0
        assert all(e.time_of_day in ("pre", "during", "post") for e in events)


@pytest.mark.skipif(not HAS_BLOOMBERG_DATA, reason="Bloomberg data not available")
class TestDividendLoading:
    """Test loading dividends from Bloomberg CSV."""

    def test_load_all_dividends(self):
        events = load_dividends_from_bloomberg()
        assert len(events) > 0

    def test_load_dividends_filtered(self):
        events = load_dividends_from_bloomberg(tickers=["AAPL"])
        assert len(events) > 0
        assert all(e.symbol == "AAPL" for e in events)

    def test_dividend_has_amount(self):
        events = load_dividends_from_bloomberg(tickers=["AAPL"])
        ex_divs = [e for e in events if e.event_type.value == "dividend_ex"]
        assert len(ex_divs) > 0
        # Most dividends should have positive amounts (some special divs may be 0)
        positive_divs = [e for e in ex_divs if e.dividend_amount and e.dividend_amount > 0]
        assert len(positive_divs) > len(ex_divs) * 0.9  # At least 90% have amounts


@pytest.mark.skipif(not HAS_BLOOMBERG_DATA, reason="Bloomberg data not available")
class TestCalendarBuilder:
    """Test building full calendar from Bloomberg data."""

    def test_build_calendar(self):
        cal = build_calendar_from_bloomberg(
            tickers=["AAPL", "MSFT"],
            years=[2025, 2026],
        )
        assert len(cal.events) > 0

    def test_calendar_has_earnings_and_dividends(self):
        cal = build_calendar_from_bloomberg(
            tickers=["AAPL"],
            years=[2025],
        )
        types = {e.event_type.value for e in cal.events}
        assert "earnings" in types
        # AAPL pays dividends
        assert "dividend_ex" in types

    def test_calendar_has_macro_events(self):
        cal = build_calendar_from_bloomberg(years=[2025], include_macro=True)
        types = {e.event_type.value for e in cal.events}
        assert "fomc" in types
        assert "cpi" in types


@pytest.mark.skipif(not HAS_BLOOMBERG_DATA, reason="Bloomberg data not available")
class TestDiscreteDividends:
    """Test discrete dividend extraction for CRR tree."""

    def test_get_dividends_for_option(self):
        divs = get_discrete_dividends_for_option(
            ticker="AAPL",
            as_of=date(2025, 1, 1),
            expiry=date(2025, 12, 31),
        )
        # AAPL pays quarterly dividends
        assert len(divs) >= 2
        assert all(d["amount"] > 0 for d in divs)
        assert all(0 < d["time_frac"] <= 1 for d in divs)


@pytest.mark.skipif(not HAS_BLOOMBERG_DATA, reason="Bloomberg data not available")
class TestRiskFreeRate:
    """Test risk-free rate lookup."""

    def test_get_latest_rate(self):
        rate = get_current_risk_free_rate()
        assert 0 < rate < 0.20  # Reasonable range
        assert isinstance(rate, float)

    def test_get_rate_at_date(self):
        rate = get_current_risk_free_rate(as_of="2024-01-15")
        assert 0 < rate < 0.20

    def test_fallback_on_missing_file(self):
        rate = get_current_risk_free_rate(data_dir="/nonexistent")
        assert rate == 0.05  # Default fallback


class TestRiskFreeRateFallback:
    """Test fallback behavior without Bloomberg data."""

    def test_missing_file_returns_default(self):
        rate = get_current_risk_free_rate(data_dir="/tmp/nonexistent_dir")
        assert rate == 0.05

    def test_missing_earnings_returns_empty(self):
        events = load_earnings_from_bloomberg(filepath="/tmp/nonexistent.csv")
        assert events == []

    def test_missing_dividends_returns_empty(self):
        events = load_dividends_from_bloomberg(filepath="/tmp/nonexistent.csv")
        assert events == []
