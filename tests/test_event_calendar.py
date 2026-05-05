"""Tests for engine/event_calendar.py — full calendar + filters + ingestion."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from engine.event_calendar import (
    CalendarIngestionManager,
    CalendarSourceConfig,
    EventCalendar,
    EventCalendarBuilder,
    EventImpact,
    EventRiskFilter,
    EventType,
    MarketEvent,
    build_default_calendar,
    validate_calendar_dates,
)


# ---------------------------------------------------------------------------
# MarketEvent
# ---------------------------------------------------------------------------

class TestMarketEvent:
    def test_basic_construction(self):
        ev = MarketEvent(
            event_date=date(2026, 5, 1),
            event_type=EventType.EARNINGS,
            symbol="AAPL",
            description="AAPL Q2 Earnings",
        )
        assert ev.symbol == "AAPL"
        assert ev.impact == EventImpact.MEDIUM  # default
        assert ev.expected_move is None  # default

    def test_str_with_symbol(self):
        ev = MarketEvent(
            event_date=date(2026, 5, 1),
            event_type=EventType.EARNINGS,
            symbol="AAPL",
            description="X",
        )
        s = str(ev)
        assert "AAPL" in s
        assert "earnings" in s

    def test_str_macro_event(self):
        ev = MarketEvent(
            event_date=date(2026, 5, 1),
            event_type=EventType.FOMC,
            symbol=None,
            description="X",
        )
        assert "[MACRO]" in str(ev)

    def test_days_until(self):
        # Future event
        future = date.today() + timedelta(days=10)
        ev = MarketEvent(
            event_date=future,
            event_type=EventType.OTHER,
            symbol=None,
            description="X",
        )
        assert ev.days_until == 10


# ---------------------------------------------------------------------------
# EventCalendar
# ---------------------------------------------------------------------------

@pytest.fixture
def cal() -> EventCalendar:
    return EventCalendar()


class TestEventCalendarBasics:
    def test_add_event_indexes_by_date_and_symbol(self, cal: EventCalendar):
        ev = MarketEvent(date(2026, 5, 1), EventType.EARNINGS, "AAPL", "X")
        cal.add_event(ev)
        assert ev in cal.events
        assert ev in cal._events_by_date[date(2026, 5, 1)]
        assert ev in cal._events_by_symbol["AAPL"]

    def test_add_macro_event_no_symbol_index(self, cal: EventCalendar):
        ev = MarketEvent(date(2026, 5, 1), EventType.FOMC, None, "X")
        cal.add_event(ev)
        assert ev in cal.events
        # No symbol → not indexed by symbol
        assert "AAPL" not in cal._events_by_symbol

    def test_add_events_batch(self, cal: EventCalendar):
        events = [
            MarketEvent(date(2026, 5, 1), EventType.EARNINGS, "AAPL", "X"),
            MarketEvent(date(2026, 6, 1), EventType.EARNINGS, "MSFT", "X"),
        ]
        cal.add_events(events)
        assert len(cal.events) == 2


class TestEventCalendarQuery:
    @pytest.fixture
    def populated(self) -> EventCalendar:
        c = EventCalendar()
        c.add_events([
            MarketEvent(date(2026, 5, 1), EventType.EARNINGS, "AAPL", "X"),
            MarketEvent(date(2026, 5, 15), EventType.EARNINGS, "MSFT", "X"),
            MarketEvent(date(2026, 5, 20), EventType.FOMC, None, "X"),
            MarketEvent(date(2026, 6, 1), EventType.DIVIDEND_EX, "AAPL", "X"),
            MarketEvent(date(2026, 7, 1), EventType.EARNINGS, "AAPL", "X"),
        ])
        return c

    def test_get_events_in_range(self, populated: EventCalendar):
        events = populated.get_events_in_range(date(2026, 5, 1), date(2026, 5, 31))
        # 3 events fall in May (AAPL earnings, MSFT earnings, FOMC)
        assert len(events) == 3
        # Sorted by date
        assert events[0].event_date <= events[-1].event_date

    def test_get_events_in_range_filtered_by_symbol(self, populated: EventCalendar):
        events = populated.get_events_in_range(
            date(2026, 5, 1), date(2026, 7, 31), symbol="AAPL",
        )
        # Returns AAPL events plus macro events (which match any-symbol query)
        symbols = {e.symbol for e in events}
        # AAPL and macro (None) — never MSFT
        assert "MSFT" not in symbols

    def test_get_events_in_range_filtered_by_type(self, populated: EventCalendar):
        events = populated.get_events_in_range(
            date(2026, 5, 1), date(2026, 7, 31),
            event_types=[EventType.EARNINGS],
        )
        assert all(e.event_type == EventType.EARNINGS for e in events)

    def test_get_events_for_symbol(self, populated: EventCalendar):
        events = populated.get_events_for_symbol("AAPL")
        # 3 AAPL events: 2 earnings + 1 div
        assert len(events) == 3

    def test_get_events_for_symbol_with_date_filter(self, populated: EventCalendar):
        events = populated.get_events_for_symbol(
            "AAPL", start_date=date(2026, 6, 1), end_date=date(2026, 7, 31),
        )
        assert len(events) == 2  # div on 6/1 + earnings on 7/1

    def test_get_next_event(self, populated: EventCalendar):
        ev = populated.get_next_event("AAPL", from_date=date(2026, 5, 10))
        assert ev is not None
        assert ev.event_date == date(2026, 6, 1)  # AAPL div

    def test_get_next_event_filtered_type(self, populated: EventCalendar):
        ev = populated.get_next_event(
            "AAPL", from_date=date(2026, 5, 10),
            event_types=[EventType.EARNINGS],
        )
        assert ev is not None
        assert ev.event_date == date(2026, 7, 1)

    def test_get_next_event_none_for_unknown(self, populated: EventCalendar):
        assert populated.get_next_event("ZZZ", from_date=date(2026, 1, 1)) is None

    def test_days_to_next_earnings(self, populated: EventCalendar):
        days = populated.days_to_next_earnings("AAPL", from_date=date(2026, 4, 1))
        assert days == 30  # 4/1 → 5/1

    def test_days_to_next_earnings_none(self, populated: EventCalendar):
        assert populated.days_to_next_earnings("ZZZ", from_date=date(2026, 1, 1)) is None

    def test_days_to_next_dividend(self, populated: EventCalendar):
        days = populated.days_to_next_dividend("AAPL", from_date=date(2026, 5, 1))
        assert days == 31  # 5/1 → 6/1

    def test_has_event_before_expiry(self, populated: EventCalendar):
        has, events = populated.has_event_before_expiry(
            "AAPL", trade_date=date(2026, 4, 25), expiry_date=date(2026, 5, 30),
        )
        assert has is True
        # AAPL earnings on 5/1 + macro FOMC on 5/20
        kinds = {e.event_type for e in events}
        assert EventType.EARNINGS in kinds
        assert EventType.FOMC in kinds

    def test_has_event_before_expiry_none(self, populated: EventCalendar):
        has, events = populated.has_event_before_expiry(
            "ZZZ", trade_date=date(2026, 8, 1), expiry_date=date(2026, 8, 30),
        )
        assert has is False
        assert events == []


# ---------------------------------------------------------------------------
# EventCalendarBuilder
# ---------------------------------------------------------------------------

class TestEventCalendarBuilder:
    def test_from_earnings_csv_missing_file(self, tmp_path: Path):
        events = EventCalendarBuilder.from_earnings_csv(str(tmp_path / "nope.csv"))
        assert events == []

    def test_from_earnings_csv_happy(self, tmp_path: Path):
        path = tmp_path / "earnings.csv"
        path.write_text(
            "symbol,date,quarter,expected_move,time\n"
            "AAPL,2026-05-01,Q2,0.05,post\n"
            "MSFT,2026-04-25,Q1,0.04,pre\n"
        )
        events = EventCalendarBuilder.from_earnings_csv(str(path))
        assert len(events) == 2
        assert events[0].event_type == EventType.EARNINGS
        assert events[0].impact == EventImpact.HIGH

    def test_from_dividends_csv_missing_file(self, tmp_path: Path):
        events = EventCalendarBuilder.from_dividends_csv(str(tmp_path / "nope.csv"))
        assert events == []

    def test_from_dividends_csv_happy(self, tmp_path: Path):
        path = tmp_path / "div.csv"
        path.write_text(
            "symbol,ex_date,pay_date,amount,yield\n"
            "AAPL,2026-05-10,2026-05-20,0.24,0.005\n"
        )
        events = EventCalendarBuilder.from_dividends_csv(str(path))
        # Both ex and pay events
        kinds = {e.event_type for e in events}
        assert EventType.DIVIDEND_EX in kinds
        assert EventType.DIVIDEND_PAY in kinds

    def test_generate_fomc_dates_2024_2025(self):
        # Both years have hardcoded dates
        ev_24 = EventCalendarBuilder.generate_fomc_dates(2024)
        ev_25 = EventCalendarBuilder.generate_fomc_dates(2025)
        # 8 FOMC meetings per year
        assert len(ev_24) == 8
        assert len(ev_25) == 8
        for ev in ev_24:
            assert ev.event_type == EventType.FOMC
            assert ev.symbol is None  # macro event

    def test_generate_cpi_dates(self):
        events = EventCalendarBuilder.generate_cpi_dates(2025)
        assert len(events) >= 1
        for ev in events:
            assert ev.event_type == EventType.CPI

    def test_generate_nfp_dates(self):
        events = EventCalendarBuilder.generate_nfp_dates(2025)
        assert len(events) >= 1
        for ev in events:
            assert ev.event_type == EventType.NFP

    def test_generate_gdp_dates(self):
        events = EventCalendarBuilder.generate_gdp_dates(2025)
        # GDP is quarterly → 4 events
        assert len(events) == 4
        for ev in events:
            assert ev.event_type == EventType.GDP

    def test_generate_monthly_expiries(self):
        events = EventCalendarBuilder.generate_monthly_expiries(2025)
        # 12 monthly expiries (3rd Friday)
        assert len(events) == 12
        for ev in events:
            assert ev.event_type == EventType.OPTIONS_EXPIRY
            # 3rd Friday is between 15-21
            assert 15 <= ev.event_date.day <= 21
            assert ev.event_date.weekday() == 4  # Friday

    def test_from_dataframe(self):
        df = pd.DataFrame([
            {"date": pd.Timestamp("2026-05-01"), "symbol": "AAPL",
             "event_type": "earnings", "description": "X", "impact": "high"},
            {"date": pd.Timestamp("2026-05-15"), "symbol": None,
             "event_type": "fomc", "description": "X", "impact": "critical"},
        ])
        events = EventCalendarBuilder.from_dataframe(df)
        assert len(events) == 2
        assert events[0].event_type == EventType.EARNINGS
        assert events[0].impact == EventImpact.HIGH
        assert events[1].event_type == EventType.FOMC

    def test_from_dataframe_unknown_event_type_falls_to_other(self):
        df = pd.DataFrame([
            {"date": pd.Timestamp("2026-05-01"), "symbol": "AAPL",
             "event_type": "unknown_thing", "description": "X"},
        ])
        events = EventCalendarBuilder.from_dataframe(df)
        assert events[0].event_type == EventType.OTHER


# ---------------------------------------------------------------------------
# EventRiskFilter
# ---------------------------------------------------------------------------

class TestEventRiskFilter:
    @pytest.fixture
    def filter_with_earnings(self) -> EventRiskFilter:
        cal = EventCalendar()
        cal.add_event(MarketEvent(date(2026, 5, 5), EventType.EARNINGS, "AAPL", "X"))
        return EventRiskFilter(cal, earnings_buffer_days=5)

    def test_avoid_when_earnings_within_buffer(self, filter_with_earnings: EventRiskFilter):
        avoid, reason = filter_with_earnings.should_avoid_trade(
            "AAPL", trade_date=date(2026, 5, 1), expiry_date=date(2026, 6, 1),
        )
        assert avoid is True
        assert "Earnings" in reason

    def test_no_avoid_when_earnings_outside_buffer(self, filter_with_earnings: EventRiskFilter):
        # Trade 10 days before earnings, buffer is 5 → outside
        avoid, _ = filter_with_earnings.should_avoid_trade(
            "AAPL", trade_date=date(2026, 4, 25), expiry_date=date(2026, 6, 1),
        )
        assert avoid is False

    def test_no_avoid_when_no_earnings(self, filter_with_earnings: EventRiskFilter):
        avoid, _ = filter_with_earnings.should_avoid_trade(
            "MSFT", trade_date=date(2026, 5, 1), expiry_date=date(2026, 6, 1),
        )
        assert avoid is False

    def test_avoid_when_fomc_within_buffer(self):
        cal = EventCalendar()
        cal.add_event(MarketEvent(date(2026, 5, 5), EventType.FOMC, None, "X"))
        f = EventRiskFilter(cal, fomc_buffer_days=2)
        avoid, reason = f.should_avoid_trade(
            "AAPL", trade_date=date(2026, 5, 4), expiry_date=date(2026, 6, 1),
        )
        assert avoid is True
        assert "FOMC" in reason

    def test_event_adjusted_sizing_no_events(self):
        cal = EventCalendar()
        f = EventRiskFilter(cal)
        size, reason = f.get_event_adjusted_sizing(
            "AAPL", date(2026, 5, 1), date(2026, 6, 1), base_size=10.0,
        )
        assert size == 10.0
        assert "No events" in reason

    def test_event_adjusted_sizing_earnings_halves(self):
        cal = EventCalendar()
        cal.add_event(MarketEvent(date(2026, 5, 10), EventType.EARNINGS, "AAPL", "X"))
        f = EventRiskFilter(cal)
        size, reason = f.get_event_adjusted_sizing(
            "AAPL", date(2026, 5, 1), date(2026, 6, 1), base_size=10.0,
        )
        assert size == pytest.approx(5.0)  # 0.5x adjustment for earnings
        assert "Earnings" in reason

    def test_event_adjusted_sizing_dividend_minor(self):
        cal = EventCalendar()
        cal.add_event(MarketEvent(date(2026, 5, 10), EventType.DIVIDEND_EX, "AAPL", "X"))
        f = EventRiskFilter(cal)
        size, _ = f.get_event_adjusted_sizing(
            "AAPL", date(2026, 5, 1), date(2026, 6, 1), base_size=10.0,
        )
        assert size == pytest.approx(9.5)  # 0.95x for dividend

    def test_premium_adjustment_no_events(self):
        cal = EventCalendar()
        f = EventRiskFilter(cal)
        m = f.get_event_premium_adjustment(
            "AAPL", date(2026, 5, 1), date(2026, 6, 1),
        )
        assert m == 1.0

    def test_premium_adjustment_earnings_with_expected_move(self):
        cal = EventCalendar()
        ev = MarketEvent(date(2026, 5, 10), EventType.EARNINGS, "AAPL", "X")
        ev.expected_move = 0.05  # 5%
        cal.add_event(ev)
        f = EventRiskFilter(cal)
        m = f.get_event_premium_adjustment(
            "AAPL", date(2026, 5, 1), date(2026, 6, 1),
        )
        assert m == pytest.approx(1.05)

    def test_premium_adjustment_earnings_default_premium(self):
        cal = EventCalendar()
        cal.add_event(MarketEvent(date(2026, 5, 10), EventType.EARNINGS, "AAPL", "X"))
        f = EventRiskFilter(cal)
        m = f.get_event_premium_adjustment(
            "AAPL", date(2026, 5, 1), date(2026, 6, 1),
        )
        assert m == pytest.approx(1.35)  # default 35% premium for earnings


# ---------------------------------------------------------------------------
# CalendarSourceConfig
# ---------------------------------------------------------------------------

class TestCalendarSourceConfig:
    def test_sources_known(self):
        assert "fomc" in CalendarSourceConfig.SOURCES
        assert "cpi" in CalendarSourceConfig.SOURCES
        assert "nfp" in CalendarSourceConfig.SOURCES

    def test_verified_years_present(self):
        assert 2024 in CalendarSourceConfig.VERIFIED_YEARS["fomc"]
        assert 2025 in CalendarSourceConfig.VERIFIED_YEARS["fomc"]


# ---------------------------------------------------------------------------
# CalendarIngestionManager
# ---------------------------------------------------------------------------

class TestCalendarIngestionManager:
    def test_load_missing_file_returns_none(self, tmp_path: Path):
        m = CalendarIngestionManager(calendar_dir=str(tmp_path))
        assert m.load_from_json("fomc", 2026) is None

    def test_load_unknown_event_type_returns_none(self, tmp_path: Path):
        m = CalendarIngestionManager(calendar_dir=str(tmp_path))
        # Write a file but request unknown type
        path = tmp_path / "unknown_2026.json"
        path.write_text(json.dumps({"dates": ["2026-01-01"]}))
        assert m.load_from_json("unknown", 2026) is None

    def test_save_then_load_roundtrip(self, tmp_path: Path):
        m = CalendarIngestionManager(calendar_dir=str(tmp_path))
        dates = [date(2026, 1, 28), date(2026, 3, 18)]
        m.save_to_json("fomc", 2026, dates, source="fed.gov")
        loaded = m.load_from_json("fomc", 2026)
        assert loaded is not None
        assert len(loaded) == 2
        assert all(ev.event_type == EventType.FOMC for ev in loaded)

    def test_save_writes_indented_json_with_metadata(self, tmp_path: Path):
        m = CalendarIngestionManager(calendar_dir=str(tmp_path))
        m.save_to_json("fomc", 2026, [date(2026, 1, 28)], source="fed.gov")
        text = (tmp_path / "fomc_2026.json").read_text()
        data = json.loads(text)
        assert data["event_type"] == "fomc"
        assert data["year"] == 2026
        assert data["source"] == "fed.gov"
        assert "verified_date" in data
        assert "dates" in data

    def test_check_staleness_warns_for_missing(self, tmp_path: Path):
        m = CalendarIngestionManager(calendar_dir=str(tmp_path))
        # No files exist for current year
        warnings = m.check_staleness(date.today().year)
        # Should warn for each missing event type
        assert len(warnings) > 0

    def test_check_staleness_no_warnings_when_fresh(self, tmp_path: Path):
        m = CalendarIngestionManager(calendar_dir=str(tmp_path))
        # Save fresh files for 4 event types
        for et in ("fomc", "cpi", "nfp", "gdp"):
            m.save_to_json(et, 2026, [date(2026, 1, 1)], source="x")
        warnings = m.check_staleness(2026)
        # All freshly verified today → may have other warnings (count mismatch)
        # but no "no JSON file" warnings
        assert all("No authoritative JSON file" not in w for w in warnings)

    def test_load_or_fallback_uses_json_when_present(self, tmp_path: Path):
        m = CalendarIngestionManager(calendar_dir=str(tmp_path))
        m.save_to_json("fomc", 2026, [date(2026, 1, 28)], source="x")
        events = m.load_or_fallback(
            "fomc", 2026, fallback_generator=EventCalendarBuilder.generate_fomc_dates,
        )
        assert events is not None
        assert all(ev.event_type == EventType.FOMC for ev in events)
        # JSON had 1 date → returned 1 event (didn't fall back)
        assert len(events) == 1

    def test_load_or_fallback_uses_hardcoded_when_missing(self, tmp_path: Path):
        m = CalendarIngestionManager(calendar_dir=str(tmp_path))
        # No JSON for 2024 → falls back to generate_fomc_dates(2024)
        events = m.load_or_fallback(
            "fomc", 2024, fallback_generator=EventCalendarBuilder.generate_fomc_dates,
        )
        assert events is not None
        assert len(events) == 8  # FOMC meets 8 times/year

    def test_load_or_fallback_strict_raises(self, tmp_path: Path):
        m = CalendarIngestionManager(calendar_dir=str(tmp_path))
        with pytest.raises(ValueError, match="STRICT CALENDAR MODE"):
            m.load_or_fallback(
                "fomc", 2099, fallback_generator=EventCalendarBuilder.generate_fomc_dates,
                strict=True,
            )


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

class TestValidateCalendarDates:
    def test_validate_with_full_calendar(self):
        cal = EventCalendar()
        # Add 8 FOMC dates for 2025 to satisfy the count check
        for d in EventCalendarBuilder.generate_fomc_dates(2025):
            cal.add_event(d)
        result = validate_calendar_dates(cal, year=2025)
        assert "warnings" in result
        # FOMC count satisfied; CPI/NFP/expiries missing → warnings
        assert any("CPI" in w for w in result["warnings"])

    def test_validate_empty_calendar_warns(self):
        cal = EventCalendar()
        result = validate_calendar_dates(cal, year=2025)
        # Empty → all 4 expected counts off
        assert len(result["warnings"]) >= 4


class TestBuildDefaultCalendar:
    def test_returns_calendar(self, tmp_path: Path):
        # Disable staleness check (no JSON files in tmp dir → would warn)
        cal = build_default_calendar(
            years=[2025],
            include_macro_events=True,
            validate=False,
            calendar_dir=str(tmp_path),
            check_staleness=False,
        )
        assert isinstance(cal, EventCalendar)
        # Should have at least FOMC + monthly expiries for 2025
        assert len(cal.events) >= 8

    def test_minimal_no_macro(self, tmp_path: Path):
        cal = build_default_calendar(
            years=[2025],
            include_macro_events=False,
            validate=False,
            calendar_dir=str(tmp_path),
            check_staleness=False,
        )
        assert isinstance(cal, EventCalendar)
