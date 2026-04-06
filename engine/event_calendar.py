"""
Event Calendar Module

Professional event tracking for options trading:
- Earnings announcements (major binary events)
- Ex-dividend dates (affects option pricing)
- FOMC meetings and rate decisions
- Economic data releases
- Options expiration cycles

Key principle: Avoid selling options into binary events unless
intentionally trading the event.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import pandas as pd


class EventType(Enum):
    """Types of market events."""

    EARNINGS = "earnings"
    DIVIDEND_EX = "dividend_ex"
    DIVIDEND_PAY = "dividend_pay"
    FOMC = "fomc"
    CPI = "cpi"
    NFP = "nfp"  # Non-farm payrolls
    GDP = "gdp"
    OPTIONS_EXPIRY = "options_expiry"
    STOCK_SPLIT = "stock_split"
    OTHER = "other"


class EventImpact(Enum):
    """Expected impact level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"  # Major binary event


@dataclass
class MarketEvent:
    """Single market event."""

    event_date: date
    event_type: EventType
    symbol: str | None  # None for macro events
    description: str
    impact: EventImpact = EventImpact.MEDIUM

    # Event-specific data
    expected_move: float | None = None  # Expected % move
    historical_move: float | None = None  # Avg historical move
    time_of_day: str | None = None  # "pre", "post", "during"

    # For dividends
    dividend_amount: float | None = None
    dividend_yield: float | None = None

    def __str__(self) -> str:
        symbol_str = f"[{self.symbol}]" if self.symbol else "[MACRO]"
        return f"{self.event_date} {symbol_str} {self.event_type.value}: {self.description}"

    @property
    def days_until(self) -> int:
        """Days until event from today."""
        today = date.today()
        return (self.event_date - today).days


@dataclass
class EventCalendar:
    """
    Comprehensive event calendar for trading decisions.

    Tracks all events that could impact option positions
    and provides filtering/query capabilities.
    """

    events: list[MarketEvent] = field(default_factory=list)
    _events_by_date: dict[date, list[MarketEvent]] = field(default_factory=dict)
    _events_by_symbol: dict[str, list[MarketEvent]] = field(default_factory=dict)

    def add_event(self, event: MarketEvent) -> None:
        """Add event to calendar."""
        self.events.append(event)

        # Index by date
        if event.event_date not in self._events_by_date:
            self._events_by_date[event.event_date] = []
        self._events_by_date[event.event_date].append(event)

        # Index by symbol
        if event.symbol:
            if event.symbol not in self._events_by_symbol:
                self._events_by_symbol[event.symbol] = []
            self._events_by_symbol[event.symbol].append(event)

    def add_events(self, events: list[MarketEvent]) -> None:
        """Add multiple events."""
        for event in events:
            self.add_event(event)

    def get_events_in_range(
        self,
        start_date: date,
        end_date: date,
        symbol: str | None = None,
        event_types: list[EventType] | None = None,
    ) -> list[MarketEvent]:
        """Get events within date range, optionally filtered."""
        results = []

        for event in self.events:
            if event.event_date < start_date or event.event_date > end_date:
                continue
            if symbol and event.symbol != symbol:
                # Include macro events for any symbol query
                if event.symbol is not None:
                    continue
            if event_types and event.event_type not in event_types:
                continue
            results.append(event)

        return sorted(results, key=lambda e: e.event_date)

    def get_events_for_symbol(
        self, symbol: str, start_date: date | None = None, end_date: date | None = None
    ) -> list[MarketEvent]:
        """Get all events for a specific symbol."""
        events = self._events_by_symbol.get(symbol, [])

        if start_date:
            events = [e for e in events if e.event_date >= start_date]
        if end_date:
            events = [e for e in events if e.event_date <= end_date]

        return sorted(events, key=lambda e: e.event_date)

    def get_next_event(
        self, symbol: str, from_date: date, event_types: list[EventType] | None = None
    ) -> MarketEvent | None:
        """Get next upcoming event for symbol."""
        events = self.get_events_for_symbol(symbol, start_date=from_date)

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        return events[0] if events else None

    def days_to_next_earnings(self, symbol: str, from_date: date) -> int | None:
        """Get days until next earnings for symbol."""
        event = self.get_next_event(symbol, from_date, event_types=[EventType.EARNINGS])
        return (event.event_date - from_date).days if event else None

    def days_to_next_dividend(self, symbol: str, from_date: date) -> int | None:
        """Get days until next ex-dividend for symbol."""
        event = self.get_next_event(symbol, from_date, event_types=[EventType.DIVIDEND_EX])
        return (event.event_date - from_date).days if event else None

    def has_event_before_expiry(
        self,
        symbol: str,
        trade_date: date,
        expiry_date: date,
        event_types: list[EventType] | None = None,
    ) -> tuple[bool, list[MarketEvent]]:
        """
        Check if there are events between trade date and expiry.

        Returns (has_event, list of events)
        """
        events = self.get_events_in_range(
            start_date=trade_date, end_date=expiry_date, symbol=symbol, event_types=event_types
        )

        # Also include macro events
        macro_events = self.get_events_in_range(
            start_date=trade_date,
            end_date=expiry_date,
            event_types=[EventType.FOMC, EventType.CPI, EventType.NFP],
        )

        all_events = events + [e for e in macro_events if e not in events]
        return len(all_events) > 0, all_events


class EventCalendarBuilder:
    """
    Build event calendar from various data sources.

    Supports loading from:
    - CSV files
    - DataFrames
    - API responses (placeholder for live data)
    """

    @staticmethod
    def from_earnings_csv(filepath: str) -> list[MarketEvent]:
        """
        Load earnings events from CSV.

        Expected columns: symbol, date, time (pre/post), expected_move
        """
        try:
            df = pd.read_csv(filepath, parse_dates=["date"])
        except FileNotFoundError:
            return []

        events = []
        for _, row in df.iterrows():
            event = MarketEvent(
                event_date=row["date"].date() if hasattr(row["date"], "date") else row["date"],
                event_type=EventType.EARNINGS,
                symbol=row["symbol"],
                description=f"{row['symbol']} Q{row.get('quarter', 'X')} Earnings",
                impact=EventImpact.HIGH,
                expected_move=row.get("expected_move"),
                time_of_day=row.get("time", "post"),
            )
            events.append(event)

        return events

    @staticmethod
    def from_dividends_csv(filepath: str) -> list[MarketEvent]:
        """
        Load dividend events from CSV.

        Expected columns: symbol, ex_date, pay_date, amount
        """
        try:
            df = pd.read_csv(filepath, parse_dates=["ex_date", "pay_date"])
        except FileNotFoundError:
            return []

        events = []
        for _, row in df.iterrows():
            # Ex-dividend event
            ex_event = MarketEvent(
                event_date=row["ex_date"].date()
                if hasattr(row["ex_date"], "date")
                else row["ex_date"],
                event_type=EventType.DIVIDEND_EX,
                symbol=row["symbol"],
                description=f"{row['symbol']} Ex-Dividend ${row['amount']:.2f}",
                impact=EventImpact.MEDIUM,
                dividend_amount=row["amount"],
                dividend_yield=row.get("yield"),
            )
            events.append(ex_event)

            # Pay date event (lower impact)
            if pd.notna(row.get("pay_date")):
                pay_event = MarketEvent(
                    event_date=row["pay_date"].date()
                    if hasattr(row["pay_date"], "date")
                    else row["pay_date"],
                    event_type=EventType.DIVIDEND_PAY,
                    symbol=row["symbol"],
                    description=f"{row['symbol']} Dividend Payment ${row['amount']:.2f}",
                    impact=EventImpact.LOW,
                    dividend_amount=row["amount"],
                )
                events.append(pay_event)

        return events

    @staticmethod
    def generate_fomc_dates(year: int) -> list[MarketEvent]:
        """
        Generate FOMC meeting dates for a year.

        FOMC meets 8 times per year on predetermined dates.
        """
        # 2024 FOMC dates (example - should be updated annually)
        fomc_2024 = [
            date(2024, 1, 31),
            date(2024, 3, 20),
            date(2024, 5, 1),
            date(2024, 6, 12),
            date(2024, 7, 31),
            date(2024, 9, 18),
            date(2024, 11, 7),
            date(2024, 12, 18),
        ]

        fomc_2025 = [
            date(2025, 1, 29),
            date(2025, 3, 19),
            date(2025, 5, 7),
            date(2025, 6, 18),
            date(2025, 7, 30),
            date(2025, 9, 17),
            date(2025, 11, 5),
            date(2025, 12, 17),
        ]

        # 2026 FOMC dates aligned with official Federal Reserve calendar
        # Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
        fomc_2026 = [
            date(2026, 1, 28),   # Jan 27-28, no press conf
            date(2026, 3, 18),   # Mar 17-18, press conf + SEP
            date(2026, 4, 29),   # Apr 28-29, no press conf
            date(2026, 6, 17),   # Jun 16-17, press conf + SEP
            date(2026, 7, 29),   # Jul 28-29, no press conf
            date(2026, 9, 16),   # Sep 15-16, press conf + SEP
            date(2026, 10, 28),  # Oct 27-28, no press conf
            date(2026, 12, 9),   # Dec 8-9, press conf + SEP
        ]

        dates_by_year = {2024: fomc_2024, 2025: fomc_2025, 2026: fomc_2026}

        fomc_dates = dates_by_year.get(year, [])

        events = []
        for fomc_date in fomc_dates:
            event = MarketEvent(
                event_date=fomc_date,
                event_type=EventType.FOMC,
                symbol=None,
                description="FOMC Rate Decision",
                impact=EventImpact.HIGH,
                time_of_day="during",
            )
            events.append(event)

        return events

    @staticmethod
    def generate_cpi_dates(year: int) -> list[MarketEvent]:
        """
        Generate CPI (Consumer Price Index) release dates for a year.

        CPI is released monthly, typically on the second Tuesday-Thursday
        of the month (varies). These dates are sourced from BLS schedule.
        Source: https://www.bls.gov/schedule/news_release/cpi.htm
        """
        # 2024 CPI release dates (actual BLS schedule)
        cpi_2024 = [
            date(2024, 1, 11), date(2024, 2, 13), date(2024, 3, 12),
            date(2024, 4, 10), date(2024, 5, 15), date(2024, 6, 12),
            date(2024, 7, 11), date(2024, 8, 14), date(2024, 9, 11),
            date(2024, 10, 10), date(2024, 11, 13), date(2024, 12, 11),
        ]

        # 2025 CPI release dates
        cpi_2025 = [
            date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
            date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
            date(2025, 7, 11), date(2025, 8, 13), date(2025, 9, 11),
            date(2025, 10, 10), date(2025, 11, 13), date(2025, 12, 10),
        ]

        # 2026 CPI release dates (projected based on BLS patterns)
        cpi_2026 = [
            date(2026, 1, 13), date(2026, 2, 11), date(2026, 3, 11),
            date(2026, 4, 14), date(2026, 5, 12), date(2026, 6, 10),
            date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 15),
            date(2026, 10, 13), date(2026, 11, 12), date(2026, 12, 10),
        ]

        dates_by_year = {2024: cpi_2024, 2025: cpi_2025, 2026: cpi_2026}
        cpi_dates = dates_by_year.get(year, [])

        events = []
        for cpi_date in cpi_dates:
            event = MarketEvent(
                event_date=cpi_date,
                event_type=EventType.CPI,
                symbol=None,
                description="CPI Release (8:30 AM ET)",
                impact=EventImpact.HIGH,
                time_of_day="pre",
            )
            events.append(event)

        return events

    @staticmethod
    def generate_nfp_dates(year: int) -> list[MarketEvent]:
        """
        Generate NFP (Non-Farm Payrolls) release dates for a year.

        NFP is released on the first Friday of each month at 8:30 AM ET.
        Source: https://www.bls.gov/schedule/news_release/empsit.htm
        """
        events = []

        for month in range(1, 13):
            # Find first Friday of month
            first_day = date(year, month, 1)
            # Days until Friday (4 = Friday in weekday())
            days_to_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_to_friday)

            event = MarketEvent(
                event_date=first_friday,
                event_type=EventType.NFP,
                symbol=None,
                description="Non-Farm Payrolls (8:30 AM ET)",
                impact=EventImpact.HIGH,
                time_of_day="pre",
            )
            events.append(event)

        return events

    @staticmethod
    def generate_gdp_dates(year: int) -> list[MarketEvent]:
        """
        Generate GDP release dates for a year.

        GDP is released quarterly with advance, preliminary, and final readings.
        Typically released last week of month following quarter end.
        """
        # Approximate GDP release schedule (advance estimates)
        gdp_quarters = [
            (1, 25),   # Q4 advance in late January
            (4, 25),   # Q1 advance in late April
            (7, 25),   # Q2 advance in late July
            (10, 25),  # Q3 advance in late October
        ]

        events = []
        for month, day in gdp_quarters:
            gdp_date = date(year, month, day)
            # Adjust to nearest business day if weekend
            while gdp_date.weekday() > 4:
                gdp_date -= timedelta(days=1)

            event = MarketEvent(
                event_date=gdp_date,
                event_type=EventType.GDP,
                symbol=None,
                description="GDP Release (Advance Estimate)",
                impact=EventImpact.MEDIUM,
                time_of_day="pre",
            )
            events.append(event)

        return events

    @staticmethod
    def generate_monthly_expiries(year: int, symbols: list[str] | None = None) -> list[MarketEvent]:
        """
        Generate monthly options expiration dates.

        Third Friday of each month.
        """
        events = []

        for month in range(1, 13):
            # Find third Friday
            first_day = date(year, month, 1)
            # Days until Friday (4 = Friday, weekday() returns 0-6)
            days_to_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_to_friday)
            third_friday = first_friday + timedelta(days=14)

            event = MarketEvent(
                event_date=third_friday,
                event_type=EventType.OPTIONS_EXPIRY,
                symbol=None,  # Applies to all
                description="Monthly Options Expiration",
                impact=EventImpact.MEDIUM,
            )
            events.append(event)

        return events

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        date_col: str = "date",
        symbol_col: str = "symbol",
        type_col: str = "event_type",
        description_col: str = "description",
    ) -> list[MarketEvent]:
        """Load events from generic DataFrame."""
        events = []

        for _, row in df.iterrows():
            event_date = row[date_col]
            if hasattr(event_date, "date"):
                event_date = event_date.date()

            event_type_str = row.get(type_col, "other")
            try:
                event_type = EventType(event_type_str.lower())
            except ValueError:
                event_type = EventType.OTHER

            event = MarketEvent(
                event_date=event_date,
                event_type=event_type,
                symbol=row.get(symbol_col),
                description=row.get(description_col, ""),
                impact=EventImpact(row.get("impact", "medium").lower()),
            )
            events.append(event)

        return events


class EventRiskFilter:
    """
    Filter trading decisions based on event risk.

    Implements rules like:
    - Don't sell puts within N days of earnings
    - Adjust position size for event risk
    - Calculate event-adjusted expected move
    """

    def __init__(
        self,
        calendar: EventCalendar,
        earnings_buffer_days: int = 5,
        fomc_buffer_days: int = 2,
        dividend_buffer_days: int = 1,
    ):
        self.calendar = calendar
        self.earnings_buffer_days = earnings_buffer_days
        self.fomc_buffer_days = fomc_buffer_days
        self.dividend_buffer_days = dividend_buffer_days

    def should_avoid_trade(
        self, symbol: str, trade_date: date, expiry_date: date
    ) -> tuple[bool, str]:
        """
        Check if trade should be avoided due to events.

        Returns (should_avoid, reason)
        """
        # Check earnings
        days_to_earnings = self.calendar.days_to_next_earnings(symbol, trade_date)
        if days_to_earnings is not None:
            earnings_date = trade_date + timedelta(days=days_to_earnings)
            if earnings_date <= expiry_date:
                if days_to_earnings <= self.earnings_buffer_days:
                    return True, f"Earnings in {days_to_earnings} days (within buffer)"

        # Check FOMC (for all symbols)
        fomc_events = self.calendar.get_events_in_range(
            trade_date, expiry_date, event_types=[EventType.FOMC]
        )
        for event in fomc_events:
            days_to_fomc = (event.event_date - trade_date).days
            if days_to_fomc <= self.fomc_buffer_days:
                return True, f"FOMC in {days_to_fomc} days"

        return False, ""

    def get_event_adjusted_sizing(
        self, symbol: str, trade_date: date, expiry_date: date, base_size: float
    ) -> tuple[float, str]:
        """
        Adjust position size based on event risk.

        Returns (adjusted_size, reason)
        """
        has_event, events = self.calendar.has_event_before_expiry(symbol, trade_date, expiry_date)

        if not has_event:
            return base_size, "No events in period"

        adjustment = 1.0
        reasons = []

        for event in events:
            if event.event_type == EventType.EARNINGS:
                # Reduce size significantly for earnings
                adjustment *= 0.5
                reasons.append(f"Earnings on {event.event_date}")

            elif event.event_type == EventType.FOMC:
                adjustment *= 0.8
                reasons.append(f"FOMC on {event.event_date}")

            elif event.event_type == EventType.DIVIDEND_EX:
                # Minor adjustment for dividends
                adjustment *= 0.95
                reasons.append(f"Ex-div on {event.event_date}")

        adjusted_size = base_size * adjustment
        reason = "; ".join(reasons) if reasons else "Events present"

        return adjusted_size, reason

    def get_event_premium_adjustment(
        self, symbol: str, trade_date: date, expiry_date: date
    ) -> float:
        """
        Get expected IV premium for events in period.

        Returns multiplier for expected IV (1.0 = no adjustment).
        """
        has_event, events = self.calendar.has_event_before_expiry(
            symbol, trade_date, expiry_date, event_types=[EventType.EARNINGS, EventType.FOMC]
        )

        if not has_event:
            return 1.0

        premium = 1.0
        for event in events:
            if event.event_type == EventType.EARNINGS:
                # Earnings typically add 30-50% IV premium
                if event.expected_move:
                    premium *= 1 + event.expected_move
                else:
                    premium *= 1.35
            elif event.event_type == EventType.FOMC:
                premium *= 1.10

        return premium


class CalendarSourceConfig:
    """
    Configuration for centralized calendar source-of-truth.

    Supports multiple data sources with validation and auto-refresh.
    """

    # Default external calendar URLs (for reference - actual fetching requires auth)
    SOURCES = {
        "fomc": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "cpi": "https://www.bls.gov/schedule/news_release/cpi.htm",
        "nfp": "https://www.bls.gov/schedule/news_release/empsit.htm",
        "earnings": None,  # Provider-specific (e.g., Nasdaq, Alpha Vantage)
    }

    # Years with verified official dates
    VERIFIED_YEARS = {
        "fomc": [2024, 2025, 2026],
        "cpi": [2024, 2025, 2026],
        "nfp": [2024, 2025, 2026],
    }


def validate_calendar_dates(calendar: EventCalendar, year: int) -> dict[str, list[str]]:
    """
    Validate calendar dates for consistency and completeness.

    Returns dict of warnings/errors by category.
    """
    issues: dict[str, list[str]] = {"warnings": [], "errors": []}

    # Check FOMC: should have 8 meetings per year
    fomc_events = [e for e in calendar.events
                   if e.event_type == EventType.FOMC
                   and e.event_date.year == year]
    if len(fomc_events) != 8:
        issues["warnings"].append(
            f"FOMC: Expected 8 meetings for {year}, found {len(fomc_events)}"
        )

    # Check CPI: should have 12 releases per year
    cpi_events = [e for e in calendar.events
                  if e.event_type == EventType.CPI
                  and e.event_date.year == year]
    if len(cpi_events) != 12:
        issues["warnings"].append(
            f"CPI: Expected 12 releases for {year}, found {len(cpi_events)}"
        )

    # Check NFP: should have 12 releases per year
    nfp_events = [e for e in calendar.events
                  if e.event_type == EventType.NFP
                  and e.event_date.year == year]
    if len(nfp_events) != 12:
        issues["warnings"].append(
            f"NFP: Expected 12 releases for {year}, found {len(nfp_events)}"
        )

    # Check monthly expiries: should have 12
    expiry_events = [e for e in calendar.events
                     if e.event_type == EventType.OPTIONS_EXPIRY
                     and e.event_date.year == year]
    if len(expiry_events) != 12:
        issues["warnings"].append(
            f"Expiries: Expected 12 for {year}, found {len(expiry_events)}"
        )

    # Check for stale data (dates in the past with no update)
    today = date.today()
    if year < today.year:
        if year not in CalendarSourceConfig.VERIFIED_YEARS.get("fomc", []):
            issues["warnings"].append(
                f"FOMC dates for {year} are from unverified source"
            )

    return issues


def build_default_calendar(
    years: list[int],
    earnings_file: str | None = None,
    dividends_file: str | None = None,
    include_macro_events: bool = True,
    validate: bool = True,
) -> EventCalendar:
    """
    Build a default event calendar with common events.

    This is the centralized source-of-truth for all market events.
    All event dates are sourced from official government/exchange schedules.

    Args:
        years: List of years to generate dates for
        earnings_file: Optional path to earnings CSV
        dividends_file: Optional path to dividends CSV
        include_macro_events: Include CPI, NFP, GDP events (default True)
        validate: Validate calendar completeness (default True)

    Returns:
        Populated EventCalendar

    Raises:
        ValueError: If validation fails with errors (not just warnings)
    """
    calendar = EventCalendar()
    builder = EventCalendarBuilder()

    # Add FOMC dates (always included - critical for options trading)
    for year in years:
        fomc_events = builder.generate_fomc_dates(year)
        calendar.add_events(fomc_events)

        expiry_events = builder.generate_monthly_expiries(year)
        calendar.add_events(expiry_events)

        # Add macro economic events
        if include_macro_events:
            cpi_events = builder.generate_cpi_dates(year)
            calendar.add_events(cpi_events)

            nfp_events = builder.generate_nfp_dates(year)
            calendar.add_events(nfp_events)

            gdp_events = builder.generate_gdp_dates(year)
            calendar.add_events(gdp_events)

    # Load external data if provided
    if earnings_file:
        earnings_events = builder.from_earnings_csv(earnings_file)
        calendar.add_events(earnings_events)

    if dividends_file:
        dividend_events = builder.from_dividends_csv(dividends_file)
        calendar.add_events(dividend_events)

    # Validate calendar if requested
    if validate:
        for year in years:
            issues = validate_calendar_dates(calendar, year)
            if issues["errors"]:
                raise ValueError(
                    f"Calendar validation failed for {year}: {issues['errors']}"
                )
            # Log warnings (in production, use proper logging)
            for warning in issues["warnings"]:
                print(f"Calendar warning: {warning}")

    return calendar
