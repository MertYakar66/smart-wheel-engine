"""
Macro Event Calendar - 2026 Official Release Schedules

This module contains official release dates from:
- Federal Reserve (FOMC meetings)
- BLS (CPI, Employment Situation, JOLTS)
- BEA (GDP, PCE/Personal Income)
- EIA (Weekly Petroleum Status Report)

All times are in Eastern Time (ET).

Sources:
- Fed: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- BLS: https://www.bls.gov/schedule/news_release/
- BEA: https://www.bea.gov/news/schedule
- EIA: Weekly Petroleum Status Report released Wednesdays at 10:30 AM ET
"""

from datetime import datetime, time, timedelta
from typing import List, Tuple
import uuid

from financial_news.schema import (
    ScheduledEvent, EventType, ImportanceLevel,
)


def _make_event_id(event_type: str, date: datetime) -> str:
    """Generate a deterministic event ID"""
    return f"{event_type}_{date.strftime('%Y%m%d')}"


# =============================================================================
# FOMC 2026 SCHEDULE
# =============================================================================

def get_fomc_2026() -> List[ScheduledEvent]:
    """
    2026 FOMC meeting schedule.

    Meetings with press conferences and SEP/dots are marked CRITICAL.
    Other meetings are marked HIGH.

    Source: Federal Reserve FOMC Calendar
    """
    # (date, has_press_conference, has_projections)
    fomc_meetings = [
        # January 27-28, 2026 - no press conf
        (datetime(2026, 1, 28, 14, 0), False, False),
        # March 17-18, 2026 - press conf + SEP
        (datetime(2026, 3, 18, 14, 0), True, True),
        # April 28-29, 2026 - no press conf
        (datetime(2026, 4, 29, 14, 0), False, False),
        # June 16-17, 2026 - press conf + SEP
        (datetime(2026, 6, 17, 14, 0), True, True),
        # July 28-29, 2026 - no press conf
        (datetime(2026, 7, 29, 14, 0), False, False),
        # September 15-16, 2026 - press conf + SEP
        (datetime(2026, 9, 16, 14, 0), True, True),
        # October 27-28, 2026 - no press conf
        (datetime(2026, 10, 28, 14, 0), False, False),
        # December 8-9, 2026 - press conf + SEP
        (datetime(2026, 12, 9, 14, 0), True, True),
    ]

    events = []
    for scheduled_at, has_presser, has_sep in fomc_meetings:
        title = "FOMC Decision"
        if has_sep:
            title += " + SEP/Dot Plot"
        if has_presser:
            title += " + Press Conference"

        importance = ImportanceLevel.CRITICAL if has_presser else ImportanceLevel.HIGH

        events.append(ScheduledEvent(
            event_id=_make_event_id("fomc", scheduled_at),
            source_id="fed",
            event_type=EventType.FOMC_DECISION,
            category_id="fed_rates",
            scheduled_at=scheduled_at,
            timezone="America/New_York",
            importance=importance,
            pre_run_offset_minutes=15,
            post_run_offset_minutes=5,
            title=title,
            description=f"FOMC {'with' if has_presser else 'without'} press conference",
            is_recurring=False,
        ))

    return events


# =============================================================================
# CPI 2026 SCHEDULE
# =============================================================================

def get_cpi_2026() -> List[ScheduledEvent]:
    """
    2026 CPI release schedule (Consumer Price Index).

    Released at 8:30 AM ET, typically second or third week of month.
    All CPI releases are CRITICAL importance.

    Source: BLS Release Calendar
    """
    # CPI release dates for 2026 (8:30 AM ET)
    cpi_dates = [
        datetime(2026, 1, 14, 8, 30),   # December 2025 CPI
        datetime(2026, 2, 12, 8, 30),   # January 2026 CPI
        datetime(2026, 3, 11, 8, 30),   # February 2026 CPI
        datetime(2026, 4, 14, 8, 30),   # March 2026 CPI
        datetime(2026, 5, 13, 8, 30),   # April 2026 CPI
        datetime(2026, 6, 11, 8, 30),   # May 2026 CPI
        datetime(2026, 7, 14, 8, 30),   # June 2026 CPI
        datetime(2026, 8, 12, 8, 30),   # July 2026 CPI
        datetime(2026, 9, 16, 8, 30),   # August 2026 CPI
        datetime(2026, 10, 13, 8, 30),  # September 2026 CPI
        datetime(2026, 11, 12, 8, 30),  # October 2026 CPI
        datetime(2026, 12, 10, 8, 30),  # November 2026 CPI
    ]

    events = []
    for i, scheduled_at in enumerate(cpi_dates):
        # Reference month is previous month
        ref_month = scheduled_at - timedelta(days=30)
        period = ref_month.strftime("%B %Y")

        events.append(ScheduledEvent(
            event_id=_make_event_id("cpi", scheduled_at),
            source_id="bls",
            event_type=EventType.CPI,
            category_id="inflation",
            scheduled_at=scheduled_at,
            timezone="America/New_York",
            importance=ImportanceLevel.CRITICAL,
            pre_run_offset_minutes=10,
            post_run_offset_minutes=3,
            title=f"CPI - {period}",
            description=f"Consumer Price Index for {period}",
            is_recurring=False,
        ))

    return events


# =============================================================================
# NFP 2026 SCHEDULE (Employment Situation)
# =============================================================================

def get_nfp_2026() -> List[ScheduledEvent]:
    """
    2026 Employment Situation (Nonfarm Payrolls) release schedule.

    Released at 8:30 AM ET, first Friday of each month.
    All NFP releases are CRITICAL importance.

    Source: BLS Release Calendar
    """
    # NFP release dates for 2026 (first Friday, 8:30 AM ET)
    nfp_dates = [
        datetime(2026, 1, 2, 8, 30),    # December 2025 jobs
        datetime(2026, 2, 6, 8, 30),    # January 2026 jobs
        datetime(2026, 3, 6, 8, 30),    # February 2026 jobs
        datetime(2026, 4, 3, 8, 30),    # March 2026 jobs
        datetime(2026, 5, 8, 8, 30),    # April 2026 jobs
        datetime(2026, 6, 5, 8, 30),    # May 2026 jobs
        datetime(2026, 7, 2, 8, 30),    # June 2026 jobs
        datetime(2026, 8, 7, 8, 30),    # July 2026 jobs
        datetime(2026, 9, 4, 8, 30),    # August 2026 jobs
        datetime(2026, 10, 2, 8, 30),   # September 2026 jobs
        datetime(2026, 11, 6, 8, 30),   # October 2026 jobs
        datetime(2026, 12, 4, 8, 30),   # November 2026 jobs
    ]

    events = []
    for scheduled_at in nfp_dates:
        ref_month = scheduled_at - timedelta(days=30)
        period = ref_month.strftime("%B %Y")

        events.append(ScheduledEvent(
            event_id=_make_event_id("nfp", scheduled_at),
            source_id="bls",
            event_type=EventType.NFP,
            category_id="labor",
            scheduled_at=scheduled_at,
            timezone="America/New_York",
            importance=ImportanceLevel.CRITICAL,
            pre_run_offset_minutes=10,
            post_run_offset_minutes=3,
            title=f"Employment Situation - {period}",
            description=f"Nonfarm payrolls and unemployment rate for {period}",
            is_recurring=False,
        ))

    return events


# =============================================================================
# GDP 2026 SCHEDULE
# =============================================================================

def get_gdp_2026() -> List[ScheduledEvent]:
    """
    2026 GDP release schedule.

    Three estimates per quarter: Advance, Second, Third.
    Released at 8:30 AM ET.
    Advance estimates are CRITICAL, revisions are HIGH.

    Source: BEA Release Schedule
    """
    # (date, estimate_type, quarter)
    gdp_releases = [
        # Q4 2025
        (datetime(2026, 1, 29, 8, 30), "advance", "Q4 2025"),
        (datetime(2026, 2, 26, 8, 30), "second", "Q4 2025"),
        (datetime(2026, 3, 26, 8, 30), "third", "Q4 2025"),
        # Q1 2026
        (datetime(2026, 4, 30, 8, 30), "advance", "Q1 2026"),
        (datetime(2026, 5, 28, 8, 30), "second", "Q1 2026"),
        (datetime(2026, 6, 25, 8, 30), "third", "Q1 2026"),
        # Q2 2026
        (datetime(2026, 7, 30, 8, 30), "advance", "Q2 2026"),
        (datetime(2026, 8, 27, 8, 30), "second", "Q2 2026"),
        (datetime(2026, 9, 24, 8, 30), "third", "Q2 2026"),
        # Q3 2026
        (datetime(2026, 10, 29, 8, 30), "advance", "Q3 2026"),
        (datetime(2026, 11, 25, 8, 30), "second", "Q3 2026"),
        (datetime(2026, 12, 23, 8, 30), "third", "Q3 2026"),
    ]

    events = []
    for scheduled_at, estimate_type, quarter in gdp_releases:
        importance = ImportanceLevel.CRITICAL if estimate_type == "advance" else ImportanceLevel.HIGH

        events.append(ScheduledEvent(
            event_id=_make_event_id(f"gdp_{estimate_type}", scheduled_at),
            source_id="bea",
            event_type=EventType.GDP if estimate_type == "advance" else EventType.GDP_REVISION,
            category_id="growth_consumer",
            scheduled_at=scheduled_at,
            timezone="America/New_York",
            importance=importance,
            pre_run_offset_minutes=10,
            post_run_offset_minutes=3,
            title=f"GDP {estimate_type.title()} - {quarter}",
            description=f"Gross Domestic Product {estimate_type} estimate for {quarter}",
            is_recurring=False,
        ))

    return events


# =============================================================================
# PCE 2026 SCHEDULE (Personal Income and Outlays)
# =============================================================================

def get_pce_2026() -> List[ScheduledEvent]:
    """
    2026 Personal Income and Outlays (PCE) release schedule.

    Contains PCE Price Index (Fed's preferred inflation measure).
    Released at 8:30 AM ET, typically last week of month.
    All PCE releases are HIGH importance (CRITICAL when before FOMC).

    Source: BEA Release Schedule
    """
    # PCE release dates for 2026 (8:30 AM ET)
    pce_dates = [
        datetime(2026, 1, 30, 8, 30),   # December 2025 PCE
        datetime(2026, 2, 27, 8, 30),   # January 2026 PCE
        datetime(2026, 3, 27, 8, 30),   # February 2026 PCE
        datetime(2026, 4, 30, 8, 30),   # March 2026 PCE
        datetime(2026, 5, 29, 8, 30),   # April 2026 PCE
        datetime(2026, 6, 26, 8, 30),   # May 2026 PCE
        datetime(2026, 7, 31, 8, 30),   # June 2026 PCE
        datetime(2026, 8, 28, 8, 30),   # July 2026 PCE
        datetime(2026, 9, 25, 8, 30),   # August 2026 PCE
        datetime(2026, 10, 30, 8, 30),  # September 2026 PCE
        datetime(2026, 11, 27, 8, 30),  # October 2026 PCE
        datetime(2026, 12, 23, 8, 30),  # November 2026 PCE
    ]

    events = []
    for scheduled_at in pce_dates:
        ref_month = scheduled_at - timedelta(days=30)
        period = ref_month.strftime("%B %Y")

        events.append(ScheduledEvent(
            event_id=_make_event_id("pce", scheduled_at),
            source_id="bea",
            event_type=EventType.PCE,
            category_id="inflation",
            scheduled_at=scheduled_at,
            timezone="America/New_York",
            importance=ImportanceLevel.HIGH,
            pre_run_offset_minutes=10,
            post_run_offset_minutes=3,
            title=f"PCE / Personal Income - {period}",
            description=f"Personal Consumption Expenditures Price Index for {period}",
            is_recurring=False,
        ))

    return events


# =============================================================================
# EIA PETROLEUM SCHEDULE
# =============================================================================

def get_eia_petroleum_schedule(year: int = 2026) -> List[ScheduledEvent]:
    """
    Generate EIA Weekly Petroleum Status Report schedule.

    Released every Wednesday at 10:30 AM ET.
    MEDIUM importance (HIGH when oil is moving).

    Source: EIA
    """
    events = []

    # Start from first Wednesday of the year
    start = datetime(year, 1, 1, 10, 30)
    # Find first Wednesday
    days_until_wednesday = (2 - start.weekday()) % 7
    current = start + timedelta(days=days_until_wednesday)

    week_num = 1
    while current.year == year:
        events.append(ScheduledEvent(
            event_id=_make_event_id(f"eia_wpsr_w{week_num}", current),
            source_id="eia",
            event_type=EventType.EIA_PETROLEUM,
            category_id="oil_energy",
            scheduled_at=current,
            timezone="America/New_York",
            importance=ImportanceLevel.MEDIUM,
            pre_run_offset_minutes=5,
            post_run_offset_minutes=5,
            title=f"EIA Petroleum Report - Week {week_num}",
            description="Weekly Petroleum Status Report including crude inventory",
            is_recurring=True,
            recurrence_rule="FREQ=WEEKLY;BYDAY=WE",
        ))
        current += timedelta(days=7)
        week_num += 1

    return events


# =============================================================================
# MACRO CALENDAR CLASS
# =============================================================================

class MacroCalendar:
    """
    Unified macro event calendar.

    Provides:
    - Full 2026 calendar with all major releases
    - Event lookup by date range
    - Pre-run and post-run event detection
    """

    def __init__(self):
        self._events: List[ScheduledEvent] = []
        self._load_2026_calendar()

    def _load_2026_calendar(self) -> None:
        """Load all 2026 events"""
        self._events.extend(get_fomc_2026())
        self._events.extend(get_cpi_2026())
        self._events.extend(get_nfp_2026())
        self._events.extend(get_gdp_2026())
        self._events.extend(get_pce_2026())
        self._events.extend(get_eia_petroleum_schedule(2026))

        # Sort by date
        self._events.sort(key=lambda e: e.scheduled_at)

    def get_all_events(self) -> List[ScheduledEvent]:
        """Get all events"""
        return self._events.copy()

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
        importance: ImportanceLevel = None,
    ) -> List[ScheduledEvent]:
        """Get events in a date range, optionally filtered by importance"""
        events = [e for e in self._events if start <= e.scheduled_at <= end]
        if importance:
            events = [e for e in events if e.importance == importance]
        return events

    def get_events_for_date(self, date: datetime) -> List[ScheduledEvent]:
        """Get all events for a specific date"""
        target_date = date.date()
        return [e for e in self._events if e.scheduled_at.date() == target_date]

    def get_upcoming_critical_events(self, hours: int = 24) -> List[ScheduledEvent]:
        """Get upcoming critical events (FOMC, CPI, NFP, GDP Advance)"""
        now = datetime.utcnow()
        end = now + timedelta(hours=hours)
        return [
            e for e in self._events
            if now <= e.scheduled_at <= end
            and e.importance == ImportanceLevel.CRITICAL
        ]

    def get_today_calendar_summary(self) -> str:
        """Get a text summary of today's events"""
        today_events = self.get_events_for_date(datetime.utcnow())

        if not today_events:
            return "No scheduled macro events today."

        lines = ["Today's Macro Calendar:"]
        for event in today_events:
            time_str = event.scheduled_at.strftime("%H:%M ET")
            imp_marker = "[!]" if event.importance == ImportanceLevel.CRITICAL else ""
            lines.append(f"  {time_str} {imp_marker} {event.title}")

        return "\n".join(lines)

    def get_tomorrow_calendar_summary(self) -> str:
        """Get a text summary of tomorrow's events"""
        tomorrow = datetime.utcnow() + timedelta(days=1)
        tomorrow_events = self.get_events_for_date(tomorrow)

        if not tomorrow_events:
            return "No scheduled macro events tomorrow."

        lines = ["Tomorrow's Macro Calendar:"]
        for event in tomorrow_events:
            time_str = event.scheduled_at.strftime("%H:%M ET")
            imp_marker = "[!]" if event.importance == ImportanceLevel.CRITICAL else ""
            lines.append(f"  {time_str} {imp_marker} {event.title}")

        return "\n".join(lines)

    def populate_database(self, db) -> int:
        """Populate a NewsDatabase with all events"""
        count = 0
        for event in self._events:
            try:
                db.add_scheduled_event(event)
                count += 1
            except Exception as e:
                pass  # Already exists or other error
        return count
