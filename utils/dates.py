"""
Date handling utilities with consistent normalization.
"""
from datetime import date, datetime
from typing import Union, Optional, List
import pandas as pd


# US Market holidays (simplified - for production use pandas_market_calendars)
US_MARKET_HOLIDAYS_2024 = {
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
}

US_MARKET_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
}

ALL_HOLIDAYS = US_MARKET_HOLIDAYS_2024 | US_MARKET_HOLIDAYS_2025


def normalize_date(d: Union[str, date, datetime, pd.Timestamp, None]) -> Optional[date]:
    """
    Convert any date-like object to datetime.date.

    Args:
        d: Date in various formats

    Returns:
        datetime.date object or None if input is None/NaT
    """
    if d is None:
        return None
    if pd.isna(d):
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, str):
        try:
            return pd.to_datetime(d).date()
        except Exception:
            return None
    raise TypeError(f"Cannot convert {type(d)} to date")


def is_weekend(d: date) -> bool:
    """Check if date is a weekend (Saturday=5, Sunday=6)."""
    return d.weekday() >= 5


def is_trading_day(d: date) -> bool:
    """
    Check if date is a US market trading day.

    Args:
        d: Date to check

    Returns:
        True if trading day
    """
    if is_weekend(d):
        return False
    if d in ALL_HOLIDAYS:
        return False
    return True


def get_previous_trading_day(d: date) -> date:
    """Get the most recent trading day on or before d."""
    while not is_trading_day(d):
        d = date.fromordinal(d.toordinal() - 1)
    return d


def get_next_trading_day(d: date) -> date:
    """Get the next trading day on or after d."""
    while not is_trading_day(d):
        d = date.fromordinal(d.toordinal() + 1)
    return d


def trading_days_between(start: date, end: date) -> int:
    """
    Count trading days between two dates (exclusive of start, inclusive of end).

    Args:
        start: Start date
        end: End date

    Returns:
        Number of trading days
    """
    if end <= start:
        return 0

    count = 0
    current = date.fromordinal(start.toordinal() + 1)
    while current <= end:
        if is_trading_day(current):
            count += 1
        current = date.fromordinal(current.toordinal() + 1)
    return count


def calendar_to_trading_dte(calendar_dte: int, reference_date: date) -> int:
    """
    Convert calendar DTE to approximate trading DTE.

    Args:
        calendar_dte: Days to expiration in calendar days
        reference_date: Reference date for calculation

    Returns:
        Approximate trading days to expiration
    """
    if calendar_dte <= 0:
        return 0

    end_date = date.fromordinal(reference_date.toordinal() + calendar_dte)
    return trading_days_between(reference_date, end_date)


def dte_to_years(dte: int, use_trading_days: bool = False) -> float:
    """
    Convert DTE to fraction of year for Black-Scholes.

    Args:
        dte: Days to expiration
        use_trading_days: If True, use 252 trading days per year

    Returns:
        Time in years
    """
    if dte <= 0:
        return 0.0
    days_per_year = 252.0 if use_trading_days else 365.0
    return dte / days_per_year
