"""
Bloomberg Data Integration for Event Calendar

Loads real earnings dates, dividend schedules, and macro events from
Bloomberg data files into the EventCalendar system.

Replaces hardcoded/projected dates with authoritative Bloomberg data.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from engine.event_calendar import (
    EventCalendar,
    EventImpact,
    EventType,
    MarketEvent,
)


def _normalize_ticker(bbg_ticker: str) -> str:
    """Convert Bloomberg ticker to standard: 'AAPL UW Equity' -> 'AAPL'."""
    t = str(bbg_ticker).strip()
    t = t.replace(" Equity", "").replace(" Index", "")
    parts = t.split()
    return parts[0] if parts else t


def load_earnings_from_bloomberg(
    filepath: str | Path = "data/bloomberg/sp500_earnings.csv",
    tickers: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[MarketEvent]:
    """
    Load earnings events from Bloomberg sp500_earnings.csv.

    Bloomberg format:
        year/period, announcement_date, announcement_time, earnings_eps,
        comparable_eps, estimate_eps, ticker

    Args:
        filepath: Path to Bloomberg earnings CSV
        tickers: Filter to specific tickers (normalized format, e.g. "AAPL")
        start_date: Only include earnings on or after this date
        end_date: Only include earnings on or before this date

    Returns:
        List of MarketEvent for earnings
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return []

    df = pd.read_csv(filepath)
    if df.empty:
        return []

    # Parse dates
    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")
    df = df.dropna(subset=["announcement_date"])

    # Normalize tickers
    df["symbol"] = df["ticker"].apply(_normalize_ticker)

    # Filter
    if tickers:
        df = df[df["symbol"].isin(tickers)]
    if start_date:
        df = df[df["announcement_date"].dt.date >= start_date]
    if end_date:
        df = df[df["announcement_date"].dt.date <= end_date]

    events = []
    for _, row in df.iterrows():
        event_date = row["announcement_date"].date()
        period = row.get("year/period", "")
        symbol = row["symbol"]

        # Determine time of day
        time_str = str(row.get("announcement_time", ""))
        if time_str:
            try:
                hour = int(time_str.split(":")[0])
                time_of_day = "pre" if hour < 9 else ("post" if hour >= 16 else "during")
            except (ValueError, IndexError):
                time_of_day = "post"
        else:
            time_of_day = "post"

        # Calculate expected move from EPS surprise history
        eps_actual = row.get("earnings_eps")
        eps_estimate = row.get("estimate_eps")
        expected_move = None
        if pd.notna(eps_actual) and pd.notna(eps_estimate) and eps_estimate != 0:
            surprise_pct = abs(eps_actual - eps_estimate) / abs(eps_estimate)
            expected_move = min(surprise_pct * 2, 0.15)  # Cap at 15%

        events.append(
            MarketEvent(
                event_date=event_date,
                event_type=EventType.EARNINGS,
                symbol=symbol,
                description=f"{symbol} {period} Earnings",
                impact=EventImpact.HIGH,
                expected_move=expected_move,
                time_of_day=time_of_day,
            )
        )

    return events


def load_dividends_from_bloomberg(
    filepath: str | Path = "data/bloomberg/sp500_dividends.csv",
    tickers: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[MarketEvent]:
    """
    Load dividend events from Bloomberg sp500_dividends.csv.

    Bloomberg format:
        declared_date, ex_date, record_date, payable_date,
        dividend_amount, dividend_frequency, dividend_type, ticker

    Returns both ex-date and pay-date events.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return []

    df = pd.read_csv(filepath)
    if df.empty:
        return []

    df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce")
    df["payable_date"] = pd.to_datetime(df["payable_date"], errors="coerce")
    df = df.dropna(subset=["ex_date"])

    df["symbol"] = df["ticker"].apply(_normalize_ticker)

    if tickers:
        df = df[df["symbol"].isin(tickers)]
    if start_date:
        df = df[df["ex_date"].dt.date >= start_date]
    if end_date:
        df = df[df["ex_date"].dt.date <= end_date]

    events = []
    for _, row in df.iterrows():
        symbol = row["symbol"]
        amount = row.get("dividend_amount", 0)
        div_type = row.get("dividend_type", "Regular Cash")

        # Ex-dividend event
        ex_date = row["ex_date"].date()
        events.append(
            MarketEvent(
                event_date=ex_date,
                event_type=EventType.DIVIDEND_EX,
                symbol=symbol,
                description=f"{symbol} Ex-Dividend ${amount:.3f} ({div_type})",
                impact=EventImpact.MEDIUM,
                dividend_amount=amount,
                time_of_day="pre",
            )
        )

        # Pay date event
        if pd.notna(row.get("payable_date")):
            pay_date = row["payable_date"].date()
            events.append(
                MarketEvent(
                    event_date=pay_date,
                    event_type=EventType.DIVIDEND_PAY,
                    symbol=symbol,
                    description=f"{symbol} Dividend Payment ${amount:.3f}",
                    impact=EventImpact.LOW,
                    dividend_amount=amount,
                )
            )

    return events


def build_calendar_from_bloomberg(
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    data_dir: str | Path = "data/bloomberg",
    include_macro: bool = True,
) -> EventCalendar:
    """
    Build a complete EventCalendar from Bloomberg data.

    Combines:
    - Earnings dates from sp500_earnings.csv
    - Dividend schedules from sp500_dividends.csv
    - FOMC/CPI/NFP from hardcoded calendar (or JSON if available)
    - Monthly options expiries (computed)

    Args:
        tickers: Filter to specific tickers (None = all)
        years: Filter to specific years (None = all available)
        data_dir: Path to Bloomberg data directory
        include_macro: Include FOMC/CPI/NFP events

    Returns:
        Populated EventCalendar
    """
    from engine.event_calendar import EventCalendarBuilder

    data_dir = Path(data_dir)
    calendar = EventCalendar()

    # Date range
    start_date = date(min(years), 1, 1) if years else None
    end_date = date(max(years), 12, 31) if years else None

    # Load earnings
    earnings = load_earnings_from_bloomberg(
        data_dir / "sp500_earnings.csv", tickers, start_date, end_date
    )
    calendar.add_events(earnings)

    # Load dividends
    dividends = load_dividends_from_bloomberg(
        data_dir / "sp500_dividends.csv", tickers, start_date, end_date
    )
    calendar.add_events(dividends)

    # Add macro events and expiries
    if include_macro and years:
        builder = EventCalendarBuilder()
        for year in years:
            calendar.add_events(builder.generate_fomc_dates(year))
            calendar.add_events(builder.generate_cpi_dates(year))
            calendar.add_events(builder.generate_nfp_dates(year))
            calendar.add_events(builder.generate_gdp_dates(year))
            calendar.add_events(builder.generate_monthly_expiries(year))

    return calendar


def get_discrete_dividends_for_option(
    ticker: str,
    as_of: date,
    expiry: date,
    data_dir: str | Path = "data/bloomberg",
) -> list[dict]:
    """
    Get discrete dividends between as_of and option expiry for CRR tree.

    Returns list of dicts compatible with engine.binomial_tree.DiscreteDividend.

    Args:
        ticker: Standard ticker (e.g. "AAPL")
        as_of: Current date (trade date)
        expiry: Option expiration date
        data_dir: Bloomberg data directory

    Returns:
        List of {"ex_date": date, "amount": float, "time_frac": float}
    """
    filepath = Path(data_dir) / "sp500_dividends.csv"
    if not filepath.exists():
        return []

    df = pd.read_csv(filepath)
    df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce")
    df = df.dropna(subset=["ex_date"])
    df["symbol"] = df["ticker"].apply(_normalize_ticker)

    # Filter to this ticker and date range
    mask = (
        (df["symbol"] == ticker)
        & (df["ex_date"].dt.date > as_of)
        & (df["ex_date"].dt.date <= expiry)
    )
    filtered = df[mask].sort_values("ex_date")

    total_days = (expiry - as_of).days
    results = []
    for _, row in filtered.iterrows():
        ex = row["ex_date"].date()
        days_to_ex = (ex - as_of).days
        results.append(
            {
                "ex_date": ex,
                "amount": row["dividend_amount"],
                "time_frac": days_to_ex / total_days if total_days > 0 else 0.0,
            }
        )

    return results


def get_current_risk_free_rate(
    as_of: str | date | None = None,
    tenor: str = "rate_3m",
    data_dir: str | Path = "data/bloomberg",
) -> float:
    """
    Get risk-free rate from treasury yield data.

    Args:
        as_of: Date to look up (None = latest available)
        tenor: Which tenor ("rate_3m", "rate_6m", "rate_2y", "rate_10y")
        data_dir: Bloomberg data directory

    Returns:
        Annual risk-free rate as decimal (e.g. 0.0435)
    """
    filepath = Path(data_dir) / "treasury_yields.csv"
    if not filepath.exists():
        return 0.05  # Fallback

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if tenor not in df.columns:
        return 0.05

    if as_of is not None:
        if isinstance(as_of, str):
            as_of = pd.Timestamp(as_of)
        df = df[df["date"] <= as_of]

    if df.empty:
        return 0.05

    rate = df[tenor].iloc[-1]
    return rate / 100.0 if rate > 1 else rate  # Handle both % and decimal formats
