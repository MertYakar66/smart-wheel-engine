"""
Signal Context Builder

Bridges Bloomberg data with the signals module to generate trading signals.
Builds the context dict that signal generators require.

Usage:
    from engine.signal_context import build_entry_context, build_exit_context

    # For entry decision
    context = build_entry_context("AAPL", option_strike=145.0, option_expiry=date(2024, 3, 15))
    signal = aggregator.evaluate_entry(context)

    # For exit decision (existing position)
    context = build_exit_context(position)
    signal = aggregator.evaluate_exit(context)
"""

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_entry_context(
    ticker: str,
    option_strike: float,
    option_expiry: date,
    option_type: str = "put",
    use_live_data: bool = True,
    data_cache: dict | None = None,
) -> dict[str, Any]:
    """
    Build context dict for entry signal evaluation.

    Args:
        ticker: Stock ticker
        option_strike: Strike price of the option
        option_expiry: Expiration date
        option_type: "put" or "call"
        use_live_data: Whether to fetch live Bloomberg data
        data_cache: Optional cache of pre-loaded data

    Returns:
        Context dict for SignalAggregator.evaluate_entry()

    Context includes:
        - iv_rank: IV percentile (0-1)
        - trend_direction: Price trend (-1 to 1)
        - dte: Days to expiration
        - days_to_earnings: Days until next earnings
        - days_to_fomc: Days until next FOMC (placeholder)
        - prices: Recent price series (for trend calculation)
    """
    context = {
        "ticker": ticker,
        "strike": option_strike,
        "expiry": option_expiry,
        "option_type": option_type,
        "is_entry": True,
    }

    # Calculate DTE
    today = date.today()
    context["dte"] = (option_expiry - today).days

    # Load data
    if use_live_data:
        context.update(_get_live_context(ticker, data_cache))
    else:
        context.update(_get_historical_context(ticker, data_cache))

    return context


def build_exit_context(
    ticker: str,
    entry_credit: float,
    current_value: float,
    entry_date: date,
    expiry_date: date,
    use_live_data: bool = True,
    data_cache: dict | None = None,
) -> dict[str, Any]:
    """
    Build context dict for exit signal evaluation.

    Args:
        ticker: Stock ticker
        entry_credit: Premium collected at entry (per share)
        current_value: Current option value (per share)
        entry_date: Date position was opened
        expiry_date: Option expiration date
        use_live_data: Whether to fetch live Bloomberg data
        data_cache: Optional cache of pre-loaded data

    Returns:
        Context dict for SignalAggregator.evaluate_exit()

    Context includes:
        - entry_credit: Premium collected
        - current_value: Current option price
        - current_pnl_pct: P&L as percentage of max profit
        - dte: Days to expiration
        - hold_days: Days position has been held
    """
    context = {
        "ticker": ticker,
        "entry_credit": entry_credit,
        "current_value": current_value,
        "is_entry": False,
    }

    today = date.today()
    context["dte"] = (expiry_date - today).days
    context["hold_days"] = (today - entry_date).days

    # Calculate P&L percentage
    if entry_credit > 0:
        context["current_pnl_pct"] = (entry_credit - current_value) / entry_credit
    else:
        context["current_pnl_pct"] = 0

    # Add market context
    if use_live_data:
        context.update(_get_live_context(ticker, data_cache))
    else:
        context.update(_get_historical_context(ticker, data_cache))

    return context


def _get_live_context(ticker: str, cache: dict | None = None) -> dict[str, Any]:
    """
    Get live context data from Bloomberg.

    Returns partial context with live market data.
    """
    context = {}

    try:
        from data.bloomberg import BloombergConnector, get_live_iv_rank

        with BloombergConnector() as bbg:
            # Get quote
            quote = bbg.get_quote(f"{ticker} US Equity")
            context["current_price"] = quote.last
            context["iv_30d"] = quote.iv_30d

            # Get IV rank (combines live IV with historical data)
            iv_rank = get_live_iv_rank(ticker)
            context["iv_rank"] = iv_rank if iv_rank is not None else 0.5

            # Get earnings date
            earnings_date = bbg.get_earnings_date(f"{ticker} US Equity")
            if earnings_date:
                days_to_earnings = (earnings_date - date.today()).days
                context["days_to_earnings"] = days_to_earnings if days_to_earnings > 0 else None
            else:
                context["days_to_earnings"] = None

            # Trend calculation requires price history
            # Use cached or load from CSV
            context["trend_direction"] = _calculate_trend(ticker, cache)

    except ImportError:
        logger.warning("Bloomberg connector not available, using historical data")
        context.update(_get_historical_context(ticker, cache))
    except Exception as e:
        logger.error(f"Error getting live data for {ticker}: {e}")
        context.update(_get_historical_context(ticker, cache))

    # FOMC dates (placeholder - would need separate calendar)
    context["days_to_fomc"] = None

    return context


def _get_historical_context(ticker: str, cache: dict | None = None) -> dict[str, Any]:
    """
    Get context data from historical CSV files.

    Used for backtesting or when live data is unavailable.
    """
    context = {}

    try:
        from data.bloomberg_loader import (
            compute_iv_rank,
            load_bloomberg_earnings,
            load_bloomberg_iv_history,
            load_bloomberg_ohlcv,
        )

        # Load OHLCV for trend
        ohlcv = None
        if cache and "ohlcv" in cache and ticker in cache["ohlcv"]:
            ohlcv = cache["ohlcv"][ticker]
        else:
            ohlcv = load_bloomberg_ohlcv(ticker)

        if ohlcv is not None and not ohlcv.empty:
            context["current_price"] = float(ohlcv.iloc[-1]["Close"])
            context["prices"] = ohlcv["Close"].tail(30)
            context["trend_direction"] = _calculate_trend_from_prices(ohlcv["Close"].tail(20))
        else:
            context["current_price"] = None
            context["trend_direction"] = 0

        # Load IV history for IV rank
        iv_hist = None
        if cache and "iv_history" in cache and ticker in cache["iv_history"]:
            iv_hist = cache["iv_history"][ticker]
        else:
            iv_hist = load_bloomberg_iv_history(ticker)

        if iv_hist is not None:
            iv_rank = compute_iv_rank(iv_hist)
            context["iv_rank"] = iv_rank if iv_rank is not None else 0.5
            if "iv_atm_30d" in iv_hist.columns:
                context["iv_30d"] = float(iv_hist.iloc[-1]["iv_atm_30d"])
        else:
            context["iv_rank"] = 0.5

        # Load earnings for days_to_earnings
        earnings = None
        if cache and "earnings" in cache and ticker in cache["earnings"]:
            earnings = cache["earnings"][ticker]
        else:
            earnings = load_bloomberg_earnings(ticker)

        if earnings is not None and not earnings.empty:
            # Find next earnings date
            future_earnings = earnings[earnings["earnings_date"] > pd.Timestamp.now()]
            if not future_earnings.empty:
                next_earnings = future_earnings.iloc[0]["earnings_date"]
                days_to = (next_earnings - pd.Timestamp.now()).days
                context["days_to_earnings"] = days_to if days_to > 0 else None
            else:
                context["days_to_earnings"] = None
        else:
            context["days_to_earnings"] = None

    except ImportError as e:
        logger.warning(f"Could not import data loaders: {e}")
        context["iv_rank"] = 0.5
        context["trend_direction"] = 0
        context["days_to_earnings"] = None

    context["days_to_fomc"] = None
    return context


def _calculate_trend(ticker: str, cache: dict | None = None) -> float:
    """Calculate trend direction from recent prices."""
    try:
        from data.bloomberg_loader import load_bloomberg_ohlcv

        if cache and "ohlcv" in cache and ticker in cache["ohlcv"]:
            ohlcv = cache["ohlcv"][ticker]
        else:
            ohlcv = load_bloomberg_ohlcv(ticker)

        if ohlcv is None or len(ohlcv) < 20:
            return 0

        prices = ohlcv["Close"].tail(20)
        return _calculate_trend_from_prices(prices)

    except Exception:
        return 0


def _calculate_trend_from_prices(prices: pd.Series) -> float:
    """
    Calculate trend direction from price series.

    Returns value from -1 (strong downtrend) to +1 (strong uptrend).
    Uses linear regression slope normalized by volatility.
    """
    if len(prices) < 5:
        return 0

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Simple trend: cumulative return over period
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

    # Normalize by volatility (so same return in low-vol is stronger signal)
    volatility = returns.std() * np.sqrt(252)
    if volatility > 0:
        # Sharpe-like metric
        trend = total_return / (volatility * np.sqrt(len(prices) / 252))
    else:
        trend = total_return * 10  # Scale up if no volatility

    # Clip to [-1, 1]
    return float(np.clip(trend, -1, 1))


def build_batch_entry_contexts(
    candidates: list[dict], use_live_data: bool = True
) -> list[dict[str, Any]]:
    """
    Build contexts for multiple entry candidates efficiently.

    Args:
        candidates: List of dicts with keys: ticker, strike, expiry, option_type
        use_live_data: Whether to use live Bloomberg data

    Returns:
        List of context dicts for each candidate
    """
    # Pre-load data for all tickers to avoid repeated I/O
    tickers = list({c["ticker"] for c in candidates})

    cache = {"ohlcv": {}, "iv_history": {}, "earnings": {}}

    try:
        from data.bloomberg_loader import (
            load_bloomberg_earnings,
            load_bloomberg_iv_history,
            load_bloomberg_ohlcv,
        )

        for ticker in tickers:
            cache["ohlcv"][ticker] = load_bloomberg_ohlcv(ticker)
            cache["iv_history"][ticker] = load_bloomberg_iv_history(ticker)
            cache["earnings"][ticker] = load_bloomberg_earnings(ticker)

    except ImportError:
        logger.warning("Could not load historical data")

    # Build contexts
    contexts = []
    for candidate in candidates:
        ctx = build_entry_context(
            ticker=candidate["ticker"],
            option_strike=candidate["strike"],
            option_expiry=candidate["expiry"],
            option_type=candidate.get("option_type", "put"),
            use_live_data=use_live_data,
            data_cache=cache,
        )
        contexts.append(ctx)

    return contexts


def evaluate_wheel_opportunities(
    tickers: list[str],
    target_dte: int = 45,
    target_delta: float = -0.30,
    use_live_data: bool = True,
) -> pd.DataFrame:
    """
    Evaluate wheel strategy opportunities for a list of tickers.

    Combines Bloomberg data with signal framework to rank opportunities.

    Args:
        tickers: List of stock tickers to evaluate
        target_dte: Target days to expiration for puts
        target_delta: Target delta for puts (negative)
        use_live_data: Whether to fetch live Bloomberg data

    Returns:
        DataFrame with columns: ticker, strike, expiry, signal_strength,
        iv_rank, trend, days_to_earnings, recommendation
    """
    from .signals import create_default_aggregator

    aggregator = create_default_aggregator()
    results = []

    # Calculate target expiry
    target_expiry = date.today() + timedelta(days=target_dte)

    for ticker in tickers:
        try:
            # Get current price for strike calculation
            current_price = None
            if use_live_data:
                try:
                    from data.bloomberg import BloombergConnector

                    with BloombergConnector() as bbg:
                        quote = bbg.get_quote(f"{ticker} US Equity")
                        current_price = quote.last
                except Exception:
                    pass

            if current_price is None:
                from data.bloomberg_loader import load_bloomberg_ohlcv

                ohlcv = load_bloomberg_ohlcv(ticker)
                if ohlcv is not None and not ohlcv.empty:
                    current_price = float(ohlcv.iloc[-1]["Close"])

            if current_price is None:
                continue

            # Estimate OTM put strike (roughly delta -0.30)
            # Delta -0.30 is typically about 5-10% OTM
            strike = round(current_price * 0.95 / 5) * 5  # Round to nearest $5

            # Build context
            context = build_entry_context(
                ticker=ticker,
                option_strike=strike,
                option_expiry=target_expiry,
                option_type="put",
                use_live_data=use_live_data,
            )

            # Evaluate signal
            signal = aggregator.evaluate_entry(context)

            results.append(
                {
                    "ticker": ticker,
                    "price": current_price,
                    "strike": strike,
                    "expiry": target_expiry,
                    "dte": context.get("dte", target_dte),
                    "iv_rank": context.get("iv_rank", 0.5),
                    "trend": context.get("trend_direction", 0),
                    "days_to_earnings": context.get("days_to_earnings"),
                    "signal_value": signal.final_value,
                    "signal_strength": signal.final_signal.name,
                    "recommended": signal.action_recommended,
                    "explanation": signal.explanation[:100] if signal.explanation else "",
                }
            )

        except Exception as e:
            logger.error(f"Error evaluating {ticker}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("signal_value", ascending=False)

    return df
