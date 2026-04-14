"""
Shared Valuation Module

This module provides consistent trade valuation logic used by BOTH:
1. The backtest simulator (for live simulation)
2. The label generator (for ML training labels)

CRITICAL: Any change to exit logic here affects both systems,
ensuring train-test consistency for ML models.

AUDIT NOTE (mark-to-market bias):
The original simulator marked every day of a trade with **constant entry IV**,
which systematically biases backtest results in any vol regime transition:
  * IV crush (vol collapses post-earnings or post-event) => true option value
    drops faster than the model estimates => backtest reports LATE profit
    exits and *understates* realized return (pessimistic bias).
  * IV spike (crash, macro shock) => true option value jumps above the
    model => backtest reports late stop-loss exits and *overstates* the loss
    that would actually have been avoided by a live trader (too pessimistic on
    the upside as well, because the model can't see the spike).

To eliminate the bias this module now optionally accepts an ``iv_trajectory``
series — an aligned pandas Series of IV (decimal) indexed by calendar date.
When supplied, the simulator uses that day's IV instead of ``entry_iv``.
If callers do not have a live IV path they should pass ``None`` and accept
the documented bias rather than silently relying on a flat surface.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd

from .option_pricer import estimate_option_price_from_iv
from .transaction_costs import (
    calculate_assignment_costs,
    calculate_total_entry_cost,
    calculate_total_exit_cost,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Result of simulating a trade to completion."""

    # Exit details
    exit_date: date
    exit_reason: (
        str  # 'profit_target', 'stop_loss', 'expiration_otm', 'expiration_itm', 'assignment'
    )
    exit_price: float  # Option price at exit (0 if expired/assigned)
    days_held: int

    # P&L breakdown
    gross_pnl: float  # Premium collected - buyback cost (before costs)
    entry_costs: float  # Commission + slippage on entry
    exit_costs: float  # Commission + slippage on exit (0 if expired)
    assignment_costs: float  # Fee if assigned
    net_pnl: float  # Gross - all costs

    # Position details at exit
    was_assigned: bool
    underlying_price_at_exit: float

    # Additional context
    max_profit_reached: float  # Highest profit during hold
    max_loss_reached: float  # Lowest profit (max loss) during hold
    iv_at_entry: float
    strike: float
    premium_collected: float


def calculate_actual_spread(bid: float, ask: float, fallback_pct: float = 0.10) -> float:
    """
    Calculate actual bid-ask spread, with fallback.

    Args:
        bid: Bid price
        ask: Ask price
        fallback_pct: Fallback percentage of mid if spread unavailable

    Returns:
        Spread in dollars
    """
    if pd.notna(bid) and pd.notna(ask) and ask >= bid >= 0:
        return ask - bid

    # Fallback: use percentage of mid
    mid = (bid + ask) / 2 if pd.notna(bid) and pd.notna(ask) else ask or bid or 0
    return mid * fallback_pct


def simulate_option_trade(
    option_type: Literal["put", "call"],
    strike: float,
    entry_premium: float,
    entry_date: date,
    expiration_date: date,
    entry_iv: float,
    ohlcv_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    profit_target_pct: float = 0.60,
    stop_loss_multiple: float = 2.0,
    entry_bid: float | None = None,
    entry_ask: float | None = None,
    dividend_yield: float = 0.0,
    iv_trajectory: pd.Series | None = None,
    early_assignment_on_div: bool = True,
    ex_div_date: date | None = None,
    expected_dividend: float = 0.0,
) -> TradeOutcome | None:
    """
    Simulate a short option trade from entry to exit.

    This function applies identical exit logic to what the simulator uses,
    ensuring that labels match backtest behavior exactly.

    Args:
        option_type: 'put' or 'call'
        strike: Option strike price
        entry_premium: Premium collected per share (mid price)
        entry_date: Trade entry date
        expiration_date: Option expiration date
        entry_iv: Implied volatility at entry (decimal, e.g., 0.25)
        ohlcv_df: OHLCV DataFrame with 'Date' and 'Close' columns
        risk_free_rate: Risk-free rate for BS pricing
        profit_target_pct: Exit when profit reaches this % of max (default 60%)
        stop_loss_multiple: Exit when loss reaches this multiple of premium (default 2x)
        entry_bid: Actual bid at entry (for spread calculation)
        entry_ask: Actual ask at entry (for spread calculation)
        dividend_yield: Continuous dividend yield
        iv_trajectory: Optional pandas Series of daily IV (decimal) indexed by
            date. When provided, mark-to-market uses that day's IV; when absent
            it falls back to ``entry_iv`` (documented biased mode). Missing
            dates are forward-filled from the previous observation to simulate
            stale quotes.
        early_assignment_on_div: If True and the option is a short ITM call,
            the simulator will force assignment on the day BEFORE ex-dividend
            when the remaining time value (estimated from BSM) is less than
            the ``expected_dividend``. This is the textbook American early
            exercise rule for calls on dividend-paying underlyings — skipping
            it biases short-call backtests upward.
        ex_div_date: Ex-dividend date during the holding period, if any.
        expected_dividend: Per-share dividend amount expected at ex_div_date.

    Returns:
        TradeOutcome dataclass or None if simulation failed
    """
    if ohlcv_df.empty:
        logger.warning("Empty OHLCV data, cannot simulate trade")
        return None

    # Normalize dates in OHLCV
    ohlcv = ohlcv_df.copy()
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"]).dt.date

    # Normalize IV trajectory index to python dates for alignment with OHLCV.
    iv_path_map: dict[date, float] = {}
    if iv_trajectory is not None and len(iv_trajectory) > 0:
        iv_series = iv_trajectory.copy()
        iv_series.index = pd.to_datetime(iv_series.index).date  # type: ignore[assignment]
        iv_path_map = iv_series.dropna().to_dict()

    # Calculate entry costs
    entry_spread = calculate_actual_spread(entry_bid, entry_ask)
    entry_cost_details = calculate_total_entry_cost(
        premium_per_share=entry_premium, bid_ask_spread=entry_spread, trade_type="option"
    )

    # AUDIT FIX: removed dead expression `entry_cost_details["net_premium_collected"]`
    # (was computed for its side-effect-free getter and immediately discarded).
    entry_costs = entry_cost_details["total_cost"]
    _last_known_iv = entry_iv

    # Track position through time
    max_profit = 0.0
    max_loss = 0.0

    # Get daily prices from entry to expiration
    date_range = ohlcv[
        (ohlcv["Date"] > entry_date) & (ohlcv["Date"] <= expiration_date)
    ].sort_values("Date")

    if date_range.empty:
        logger.warning(f"No price data between {entry_date} and {expiration_date}")
        return None

    # Simulate each day
    for _, row in date_range.iterrows():
        current_date = row["Date"]
        current_stock_price = row["Close"]

        # Calculate days remaining
        days_remaining = (expiration_date - current_date).days

        # Handle expiration
        if current_date >= expiration_date or days_remaining <= 0:
            # At expiration - check ITM/OTM
            if option_type == "put":
                is_itm = current_stock_price < strike
                intrinsic = max(0, strike - current_stock_price)
            else:  # call
                is_itm = current_stock_price > strike
                intrinsic = max(0, current_stock_price - strike)

            if is_itm:
                # Assignment
                assignment_details = calculate_assignment_costs(strike, 100)
                gross_pnl = entry_premium * 100 - intrinsic * 100
                return TradeOutcome(
                    exit_date=current_date,
                    exit_reason="assignment",
                    exit_price=intrinsic,
                    days_held=(current_date - entry_date).days,
                    gross_pnl=gross_pnl,
                    entry_costs=entry_costs,
                    exit_costs=0,  # No buyback
                    assignment_costs=assignment_details["assignment_fee"],
                    net_pnl=gross_pnl - entry_costs - assignment_details["assignment_fee"],
                    was_assigned=True,
                    underlying_price_at_exit=current_stock_price,
                    max_profit_reached=max_profit,
                    max_loss_reached=max_loss,
                    iv_at_entry=entry_iv,
                    strike=strike,
                    premium_collected=entry_premium * 100,
                )
            else:
                # Expired worthless - keep full premium
                gross_pnl = entry_premium * 100
                return TradeOutcome(
                    exit_date=current_date,
                    exit_reason="expiration_otm",
                    exit_price=0,
                    days_held=(current_date - entry_date).days,
                    gross_pnl=gross_pnl,
                    entry_costs=entry_costs,
                    exit_costs=0,
                    assignment_costs=0,
                    net_pnl=gross_pnl - entry_costs,
                    was_assigned=False,
                    underlying_price_at_exit=current_stock_price,
                    max_profit_reached=max_profit,
                    max_loss_reached=max_loss,
                    iv_at_entry=entry_iv,
                    strike=strike,
                    premium_collected=entry_premium * 100,
                )

        # Use the day's IV if an iv_trajectory was provided, otherwise fall
        # back to entry_iv (documented biased mode). Forward-fill with the
        # last known observed IV to simulate realistic stale-quote behaviour.
        if iv_path_map:
            if current_date in iv_path_map:
                _last_known_iv = iv_path_map[current_date]
            # else keep _last_known_iv from previous step
            iv_for_mark = _last_known_iv
        else:
            iv_for_mark = entry_iv

        estimated_value = estimate_option_price_from_iv(
            underlying_price=current_stock_price,
            strike=strike,
            dte=days_remaining,
            iv=iv_for_mark,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )

        # --------------------------------------------------------------
        # American early-assignment on dividend (short ITM call only).
        #
        # Textbook rule: a short call on a dividend-paying underlying can be
        # optimally early-exercised the day before ex-dividend when the
        # remaining time value is below the dividend. We implement the rule
        # exactly by computing time_value = option_value - intrinsic on the
        # day before ex-div and comparing to expected_dividend.
        # --------------------------------------------------------------
        if (
            early_assignment_on_div
            and option_type == "call"
            and ex_div_date is not None
            and expected_dividend > 0
        ):
            days_to_ex = (ex_div_date - current_date).days
            if days_to_ex == 1:
                intrinsic_now = max(0.0, current_stock_price - strike)
                time_value = max(0.0, estimated_value - intrinsic_now)
                if intrinsic_now > 0 and time_value < expected_dividend:
                    # Forced assignment at strike the day before ex-div.
                    assignment_details = calculate_assignment_costs(strike, 100)
                    # P&L: kept entry premium, paid intrinsic buy-back (lost
                    # intrinsic value), and skipped the dividend we would
                    # have collected as a stockholder.
                    gross_pnl = entry_premium * 100 - intrinsic_now * 100
                    return TradeOutcome(
                        exit_date=current_date,
                        exit_reason="early_assignment_div",
                        exit_price=intrinsic_now,
                        days_held=(current_date - entry_date).days,
                        gross_pnl=gross_pnl,
                        entry_costs=entry_costs,
                        exit_costs=0,
                        assignment_costs=assignment_details["assignment_fee"],
                        net_pnl=(
                            gross_pnl
                            - entry_costs
                            - assignment_details["assignment_fee"]
                            - expected_dividend * 100
                        ),
                        was_assigned=True,
                        underlying_price_at_exit=current_stock_price,
                        max_profit_reached=max_profit,
                        max_loss_reached=max_loss,
                        iv_at_entry=entry_iv,
                        strike=strike,
                        premium_collected=entry_premium * 100,
                    )

        # Current P&L (before exit costs)
        current_profit = entry_premium - estimated_value  # Per share
        current_profit_pct = current_profit / entry_premium if entry_premium > 0 else 0

        # Track max profit/loss
        max_profit = max(max_profit, current_profit * 100)
        max_loss = min(max_loss, current_profit * 100)

        # Check profit target
        if current_profit_pct >= profit_target_pct:
            # Exit at profit target
            exit_spread = estimated_value * 0.10  # Approximate exit spread
            exit_cost_details = calculate_total_exit_cost(
                buyback_price_per_share=estimated_value,
                bid_ask_spread=exit_spread,
                trade_type="option",
            )

            gross_pnl = entry_premium * 100 - exit_cost_details["gross_buyback_cost"]
            return TradeOutcome(
                exit_date=current_date,
                exit_reason="profit_target",
                exit_price=estimated_value,
                days_held=(current_date - entry_date).days,
                gross_pnl=gross_pnl,
                entry_costs=entry_costs,
                exit_costs=exit_cost_details["total_cost"],
                assignment_costs=0,
                net_pnl=gross_pnl - entry_costs - exit_cost_details["total_cost"],
                was_assigned=False,
                underlying_price_at_exit=current_stock_price,
                max_profit_reached=max_profit,
                max_loss_reached=max_loss,
                iv_at_entry=entry_iv,
                strike=strike,
                premium_collected=entry_premium * 100,
            )

        # Check stop loss
        if estimated_value >= stop_loss_multiple * entry_premium:
            # Exit at stop loss
            exit_spread = estimated_value * 0.10
            exit_cost_details = calculate_total_exit_cost(
                buyback_price_per_share=estimated_value,
                bid_ask_spread=exit_spread,
                trade_type="option",
            )

            gross_pnl = entry_premium * 100 - exit_cost_details["gross_buyback_cost"]
            return TradeOutcome(
                exit_date=current_date,
                exit_reason="stop_loss",
                exit_price=estimated_value,
                days_held=(current_date - entry_date).days,
                gross_pnl=gross_pnl,
                entry_costs=entry_costs,
                exit_costs=exit_cost_details["total_cost"],
                assignment_costs=0,
                net_pnl=gross_pnl - entry_costs - exit_cost_details["total_cost"],
                was_assigned=False,
                underlying_price_at_exit=current_stock_price,
                max_profit_reached=max_profit,
                max_loss_reached=max_loss,
                iv_at_entry=entry_iv,
                strike=strike,
                premium_collected=entry_premium * 100,
            )

    # Should not reach here, but handle edge case
    logger.warning(f"Trade simulation ended without exit for {option_type} {strike}")
    return None


def simulate_wheel_cycle(
    put_strike: float,
    put_premium: float,
    put_entry_date: date,
    put_expiration_date: date,
    put_iv: float,
    ohlcv_df: pd.DataFrame,
    call_selector: Callable | None = None,
    risk_free_rate: float = 0.04,
    profit_target_pct: float = 0.60,
    stop_loss_multiple: float = 2.0,
) -> dict[str, Any]:
    """
    Simulate a complete Wheel cycle (put -> potential assignment -> covered calls).

    This is a more complex simulation that handles the full Wheel strategy.

    Args:
        put_strike: Initial put strike
        put_premium: Put premium collected per share
        put_entry_date: Put entry date
        put_expiration_date: Put expiration date
        put_iv: Put implied volatility
        ohlcv_df: OHLCV data
        call_selector: Optional function to select covered calls (receives stock price, date)
        risk_free_rate: Risk-free rate
        profit_target_pct: Profit target for exits
        stop_loss_multiple: Stop loss multiple

    Returns:
        Dictionary with cycle results
    """
    results = {
        "put_outcome": None,
        "call_outcomes": [],
        "total_gross_pnl": 0,
        "total_costs": 0,
        "total_net_pnl": 0,
        "cycle_complete": False,
        "final_state": "put_open",
    }

    # Simulate put leg
    put_outcome = simulate_option_trade(
        option_type="put",
        strike=put_strike,
        entry_premium=put_premium,
        entry_date=put_entry_date,
        expiration_date=put_expiration_date,
        entry_iv=put_iv,
        ohlcv_df=ohlcv_df,
        risk_free_rate=risk_free_rate,
        profit_target_pct=profit_target_pct,
        stop_loss_multiple=stop_loss_multiple,
    )

    if put_outcome is None:
        return results

    results["put_outcome"] = put_outcome
    results["total_gross_pnl"] += put_outcome.gross_pnl
    results["total_costs"] += (
        put_outcome.entry_costs + put_outcome.exit_costs + put_outcome.assignment_costs
    )
    results["total_net_pnl"] += put_outcome.net_pnl

    if not put_outcome.was_assigned:
        # Put expired OTM or closed early - cycle complete
        results["cycle_complete"] = True
        results["final_state"] = "put_closed"
        return results

    # Put was assigned - now own stock at strike price
    results["final_state"] = "stock_owned"
    stock_basis = put_strike

    # If call_selector provided, simulate covered calls
    # This is a simplified version - full implementation would loop until called away
    if call_selector is not None:
        # Get call parameters from selector
        call_params = call_selector(put_outcome.underlying_price_at_exit, put_outcome.exit_date)
        if call_params:
            call_outcome = simulate_option_trade(
                option_type="call",
                strike=call_params["strike"],
                entry_premium=call_params["premium"],
                entry_date=call_params["entry_date"],
                expiration_date=call_params["expiration_date"],
                entry_iv=call_params["iv"],
                ohlcv_df=ohlcv_df,
                risk_free_rate=risk_free_rate,
                profit_target_pct=profit_target_pct,
                stop_loss_multiple=stop_loss_multiple,
            )

            if call_outcome:
                results["call_outcomes"].append(call_outcome)
                results["total_gross_pnl"] += call_outcome.gross_pnl
                results["total_costs"] += call_outcome.entry_costs + call_outcome.exit_costs
                results["total_net_pnl"] += call_outcome.net_pnl

                if call_outcome.was_assigned:
                    # Stock called away - add stock P&L
                    stock_pnl = (call_params["strike"] - stock_basis) * 100
                    results["total_gross_pnl"] += stock_pnl
                    results["total_net_pnl"] += stock_pnl
                    results["cycle_complete"] = True
                    results["final_state"] = "called_away"

    return results
