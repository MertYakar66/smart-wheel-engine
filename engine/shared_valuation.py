"""
Shared Valuation Module

This module provides consistent trade valuation logic used by BOTH:
1. The backtest simulator (for live simulation)
2. The label generator (for ML training labels)

CRITICAL: Any change to exit logic here affects both systems,
ensuring train-test consistency for ML models.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import date
from typing import Optional, Literal, Dict, Any
import logging

from .option_pricer import estimate_option_price_from_iv
from .transaction_costs import (
    calculate_total_entry_cost,
    calculate_total_exit_cost,
    calculate_assignment_costs
)

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Result of simulating a trade to completion."""
    # Exit details
    exit_date: date
    exit_reason: str  # 'profit_target', 'stop_loss', 'expiration_otm', 'expiration_itm', 'assignment'
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
    option_type: Literal['put', 'call'],
    strike: float,
    entry_premium: float,
    entry_date: date,
    expiration_date: date,
    entry_iv: float,
    ohlcv_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    profit_target_pct: float = 0.60,
    stop_loss_multiple: float = 2.0,
    entry_bid: Optional[float] = None,
    entry_ask: Optional[float] = None,
    dividend_yield: float = 0.0
) -> Optional[TradeOutcome]:
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

    Returns:
        TradeOutcome dataclass or None if simulation failed
    """
    if ohlcv_df.empty:
        logger.warning("Empty OHLCV data, cannot simulate trade")
        return None

    # Normalize dates in OHLCV
    ohlcv = ohlcv_df.copy()
    ohlcv['Date'] = pd.to_datetime(ohlcv['Date']).dt.date

    # Calculate entry costs
    entry_spread = calculate_actual_spread(entry_bid, entry_ask)
    entry_cost_details = calculate_total_entry_cost(
        premium_per_share=entry_premium,
        bid_ask_spread=entry_spread,
        trade_type="option"
    )

    premium_collected_per_contract = entry_cost_details["net_premium_collected"]
    entry_costs = entry_cost_details["total_cost"]

    # Track position through time
    max_profit = 0.0
    max_loss = 0.0

    # Get daily prices from entry to expiration
    date_range = ohlcv[
        (ohlcv['Date'] > entry_date) &
        (ohlcv['Date'] <= expiration_date)
    ].sort_values('Date')

    if date_range.empty:
        logger.warning(f"No price data between {entry_date} and {expiration_date}")
        return None

    # Simulate each day
    for _, row in date_range.iterrows():
        current_date = row['Date']
        current_stock_price = row['Close']

        # Calculate days remaining
        days_remaining = (expiration_date - current_date).days

        # Handle expiration
        if current_date >= expiration_date or days_remaining <= 0:
            # At expiration - check ITM/OTM
            if option_type == 'put':
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
                    exit_reason='assignment',
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
                    premium_collected=entry_premium * 100
                )
            else:
                # Expired worthless - keep full premium
                gross_pnl = entry_premium * 100
                return TradeOutcome(
                    exit_date=current_date,
                    exit_reason='expiration_otm',
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
                    premium_collected=entry_premium * 100
                )

        # Estimate current option value using Black-Scholes
        # LIMITATION: Uses constant IV (entry IV) - will be improved with daily IV data
        estimated_value = estimate_option_price_from_iv(
            underlying_price=current_stock_price,
            strike=strike,
            dte=days_remaining,
            iv=entry_iv,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
            dividend_yield=dividend_yield
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
                trade_type="option"
            )

            gross_pnl = entry_premium * 100 - exit_cost_details["gross_buyback_cost"]
            return TradeOutcome(
                exit_date=current_date,
                exit_reason='profit_target',
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
                premium_collected=entry_premium * 100
            )

        # Check stop loss
        if estimated_value >= stop_loss_multiple * entry_premium:
            # Exit at stop loss
            exit_spread = estimated_value * 0.10
            exit_cost_details = calculate_total_exit_cost(
                buyback_price_per_share=estimated_value,
                bid_ask_spread=exit_spread,
                trade_type="option"
            )

            gross_pnl = entry_premium * 100 - exit_cost_details["gross_buyback_cost"]
            return TradeOutcome(
                exit_date=current_date,
                exit_reason='stop_loss',
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
                premium_collected=entry_premium * 100
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
    call_selector: Optional[callable] = None,
    risk_free_rate: float = 0.04,
    profit_target_pct: float = 0.60,
    stop_loss_multiple: float = 2.0
) -> Dict[str, Any]:
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
        'put_outcome': None,
        'call_outcomes': [],
        'total_gross_pnl': 0,
        'total_costs': 0,
        'total_net_pnl': 0,
        'cycle_complete': False,
        'final_state': 'put_open'
    }

    # Simulate put leg
    put_outcome = simulate_option_trade(
        option_type='put',
        strike=put_strike,
        entry_premium=put_premium,
        entry_date=put_entry_date,
        expiration_date=put_expiration_date,
        entry_iv=put_iv,
        ohlcv_df=ohlcv_df,
        risk_free_rate=risk_free_rate,
        profit_target_pct=profit_target_pct,
        stop_loss_multiple=stop_loss_multiple
    )

    if put_outcome is None:
        return results

    results['put_outcome'] = put_outcome
    results['total_gross_pnl'] += put_outcome.gross_pnl
    results['total_costs'] += put_outcome.entry_costs + put_outcome.exit_costs + put_outcome.assignment_costs
    results['total_net_pnl'] += put_outcome.net_pnl

    if not put_outcome.was_assigned:
        # Put expired OTM or closed early - cycle complete
        results['cycle_complete'] = True
        results['final_state'] = 'put_closed'
        return results

    # Put was assigned - now own stock at strike price
    results['final_state'] = 'stock_owned'
    stock_basis = put_strike

    # If call_selector provided, simulate covered calls
    # This is a simplified version - full implementation would loop until called away
    if call_selector is not None:
        # Get call parameters from selector
        call_params = call_selector(put_outcome.underlying_price_at_exit, put_outcome.exit_date)
        if call_params:
            call_outcome = simulate_option_trade(
                option_type='call',
                strike=call_params['strike'],
                entry_premium=call_params['premium'],
                entry_date=call_params['entry_date'],
                expiration_date=call_params['expiration_date'],
                entry_iv=call_params['iv'],
                ohlcv_df=ohlcv_df,
                risk_free_rate=risk_free_rate,
                profit_target_pct=profit_target_pct,
                stop_loss_multiple=stop_loss_multiple
            )

            if call_outcome:
                results['call_outcomes'].append(call_outcome)
                results['total_gross_pnl'] += call_outcome.gross_pnl
                results['total_costs'] += call_outcome.entry_costs + call_outcome.exit_costs
                results['total_net_pnl'] += call_outcome.net_pnl

                if call_outcome.was_assigned:
                    # Stock called away - add stock P&L
                    stock_pnl = (call_params['strike'] - stock_basis) * 100
                    results['total_gross_pnl'] += stock_pnl
                    results['total_net_pnl'] += stock_pnl
                    results['cycle_complete'] = True
                    results['final_state'] = 'called_away'

    return results
