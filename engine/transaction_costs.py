"""
Transaction Cost Model for Wheel Strategy

Centralizes all trading cost calculations including commissions, slippage, and fees.
Enables consistent cost application across simulator and label generator.

Cost Model Assumptions:
- Commission: $0.65 per contract (typical retail broker rate)
- Slippage: Adaptive based on spread and liquidity
- Assignment: $5.00 per assignment (stock delivery fee)
- Fill model: Sell at mid - slippage, buy at mid + slippage

LIMITATIONS:
- Does not account for order size impact beyond basic liquidity adjustment
- Commission structure is simplified (no tiered pricing)
"""

from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)


# Default cost parameters (can be overridden)
DEFAULT_COMMISSION_PER_CONTRACT = 0.65
DEFAULT_ASSIGNMENT_FEE = 5.00
DEFAULT_SLIPPAGE_PCT = 0.15  # 15% of spread


def calculate_commission(trade_type: str = "option", num_contracts: int = 1) -> float:
    """
    Calculate commission for a trade.

    Args:
        trade_type: Type of trade (option, stock)
        num_contracts: Number of contracts (for options) or 100-share lots (for stock)

    Returns:
        Total commission in dollars
    """
    commission_schedule = {
        "option": DEFAULT_COMMISSION_PER_CONTRACT,
        "stock": 0.0,  # Most brokers now offer zero-commission stock trades
    }
    per_unit = commission_schedule.get(trade_type, DEFAULT_COMMISSION_PER_CONTRACT)
    return per_unit * num_contracts


def calculate_actual_spread(
    bid: Optional[float],
    ask: Optional[float],
    mid_price: Optional[float] = None,
    fallback_pct: float = 0.10
) -> float:
    """
    Calculate actual bid-ask spread from market data.

    Args:
        bid: Bid price (can be None or 0)
        ask: Ask price (can be None)
        mid_price: Mid price for fallback calculation
        fallback_pct: Fallback percentage if spread unavailable

    Returns:
        Spread in dollars
    """
    # If both bid and ask are valid
    if bid is not None and ask is not None and ask >= bid >= 0:
        return ask - bid

    # Fallback: use percentage of mid
    if mid_price is not None and mid_price > 0:
        return mid_price * fallback_pct

    # Last resort: use ask or bid as basis
    basis = ask if ask is not None and ask > 0 else (bid if bid is not None and bid > 0 else 0)
    return basis * fallback_pct


def calculate_slippage(
    mid_price: float,
    bid_ask_spread: float,
    trade_direction: Literal["buy", "sell"],
    open_interest: Optional[int] = None,
    volume: Optional[int] = None
) -> float:
    """
    Calculate slippage based on spread and liquidity indicators.

    Args:
        mid_price: Theoretical mid-point price
        bid_ask_spread: Width of bid-ask spread
        trade_direction: "buy" (pay slippage) or "sell" (lose slippage)
        open_interest: Option open interest (for liquidity adjustment)
        volume: Trading volume (for liquidity adjustment)

    Returns:
        Slippage amount in dollars (always positive)

    Note:
        Base slippage is 15% of spread width, adjusted for liquidity.
        For sells: execute at mid - slippage
        For buys: execute at mid + slippage
    """
    base_factor = DEFAULT_SLIPPAGE_PCT

    # Adjust for liquidity (increase slippage for illiquid options)
    if open_interest is not None:
        if open_interest < 50:
            base_factor *= 2.5  # Very illiquid
        elif open_interest < 100:
            base_factor *= 2.0
        elif open_interest < 500:
            base_factor *= 1.5

    # Adjust for wide spreads (already captured in spread, but add penalty for very wide)
    if mid_price > 0:
        spread_pct = bid_ask_spread / mid_price
        if spread_pct > 0.30:
            base_factor *= 1.5  # Penalize very wide spreads
        elif spread_pct > 0.50:
            base_factor *= 2.0  # Severe penalty

    # Cap slippage factor at 50% of spread
    base_factor = min(base_factor, 0.50)

    slippage_amount = bid_ask_spread * base_factor
    return abs(slippage_amount)


def calculate_assignment_fee() -> float:
    """
    Calculate fee charged when option is assigned/exercised.

    Returns:
        Assignment fee in dollars
    """
    return DEFAULT_ASSIGNMENT_FEE


def calculate_reg_t_margin_short_put(
    strike: float,
    underlying_price: float,
    premium: float
) -> float:
    """
    Calculate Reg-T margin requirement for short put.

    Reg-T formula for short puts:
    margin = max(
        20% of underlying - OTM amount + premium,
        10% of strike + premium,
        $100 per contract minimum
    )

    Args:
        strike: Put strike price
        underlying_price: Current underlying price
        premium: Premium collected per share

    Returns:
        Margin required per contract in dollars
    """
    otm_amount = max(0, strike - underlying_price)

    # Method 1: 20% of underlying minus OTM amount plus premium
    margin_1 = 0.20 * underlying_price * 100 - otm_amount * 100 + premium * 100

    # Method 2: 10% of strike plus premium
    margin_2 = 0.10 * strike * 100 + premium * 100

    # Minimum margin
    margin_3 = 100.0

    return max(margin_1, margin_2, margin_3)


def calculate_total_entry_cost(
    premium_per_share: float,
    bid_ask_spread: Optional[float] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    trade_type: str = "option",
    open_interest: Optional[int] = None,
    volume: Optional[int] = None
) -> dict:
    """
    Calculate all costs for opening a short option position.

    Args:
        premium_per_share: Premium collected per share (typically mid price)
        bid_ask_spread: Current bid-ask spread (if known)
        bid: Bid price (used to calculate spread if bid_ask_spread not provided)
        ask: Ask price (used to calculate spread if bid_ask_spread not provided)
        trade_type: Type of trade
        open_interest: Open interest for liquidity adjustment
        volume: Volume for liquidity adjustment

    Returns:
        Dict with commission, slippage, total_cost, and net_premium_collected
    """
    # Calculate spread
    if bid_ask_spread is None:
        bid_ask_spread = calculate_actual_spread(bid, ask, premium_per_share)

    commission = calculate_commission(trade_type)

    # When selling options, we receive less than mid due to slippage
    slippage = calculate_slippage(
        mid_price=premium_per_share,
        bid_ask_spread=bid_ask_spread,
        trade_direction="sell",
        open_interest=open_interest,
        volume=volume
    )

    # Convert per-share values to per-contract (100 shares)
    slippage_per_contract = slippage * 100
    gross_premium = premium_per_share * 100
    net_premium = gross_premium - slippage_per_contract

    return {
        "commission": commission,
        "slippage": slippage_per_contract,
        "total_cost": commission + slippage_per_contract,
        "gross_premium": gross_premium,
        "net_premium_collected": net_premium - commission,
        "effective_fill_price": premium_per_share - slippage  # Per share
    }


def calculate_total_exit_cost(
    buyback_price_per_share: float,
    bid_ask_spread: Optional[float] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    trade_type: str = "option",
    open_interest: Optional[int] = None,
    volume: Optional[int] = None
) -> dict:
    """
    Calculate all costs for closing a short option position via buyback.

    Args:
        buyback_price_per_share: Price to buy back option per share (mid price)
        bid_ask_spread: Current bid-ask spread (if known)
        bid: Bid price (used to calculate spread if not provided)
        ask: Ask price (used to calculate spread if not provided)
        trade_type: Type of trade
        open_interest: Open interest for liquidity adjustment
        volume: Volume for liquidity adjustment

    Returns:
        Dict with commission, slippage, total_cost, and gross_buyback_cost
    """
    # Calculate spread
    if bid_ask_spread is None:
        bid_ask_spread = calculate_actual_spread(bid, ask, buyback_price_per_share)

    commission = calculate_commission(trade_type)

    # When buying options, we pay more than mid due to slippage
    slippage = calculate_slippage(
        mid_price=buyback_price_per_share,
        bid_ask_spread=bid_ask_spread,
        trade_direction="buy",
        open_interest=open_interest,
        volume=volume
    )

    # Convert per-share values to per-contract (100 shares)
    slippage_per_contract = slippage * 100
    gross_cost = buyback_price_per_share * 100
    total_cost_with_slippage = gross_cost + slippage_per_contract

    return {
        "commission": commission,
        "slippage": slippage_per_contract,
        "total_cost": commission + slippage_per_contract,
        "gross_buyback_cost": gross_cost,
        "total_buyback_cost": total_cost_with_slippage + commission,
        "effective_fill_price": buyback_price_per_share + slippage  # Per share
    }


def calculate_assignment_costs(strike_price: float, shares: int = 100) -> dict:
    """
    Calculate costs when option is assigned and stock is delivered.

    Args:
        strike_price: Strike price for assignment
        shares: Number of shares (default 100 for one contract)

    Returns:
        Dict with assignment_fee, stock_cost, and total_cash_required
    """
    assignment_fee = calculate_assignment_fee()
    stock_cost = strike_price * shares

    return {
        "assignment_fee": assignment_fee,
        "stock_cost": stock_cost,
        "total_cash_required": stock_cost + assignment_fee
    }


def estimate_round_trip_cost(
    entry_premium: float,
    expected_exit_premium: float,
    entry_spread: Optional[float] = None,
    exit_spread: Optional[float] = None,
    open_interest: Optional[int] = None
) -> dict:
    """
    Estimate total round-trip costs for a trade.

    Useful for expected value calculations before entering a trade.

    Args:
        entry_premium: Expected entry premium per share
        expected_exit_premium: Expected exit premium per share
        entry_spread: Spread at entry
        exit_spread: Spread at exit (defaults to entry_spread)
        open_interest: OI for liquidity adjustment

    Returns:
        Dict with entry_costs, exit_costs, total_costs, and breakeven_move
    """
    if entry_spread is None:
        entry_spread = entry_premium * 0.10

    if exit_spread is None:
        exit_spread = entry_spread

    entry = calculate_total_entry_cost(
        premium_per_share=entry_premium,
        bid_ask_spread=entry_spread,
        open_interest=open_interest
    )

    exit_costs = calculate_total_exit_cost(
        buyback_price_per_share=expected_exit_premium,
        bid_ask_spread=exit_spread,
        open_interest=open_interest
    )

    total = entry["total_cost"] + exit_costs["total_cost"]

    return {
        "entry_costs": entry["total_cost"],
        "exit_costs": exit_costs["total_cost"],
        "total_costs": total,
        "cost_as_pct_of_premium": total / (entry_premium * 100) if entry_premium > 0 else 0,
        "breakeven_decay_needed": total / 100  # Per share decay needed to break even
    }
