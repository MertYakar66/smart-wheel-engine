"""
Transaction Cost Model for Wheel Strategy

Centralizes all trading cost calculations including commissions, slippage, and fees.
Enables consistent cost application across simulator and label generator.

Cost Model Assumptions:
- Commission: $0.65 per contract (typical retail broker rate)
- Slippage: 15% of bid-ask spread (market impact for small retail size)
- Assignment: $5.00 per assignment (stock delivery fee)
- Fill model: Sell at mid - slippage, buy at mid + slippage

LIMITATIONS:
- Slippage model assumes constant percentage regardless of liquidity
- Does not account for order size impact
- Commission structure is simplified (no tiered pricing)
"""

from typing import Literal


def calculate_commission(trade_type: str = "option") -> float:
    """
    Calculate per-contract commission.
    
    Args:
        trade_type: Type of trade (option, stock, etc.)
        
    Returns:
        Commission in dollars per contract
    """
    # Currently flat rate, parameterized for future enhancement
    commission_schedule = {
        "option": 0.65,
        "stock": 0.0,  # Most brokers now offer zero-commission stock trades
    }
    return commission_schedule.get(trade_type, 0.65)


def calculate_slippage(
    mid_price: float,
    bid_ask_spread: float,
    trade_direction: Literal["buy", "sell"]
) -> float:
    """
    Calculate slippage as percentage of bid-ask spread.
    
    Args:
        mid_price: Theoretical mid-point price
        bid_ask_spread: Width of bid-ask spread
        trade_direction: "buy" (pay slippage) or "sell" (lose slippage)
        
    Returns:
        Slippage amount in dollars (always positive)
        
    Note:
        Slippage is 15% of spread width applied in direction unfavorable to trader.
        For sells: execute at mid - slippage
        For buys: execute at mid + slippage
    """
    slippage_factor = 0.15  # 15% of spread
    slippage_amount = bid_ask_spread * slippage_factor
    
    # Slippage is always a cost, return absolute value
    return abs(slippage_amount)


def calculate_assignment_fee() -> float:
    """
    Calculate fee charged when option is assigned/exercised.
    
    Returns:
        Assignment fee in dollars
    """
    return 5.0


def calculate_total_entry_cost(
    premium_per_share: float,
    bid_ask_spread: float,
    trade_type: str = "option"
) -> dict:
    """
    Calculate all costs for opening a short option position.
    
    Args:
        premium_per_share: Premium collected per share
        bid_ask_spread: Current bid-ask spread
        trade_type: Type of trade
        
    Returns:
        Dict with commission, slippage, total_cost, and net_premium_collected
    """
    commission = calculate_commission(trade_type)
    
    # When selling options, we receive less than mid due to slippage
    slippage = calculate_slippage(
        mid_price=premium_per_share,
        bid_ask_spread=bid_ask_spread,
        trade_direction="sell"
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
        "net_premium_collected": net_premium - commission
    }


def calculate_total_exit_cost(
    buyback_price_per_share: float,
    bid_ask_spread: float,
    trade_type: str = "option"
) -> dict:
    """
    Calculate all costs for closing a short option position via buyback.
    
    Args:
        buyback_price_per_share: Price to buy back option per share
        bid_ask_spread: Current bid-ask spread
        trade_type: Type of trade
        
    Returns:
        Dict with commission, slippage, total_cost, and gross_buyback_cost
    """
    commission = calculate_commission(trade_type)
    
    # When buying options, we pay more than mid due to slippage
    slippage = calculate_slippage(
        mid_price=buyback_price_per_share,
        bid_ask_spread=bid_ask_spread,
        trade_direction="buy"
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
        "total_buyback_cost": total_cost_with_slippage + commission
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
