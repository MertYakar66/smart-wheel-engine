"""
Wheel Strategy Position Tracker
Manages the full lifecycle: Short Put → Stock Assignment → Covered Call → Exit
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional, List, Dict
import pandas as pd

from .transaction_costs import (
    calculate_total_entry_cost,
    calculate_total_exit_cost,
    calculate_assignment_costs
)


class PositionState(Enum):
    """State machine for Wheel positions"""
    NO_POSITION = "no_position"
    SHORT_PUT = "short_put"
    STOCK_OWNED = "stock_owned"
    COVERED_CALL = "covered_call"


@dataclass
class WheelPosition:
    """
    Tracks a single Wheel position through its lifecycle.
    All prices in dollars, P&L in dollars (not per-share).
    """
    ticker: str
    state: PositionState
    entry_date: date
    
    # Short put phase
    put_strike: Optional[float] = None
    put_premium: Optional[float] = None  # Per share
    put_entry_date: Optional[date] = None
    put_dte_at_entry: Optional[int] = None
    put_entry_iv: Optional[float] = None
    put_expiration_date: Optional[date] = None
    
    # Stock ownership phase
    stock_shares: int = 0
    stock_basis: Optional[float] = None  # Cost per share
    stock_acquisition_date: Optional[date] = None
    
    # Covered call phase
    call_strike: Optional[float] = None
    call_premium: Optional[float] = None  # Per share
    call_entry_date: Optional[date] = None
    call_dte_at_entry: Optional[int] = None
    call_entry_iv: Optional[float] = None
    call_expiration_date: Optional[date] = None
    
    # P&L tracking (cumulative, in dollars)
    realized_pnl: float = 0.0
    transaction_costs: float = 0.0
    
    # Metadata
    notes: List[str] = field(default_factory=list)


class WheelTracker:
    """
    Portfolio-level tracker for all Wheel positions.
    Handles state transitions and P&L accounting.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, WheelPosition] = {}
        self.closed_positions: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
    def open_short_put(
        self,
        ticker: str,
        strike: float,
        premium: float,
        entry_date: date,
        expiration_date: date,  # CHANGED: explicit date instead of dte
        iv: float
    ) -> bool:
        """
        Enter short put position.
        
        Args:
            ticker: Stock ticker
            strike: Put strike price
            premium: Premium collected per share
            entry_date: Trade entry date
            expiration_date: Explicit calendar expiration date
            iv: Implied volatility at entry
            
        Returns:
            True if position opened successfully
        """
        # Check if already have position in this ticker
        if ticker in self.positions:
            print(f"[SKIP] Already have position in {ticker}")
            return False
        
        # Check buying power (simplified: 20% of strike value as margin)
        margin_required = strike * 100 * 0.20
        if self.cash < margin_required:
            print(f"[SKIP] Insufficient buying power for {ticker} put")
            return False
        
        # Calculate entry costs (commission + slippage) and net premium collected
        cost_details = calculate_total_entry_cost(
            premium_per_share=premium,
            bid_ask_spread=premium * 0.10,  # temporary approx: 10% of option price
            trade_type="option"
        )

        premium_collected = cost_details["net_premium_collected"]
        commission = cost_details["commission"]
        slippage = cost_details["slippage"]

        # Credit net premium to cash (per-contract)
        self.cash += premium_collected
        
        # Create position (store explicit expiration date and derived DTE)
        derived_dte = (expiration_date - entry_date).days
        self.positions[ticker] = WheelPosition(
            ticker=ticker,
            state=PositionState.SHORT_PUT,
            entry_date=entry_date,
            put_strike=strike,
            put_premium=premium,
            put_entry_date=entry_date,
            put_dte_at_entry=derived_dte,  # Derived from dates
            put_entry_iv=iv,
            put_expiration_date=expiration_date,  # NEW: store explicit date
            realized_pnl=cost_details["gross_premium"],
            transaction_costs=cost_details["total_cost"]
        )

        self.positions[ticker].notes.append(
            f"Sold {derived_dte}d {strike}P for ${premium:.2f} premium"
        )
        
        return True
    
    def close_short_put(
        self,
        ticker: str,
        buyback_price: float,
        exit_date: date,
        reason: str = "early_exit"
    ) -> Optional[Dict]:
        """
        Buy back short put (early exit before expiration).
        
        Args:
            ticker: Stock ticker
            buyback_price: Price paid to buy back put (per share)
            exit_date: Exit date
            reason: Exit reason (profit_target, stop_loss, time_decay)
            
        Returns:
            Closed position summary dict
        """
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        if pos.state != PositionState.SHORT_PUT:
            return None
        
        # Calculate exit costs (buyback + slippage + commission)
        cost_details = calculate_total_exit_cost(
            buyback_price_per_share=buyback_price,
            bid_ask_spread=buyback_price * 0.10,
            trade_type="option"
        )

        gross_pnl = pos.put_premium * 100 - cost_details["gross_buyback_cost"]
        pos.realized_pnl = gross_pnl  # Store GROSS P&L
        pos.transaction_costs += cost_details["total_cost"]

        # Deduct total buyback cash (includes slippage and commission)
        self.cash -= cost_details["total_buyback_cost"]
        
        # Record closed position
        closed = self._finalize_position(pos, exit_date, reason)
        del self.positions[ticker]
        
        return closed
    
    def handle_put_assignment(
        self,
        ticker: str,
        assignment_date: date,
        stock_price: float
    ) -> bool:
        """
        Handle put assignment: acquire 100 shares at strike price.
        
        Args:
            ticker: Stock ticker
            assignment_date: Assignment date
            stock_price: Current stock price (for unrealized P&L tracking)
            
        Returns:
            True if assignment handled successfully
        """
        if ticker not in self.positions:
            return False
        
        pos = self.positions[ticker]
        if pos.state != PositionState.SHORT_PUT:
            return False
        
        # Acquire stock at strike price (include assignment fee)
        assignment_details = calculate_assignment_costs(
            strike_price=pos.put_strike,
            shares=100
        )

        stock_cost = assignment_details["stock_cost"]
        assignment_fee = assignment_details["assignment_fee"]
        
        if self.cash < assignment_details["total_cash_required"]:
            # In reality, broker would margin call or auto-liquidate
            # For simulation, we'll allow it but flag
            pos.notes.append(f"WARNING: Assignment required ${stock_cost:.2f}, only ${self.cash:.2f} available")
        
        self.cash -= assignment_details["total_cash_required"]
        
        # Update position state
        pos.state = PositionState.STOCK_OWNED
        pos.stock_shares = 100
        pos.stock_basis = pos.put_strike  # Basis is strike, not current price
        pos.stock_acquisition_date = assignment_date
        pos.transaction_costs += assignment_fee
        
        pos.notes.append(
            f"Assigned: Bought 100 shares at ${pos.put_strike:.2f} (market: ${stock_price:.2f})"
        )
        
        return True
    
    def handle_put_expiration(
        self,
        ticker: str,
        expiry_date: date,
        stock_price: float
    ) -> Optional[Dict]:
        """
        Handle put expiration: either assign or expire worthless.
        
        Args:
            ticker: Stock ticker
            expiry_date: Expiration date
            stock_price: Stock price at expiration
            
        Returns:
            Closed position summary if expired worthless, else None
        """
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        if pos.state != PositionState.SHORT_PUT:
            return None
        
        if stock_price < pos.put_strike:
            # Assigned
            self.handle_put_assignment(ticker, expiry_date, stock_price)
            return None  # Position remains open (now stock)
        else:
            # Expired worthless (keep full premium)
            pos.notes.append(f"Put expired worthless (stock at ${stock_price:.2f})")
            closed = self._finalize_position(pos, expiry_date, "put_expired_otm")
            del self.positions[ticker]
            return closed
    
    def open_covered_call(
        self,
        ticker: str,
        strike: float,
        premium: float,
        entry_date: date,
        expiration_date: date,  # CHANGED: explicit date
        iv: float
    ) -> bool:
        """
        Sell covered call on owned stock.
        
        Args:
            ticker: Stock ticker
            strike: Call strike price
            premium: Premium collected per share
            entry_date: Trade entry date
            expiration_date: Explicit calendar expiration date
            iv: Implied volatility at entry
            
        Returns:
            True if call opened successfully
        """
        if ticker not in self.positions:
            return False
        
        pos = self.positions[ticker]
        if pos.state != PositionState.STOCK_OWNED:
            return False
        
        # Calculate entry costs for covered call (commission + slippage)
        cost_details = calculate_total_entry_cost(
            premium_per_share=premium,
            bid_ask_spread=premium * 0.10,
            trade_type="option"
        )

        # Credit net premium to cash
        self.cash += cost_details["net_premium_collected"]

        # Update position state and store explicit expiration date
        derived_dte = (expiration_date - entry_date).days
        pos.state = PositionState.COVERED_CALL
        pos.call_strike = strike
        pos.call_premium = premium
        pos.call_entry_date = entry_date
        pos.call_dte_at_entry = derived_dte  # Derived
        pos.call_entry_iv = iv
        pos.call_expiration_date = expiration_date  # NEW: explicit date
        pos.realized_pnl += cost_details["gross_premium"]
        pos.transaction_costs += cost_details["total_cost"]

        pos.notes.append(
            f"Sold {derived_dte}d {strike}C for ${premium:.2f} premium"
        )
        
        return True

    def close_covered_call(
        self,
        ticker: str,
        buyback_price: float,
        exit_date: date,
        reason: str = "early_exit"
    ) -> Optional[Dict]:
        """
        Buy back an outstanding covered call early (before expiration).

        Returns a dict summarizing the buyback including the call-leg P&L.
        """
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if pos.state != PositionState.COVERED_CALL:
            return None

        # Compute entry premium collected (per-contract)
        premium_collected = (pos.call_premium or 0.0) * 100

        # Compute exit costs including slippage and commission
        cost_details = calculate_total_exit_cost(
            buyback_price_per_share=buyback_price,
            bid_ask_spread=buyback_price * 0.10,
            trade_type="option"
        )

        gross_pnl = premium_collected - cost_details["gross_buyback_cost"]
        pos.realized_pnl += gross_pnl
        pos.transaction_costs += cost_details["total_cost"]

        # Deduct total cash paid to buy back the call
        self.cash -= cost_details["total_buyback_cost"]

        # Revert to stock-owned state (we keep the shares)
        pos.state = PositionState.STOCK_OWNED
        pos.call_strike = None
        pos.call_premium = None
        pos.call_entry_date = None
        pos.call_dte_at_entry = None
        pos.call_entry_iv = None
        pos.call_expiration_date = None

        pos.notes.append(f"Bought back call for ${buyback_price:.2f} due to {reason}")

        return {
            'ticker': ticker,
            'call_leg_pnl': gross_pnl,
            'transaction_costs': cost_details["total_cost"],
            'cash_after': self.cash,
            'reason': reason
        }
    
    def handle_call_assignment(
        self,
        ticker: str,
        assignment_date: date
    ) -> Optional[Dict]:
        """
        Handle call assignment: sell stock at call strike, close Wheel cycle.
        
        Args:
            ticker: Stock ticker
            assignment_date: Assignment date
            
        Returns:
            Closed position summary
        """
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        if pos.state != PositionState.COVERED_CALL:
            return None
        
        # Sell stock at call strike (account for assignment fee)
        stock_proceeds = pos.call_strike * 100
        assignment_details = calculate_assignment_costs(
            strike_price=pos.call_strike,
            shares=100
        )
        assignment_fee = assignment_details["assignment_fee"]

        self.cash += stock_proceeds - assignment_fee
        
        # Calculate stock P&L
        stock_pnl = (pos.call_strike - pos.stock_basis) * 100
        pos.realized_pnl += stock_pnl
        pos.transaction_costs += assignment_fee
        
        pos.notes.append(
            f"Called away: Sold 100 shares at ${pos.call_strike:.2f} (basis: ${pos.stock_basis:.2f})"
        )
        pos.notes.append(
            f"Wheel cycle complete: Total P&L = ${pos.realized_pnl:.2f}"
        )
        
        # Close position
        closed = self._finalize_position(pos, assignment_date, "call_assigned")
        del self.positions[ticker]
        
        return closed
    
    def handle_call_expiration(
        self,
        ticker: str,
        expiry_date: date,
        stock_price: float
    ) -> bool:
        """
        Handle call expiration: either assign or keep stock.
        
        Args:
            ticker: Stock ticker
            expiry_date: Expiration date
            stock_price: Stock price at expiration
            
        Returns:
            True if handled successfully
        """
        if ticker not in self.positions:
            return False
        
        pos = self.positions[ticker]
        if pos.state != PositionState.COVERED_CALL:
            return False
        
        if stock_price > pos.call_strike:
            # Assigned
            self.handle_call_assignment(ticker, expiry_date)
        else:
            # Expired worthless, keep stock
            pos.state = PositionState.STOCK_OWNED
            pos.call_strike = None
            pos.call_premium = None
            pos.notes.append(
                f"Call expired worthless (stock at ${stock_price:.2f}), still holding shares"
            )
        
        return True
    
    def mark_to_market(self, current_date: date, prices: Dict[str, float], risk_free_rate: float = 0.04) -> float:
        """
        Calculate current portfolio value (cash + stock + option liabilities).
        
        Args:
            current_date: Current date for mark
            prices: Dict of {ticker: current_stock_price}
            risk_free_rate: Risk-free rate for option pricing (default 4%)
            
        Returns:
            Total portfolio value including option liabilities
        """
        from .option_pricer import estimate_option_price_from_iv

        total_value = self.cash

        for ticker, pos in self.positions.items():
            if ticker not in prices:
                continue

            stock_price = prices[ticker]

            # Add stock value (if owned)
            if pos.state in [PositionState.STOCK_OWNED, PositionState.COVERED_CALL]:
                total_value += stock_price * pos.stock_shares

            # Subtract short put liability
            if pos.state == PositionState.SHORT_PUT:
                if pos.put_expiration_date and current_date < pos.put_expiration_date:
                    days_to_expiry = (pos.put_expiration_date - current_date).days
                    if days_to_expiry > 0:
                        # TODO: Replace pos.put_entry_iv with actual daily IV when Bloomberg data available
                        put_value = estimate_option_price_from_iv(
                            underlying_price=stock_price,
                            strike=pos.put_strike,
                            dte=days_to_expiry,
                            iv=pos.put_entry_iv,  # Constant IV approximation
                            risk_free_rate=risk_free_rate,
                            option_type='put'
                        )
                        total_value -= put_value * 100  # Short = liability

            # Subtract short call liability
            if pos.state == PositionState.COVERED_CALL:
                if pos.call_expiration_date and current_date < pos.call_expiration_date:
                    days_to_expiry = (pos.call_expiration_date - current_date).days
                    if days_to_expiry > 0:
                        # TODO: Replace pos.call_entry_iv with actual daily IV when Bloomberg data available
                        call_value = estimate_option_price_from_iv(
                            underlying_price=stock_price,
                            strike=pos.call_strike,
                            dte=days_to_expiry,
                            iv=pos.call_entry_iv,  # Constant IV approximation
                            risk_free_rate=risk_free_rate,
                            option_type='call'
                        )
                        total_value -= call_value * 100  # Short = liability
        
        # Record equity curve
        self.equity_curve.append({
            'date': current_date,
            'portfolio_value': total_value,
            'cash': self.cash,
            'num_positions': len(self.positions)
        })
        
        return total_value
    
    def _finalize_position(self, pos: WheelPosition, exit_date: date, exit_reason: str) -> Dict:
        """Internal: Convert position to closed trade record"""
        closed = {
            'ticker': pos.ticker,
            'entry_date': pos.entry_date,
            'exit_date': exit_date,
            'exit_reason': exit_reason,
            'hold_days': (exit_date - pos.entry_date).days,
            'realized_pnl': pos.realized_pnl,
            'transaction_costs': pos.transaction_costs,
            # ADD COMMENT: Net P&L = Gross P&L - All Transaction Costs (computed once)
            'net_pnl': pos.realized_pnl - pos.transaction_costs,
            'put_premium': pos.put_premium * 100 if pos.put_premium else 0,
            'call_premium': pos.call_premium * 100 if pos.call_premium else 0,
            'notes': ' | '.join(pos.notes)
        }
        
        self.closed_positions.append(closed)
        return closed
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Generate performance report"""
        if not self.closed_positions:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.closed_positions)
        
        summary = {
            'total_trades': len(df),
            'winners': len(df[df['net_pnl'] > 0]),
            'losers': len(df[df['net_pnl'] < 0]),
            'win_rate': len(df[df['net_pnl'] > 0]) / len(df),
            'total_pnl': df['net_pnl'].sum(),
            'avg_pnl_per_trade': df['net_pnl'].mean(),
            'total_commissions': df['transaction_costs'].sum(),
            'largest_win': df['net_pnl'].max(),
            'largest_loss': df['net_pnl'].min(),
        }
        
        return pd.DataFrame([summary])