"""
Simplified backtest simulator that works with limited data.

LIMITATIONS (until Bloomberg data arrives):
- Early exits use constant IV approximation (no daily IV available)
- Stop-loss/profit targets based on Black-Scholes reconstruction
- Execution assumes mid-price fills with fixed slippage

This is a PLACEHOLDER that will be upgraded when daily option prices arrive.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from engine.wheel_tracker import WheelTracker, PositionState
from engine.option_pricer import estimate_option_price_from_iv


class WheelBacktester:
    """
    Backtest engine for Wheel strategy.
    Uses constant IV approximation for daily valuations (temporary).
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        profit_target_pct: float = 0.60,
        stop_loss_multiple: float = 2.0,
        max_positions: int = 10,
        risk_free_rate: float = 0.04
    ):
        self.tracker = WheelTracker(initial_capital)
        self.profit_target_pct = profit_target_pct
        self.stop_loss_multiple = stop_loss_multiple
        self.max_positions = max_positions
        self.risk_free_rate = risk_free_rate
        
        self.trade_log = []
    
    def run_backtest(
        self,
        trade_universe: pd.DataFrame,
        ohlcv_data: dict,  # {ticker: DataFrame with 'date', 'close'}
        start_date: str,
        end_date: str
    ):
        """
        Run backtest over date range.
        
        Args:
            trade_universe: DataFrame with potential trades (from trade_universe.py)
            ohlcv_data: Dict of OHLCV DataFrames per ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        # Filter trades to date range
        trades = trade_universe[
            (trade_universe['date'] >= start_date) &
            (trade_universe['date'] <= end_date)
        ].copy()
        
        # Sort by date
        trades = trades.sort_values('date')
        
        # Group trades by date
        for trade_date, day_trades in trades.groupby('date'):
            self._process_day(trade_date, day_trades, ohlcv_data)
        
        # Final mark-to-market
        final_date = pd.to_datetime(end_date).date()
        final_prices = {
            ticker: ohlcv_data[ticker].loc[ohlcv_data[ticker]['date'] == final_date, 'close'].values[0]
            for ticker in ohlcv_data.keys()
            if len(ohlcv_data[ticker].loc[ohlcv_data[ticker]['date'] == final_date]) > 0
        }
        self.tracker.mark_to_market(final_date, final_prices)
        
        return self.tracker.get_performance_summary()
    
    def _process_day(self, current_date, day_trades, ohlcv_data):
        """Process a single day: check exits, manage stock, enter new trades"""

        # 1. Check existing positions for exits (stop-loss, profit target, expiration)
        self._check_exits(current_date, ohlcv_data)

        # 2. Sell covered calls on owned stock (complete Wheel cycle)
        self._manage_stock_positions(current_date, day_trades, ohlcv_data)

        # 3. Enter new short puts if we have capacity
        if len(self.tracker.positions) < self.max_positions:
            self._enter_new_trades(current_date, day_trades, ohlcv_data)

        # 4. Mark to market (include option liabilities)
        current_prices = self._get_current_prices(current_date, ohlcv_data)
        self.tracker.mark_to_market(current_date, current_prices, self.risk_free_rate)
    
    def _check_exits(self, current_date, ohlcv_data):
        """Check all open positions for exit conditions"""
        
        positions_to_close = []
        
        for ticker, pos in list(self.tracker.positions.items()):
            if ticker not in ohlcv_data:
                continue
            
            # Get current stock price
            price_df = ohlcv_data[ticker]
            price_row = price_df[price_df['date'] == current_date]
            if price_row.empty:
                continue
            
            current_stock_price = price_row['close'].values[0]
            
            # Handle SHORT PUT positions
            if pos.state.value == 'short_put':
                # Check if expiration (use explicit expiration date)
                if pos.put_expiration_date and current_date >= pos.put_expiration_date:
                    # Expiration logic
                    if current_stock_price < pos.put_strike:
                        # Assigned
                        self.tracker.handle_put_assignment(ticker, current_date, current_stock_price)
                    else:
                        # Expired worthless
                        self.tracker.handle_put_expiration(ticker, current_date, current_stock_price)
                    continue
                
                # APPROXIMATION: Use constant IV to estimate current option value
                days_remaining = (pos.put_expiration_date - current_date).days if pos.put_expiration_date else pos.put_dte_at_entry
                
                estimated_put_value = estimate_option_price_from_iv(
                    underlying_price=current_stock_price,
                    strike=pos.put_strike,
                    dte=days_remaining,
                    iv=pos.put_entry_iv,  # CONSTANT IV ASSUMPTION (limitation)
                    risk_free_rate=self.risk_free_rate,
                    option_type='put'
                )
                
                # Check profit target (60% of max profit)
                max_profit = pos.put_premium
                current_profit = pos.put_premium - estimated_put_value
                
                if current_profit >= self.profit_target_pct * max_profit:
                    self.tracker.close_short_put(ticker, estimated_put_value, current_date, "profit_target")
                    continue
                
                # Check stop-loss (2x entry credit)
                if estimated_put_value >= self.stop_loss_multiple * pos.put_premium:
                    self.tracker.close_short_put(ticker, estimated_put_value, current_date, "stop_loss")
                    continue
            
            # Handle COVERED CALL positions
            elif pos.state.value == 'covered_call':
                # Check expiration (use explicit expiration date)
                if pos.call_expiration_date and current_date >= pos.call_expiration_date:
                    if current_stock_price > pos.call_strike:
                        self.tracker.handle_call_assignment(ticker, current_date)
                    else:
                        self.tracker.handle_call_expiration(ticker, current_date, current_stock_price)
                    continue

                # Early exit logic for covered calls (similar to puts)
                days_remaining = (pos.call_expiration_date - current_date).days if pos.call_expiration_date else pos.call_dte_at_entry
                
                estimated_call_value = estimate_option_price_from_iv(
                    underlying_price=current_stock_price,
                    strike=pos.call_strike,
                    dte=days_remaining,
                    iv=pos.call_entry_iv,
                    risk_free_rate=self.risk_free_rate,
                    option_type='call'
                )
                
                current_profit = pos.call_premium - estimated_call_value
                if current_profit >= self.profit_target_pct * pos.call_premium:
                    # Buy back call (simplified: assume we can close, stay in stock)
                    # In reality, you'd then sell another call or exit stock
                    pass  # TODO: Implement call buyback logic
            
            # Handle STOCK OWNED (no call sold yet)
            elif pos.state.value == 'stock_owned':
                # Covered call logic handled in _manage_stock_positions()
                pass

    def _manage_stock_positions(self, current_date, day_trades, ohlcv_data):
        """
        For tickers where we own stock but no call is sold, look for covered call opportunities.
        
        Args:
            current_date: Current simulation date
            day_trades: Available trades for this date
            ohlcv_data: OHLCV data for price lookups
        """
        for ticker, pos in list(self.tracker.positions.items()):
            # Only manage positions where we own stock with no call sold
            if pos.state != PositionState.STOCK_OWNED:
                continue

            # Find available covered calls for this ticker on this date
            available_calls = day_trades[
                (day_trades['ticker'] == ticker) &
                (day_trades['strategy_leg'] == 'covered_call')
            ]

            if available_calls.empty:
                continue

            # Filter to OTM calls only (strike > current stock price)
            if ticker in ohlcv_data:
                price_df = ohlcv_data[ticker]
                price_row = price_df[price_df['date'] == current_date]
                if not price_row.empty:
                    current_price = price_row['close'].values[0]
                    available_calls = available_calls[available_calls['strike'] > current_price]

            if available_calls.empty:
                continue

            # Select call with highest premium (among OTM strikes)
            best_call = available_calls.sort_values('mid_price', ascending=False).iloc[0]

            # Get expiration date
            expiration_date = pd.to_datetime(best_call['expiration']).date()

            # Sell covered call
            success = self.tracker.open_covered_call(
                ticker=ticker,
                strike=best_call['strike'],
                premium=best_call['mid_price'],
                entry_date=current_date,
                expiration_date=expiration_date,
                iv=best_call['implied_vol']
            )

            if success:
                self.trade_log.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'open_covered_call',
                    'strike': best_call['strike'],
                    'premium': best_call['mid_price'],
                    'dte': (expiration_date - current_date).days
                })
    
    def _enter_new_trades(self, current_date, day_trades, ohlcv_data):
        """Enter new short put positions if capital available"""
        
        # Filter to short puts only (initial entry)
        puts = day_trades[day_trades['strategy_leg'] == 'short_put'].copy()
        
        # Sort by expected value (if available) or premium
        if 'expected_value' in puts.columns:
            puts = puts.sort_values('expected_value', ascending=False)
        else:
            puts = puts.sort_values('mid_price', ascending=False)
        
        # Enter trades until max positions reached
        for _, trade in puts.iterrows():
            if len(self.tracker.positions) >= self.max_positions:
                break
            
            ticker = trade['ticker']
            if ticker in self.tracker.positions:
                continue  # Already have position
            
            # Extract explicit expiration date from trade data
            expiration_date = pd.to_datetime(trade['expiration']).date()

            # Enter short put
            success = self.tracker.open_short_put(
                ticker=ticker,
                strike=trade['strike'],
                premium=trade['mid_price'],
                entry_date=current_date,
                expiration_date=expiration_date,
                iv=trade['implied_vol']
            )
            
            if success:
                self.trade_log.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'open_short_put',
                    'strike': trade['strike'],
                    'premium': trade['mid_price'],
                    'dte': trade['dte']
                })
    
    def _get_current_prices(self, current_date, ohlcv_data):
        """Get current stock prices for all tickers"""
        prices = {}
        for ticker, df in ohlcv_data.items():
            price_row = df[df['date'] == current_date]
            if not price_row.empty:
                prices[ticker] = price_row['close'].values[0]
        return prices