"""
Performance Metrics Module

Calculates comprehensive risk-adjusted performance metrics for backtests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    # Return metrics
    total_return: float
    annualized_return: float
    total_pnl: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    # P&L statistics
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_pnl_per_trade: float

    # Time statistics
    avg_hold_days: float
    avg_win_hold_days: float
    avg_loss_hold_days: float

    # Cost analysis
    total_transaction_costs: float
    cost_as_pct_of_pnl: float

    # Risk-adjusted
    calmar_ratio: float
    ulcer_index: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'total_pnl': self.total_pnl,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_pnl_per_trade': self.avg_pnl_per_trade,
            'avg_hold_days': self.avg_hold_days,
            'avg_win_hold_days': self.avg_win_hold_days,
            'avg_loss_hold_days': self.avg_loss_hold_days,
            'total_transaction_costs': self.total_transaction_costs,
            'cost_as_pct_of_pnl': self.cost_as_pct_of_pnl,
            'calmar_ratio': self.calmar_ratio,
            'ulcer_index': self.ulcer_index,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])


def calculate_returns(equity_curve: pd.DataFrame, initial_capital: float) -> pd.Series:
    """
    Calculate daily returns from equity curve.

    Args:
        equity_curve: DataFrame with 'portfolio_value' column
        initial_capital: Starting capital

    Returns:
        Series of daily returns
    """
    if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
        return pd.Series(dtype=float)

    values = equity_curve['portfolio_value'].values
    returns = np.diff(values) / values[:-1]
    return pd.Series(returns)


def calculate_max_drawdown(equity_curve: pd.DataFrame) -> tuple:
    """
    Calculate maximum drawdown and duration.

    Args:
        equity_curve: DataFrame with 'portfolio_value' column

    Returns:
        Tuple of (max_drawdown, duration_in_days)
    """
    if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
        return 0.0, 0

    values = equity_curve['portfolio_value'].values
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak

    max_dd = drawdown.min()

    # Calculate duration
    in_drawdown = drawdown < 0
    if not any(in_drawdown):
        return abs(max_dd), 0

    # Find longest drawdown period
    max_duration = 0
    current_duration = 0
    for dd in in_drawdown:
        if dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return abs(max_dd), max_duration


def calculate_ulcer_index(equity_curve: pd.DataFrame) -> float:
    """
    Calculate Ulcer Index (quadratic mean of drawdowns).
    Lower is better.

    Args:
        equity_curve: DataFrame with 'portfolio_value' column

    Returns:
        Ulcer Index value
    """
    if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
        return 0.0

    values = equity_curve['portfolio_value'].values
    peak = np.maximum.accumulate(values)
    drawdown_pct = ((values - peak) / peak) * 100  # In percentage

    return np.sqrt(np.mean(drawdown_pct ** 2))


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading days per year

    Returns:
        Sharpe ratio
    """
    if returns.empty or returns.std() == 0:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation only).

    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading days per year

    Returns:
        Sortino ratio
    """
    if returns.empty:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    downside_returns = returns[returns < 0]

    if downside_returns.empty or downside_returns.std() == 0:
        return 0.0 if excess_returns.mean() <= 0 else float('inf')

    downside_std = downside_returns.std()
    return excess_returns.mean() / downside_std * np.sqrt(periods_per_year)


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        trades: DataFrame with 'net_pnl' column

    Returns:
        Profit factor (>1 is profitable)
    """
    if trades.empty or 'net_pnl' not in trades.columns:
        return 0.0

    gross_profit = trades[trades['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = abs(trades[trades['net_pnl'] < 0]['net_pnl'].sum())

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_performance_report(
    closed_trades: List[Dict],
    equity_curve: List[Dict],
    initial_capital: float,
    risk_free_rate: float = 0.04
) -> PerformanceReport:
    """
    Calculate comprehensive performance report.

    Args:
        closed_trades: List of closed trade dicts
        equity_curve: List of equity curve dicts
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceReport dataclass
    """
    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()

    # Handle empty data
    if trades_df.empty:
        return PerformanceReport(
            total_return=0.0, annualized_return=0.0, total_pnl=0.0,
            volatility=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
            max_drawdown=0.0, max_drawdown_duration=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
            avg_pnl_per_trade=0.0,
            avg_hold_days=0.0, avg_win_hold_days=0.0, avg_loss_hold_days=0.0,
            total_transaction_costs=0.0, cost_as_pct_of_pnl=0.0,
            calmar_ratio=0.0, ulcer_index=0.0
        )

    # Calculate returns
    returns = calculate_returns(equity_df, initial_capital)

    # Return metrics
    final_value = equity_df['portfolio_value'].iloc[-1] if not equity_df.empty else initial_capital
    total_return = (final_value - initial_capital) / initial_capital
    num_days = len(equity_df) if not equity_df.empty else 1
    annualized_return = ((1 + total_return) ** (252 / max(num_days, 1))) - 1

    # Risk metrics
    volatility = returns.std() * np.sqrt(252) if not returns.empty else 0.0
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    max_dd, max_dd_duration = calculate_max_drawdown(equity_df)
    ulcer = calculate_ulcer_index(equity_df)

    # Trade statistics
    total_trades = len(trades_df)
    winners = trades_df[trades_df['net_pnl'] > 0]
    losers = trades_df[trades_df['net_pnl'] < 0]
    winning_trades = len(winners)
    losing_trades = len(losers)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    profit_factor = calculate_profit_factor(trades_df)

    # P&L statistics
    total_pnl = trades_df['net_pnl'].sum()
    avg_win = winners['net_pnl'].mean() if not winners.empty else 0.0
    avg_loss = losers['net_pnl'].mean() if not losers.empty else 0.0
    largest_win = trades_df['net_pnl'].max() if not trades_df.empty else 0.0
    largest_loss = trades_df['net_pnl'].min() if not trades_df.empty else 0.0
    avg_pnl = trades_df['net_pnl'].mean() if not trades_df.empty else 0.0

    # Time statistics
    avg_hold = trades_df['hold_days'].mean() if 'hold_days' in trades_df.columns else 0.0
    avg_win_hold = winners['hold_days'].mean() if 'hold_days' in winners.columns and not winners.empty else 0.0
    avg_loss_hold = losers['hold_days'].mean() if 'hold_days' in losers.columns and not losers.empty else 0.0

    # Cost analysis
    total_costs = trades_df['transaction_costs'].sum() if 'transaction_costs' in trades_df.columns else 0.0
    gross_pnl = trades_df['realized_pnl'].sum() if 'realized_pnl' in trades_df.columns else total_pnl + total_costs
    cost_pct = total_costs / gross_pnl if gross_pnl != 0 else 0.0

    # Risk-adjusted
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0

    return PerformanceReport(
        total_return=total_return,
        annualized_return=annualized_return,
        total_pnl=total_pnl,
        volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        avg_pnl_per_trade=avg_pnl,
        avg_hold_days=avg_hold,
        avg_win_hold_days=avg_win_hold,
        avg_loss_hold_days=avg_loss_hold,
        total_transaction_costs=total_costs,
        cost_as_pct_of_pnl=cost_pct,
        calmar_ratio=calmar,
        ulcer_index=ulcer
    )


def generate_trade_analysis(closed_trades: List[Dict]) -> pd.DataFrame:
    """
    Generate detailed trade-by-trade analysis.

    Args:
        closed_trades: List of closed trade dicts

    Returns:
        DataFrame with trade analysis
    """
    if not closed_trades:
        return pd.DataFrame()

    df = pd.DataFrame(closed_trades)

    # Add analysis columns
    if 'net_pnl' in df.columns:
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        df['is_winner'] = df['net_pnl'] > 0

    if 'realized_pnl' in df.columns and 'transaction_costs' in df.columns:
        df['cost_ratio'] = df['transaction_costs'] / df['realized_pnl'].abs()

    return df


def generate_monthly_returns(equity_curve: List[Dict]) -> pd.DataFrame:
    """
    Generate monthly returns table.

    Args:
        equity_curve: List of equity curve dicts

    Returns:
        DataFrame with monthly returns by year
    """
    if not equity_curve:
        return pd.DataFrame()

    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Resample to month-end
    monthly = df['portfolio_value'].resample('M').last()
    monthly_returns = monthly.pct_change()

    # Pivot to year x month format
    monthly_returns = monthly_returns.to_frame('return')
    monthly_returns['year'] = monthly_returns.index.year
    monthly_returns['month'] = monthly_returns.index.month

    pivot = monthly_returns.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

    return pivot
