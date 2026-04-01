"""
Wheel Strategy Backtesting Engine.

Simulates wheel strategy execution with:
- ML-based entry signals
- Realistic option pricing
- Transaction costs
- Position management
"""

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.option_pricer import estimate_option_price_from_iv
from engine.signals import create_default_aggregator
from engine.wheel_tracker import PositionState, WheelTracker


@dataclass
class BacktestConfig:
    """Configuration for wheel backtest."""

    # Capital
    initial_capital: float = 100_000
    max_positions: int = 10
    max_position_pct: float = 0.15  # Max 15% per position

    # Entry criteria
    min_entry_score: float = 0.6  # ML model threshold
    min_iv_rank: float = 0.30  # Min IV rank for entry
    target_delta: float = 0.30  # Target put delta
    target_dte: int = 30  # Target DTE

    # Exit criteria
    profit_target: float = 0.50  # Take profit at 50% max gain
    stop_loss: float = 2.0  # Stop at 2x premium
    min_exit_dte: int = 5  # Exit if DTE < 5

    # Covered call
    cc_delta: float = 0.30  # Covered call delta
    cc_dte: int = 30  # Covered call DTE

    # Risk
    avoid_earnings: bool = True
    earnings_buffer_days: int = 5


@dataclass
class BacktestResult:
    """Results from backtest run."""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    config: BacktestConfig


class WheelBacktest:
    """
    Event-driven wheel strategy backtester.
    """

    def __init__(self, config: BacktestConfig | None = None, model=None, signal_aggregator=None):
        self.config = config or BacktestConfig()
        self.model = model  # Optional ML model for entry scoring
        self.aggregator = signal_aggregator or create_default_aggregator()

    def run(
        self, price_data: pd.DataFrame, start_date: date | None = None, end_date: date | None = None
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            price_data: DataFrame with columns [date, ticker, open, high, low, close, volume, ...]
            start_date: Start date for simulation
            end_date: End date for simulation

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        # Initialize tracker
        tracker = WheelTracker(initial_capital=self.config.initial_capital)

        # Prepare data
        data = price_data.copy()
        data["date"] = pd.to_datetime(data["date"]).dt.date

        if start_date:
            data = data[data["date"] >= start_date]
        if end_date:
            data = data[data["date"] <= end_date]

        dates = sorted(data["date"].unique())
        tickers = data["ticker"].unique()

        # Pre-compute ticker data for faster access
        ticker_data = {t: data[data["ticker"] == t].set_index("date") for t in tickers}

        equity_history = []

        for current_date in dates:
            # Get current prices for all tickers
            prices_today = {}
            for ticker in tickers:
                if current_date in ticker_data[ticker].index:
                    prices_today[ticker] = ticker_data[ticker].loc[current_date, "close"]

            # 1. Process expirations and exits
            self._process_expirations(tracker, current_date, prices_today, ticker_data)

            # 2. Process new entries
            self._process_entries(tracker, current_date, prices_today, ticker_data)

            # 3. Mark to market
            portfolio_value = tracker.mark_to_market(current_date, prices_today)

            equity_history.append(
                {
                    "date": current_date,
                    "portfolio_value": portfolio_value,
                    "cash": tracker.cash,
                    "num_positions": len(tracker.positions),
                }
            )

        # Build results
        equity_df = pd.DataFrame(equity_history)
        trades_df = (
            pd.DataFrame(tracker.closed_positions) if tracker.closed_positions else pd.DataFrame()
        )

        metrics = self._compute_metrics(equity_df, trades_df)

        return BacktestResult(
            equity_curve=equity_df, trades=trades_df, metrics=metrics, config=self.config
        )

    def _process_expirations(
        self,
        tracker: WheelTracker,
        current_date: date,
        prices: dict[str, float],
        ticker_data: dict[str, pd.DataFrame],
    ):
        """Handle option expirations and early exits."""
        positions_to_check = list(tracker.positions.keys())

        for ticker in positions_to_check:
            if ticker not in tracker.positions:
                continue

            pos = tracker.positions[ticker]
            price = prices.get(ticker, 0)

            if pos.state == PositionState.SHORT_PUT:
                # Check put expiration
                if pos.put_expiration_date and current_date >= pos.put_expiration_date:
                    tracker.handle_put_expiration(ticker, current_date, price)
                    continue

                # Check early exit conditions
                if self._should_exit_put(pos, current_date, price, ticker_data.get(ticker)):
                    # Calculate current put value for buyback
                    days_left = (
                        (pos.put_expiration_date - current_date).days
                        if pos.put_expiration_date
                        else 1
                    )
                    buyback_price = self._estimate_put_price(
                        pos.put_strike, price, days_left, pos.put_entry_iv or 0.25
                    )
                    tracker.close_short_put(ticker, buyback_price, current_date, "early_exit")

            elif pos.state == PositionState.COVERED_CALL:
                # Check call expiration
                if pos.call_expiration_date and current_date >= pos.call_expiration_date:
                    tracker.handle_call_expiration(ticker, current_date, price)
                    continue

            elif pos.state == PositionState.STOCK_OWNED:
                # Sell covered call on owned stock
                if price > 0:
                    self._try_sell_covered_call(
                        tracker, ticker, current_date, price, ticker_data.get(ticker)
                    )

    def _process_entries(
        self,
        tracker: WheelTracker,
        current_date: date,
        prices: dict[str, float],
        ticker_data: dict[str, pd.DataFrame],
    ):
        """Process new put entries."""
        # Check if we can take more positions
        if len(tracker.positions) >= self.config.max_positions:
            return

        # Calculate available capital per position
        portfolio_value = tracker.cash + sum(
            prices.get(t, 0) * pos.stock_shares
            for t, pos in tracker.positions.items()
            if pos.state in [PositionState.STOCK_OWNED, PositionState.COVERED_CALL]
        )
        max_per_position = portfolio_value * self.config.max_position_pct

        # Score and rank candidates
        candidates = []
        for ticker, df in ticker_data.items():
            if ticker in tracker.positions:
                continue

            if current_date not in df.index:
                continue

            row = df.loc[current_date]
            price = row["close"]

            # Skip if position would be too large
            strike = price * (1 - 0.05)  # ~5% OTM
            notional = strike * 100
            if notional > max_per_position:
                continue

            # Score candidate
            score = self._score_entry(row, current_date, df)
            if score >= self.config.min_entry_score:
                candidates.append((ticker, price, row, score))

        # Sort by score and enter top candidates
        candidates.sort(key=lambda x: x[3], reverse=True)

        for ticker, price, row, _score in candidates:
            if len(tracker.positions) >= self.config.max_positions:
                break

            # Calculate put parameters
            strike = round(price * (1 - 0.05), 2)  # 5% OTM
            iv = row.get("realized_vol_20", 0.25)  # Use realized vol as proxy
            premium = self._estimate_put_price(strike, price, self.config.target_dte, iv)

            if premium / strike < 0.01:  # Min 1% premium
                continue

            expiration_date = current_date + timedelta(days=self.config.target_dte)

            tracker.open_short_put(
                ticker=ticker,
                strike=strike,
                premium=premium,
                entry_date=current_date,
                expiration_date=expiration_date,
                iv=iv,
            )

    def _should_exit_put(
        self, pos, current_date: date, price: float, df: pd.DataFrame | None
    ) -> bool:
        """Check if we should exit put early."""
        if not pos.put_expiration_date:
            return False

        days_left = (pos.put_expiration_date - current_date).days

        # Exit at min DTE
        if days_left <= self.config.min_exit_dte:
            return True

        # Calculate current value
        iv = pos.put_entry_iv or 0.25
        current_value = self._estimate_put_price(pos.put_strike, price, days_left, iv)

        # Profit target
        entry_premium = pos.put_premium or 0
        if entry_premium > 0:
            pnl_pct = (entry_premium - current_value) / entry_premium
            if pnl_pct >= self.config.profit_target:
                return True

            # Stop loss
            if current_value / entry_premium >= self.config.stop_loss:
                return True

        return False

    def _try_sell_covered_call(
        self,
        tracker: WheelTracker,
        ticker: str,
        current_date: date,
        price: float,
        df: pd.DataFrame | None,
    ):
        """Try to sell covered call on owned stock."""
        pos = tracker.positions.get(ticker)
        if not pos or pos.state != PositionState.STOCK_OWNED:
            return

        # Calculate call parameters
        strike = round(price * 1.05, 2)  # 5% OTM
        iv = 0.25  # Default IV

        if df is not None and current_date in df.index:
            iv = df.loc[current_date].get("realized_vol_20", 0.25)

        premium = self._estimate_call_price(strike, price, self.config.cc_dte, iv)
        expiration_date = current_date + timedelta(days=self.config.cc_dte)

        tracker.open_covered_call(
            ticker=ticker,
            strike=strike,
            premium=premium,
            entry_date=current_date,
            expiration_date=expiration_date,
            iv=iv,
        )

    def _score_entry(self, row: pd.Series, current_date: date, df: pd.DataFrame) -> float:
        """
        Score a potential entry.

        Returns score 0-1 (higher = better entry).
        """
        score = 0.5  # Base score

        # IV rank bonus
        rv_rank = row.get("rv_rank_252", 0.5)
        if rv_rank > 0.50:
            score += 0.15
        elif rv_rank > 0.30:
            score += 0.05
        elif rv_rank < 0.20:
            score -= 0.10

        # Trend: prefer uptrend
        trend_20d = row.get("trend_20d", 0)
        if trend_20d > 0.02:
            score += 0.10
        elif trend_20d < -0.05:
            score -= 0.15

        # RSI: prefer mid-range
        rsi = row.get("rsi_14", 50)
        if 40 <= rsi <= 60:
            score += 0.05
        elif rsi < 30:  # Oversold - risky
            score -= 0.10
        elif rsi > 70:  # Overbought
            score -= 0.05

        # Above moving averages
        if row.get("above_sma_200", 0) == 1:
            score += 0.10
        else:
            score -= 0.10

        # Drawdown: avoid stocks in freefall
        dd = row.get("drawdown_52w", 0)
        if dd < -0.30:
            score -= 0.20
        elif dd < -0.15:
            score -= 0.10

        # Use ML model if available
        if self.model is not None:
            try:
                ml_score = self.model.predict_proba(row.to_frame().T)[0]
                score = 0.3 * score + 0.7 * ml_score  # Weight ML heavily
            except Exception:
                pass

        return max(0, min(1, score))

    def _estimate_put_price(self, strike: float, spot: float, dte: int, iv: float) -> float:
        """Estimate put option price."""
        return estimate_option_price_from_iv(
            underlying_price=spot,
            strike=strike,
            dte=max(1, dte),
            iv=iv,
            risk_free_rate=0.04,
            option_type="put",
        )

    def _estimate_call_price(self, strike: float, spot: float, dte: int, iv: float) -> float:
        """Estimate call option price."""
        return estimate_option_price_from_iv(
            underlying_price=spot,
            strike=strike,
            dte=max(1, dte),
            iv=iv,
            risk_free_rate=0.04,
            option_type="call",
        )

    def _compute_metrics(self, equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
        """Compute backtest performance metrics."""
        if equity.empty:
            return {}

        initial = equity["portfolio_value"].iloc[0]
        final = equity["portfolio_value"].iloc[-1]
        total_return = (final - initial) / initial

        # Calculate returns
        equity["daily_return"] = equity["portfolio_value"].pct_change()
        daily_returns = equity["daily_return"].dropna()

        # Annualized metrics
        n_days = len(equity)
        ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        ann_vol = daily_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        rolling_max = equity["portfolio_value"].cummax()
        drawdown = (equity["portfolio_value"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Trade metrics
        if not trades.empty and "net_pnl" in trades.columns:
            win_rate = (trades["net_pnl"] > 0).mean()
            avg_pnl = trades["net_pnl"].mean()
            total_pnl = trades["net_pnl"].sum()
            n_trades = len(trades)
        else:
            win_rate = avg_pnl = total_pnl = 0
            n_trades = 0

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_pnl_per_trade": avg_pnl,
            "total_pnl": total_pnl,
            "final_value": final,
        }


def run_backtest(
    data_path: str, config: BacktestConfig | None = None, model_path: str | None = None
) -> BacktestResult:
    """
    Convenience function to run backtest.

    Args:
        data_path: Path to processed features parquet
        config: Backtest configuration
        model_path: Optional path to trained ML model

    Returns:
        BacktestResult
    """
    print("Loading data...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers")

    model = None
    if model_path:
        from ml.wheel_model import WheelEntryModel

        model = WheelEntryModel.load(model_path)
        print(f"  Loaded model from {model_path}")

    config = config or BacktestConfig()
    backtest = WheelBacktest(config=config, model=model)

    print("\nRunning backtest...")
    print(f"  Initial capital: ${config.initial_capital:,.0f}")
    print(f"  Max positions: {config.max_positions}")

    result = backtest.run(df)

    print(f"\n{'=' * 50}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 50}")
    print(f"Total Return:     {result.metrics['total_return']:>10.1%}")
    print(f"Ann. Return:      {result.metrics['annualized_return']:>10.1%}")
    print(f"Ann. Volatility:  {result.metrics['annualized_vol']:>10.1%}")
    print(f"Sharpe Ratio:     {result.metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:     {result.metrics['max_drawdown']:>10.1%}")
    print(f"Number of Trades: {result.metrics['n_trades']:>10}")
    print(f"Win Rate:         {result.metrics['win_rate']:>10.1%}")
    print(f"Avg P&L/Trade:    ${result.metrics['avg_pnl_per_trade']:>9.2f}")
    print(f"Final Value:      ${result.metrics['final_value']:>9,.0f}")
    print(f"{'=' * 50}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to processed features parquet")
    parser.add_argument("--model", help="Path to ML model")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--max-positions", type=int, default=10)
    args = parser.parse_args()

    config = BacktestConfig(initial_capital=args.capital, max_positions=args.max_positions)

    result = run_backtest(args.data_path, config, args.model)
