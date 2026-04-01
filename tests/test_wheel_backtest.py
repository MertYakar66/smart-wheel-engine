"""
Tests for the wheel backtest module.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.wheel_backtest import (
    BacktestConfig,
    BacktestResult,
    WheelBacktest,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_price_data():
    """Create sample price data for backtesting."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02", periods=100, freq="B")

    tickers = ["AAPL", "MSFT", "GOOGL"]
    rows = []

    for ticker in tickers:
        # Generate realistic price paths
        base_price = {"AAPL": 180, "MSFT": 350, "GOOGL": 140}[ticker]
        price = base_price

        for _i, d in enumerate(dates):
            # Random walk with slight upward drift
            ret = np.random.normal(0.0003, 0.015)
            price = price * (1 + ret)

            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = price * (1 + np.random.normal(0, 0.002))

            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": int(np.random.uniform(10_000_000, 50_000_000)),
                    "realized_vol_20": np.random.uniform(0.15, 0.35),
                    "rv_rank_252": np.random.uniform(0.2, 0.8),
                    "trend_20d": np.random.uniform(-0.05, 0.05),
                    "rsi_14": np.random.uniform(30, 70),
                    "above_sma_200": np.random.choice([0, 1]),
                    "drawdown_52w": np.random.uniform(-0.20, 0),
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def default_config():
    """Create default backtest config."""
    return BacktestConfig()


@pytest.fixture
def aggressive_config():
    """Create aggressive backtest config."""
    return BacktestConfig(
        initial_capital=50_000,
        max_positions=5,
        min_entry_score=0.4,  # Lower threshold
        profit_target=0.40,
        stop_loss=1.5,
    )


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.initial_capital == 100_000
        assert config.max_positions == 10
        assert config.max_position_pct == 0.15
        assert config.min_entry_score == 0.6
        assert config.target_delta == 0.30
        assert config.target_dte == 30
        assert config.profit_target == 0.50
        assert config.stop_loss == 2.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BacktestConfig(initial_capital=200_000, max_positions=15, min_entry_score=0.7)

        assert config.initial_capital == 200_000
        assert config.max_positions == 15
        assert config.min_entry_score == 0.7

    def test_risk_parameters(self):
        """Test risk-related parameters."""
        config = BacktestConfig(avoid_earnings=True, earnings_buffer_days=10)

        assert config.avoid_earnings is True
        assert config.earnings_buffer_days == 10

    def test_covered_call_parameters(self):
        """Test covered call parameters."""
        config = BacktestConfig()

        assert config.cc_delta == 0.30
        assert config.cc_dte == 30


# =============================================================================
# BACKTEST RESULT TESTS
# =============================================================================


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_result_creation(self):
        """Test creating a BacktestResult."""
        config = BacktestConfig()
        equity = pd.DataFrame({"date": [date.today()], "portfolio_value": [100000]})
        trades = pd.DataFrame()
        metrics = {"total_return": 0.10}

        result = BacktestResult(equity_curve=equity, trades=trades, metrics=metrics, config=config)

        assert len(result.equity_curve) == 1
        assert result.metrics["total_return"] == 0.10

    def test_result_has_config(self):
        """Test that result includes config."""
        config = BacktestConfig(initial_capital=50000)
        equity = pd.DataFrame()
        trades = pd.DataFrame()
        metrics = {}

        result = BacktestResult(equity, trades, metrics, config)

        assert result.config.initial_capital == 50000


# =============================================================================
# WHEEL BACKTEST TESTS
# =============================================================================


class TestWheelBacktest:
    """Test WheelBacktest class."""

    def test_initialization_default(self):
        """Test default initialization."""
        backtest = WheelBacktest()

        assert backtest.config is not None
        assert backtest.config.initial_capital == 100_000
        assert backtest.aggregator is not None

    def test_initialization_with_config(self, default_config):
        """Test initialization with custom config."""
        backtest = WheelBacktest(config=default_config)

        assert backtest.config == default_config

    def test_run_returns_result(self, sample_price_data, default_config):
        """Test that run returns a BacktestResult."""
        backtest = WheelBacktest(config=default_config)
        result = backtest.run(sample_price_data)

        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert result.metrics is not None

    def test_run_with_date_range(self, sample_price_data, default_config):
        """Test run with start and end dates."""
        backtest = WheelBacktest(config=default_config)

        start = date(2024, 2, 1)
        end = date(2024, 4, 1)

        result = backtest.run(sample_price_data, start_date=start, end_date=end)

        assert len(result.equity_curve) > 0
        # Verify dates are within range
        dates = pd.to_datetime(result.equity_curve["date"]).dt.date
        assert dates.min() >= start
        assert dates.max() <= end

    def test_equity_curve_columns(self, sample_price_data, default_config):
        """Test equity curve has required columns."""
        backtest = WheelBacktest(config=default_config)
        result = backtest.run(sample_price_data)

        required_cols = ["date", "portfolio_value", "cash", "num_positions"]
        for col in required_cols:
            assert col in result.equity_curve.columns

    def test_portfolio_value_starts_at_initial(self, sample_price_data, default_config):
        """Test portfolio starts at initial capital."""
        backtest = WheelBacktest(config=default_config)
        result = backtest.run(sample_price_data)

        initial_value = result.equity_curve["portfolio_value"].iloc[0]
        # Should be close to initial capital (may vary slightly due to timing)
        assert abs(initial_value - default_config.initial_capital) < 1000

    def test_metrics_computed(self, sample_price_data, default_config):
        """Test that metrics are computed."""
        backtest = WheelBacktest(config=default_config)
        result = backtest.run(sample_price_data)

        expected_metrics = [
            "total_return",
            "annualized_return",
            "annualized_vol",
            "sharpe_ratio",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "final_value",
        ]

        for metric in expected_metrics:
            assert metric in result.metrics


class TestBacktestScoring:
    """Test entry scoring logic."""

    def test_score_entry_basic(self, sample_price_data, default_config):
        """Test basic entry scoring."""
        backtest = WheelBacktest(config=default_config)

        # Get a sample row
        df = sample_price_data[sample_price_data["ticker"] == "AAPL"]
        df = df.set_index("date")
        row = df.iloc[0]

        score = backtest._score_entry(row, df.index[0], df)

        assert 0 <= score <= 1

    def test_high_iv_rank_boosts_score(self, sample_price_data):
        """Test that high IV rank increases score."""
        backtest = WheelBacktest()

        df = sample_price_data[sample_price_data["ticker"] == "AAPL"].copy()
        df = df.set_index("date")

        # Modify to have high IV rank
        row_high_iv = df.iloc[0].copy()
        row_high_iv["rv_rank_252"] = 0.80

        row_low_iv = df.iloc[0].copy()
        row_low_iv["rv_rank_252"] = 0.15

        score_high = backtest._score_entry(row_high_iv, df.index[0], df)
        score_low = backtest._score_entry(row_low_iv, df.index[0], df)

        assert score_high > score_low

    def test_uptrend_boosts_score(self, sample_price_data):
        """Test that uptrend increases score."""
        backtest = WheelBacktest()

        df = sample_price_data[sample_price_data["ticker"] == "AAPL"].copy()
        df = df.set_index("date")

        row_uptrend = df.iloc[0].copy()
        row_uptrend["trend_20d"] = 0.05

        row_downtrend = df.iloc[0].copy()
        row_downtrend["trend_20d"] = -0.10

        score_up = backtest._score_entry(row_uptrend, df.index[0], df)
        score_down = backtest._score_entry(row_downtrend, df.index[0], df)

        assert score_up > score_down


class TestOptionPricing:
    """Test option pricing estimation."""

    def test_estimate_put_price(self):
        """Test put price estimation."""
        backtest = WheelBacktest()

        price = backtest._estimate_put_price(strike=95, spot=100, dte=30, iv=0.25)

        assert price > 0
        assert price < 100  # Should be reasonable

    def test_estimate_call_price(self):
        """Test call price estimation."""
        backtest = WheelBacktest()

        price = backtest._estimate_call_price(strike=105, spot=100, dte=30, iv=0.25)

        assert price > 0
        assert price < 100

    def test_put_price_increases_with_iv(self):
        """Test that put price increases with IV."""
        backtest = WheelBacktest()

        low_iv_price = backtest._estimate_put_price(95, 100, 30, 0.15)
        high_iv_price = backtest._estimate_put_price(95, 100, 30, 0.40)

        assert high_iv_price > low_iv_price

    def test_put_price_increases_with_dte(self):
        """Test that put price increases with DTE."""
        backtest = WheelBacktest()

        short_dte_price = backtest._estimate_put_price(95, 100, 7, 0.25)
        long_dte_price = backtest._estimate_put_price(95, 100, 45, 0.25)

        assert long_dte_price > short_dte_price


class TestExitLogic:
    """Test exit conditions."""

    def test_should_exit_no_expiration(self):
        """Test exit logic with no expiration date."""
        from engine.wheel_tracker import PositionState, WheelPosition

        backtest = WheelBacktest()

        pos = WheelPosition(
            ticker="AAPL",
            state=PositionState.SHORT_PUT,
            entry_date=date.today(),
            put_strike=95,
            put_premium=2.0,
        )
        # No expiration date set

        should_exit = backtest._should_exit_put(pos, date.today(), 100, None)
        assert should_exit is False


class TestMetricsComputation:
    """Test metrics computation."""

    def test_compute_metrics_empty_equity(self):
        """Test metrics with empty equity curve."""
        backtest = WheelBacktest()

        empty_equity = pd.DataFrame()
        empty_trades = pd.DataFrame()

        metrics = backtest._compute_metrics(empty_equity, empty_trades)

        assert metrics == {}

    def test_compute_metrics_basic(self, sample_price_data, default_config):
        """Test basic metrics computation."""
        backtest = WheelBacktest(config=default_config)
        result = backtest.run(sample_price_data)

        metrics = result.metrics

        # Check total return is reasonable
        assert -1 < metrics["total_return"] < 10  # Between -100% and +1000%

        # Check Sharpe is reasonable
        assert -10 < metrics["sharpe_ratio"] < 10

        # Max drawdown should be negative or zero
        assert metrics["max_drawdown"] <= 0


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_data(self, default_config):
        """Test with empty price data."""
        backtest = WheelBacktest(config=default_config)
        empty_data = pd.DataFrame(columns=["date", "ticker", "close"])

        result = backtest.run(empty_data)

        assert result.equity_curve.empty or len(result.equity_curve) == 0

    def test_single_ticker(self, default_config):
        """Test with single ticker."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-02", periods=50, freq="B")

        data = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAPL"] * 50,
                "open": np.random.uniform(175, 185, 50),
                "high": np.random.uniform(180, 190, 50),
                "low": np.random.uniform(170, 180, 50),
                "close": np.random.uniform(175, 185, 50),
                "volume": [10_000_000] * 50,
                "realized_vol_20": [0.25] * 50,
                "rv_rank_252": [0.5] * 50,
                "trend_20d": [0.02] * 50,
                "rsi_14": [50] * 50,
                "above_sma_200": [1] * 50,
                "drawdown_52w": [-0.05] * 50,
            }
        )

        backtest = WheelBacktest(config=default_config)
        result = backtest.run(data)

        assert len(result.equity_curve) > 0

    def test_max_positions_respected(self, sample_price_data):
        """Test that max positions is respected."""
        config = BacktestConfig(
            max_positions=2,
            min_entry_score=0.3,  # Low threshold to encourage entries
        )
        backtest = WheelBacktest(config=config)
        result = backtest.run(sample_price_data)

        # Check that num_positions never exceeds max
        assert result.equity_curve["num_positions"].max() <= config.max_positions


class TestIntegration:
    """Integration tests."""

    def test_full_backtest_cycle(self, sample_price_data, aggressive_config):
        """Test a complete backtest cycle."""
        backtest = WheelBacktest(config=aggressive_config)
        result = backtest.run(sample_price_data)

        # Should have equity history
        assert len(result.equity_curve) > 0

        # Should have valid metrics
        assert "final_value" in result.metrics
        assert result.metrics["final_value"] > 0

        # Config should be preserved
        assert result.config == aggressive_config

    def test_backtest_determinism(self, sample_price_data, default_config):
        """Test that backtest is deterministic."""
        backtest = WheelBacktest(config=default_config)

        result1 = backtest.run(sample_price_data)
        result2 = backtest.run(sample_price_data)

        # Final values should match
        assert result1.metrics["final_value"] == result2.metrics["final_value"]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
