"""
Tests for Portfolio Tracker Module

Comprehensive tests for:
- Transaction processing
- Holdings management
- Return calculations (1D, 1W, 1M, 3M, YTD, 1Y, All-time)
- Portfolio snapshots
- Allocation breakdown
- Import/Export functionality
"""

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from engine.portfolio_tracker import (
    Holding,
    PortfolioTracker,
    Transaction,
    TransactionType,
    create_portfolio_from_holdings,
    quick_snapshot,
)


class TestTransaction:
    """Tests for Transaction dataclass."""

    def test_transaction_creation(self):
        """Test basic transaction creation."""
        txn = Transaction(
            ticker="AAPL",
            action="buy",
            shares=100,
            price=175.50,
            date=date.today(),
            fees=1.00,
        )
        assert txn.ticker == "AAPL"
        assert txn.action == TransactionType.BUY
        assert txn.shares == 100
        assert txn.price == 175.50
        assert txn.fees == 1.00

    def test_transaction_total_value(self):
        """Test total value calculation."""
        txn = Transaction(
            ticker="MSFT",
            action=TransactionType.BUY,
            shares=50,
            price=400.00,
            date=date.today(),
            fees=5.00,
        )
        assert txn.total_value == 50 * 400.00 + 5.00

    def test_transaction_enum_conversion(self):
        """Test string to enum conversion."""
        txn = Transaction(
            ticker="GOOGL",
            action="sell",  # String should convert to enum
            shares=10,
            price=150.00,
            date=date.today(),
        )
        assert txn.action == TransactionType.SELL

    def test_transaction_to_dict(self):
        """Test serialization to dict."""
        txn = Transaction(
            ticker="NVDA",
            action=TransactionType.BUY,
            shares=25,
            price=800.00,
            date=date(2025, 1, 15),
            fees=2.50,
            notes="Test purchase",
        )
        d = txn.to_dict()
        assert d["ticker"] == "NVDA"
        assert d["action"] == "buy"
        assert d["date"] == "2025-01-15"


class TestHolding:
    """Tests for Holding dataclass."""

    def test_holding_creation(self):
        """Test basic holding creation."""
        holding = Holding(
            ticker="AAPL",
            shares=100,
            cost_basis=150.00,
            current_price=175.00,
        )
        assert holding.ticker == "AAPL"
        assert holding.shares == 100
        assert holding.cost_basis == 150.00
        assert holding.current_price == 175.00

    def test_holding_market_value(self):
        """Test market value calculation."""
        holding = Holding(
            ticker="AAPL",
            shares=100,
            cost_basis=150.00,
            current_price=175.00,
        )
        assert holding.market_value == 100 * 175.00

    def test_holding_total_cost(self):
        """Test total cost calculation."""
        holding = Holding(
            ticker="MSFT",
            shares=50,
            cost_basis=400.00,
            current_price=420.00,
        )
        assert holding.total_cost == 50 * 400.00

    def test_holding_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        holding = Holding(
            ticker="GOOGL",
            shares=100,
            cost_basis=100.00,
            current_price=120.00,
        )
        assert holding.unrealized_pnl == 100 * (120.00 - 100.00)
        assert holding.unrealized_pnl == 2000.00

    def test_holding_unrealized_pnl_percentage(self):
        """Test unrealized P&L percentage."""
        holding = Holding(
            ticker="AMZN",
            shares=10,
            cost_basis=100.00,
            current_price=125.00,
        )
        # 25% gain
        assert holding.unrealized_pnl_pct == pytest.approx(0.25)

    def test_holding_total_return_with_dividends(self):
        """Test total return including dividends."""
        holding = Holding(
            ticker="JNJ",
            shares=100,
            cost_basis=150.00,
            current_price=155.00,
            dividends_received=200.00,
        )
        # Unrealized: 100 * (155 - 150) = 500
        # Dividends: 200
        # Total: 700
        assert holding.total_return == 700.00
        # Total return %: 700 / (100 * 150) = 4.67%
        assert holding.total_return_pct == pytest.approx(700.00 / 15000.00)


class TestPortfolioTracker:
    """Tests for PortfolioTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = PortfolioTracker(initial_cash=100_000)
        assert tracker.cash == 100_000
        assert len(tracker.holdings) == 0
        assert len(tracker.transactions) == 0
        assert tracker.total_deposits == 100_000

    def test_buy_transaction(self):
        """Test buy transaction processing."""
        tracker = PortfolioTracker(initial_cash=100_000)

        txn = Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
            fees=5.00,
        )

        result = tracker.add_transaction(txn)

        assert result is True
        assert "AAPL" in tracker.holdings
        assert tracker.holdings["AAPL"].shares == 100
        # Cost basis includes fees: 150 + (5 / 100) = 150.05 per share
        assert tracker.holdings["AAPL"].cost_basis == 150.05
        # Cash should be reduced by cost + fees
        assert tracker.cash == 100_000 - (100 * 150.00 + 5.00)

    def test_buy_insufficient_funds(self):
        """Test buy with insufficient funds."""
        tracker = PortfolioTracker(initial_cash=1_000)

        txn = Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,  # Would cost $15,000
            date=date.today(),
        )

        result = tracker.add_transaction(txn)
        assert result is False
        assert "AAPL" not in tracker.holdings
        assert tracker.cash == 1_000

    def test_sell_transaction(self):
        """Test sell transaction processing."""
        tracker = PortfolioTracker(initial_cash=100_000)

        # First buy
        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))

        # Then sell
        result = tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.SELL,
            shares=50,
            price=175.00,
            date=date.today(),
            fees=2.00,
        ))

        assert result is True
        assert tracker.holdings["AAPL"].shares == 50
        # Realized P&L: 50 * (175 - 150) - 2 = 1248
        assert tracker.realized_pnl == 50 * 175.00 - 2.00 - 50 * 150.00

    def test_sell_all_shares(self):
        """Test selling all shares removes holding."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.SELL,
            shares=100,
            price=175.00,
            date=date.today(),
        ))

        assert "AAPL" not in tracker.holdings

    def test_dividend_transaction(self):
        """Test dividend transaction."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))

        # Receive dividend
        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.DIVIDEND,
            shares=100,  # Number of shares
            price=0.25,  # Dividend per share
            date=date.today(),
        ))

        assert tracker.total_dividends == 25.00
        assert tracker.holdings["AAPL"].dividends_received == 25.00
        # Cash should increase
        expected_cash = 100_000 - 15_000 + 25.00
        assert tracker.cash == expected_cash

    def test_deposit_withdrawal(self):
        """Test deposit and withdrawal transactions."""
        tracker = PortfolioTracker(initial_cash=50_000)

        # Deposit
        tracker.add_transaction(Transaction(
            ticker="CASH",
            action=TransactionType.DEPOSIT,
            shares=1,
            price=25_000,
            date=date.today(),
        ))

        assert tracker.cash == 75_000
        assert tracker.total_deposits == 75_000

        # Withdrawal
        tracker.add_transaction(Transaction(
            ticker="CASH",
            action=TransactionType.WITHDRAWAL,
            shares=1,
            price=10_000,
            date=date.today(),
        ))

        assert tracker.cash == 65_000
        assert tracker.total_withdrawals == 10_000

    def test_cost_basis_averaging(self):
        """Test cost basis averaging on multiple buys."""
        tracker = PortfolioTracker(initial_cash=100_000)

        # Buy 100 @ $100
        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=100.00,
            date=date.today(),
        ))

        # Buy 100 @ $120
        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=120.00,
            date=date.today(),
        ))

        # Average cost: (100*100 + 100*120) / 200 = 110
        assert tracker.holdings["AAPL"].shares == 200
        assert tracker.holdings["AAPL"].cost_basis == pytest.approx(110.00)


class TestPortfolioSnapshot:
    """Tests for portfolio snapshots."""

    def test_snapshot_creation(self):
        """Test taking a portfolio snapshot."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))

        prices = {"AAPL": 175.00}
        snapshot = tracker.snapshot(prices)

        assert snapshot.total_value == tracker.cash + 100 * 175.00
        assert snapshot.invested_value == 100 * 175.00
        assert snapshot.unrealized_pnl == 100 * (175.00 - 150.00)
        assert snapshot.holdings_count == 1

    def test_multiple_snapshots(self):
        """Test taking multiple snapshots over time."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today() - timedelta(days=7),
        ))

        # First snapshot
        tracker.snapshot({"AAPL": 155.00}, snapshot_date=date.today() - timedelta(days=5))

        # Second snapshot
        tracker.snapshot({"AAPL": 160.00}, snapshot_date=date.today() - timedelta(days=3))

        # Third snapshot
        tracker.snapshot({"AAPL": 170.00}, snapshot_date=date.today())

        assert len(tracker.snapshots) >= 3


class TestReturnCalculations:
    """Tests for return calculations."""

    def test_returns_empty_portfolio(self):
        """Test returns for empty portfolio."""
        tracker = PortfolioTracker(initial_cash=0)
        metrics = tracker.get_returns()
        assert metrics.return_1d == 0.0
        assert metrics.return_all_time == 0.0

    def test_returns_with_history(self):
        """Test return calculations with snapshot history."""
        tracker = PortfolioTracker(initial_cash=100_000)

        # Buy stock
        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=100.00,
            date=date.today() - timedelta(days=35),
        ))

        # Create snapshots showing growth over 35 days
        for i in range(35, -1, -1):
            price = 100.00 + (35 - i) * 1.0  # Price increases by $1/day
            tracker.snapshot(
                {"AAPL": price},
                snapshot_date=date.today() - timedelta(days=i)
            )

        metrics = tracker.get_returns()

        # Verify total gain is positive (confirms portfolio value increase)
        # Total gain = 100 shares * $35 price increase = $3500
        assert metrics.total_gain > 0
        # Verify volatility is calculated
        assert metrics.volatility_annualized > 0

    def test_time_weighted_return_with_deposit(self):
        """Test TWR handles deposits correctly."""
        tracker = PortfolioTracker(initial_cash=10_000)

        # Initial snapshot
        tracker.snapshot({}, snapshot_date=date.today() - timedelta(days=10))

        # Deposit midway
        tracker.add_transaction(Transaction(
            ticker="CASH",
            action=TransactionType.DEPOSIT,
            shares=1,
            price=10_000,
            date=date.today() - timedelta(days=5),
        ))
        tracker.snapshot({}, snapshot_date=date.today() - timedelta(days=5))

        # Final snapshot
        tracker.snapshot({}, snapshot_date=date.today())

        metrics = tracker.get_returns()
        # TWR should account for the deposit, showing near-zero return
        # since the portfolio just held cash
        assert abs(metrics.return_all_time) < 0.01  # Within 1%


class TestAllocation:
    """Tests for allocation breakdown."""

    def test_allocation_by_sector(self):
        """Test sector allocation breakdown."""
        tracker = PortfolioTracker(initial_cash=100_000)

        # Buy tech stocks
        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))
        tracker.add_transaction(Transaction(
            ticker="MSFT",
            action=TransactionType.BUY,
            shares=50,
            price=400.00,
            date=date.today(),
        ))

        # Update prices
        tracker.snapshot({"AAPL": 150.00, "MSFT": 400.00})

        allocation = tracker.get_allocation()

        assert "Technology" in allocation["by_sector"]
        # AAPL: 15000, MSFT: 20000, Cash: 65000, Total: 100000
        tech_pct = (15000 + 20000) / 100000
        assert allocation["by_sector"]["Technology"] == pytest.approx(tech_pct)

    def test_allocation_cash_percentage(self):
        """Test cash percentage in allocation."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))

        tracker.snapshot({"AAPL": 150.00})

        allocation = tracker.get_allocation()

        # Invested: $15,000, Cash: $85,000
        assert allocation["cash"] == 85_000
        assert allocation["cash_pct"] == pytest.approx(0.85)

    def test_allocation_top_holdings(self):
        """Test top holdings in allocation."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))
        tracker.add_transaction(Transaction(
            ticker="MSFT",
            action=TransactionType.BUY,
            shares=25,
            price=400.00,
            date=date.today(),
        ))

        tracker.snapshot({"AAPL": 150.00, "MSFT": 400.00})

        allocation = tracker.get_allocation()

        assert len(allocation["top_holdings"]) == 2
        # AAPL: $15,000, MSFT: $10,000 - AAPL should be first
        assert allocation["top_holdings"][0]["ticker"] == "AAPL"


class TestImportExport:
    """Tests for import/export functionality."""

    def test_import_holdings(self):
        """Test importing holdings."""
        tracker = PortfolioTracker(initial_cash=50_000)

        holdings = [
            {"ticker": "AAPL", "shares": 100, "cost_basis": 150.00, "current_price": 175.00},
            {"ticker": "MSFT", "shares": 50, "cost_basis": 400.00, "current_price": 420.00},
        ]

        count = tracker.import_holdings(holdings)

        assert count == 2
        assert "AAPL" in tracker.holdings
        assert "MSFT" in tracker.holdings
        assert tracker.holdings["AAPL"].shares == 100
        assert tracker.holdings["MSFT"].cost_basis == 400.00

    def test_export_to_json(self):
        """Test exporting portfolio to JSON."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        tracker.export_to_json(filepath)

        # Verify file contents
        with open(filepath) as f:
            data = json.load(f)

        assert data["cash"] < 100_000  # Some spent
        assert len(data["holdings"]) == 1
        assert len(data["transactions"]) == 1

        # Cleanup
        Path(filepath).unlink()

    def test_load_from_json(self):
        """Test loading portfolio from JSON."""
        tracker1 = PortfolioTracker(initial_cash=100_000)

        tracker1.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))
        tracker1.realized_pnl = 500.00
        tracker1.total_dividends = 100.00

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        tracker1.export_to_json(filepath)

        # Load into new tracker
        tracker2 = PortfolioTracker()
        tracker2.load_from_json(filepath)

        assert tracker2.cash == tracker1.cash
        assert len(tracker2.holdings) == len(tracker1.holdings)
        assert tracker2.realized_pnl == 500.00
        assert tracker2.total_dividends == 100.00

        # Cleanup
        Path(filepath).unlink()

    def test_import_from_csv(self):
        """Test importing transactions from CSV."""
        tracker = PortfolioTracker(initial_cash=100_000)

        csv_content = """date,ticker,action,shares,price,fees
2025-01-15,AAPL,buy,100,150.00,5.00
2025-01-16,MSFT,buy,50,400.00,5.00
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            filepath = f.name

        count = tracker.import_from_csv(filepath)

        assert count == 2
        assert "AAPL" in tracker.holdings
        assert "MSFT" in tracker.holdings

        # Cleanup
        Path(filepath).unlink()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_portfolio_from_holdings(self):
        """Test quick portfolio creation."""
        holdings = [
            {"ticker": "AAPL", "shares": 100, "cost_basis": 150.00, "current_price": 175.00},
        ]

        tracker = create_portfolio_from_holdings(holdings, cash=50_000)

        assert tracker.cash == 50_000
        assert "AAPL" in tracker.holdings

    def test_quick_snapshot(self):
        """Test quick snapshot helper."""
        holdings = [
            {"ticker": "AAPL", "shares": 100, "cost_basis": 150.00, "current_price": 175.00},
        ]
        tracker = create_portfolio_from_holdings(holdings, cash=50_000)

        result = quick_snapshot(tracker, {"AAPL": 180.00})

        assert "total_value" in result
        assert "returns" in result
        assert "allocation" in result
        assert result["total_value"] == 50_000 + 100 * 180.00


class TestRiskMetrics:
    """Tests for risk metrics calculation."""

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=1000,
            price=100.00,
            date=date.today() - timedelta(days=10),
        ))

        # Create snapshots with a drawdown
        # Day 0: $100,000
        tracker.snapshot({"AAPL": 100.00}, snapshot_date=date.today() - timedelta(days=10))

        # Day 3: Peak at $110,000
        tracker.snapshot({"AAPL": 110.00}, snapshot_date=date.today() - timedelta(days=7))

        # Day 5: Drop to $90,000 (18% drawdown from peak)
        tracker.snapshot({"AAPL": 90.00}, snapshot_date=date.today() - timedelta(days=5))

        # Day 10: Recovery to $105,000
        tracker.snapshot({"AAPL": 105.00}, snapshot_date=date.today())

        metrics = tracker.get_returns()

        # Max drawdown should be approximately 18% (from 110k to 90k)
        assert metrics.max_drawdown > 0.15
        assert metrics.max_drawdown < 0.25


class TestReports:
    """Tests for report generation."""

    def test_summary_report(self):
        """Test summary report generation."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))
        tracker.snapshot({"AAPL": 175.00})

        report = tracker.summary_report()

        assert "PORTFOLIO SUMMARY" in report
        assert "PERFORMANCE" in report
        assert "ALLOCATION BY SECTOR" in report

    def test_positions_report(self):
        """Test positions report generation."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))
        tracker.holdings["AAPL"].current_price = 175.00

        report = tracker.positions_report()

        assert "POSITIONS DETAIL" in report
        assert "AAPL" in report


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_portfolio_allocation(self):
        """Test allocation on empty portfolio."""
        tracker = PortfolioTracker(initial_cash=0)
        allocation = tracker.get_allocation()
        assert allocation == {}

    def test_sell_non_existent_holding(self):
        """Test selling stock not owned."""
        tracker = PortfolioTracker(initial_cash=100_000)

        result = tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.SELL,
            shares=100,
            price=150.00,
            date=date.today(),
        ))

        assert result is False

    def test_sell_more_than_owned(self):
        """Test selling more shares than owned."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=50,
            price=150.00,
            date=date.today(),
        ))

        result = tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.SELL,
            shares=100,  # More than owned
            price=175.00,
            date=date.today(),
        ))

        assert result is False
        assert tracker.holdings["AAPL"].shares == 50  # Unchanged

    def test_withdrawal_more_than_cash(self):
        """Test withdrawal exceeding cash balance."""
        tracker = PortfolioTracker(initial_cash=10_000)

        result = tracker.add_transaction(Transaction(
            ticker="CASH",
            action=TransactionType.WITHDRAWAL,
            shares=1,
            price=50_000,  # More than available
            date=date.today(),
        ))

        assert result is False
        assert tracker.cash == 10_000

    def test_zero_cost_basis_pnl(self):
        """Test P&L calculation with zero cost basis."""
        holding = Holding(
            ticker="FREE",
            shares=100,
            cost_basis=0.0,
            current_price=10.00,
        )
        assert holding.unrealized_pnl == 1000.00
        assert holding.unrealized_pnl_pct == 0.0  # Avoid division by zero

    def test_holdings_dataframe(self):
        """Test getting holdings as DataFrame."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))
        tracker.holdings["AAPL"].current_price = 175.00

        df = tracker.get_holdings_df()

        assert len(df) == 1
        assert "ticker" in df.columns
        assert "shares" in df.columns
        assert df.iloc[0]["ticker"] == "AAPL"

    def test_transactions_dataframe(self):
        """Test getting transactions as DataFrame."""
        tracker = PortfolioTracker(initial_cash=100_000)

        tracker.add_transaction(Transaction(
            ticker="AAPL",
            action=TransactionType.BUY,
            shares=100,
            price=150.00,
            date=date.today(),
        ))

        df = tracker.get_transactions_df()

        assert len(df) == 1
        assert "ticker" in df.columns
        assert "action" in df.columns
