"""
Portfolio Tracker Module

Comprehensive portfolio tracking inspired by Wealthsimple, IBKR, and Robinhood:
- Holdings management (stocks, options, cash)
- Time-period returns (1D, 1W, 1M, 3M, YTD, 1Y, All-time)
- Time-weighted return (TWR) calculations
- Position-level P&L and cost basis tracking
- Portfolio snapshot history
- Allocation breakdown by sector/asset class
- Dividend tracking
- Benchmark comparison
- Import/export functionality

Usage:
    from engine.portfolio_tracker import PortfolioTracker, Holding, Transaction

    # Create tracker
    tracker = PortfolioTracker(initial_cash=100_000)

    # Add holdings
    tracker.add_transaction(Transaction(
        ticker="AAPL",
        action="BUY",
        shares=100,
        price=175.50,
        date=date.today()
    ))

    # Take daily snapshot
    tracker.snapshot({"AAPL": 178.25})

    # Get returns
    returns = tracker.get_returns()
    print(f"1W Return: {returns['1W']:.2%}")
"""

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# =============================================================================
# Enums and Constants
# =============================================================================


class AssetClass(Enum):
    """Asset classification."""

    EQUITY = "equity"
    OPTION = "option"
    ETF = "etf"
    CASH = "cash"
    FIXED_INCOME = "fixed_income"
    CRYPTO = "crypto"
    OTHER = "other"


class TransactionType(Enum):
    """Transaction types."""

    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    OPTION_OPEN = "option_open"
    OPTION_CLOSE = "option_close"
    OPTION_ASSIGNMENT = "option_assignment"
    OPTION_EXPIRATION = "option_expiration"
    INTEREST = "interest"
    FEE = "fee"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"


# Standard sector mappings (inspired by GICS)
DEFAULT_SECTOR_MAP = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "META": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "CRM": "Technology",
    "ORCL": "Technology",
    "ADBE": "Technology",
    "CSCO": "Technology",
    "AVGO": "Technology",
    "QCOM": "Technology",
    "TXN": "Technology",
    # Financial
    "JPM": "Financial",
    "BAC": "Financial",
    "WFC": "Financial",
    "GS": "Financial",
    "MS": "Financial",
    "C": "Financial",
    "BLK": "Financial",
    "SCHW": "Financial",
    "AXP": "Financial",
    "V": "Financial",
    "MA": "Financial",
    "PYPL": "Financial",
    # Healthcare
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "LLY": "Healthcare",
    "TMO": "Healthcare",
    "ABT": "Healthcare",
    "DHR": "Healthcare",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "TGT": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "OXY": "Energy",
    # Industrials
    "CAT": "Industrials",
    "BA": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "RTX": "Industrials",
    "DE": "Industrials",
    "GE": "Industrials",
    "MMM": "Industrials",
    "LMT": "Industrials",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    # Real Estate
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    "CCI": "Real Estate",
    "EQIX": "Real Estate",
    "SPG": "Real Estate",
    # Communication Services
    "DIS": "Communication Services",
    "NFLX": "Communication Services",
    "CMCSA": "Communication Services",
    "VZ": "Communication Services",
    "T": "Communication Services",
    "TMUS": "Communication Services",
    # Materials
    "LIN": "Materials",
    "APD": "Materials",
    "SHW": "Materials",
    "FCX": "Materials",
    "NEM": "Materials",
    # ETFs
    "SPY": "Broad Market ETF",
    "QQQ": "Tech ETF",
    "IWM": "Small Cap ETF",
    "DIA": "Broad Market ETF",
    "VTI": "Broad Market ETF",
    "VOO": "Broad Market ETF",
    "IVV": "Broad Market ETF",
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Transaction:
    """A single portfolio transaction."""

    ticker: str
    action: TransactionType | str
    shares: float
    price: float
    date: date
    fees: float = 0.0
    notes: str = ""
    asset_class: AssetClass = AssetClass.EQUITY

    # Option-specific fields
    option_type: Literal["call", "put"] | None = None
    strike: float | None = None
    expiration: date | None = None

    def __post_init__(self):
        if isinstance(self.action, str):
            self.action = TransactionType(self.action.lower())
        if isinstance(self.asset_class, str):
            self.asset_class = AssetClass(self.asset_class.lower())

    @property
    def total_value(self) -> float:
        """Total transaction value including fees."""
        return self.shares * self.price + self.fees

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "action": self.action.value,
            "shares": self.shares,
            "price": self.price,
            "date": self.date.isoformat(),
            "fees": self.fees,
            "notes": self.notes,
            "asset_class": self.asset_class.value,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiration": self.expiration.isoformat() if self.expiration else None,
        }


class CostBasisMethod(Enum):
    """Cost basis calculation method for tax lots."""

    AVERAGE = "average"  # Weighted average cost
    FIFO = "fifo"  # First in, first out
    LIFO = "lifo"  # Last in, first out
    SPECIFIC_ID = "specific_id"  # Specific lot identification


@dataclass
class TaxLot:
    """Individual tax lot for precise cost basis tracking."""

    lot_id: str
    ticker: str
    shares: float
    cost_per_share: float
    purchase_date: date
    remaining_shares: float = field(default=None)

    def __post_init__(self):
        if self.remaining_shares is None:
            self.remaining_shares = self.shares

    @property
    def total_cost(self) -> float:
        """Total cost of remaining shares in this lot."""
        return self.remaining_shares * self.cost_per_share

    @property
    def is_long_term(self) -> bool:
        """Whether lot qualifies for long-term capital gains (held > 1 year)."""
        return (date.today() - self.purchase_date).days > 365

    def to_dict(self) -> dict:
        return {
            "lot_id": self.lot_id,
            "ticker": self.ticker,
            "shares": self.shares,
            "cost_per_share": self.cost_per_share,
            "purchase_date": self.purchase_date.isoformat(),
            "remaining_shares": self.remaining_shares,
            "is_long_term": self.is_long_term,
        }


@dataclass
class Holding:
    """A current portfolio holding."""

    ticker: str
    shares: float
    cost_basis: float  # Average cost per share
    current_price: float = 0.0
    asset_class: AssetClass = AssetClass.EQUITY
    sector: str = "Unknown"

    # Tax lot tracking for FIFO/LIFO/Specific-ID
    tax_lots: list[TaxLot] = field(default_factory=list)

    # Option-specific
    option_type: Literal["call", "put"] | None = None
    strike: float | None = None
    expiration: date | None = None

    # Dividend tracking
    dividends_received: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value."""
        return self.shares * self.current_price

    @property
    def total_cost(self) -> float:
        """Total cost basis."""
        return self.shares * self.cost_basis

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L (excluding dividends)."""
        return self.market_value - self.total_cost

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.total_cost == 0:
            return 0.0
        return self.unrealized_pnl / self.total_cost

    @property
    def total_return(self) -> float:
        """Total return including dividends."""
        return self.unrealized_pnl + self.dividends_received

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage."""
        if self.total_cost == 0:
            return 0.0
        return self.total_return / self.total_cost

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "shares": self.shares,
            "cost_basis": self.cost_basis,
            "current_price": self.current_price,
            "asset_class": self.asset_class.value,
            "sector": self.sector,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "dividends_received": self.dividends_received,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
        }


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""

    date: date
    total_value: float
    cash: float
    invested_value: float
    unrealized_pnl: float
    realized_pnl_cumulative: float
    dividends_cumulative: float
    deposits_cumulative: float
    withdrawals_cumulative: float

    # Breakdown
    holdings_count: int
    holdings_summary: dict = field(default_factory=dict)  # {ticker: market_value}

    @property
    def net_deposits(self) -> float:
        """Net deposits (deposits - withdrawals)."""
        return self.deposits_cumulative - self.withdrawals_cumulative

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "total_value": self.total_value,
            "cash": self.cash,
            "invested_value": self.invested_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl_cumulative": self.realized_pnl_cumulative,
            "dividends_cumulative": self.dividends_cumulative,
            "deposits_cumulative": self.deposits_cumulative,
            "withdrawals_cumulative": self.withdrawals_cumulative,
            "holdings_count": self.holdings_count,
        }


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns by period
    return_1d: float = 0.0
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_ytd: float = 0.0
    return_1y: float = 0.0
    return_all_time: float = 0.0

    # Risk metrics
    volatility_annualized: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0

    # Benchmark comparison
    benchmark_return_1y: float = 0.0
    alpha_1y: float = 0.0
    beta: float = 0.0

    # Summary
    total_gain: float = 0.0
    total_dividends: float = 0.0
    total_fees: float = 0.0

    def to_dict(self) -> dict:
        return {
            "1D": self.return_1d,
            "1W": self.return_1w,
            "1M": self.return_1m,
            "3M": self.return_3m,
            "YTD": self.return_ytd,
            "1Y": self.return_1y,
            "All": self.return_all_time,
            "volatility": self.volatility_annualized,
            "sharpe": self.sharpe_ratio,
            "sortino": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "total_gain": self.total_gain,
            "total_dividends": self.total_dividends,
        }


# =============================================================================
# Portfolio Tracker
# =============================================================================


class PortfolioTracker:
    """
    Comprehensive portfolio tracking system.

    Features inspired by Wealthsimple, IBKR, Robinhood:
    - Real-time holdings management
    - Historical performance tracking
    - Time-weighted return calculations
    - Position-level analytics
    - Sector/asset class allocation
    - Dividend tracking
    - Benchmark comparison
    """

    def __init__(
        self,
        initial_cash: float = 0.0,
        sector_map: dict[str, str] | None = None,
        benchmark_ticker: str = "SPY",
        cost_basis_method: CostBasisMethod = CostBasisMethod.FIFO,
        risk_free_rate: float = 0.04,
    ):
        """
        Initialize portfolio tracker.

        Args:
            initial_cash: Starting cash balance
            sector_map: Custom ticker to sector mapping
            benchmark_ticker: Benchmark for comparison (default SPY)
            cost_basis_method: Method for calculating realized gains (FIFO/LIFO/AVERAGE)
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 4%)
        """
        self.cash = initial_cash
        self.sector_map = sector_map or DEFAULT_SECTOR_MAP
        self.benchmark_ticker = benchmark_ticker
        self.cost_basis_method = cost_basis_method
        self.risk_free_rate = risk_free_rate

        # Holdings
        self.holdings: dict[str, Holding] = {}

        # Tax lot counter for unique IDs
        self._lot_counter = 0

        # Transaction history
        self.transactions: list[Transaction] = []

        # Snapshot history (for return calculations)
        self.snapshots: list[PortfolioSnapshot] = []

        # Cumulative tracking
        self.realized_pnl = 0.0
        self.realized_pnl_short_term = 0.0  # Short-term capital gains
        self.realized_pnl_long_term = 0.0  # Long-term capital gains
        self.total_dividends = 0.0
        self.total_fees = 0.0
        self.total_deposits = initial_cash
        self.total_withdrawals = 0.0

        # Benchmark prices (for comparison)
        self.benchmark_snapshots: list[tuple[date, float]] = []

        # Record initial snapshot if we have starting cash
        if initial_cash > 0:
            self._record_initial_snapshot()

    def _record_initial_snapshot(self, snapshot_date: date | None = None):
        """Record initial portfolio snapshot."""
        snapshot = PortfolioSnapshot(
            date=snapshot_date or date.today(),
            total_value=self.cash,
            cash=self.cash,
            invested_value=0.0,
            unrealized_pnl=0.0,
            realized_pnl_cumulative=0.0,
            dividends_cumulative=0.0,
            deposits_cumulative=self.total_deposits,
            withdrawals_cumulative=0.0,
            holdings_count=0,
            holdings_summary={},
        )
        self.snapshots.append(snapshot)

    # =========================================================================
    # Transaction Processing
    # =========================================================================

    def add_transaction(self, txn: Transaction) -> bool:
        """
        Process a new transaction.

        Args:
            txn: Transaction to process

        Returns:
            True if transaction was processed successfully
        """
        self.transactions.append(txn)
        self.total_fees += txn.fees

        if txn.action == TransactionType.BUY:
            return self._process_buy(txn)
        elif txn.action == TransactionType.SELL:
            return self._process_sell(txn)
        elif txn.action == TransactionType.DIVIDEND:
            return self._process_dividend(txn)
        elif txn.action == TransactionType.DEPOSIT:
            return self._process_deposit(txn)
        elif txn.action == TransactionType.WITHDRAWAL:
            return self._process_withdrawal(txn)
        elif txn.action in (TransactionType.OPTION_OPEN, TransactionType.OPTION_CLOSE):
            return self._process_option(txn)
        elif txn.action == TransactionType.INTEREST:
            return self._process_interest(txn)
        elif txn.action == TransactionType.FEE:
            self.cash -= txn.total_value
            return True
        else:
            return False

    def _generate_lot_id(self, ticker: str) -> str:
        """Generate unique tax lot ID."""
        self._lot_counter += 1
        return f"{ticker}-{self._lot_counter:06d}"

    def _process_buy(self, txn: Transaction) -> bool:
        """Process buy transaction with tax lot tracking."""
        cost = txn.shares * txn.price + txn.fees
        if cost > self.cash:
            return False  # Insufficient funds

        self.cash -= cost

        # Create new tax lot for this purchase
        new_lot = TaxLot(
            lot_id=self._generate_lot_id(txn.ticker),
            ticker=txn.ticker,
            shares=txn.shares,
            cost_per_share=txn.price + (txn.fees / txn.shares),  # Include fees in cost basis
            purchase_date=txn.date,
        )

        if txn.ticker in self.holdings:
            # Add to existing position
            holding = self.holdings[txn.ticker]
            holding.tax_lots.append(new_lot)
            holding.shares += txn.shares
            # Recalculate average cost basis from all lots
            total_cost = sum(lot.remaining_shares * lot.cost_per_share for lot in holding.tax_lots)
            holding.cost_basis = total_cost / holding.shares
        else:
            # New position
            self.holdings[txn.ticker] = Holding(
                ticker=txn.ticker,
                shares=txn.shares,
                cost_basis=new_lot.cost_per_share,
                asset_class=txn.asset_class,
                sector=self.sector_map.get(txn.ticker, "Unknown"),
                tax_lots=[new_lot],
            )

        return True

    def _process_sell(self, txn: Transaction) -> bool:
        """Process sell transaction with FIFO/LIFO/Specific-ID cost basis."""
        if txn.ticker not in self.holdings:
            return False

        holding = self.holdings[txn.ticker]
        if txn.shares > holding.shares:
            return False  # Can't sell more than owned

        proceeds = txn.shares * txn.price - txn.fees
        self.cash += proceeds

        # Calculate realized P&L using selected cost basis method
        shares_to_sell = txn.shares
        total_cost = 0.0
        short_term_gain = 0.0
        long_term_gain = 0.0

        # Order lots based on cost basis method
        if self.cost_basis_method == CostBasisMethod.FIFO:
            lots_ordered = sorted(holding.tax_lots, key=lambda lot: lot.purchase_date)
        elif self.cost_basis_method == CostBasisMethod.LIFO:
            lots_ordered = sorted(holding.tax_lots, key=lambda lot: lot.purchase_date, reverse=True)
        else:  # AVERAGE - use all lots proportionally
            lots_ordered = holding.tax_lots.copy()

        if self.cost_basis_method == CostBasisMethod.AVERAGE:
            # Simple average cost calculation
            avg_cost = holding.cost_basis * shares_to_sell
            realized_gain = proceeds - avg_cost
            self.realized_pnl += realized_gain
            # For average, proportionally reduce all lots
            ratio = shares_to_sell / holding.shares
            for lot in holding.tax_lots:
                lot.remaining_shares *= 1 - ratio
        else:
            # FIFO or LIFO: consume lots in order
            for lot in lots_ordered:
                if shares_to_sell <= 0:
                    break

                shares_from_lot = min(lot.remaining_shares, shares_to_sell)
                lot_cost = shares_from_lot * lot.cost_per_share
                total_cost += lot_cost

                # Calculate gain for this lot
                lot_proceeds = (shares_from_lot / txn.shares) * proceeds
                lot_gain = lot_proceeds - lot_cost

                # Track short-term vs long-term gains
                if lot.is_long_term:
                    long_term_gain += lot_gain
                else:
                    short_term_gain += lot_gain

                lot.remaining_shares -= shares_from_lot
                shares_to_sell -= shares_from_lot

            self.realized_pnl += short_term_gain + long_term_gain
            self.realized_pnl_short_term += short_term_gain
            self.realized_pnl_long_term += long_term_gain

        # Remove depleted lots
        holding.tax_lots = [lot for lot in holding.tax_lots if lot.remaining_shares > 0.001]

        # Update or remove holding
        holding.shares -= txn.shares
        if holding.shares <= 0.001:  # Effectively zero
            del self.holdings[txn.ticker]
        else:
            # Recalculate average cost basis from remaining lots
            total_remaining_cost = sum(
                lot.remaining_shares * lot.cost_per_share for lot in holding.tax_lots
            )
            holding.cost_basis = total_remaining_cost / holding.shares

        return True

    def _process_dividend(self, txn: Transaction) -> bool:
        """Process dividend payment."""
        dividend_amount = txn.shares * txn.price  # shares = dividend per share count
        self.cash += dividend_amount
        self.total_dividends += dividend_amount

        if txn.ticker in self.holdings:
            self.holdings[txn.ticker].dividends_received += dividend_amount

        return True

    def _process_deposit(self, txn: Transaction) -> bool:
        """Process cash deposit."""
        self.cash += txn.total_value
        self.total_deposits += txn.total_value
        return True

    def _process_withdrawal(self, txn: Transaction) -> bool:
        """Process cash withdrawal."""
        if txn.total_value > self.cash:
            return False
        self.cash -= txn.total_value
        self.total_withdrawals += txn.total_value
        return True

    def _process_option(self, txn: Transaction) -> bool:
        """Process option transaction."""
        if txn.action == TransactionType.OPTION_OPEN:
            # Selling option: receive premium
            premium = txn.shares * txn.price * 100  # Options are per 100 shares
            self.cash += premium - txn.fees

            # Track as holding
            option_key = f"{txn.ticker}_{txn.option_type}_{txn.strike}_{txn.expiration}"
            self.holdings[option_key] = Holding(
                ticker=option_key,
                shares=-txn.shares,  # Negative = short
                cost_basis=txn.price,
                asset_class=AssetClass.OPTION,
                sector="Options",
                option_type=txn.option_type,
                strike=txn.strike,
                expiration=txn.expiration,
            )
        else:  # OPTION_CLOSE
            option_key = f"{txn.ticker}_{txn.option_type}_{txn.strike}_{txn.expiration}"
            if option_key in self.holdings:
                holding = self.holdings[option_key]
                # Buying back: pay premium
                cost = txn.shares * txn.price * 100 + txn.fees
                realized = (holding.cost_basis - txn.price) * abs(holding.shares) * 100
                self.realized_pnl += realized
                self.cash -= cost
                del self.holdings[option_key]

        return True

    def _process_interest(self, txn: Transaction) -> bool:
        """Process interest income."""
        self.cash += txn.total_value
        return True

    # =========================================================================
    # Portfolio Snapshots
    # =========================================================================

    def snapshot(
        self,
        prices: dict[str, float],
        snapshot_date: date | None = None,
        benchmark_price: float | None = None,
    ) -> PortfolioSnapshot:
        """
        Take a portfolio snapshot with current prices.

        Args:
            prices: Dict of {ticker: current_price}
            snapshot_date: Date of snapshot (default: today)
            benchmark_price: Benchmark price for comparison

        Returns:
            PortfolioSnapshot
        """
        snapshot_date = snapshot_date or date.today()

        # Update holding prices
        invested_value = 0.0
        unrealized_pnl = 0.0
        holdings_summary = {}

        for ticker, holding in self.holdings.items():
            if ticker in prices:
                holding.current_price = prices[ticker]

            market_val = holding.market_value
            invested_value += market_val
            unrealized_pnl += holding.unrealized_pnl
            holdings_summary[ticker] = market_val

        total_value = self.cash + invested_value

        snapshot = PortfolioSnapshot(
            date=snapshot_date,
            total_value=total_value,
            cash=self.cash,
            invested_value=invested_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl_cumulative=self.realized_pnl,
            dividends_cumulative=self.total_dividends,
            deposits_cumulative=self.total_deposits,
            withdrawals_cumulative=self.total_withdrawals,
            holdings_count=len(self.holdings),
            holdings_summary=holdings_summary,
        )

        self.snapshots.append(snapshot)

        # Track benchmark
        if benchmark_price is not None:
            self.benchmark_snapshots.append((snapshot_date, benchmark_price))

        return snapshot

    # =========================================================================
    # Return Calculations
    # =========================================================================

    def get_returns(self, as_of: date | None = None) -> PerformanceMetrics:
        """
        Calculate returns for all standard time periods.

        Args:
            as_of: Calculate returns as of this date (default: latest snapshot)

        Returns:
            PerformanceMetrics with returns for all periods
        """
        if not self.snapshots:
            return PerformanceMetrics()

        # Get snapshot series
        df = pd.DataFrame([s.to_dict() for s in self.snapshots])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        if df.empty:
            return PerformanceMetrics()

        as_of_date = pd.Timestamp(as_of) if as_of else df.index[-1]

        # Calculate time-weighted returns for each period
        metrics = PerformanceMetrics()

        # 1-Day return
        metrics.return_1d = self._calculate_twr(df, as_of_date, days=1)

        # 1-Week return
        metrics.return_1w = self._calculate_twr(df, as_of_date, days=7)

        # 1-Month return
        metrics.return_1m = self._calculate_twr(df, as_of_date, days=30)

        # 3-Month return
        metrics.return_3m = self._calculate_twr(df, as_of_date, days=90)

        # YTD return
        year_start = pd.Timestamp(as_of_date.year, 1, 1)
        metrics.return_ytd = self._calculate_twr(df, as_of_date, start_date=year_start)

        # 1-Year return
        metrics.return_1y = self._calculate_twr(df, as_of_date, days=365)

        # All-time return
        metrics.return_all_time = self._calculate_twr(df, as_of_date, start_date=df.index[0])

        # Risk metrics
        if len(df) > 1:
            daily_returns = df["total_value"].pct_change().dropna()
            if len(daily_returns) > 0:
                metrics.volatility_annualized = daily_returns.std() * np.sqrt(252)

                # Sharpe ratio using configured risk-free rate
                rf_daily = self.risk_free_rate / 252
                excess_returns = daily_returns - rf_daily
                if daily_returns.std() > 0:
                    metrics.sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(
                        252
                    )

                # Sortino ratio
                downside_returns = daily_returns[daily_returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    metrics.sortino_ratio = (
                        excess_returns.mean() / downside_returns.std()
                    ) * np.sqrt(252)

                # Max drawdown
                metrics.max_drawdown, metrics.max_drawdown_duration_days = (
                    self._calculate_max_drawdown(df)
                )

        # Totals
        metrics.total_gain = self.realized_pnl + sum(
            h.unrealized_pnl for h in self.holdings.values()
        )
        metrics.total_dividends = self.total_dividends
        metrics.total_fees = self.total_fees

        # Benchmark comparison
        if self.benchmark_snapshots:
            metrics.benchmark_return_1y = self._calculate_benchmark_return(365)
            metrics.alpha_1y = metrics.return_1y - metrics.benchmark_return_1y

        return metrics

    def _calculate_twr(
        self,
        df: pd.DataFrame,
        end_date: pd.Timestamp,
        days: int | None = None,
        start_date: pd.Timestamp | None = None,
    ) -> float:
        """
        Calculate time-weighted return using GIPS-compliant geometric chain-linking.

        This implements true TWR by:
        1. Identifying external cash flows (deposits/withdrawals)
        2. Creating subperiods around each cash flow
        3. Calculating holding period return for each subperiod
        4. Geometrically linking subperiod returns

        GIPS (Global Investment Performance Standards) requires:
        - Subperiods start/end at each external cash flow
        - Beginning-of-period valuation for cash flows
        - Geometric (multiplicative) linking of subperiod returns

        TWR eliminates the impact of cash flows to show true investment performance,
        making it suitable for institutional reporting and manager comparison.
        """
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Filter to date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        period_df = df[mask]

        # Deduplicate: keep last entry per date (most current state)
        period_df = period_df[~period_df.index.duplicated(keep="last")]

        if len(period_df) < 2:
            return 0.0

        # Identify cash flow dates (where deposits or withdrawals changed)
        deposits = period_df["deposits_cumulative"]
        withdrawals = period_df["withdrawals_cumulative"]

        # Detect cash flow events (change in cumulative deposits/withdrawals)
        deposit_changes = deposits.diff().fillna(0) != 0
        withdrawal_changes = withdrawals.diff().fillna(0) != 0
        cash_flow_dates = period_df.index[deposit_changes | withdrawal_changes].tolist()

        # Build subperiod boundaries: start, all cash flow dates, end
        subperiod_dates = [period_df.index[0]]
        for cf_date in cash_flow_dates:
            if cf_date > subperiod_dates[-1]:
                subperiod_dates.append(cf_date)
        if period_df.index[-1] > subperiod_dates[-1]:
            subperiod_dates.append(period_df.index[-1])

        if len(subperiod_dates) < 2:
            return 0.0

        # Calculate and link subperiod returns
        cumulative_return = 1.0

        # Helper to safely extract scalar from potentially duplicated index
        def _get_scalar(df: pd.DataFrame, idx: pd.Timestamp, col: str) -> float:
            val = df.loc[idx, col]
            if isinstance(val, pd.Series):
                return float(val.iloc[0])
            return float(val)

        for i in range(len(subperiod_dates) - 1):
            sub_start = subperiod_dates[i]
            sub_end = subperiod_dates[i + 1]

            # Get values at subperiod boundaries
            start_value = _get_scalar(period_df, sub_start, "total_value")

            # GIPS-compliant TWR: snapshots are taken AFTER cash flows, so
            # start_value at a cash-flow boundary already includes the deposit/withdrawal.
            # The subperiod return uses the post-cash-flow value as the denominator.
            # This is correct because:
            #   - Previous subperiod ended at the pre-CF value (handled via end adjustment)
            #   - This subperiod starts at the post-CF value (the snapshot)
            # Net effect: the cash flow is excluded from return calculation.

            # Compute cash flow at the END of this subperiod (i.e. at sub_end)
            cash_flow_at_end = 0.0
            if i < len(subperiod_dates) - 2:  # Not the last subperiod
                curr_deposits = _get_scalar(period_df, sub_end, "deposits_cumulative")
                curr_withdrawals = _get_scalar(period_df, sub_end, "withdrawals_cumulative")
                start_deposits = _get_scalar(period_df, sub_start, "deposits_cumulative")
                start_withdrawals = _get_scalar(period_df, sub_start, "withdrawals_cumulative")
                cash_flow_at_end = (curr_deposits - start_deposits) - (
                    curr_withdrawals - start_withdrawals
                )

            end_value = _get_scalar(period_df, sub_end, "total_value")

            # Subtract cash flow from end value to get the pre-cash-flow portfolio value.
            # This isolates investment return from external flows.
            adjusted_end = end_value - cash_flow_at_end

            # Calculate subperiod return
            if start_value > 0:
                subperiod_return = adjusted_end / start_value
                cumulative_return *= subperiod_return

        # Convert from growth factor to return
        return cumulative_return - 1.0

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        values = df["total_value"].values
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak

        max_dd = abs(drawdown.min())

        # Duration calculation
        in_drawdown = drawdown < 0
        max_duration = 0
        current_duration = 0
        for dd in in_drawdown:
            if dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, max_duration

    def _calculate_benchmark_return(self, days: int) -> float:
        """Calculate benchmark return for period."""
        if len(self.benchmark_snapshots) < 2:
            return 0.0

        end_date = self.benchmark_snapshots[-1][0]
        start_date = end_date - timedelta(days=days)

        # Find closest snapshots
        start_price = None
        end_price = self.benchmark_snapshots[-1][1]

        for d, price in self.benchmark_snapshots:
            if d >= start_date and start_price is None:
                start_price = price
                break

        if start_price is None or start_price == 0:
            return 0.0

        return (end_price - start_price) / start_price

    # =========================================================================
    # Portfolio Analytics
    # =========================================================================

    def get_allocation(self) -> dict:
        """
        Get portfolio allocation breakdown.

        Returns:
            Dict with allocations by sector, asset class, and top holdings
        """
        total_value = self.get_total_value()
        if total_value == 0:
            return {}

        # By sector
        sector_allocation = {}
        for holding in self.holdings.values():
            sector = holding.sector
            if sector not in sector_allocation:
                sector_allocation[sector] = 0.0
            sector_allocation[sector] += holding.market_value

        sector_pct = {k: v / total_value for k, v in sector_allocation.items()}

        # By asset class
        asset_allocation = {}
        for holding in self.holdings.values():
            asset_class = holding.asset_class.value
            if asset_class not in asset_allocation:
                asset_allocation[asset_class] = 0.0
            asset_allocation[asset_class] += holding.market_value

        asset_pct = {k: v / total_value for k, v in asset_allocation.items()}

        # Cash percentage
        cash_pct = self.cash / total_value

        # Top holdings
        holdings_by_value = sorted(
            self.holdings.values(), key=lambda h: h.market_value, reverse=True
        )
        top_holdings = [
            {
                "ticker": h.ticker,
                "value": h.market_value,
                "pct": h.market_value / total_value,
                "pnl": h.unrealized_pnl,
                "pnl_pct": h.unrealized_pnl_pct,
            }
            for h in holdings_by_value[:10]
        ]

        return {
            "total_value": total_value,
            "cash": self.cash,
            "cash_pct": cash_pct,
            "invested": total_value - self.cash,
            "invested_pct": 1 - cash_pct,
            "by_sector": sector_pct,
            "by_asset_class": asset_pct,
            "top_holdings": top_holdings,
            "holdings_count": len(self.holdings),
        }

    def get_holdings_df(self) -> pd.DataFrame:
        """Get all holdings as a DataFrame."""
        if not self.holdings:
            return pd.DataFrame()

        return pd.DataFrame([h.to_dict() for h in self.holdings.values()])

    def get_transactions_df(self) -> pd.DataFrame:
        """Get all transactions as a DataFrame."""
        if not self.transactions:
            return pd.DataFrame()

        return pd.DataFrame([t.to_dict() for t in self.transactions])

    def get_total_value(self) -> float:
        """Get current total portfolio value."""
        return self.cash + sum(h.market_value for h in self.holdings.values())

    def get_daily_values(self) -> pd.DataFrame:
        """Get daily portfolio values for charting."""
        if not self.snapshots:
            return pd.DataFrame()

        df = pd.DataFrame([{"date": s.date, "value": s.total_value} for s in self.snapshots])
        return df.set_index("date")

    # =========================================================================
    # Import/Export
    # =========================================================================

    def import_from_csv(self, filepath: str | Path) -> int:
        """
        Import transactions from CSV file.

        Expected columns: date, ticker, action, shares, price, fees (optional)

        Args:
            filepath: Path to CSV file

        Returns:
            Number of transactions imported
        """
        df = pd.read_csv(filepath)
        required_cols = ["date", "ticker", "action", "shares", "price"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        count = 0
        for _, row in df.iterrows():
            txn = Transaction(
                ticker=row["ticker"],
                action=row["action"],
                shares=float(row["shares"]),
                price=float(row["price"]),
                date=pd.to_datetime(row["date"]).date(),
                fees=float(row.get("fees", 0)),
                notes=str(row.get("notes", "")),
            )
            if self.add_transaction(txn):
                count += 1

        return count

    def import_holdings(self, holdings: list[dict]) -> int:
        """
        Import current holdings directly (for initial setup).

        Args:
            holdings: List of dicts with {ticker, shares, cost_basis, current_price}

        Returns:
            Number of holdings imported
        """
        count = 0
        for h in holdings:
            self.holdings[h["ticker"]] = Holding(
                ticker=h["ticker"],
                shares=h["shares"],
                cost_basis=h["cost_basis"],
                current_price=h.get("current_price", h["cost_basis"]),
                asset_class=AssetClass(h.get("asset_class", "equity")),
                sector=self.sector_map.get(h["ticker"], h.get("sector", "Unknown")),
            )
            count += 1
        return count

    def export_to_json(self, filepath: str | Path) -> None:
        """Export portfolio data to JSON."""
        data = {
            "cash": self.cash,
            "holdings": [h.to_dict() for h in self.holdings.values()],
            "transactions": [t.to_dict() for t in self.transactions],
            "snapshots": [s.to_dict() for s in self.snapshots],
            "realized_pnl": self.realized_pnl,
            "total_dividends": self.total_dividends,
            "total_fees": self.total_fees,
            "total_deposits": self.total_deposits,
            "total_withdrawals": self.total_withdrawals,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_from_json(self, filepath: str | Path) -> None:
        """Load portfolio data from JSON."""
        with open(filepath) as f:
            data = json.load(f)

        self.cash = data["cash"]
        self.realized_pnl = data.get("realized_pnl", 0.0)
        self.total_dividends = data.get("total_dividends", 0.0)
        self.total_fees = data.get("total_fees", 0.0)
        self.total_deposits = data.get("total_deposits", 0.0)
        self.total_withdrawals = data.get("total_withdrawals", 0.0)

        # Load holdings
        self.holdings = {}
        for h in data.get("holdings", []):
            self.holdings[h["ticker"]] = Holding(
                ticker=h["ticker"],
                shares=h["shares"],
                cost_basis=h["cost_basis"],
                current_price=h.get("current_price", 0),
                asset_class=AssetClass(h.get("asset_class", "equity")),
                sector=h.get("sector", "Unknown"),
                dividends_received=h.get("dividends_received", 0),
            )

        # Load transactions
        self.transactions = []
        for t in data.get("transactions", []):
            self.transactions.append(
                Transaction(
                    ticker=t["ticker"],
                    action=t["action"],
                    shares=t["shares"],
                    price=t["price"],
                    date=date.fromisoformat(t["date"]),
                    fees=t.get("fees", 0),
                    notes=t.get("notes", ""),
                    asset_class=t.get("asset_class", "equity"),
                )
            )

    # =========================================================================
    # Reports
    # =========================================================================

    def summary_report(self) -> str:
        """Generate a summary report similar to brokerage apps."""
        metrics = self.get_returns()
        allocation = self.get_allocation()

        report = f"""
{"=" * 60}
PORTFOLIO SUMMARY
{"=" * 60}

TOTAL VALUE: ${allocation.get("total_value", 0):,.2f}
  Cash:      ${allocation.get("cash", 0):,.2f} ({allocation.get("cash_pct", 0):.1%})
  Invested:  ${allocation.get("invested", 0):,.2f} ({allocation.get("invested_pct", 0):.1%})

PERFORMANCE
{"-" * 40}
  Today:      {metrics.return_1d:+.2%}
  1 Week:     {metrics.return_1w:+.2%}
  1 Month:    {metrics.return_1m:+.2%}
  3 Months:   {metrics.return_3m:+.2%}
  YTD:        {metrics.return_ytd:+.2%}
  1 Year:     {metrics.return_1y:+.2%}
  All Time:   {metrics.return_all_time:+.2%}

RISK METRICS
{"-" * 40}
  Volatility (Ann.): {metrics.volatility_annualized:.1%}
  Sharpe Ratio:      {metrics.sharpe_ratio:.2f}
  Max Drawdown:      {metrics.max_drawdown:.1%}

TOTALS
{"-" * 40}
  Total Gain/Loss:   ${metrics.total_gain:+,.2f}
  Dividends:         ${metrics.total_dividends:,.2f}
  Fees Paid:         ${metrics.total_fees:,.2f}
  Net Deposits:      ${self.total_deposits - self.total_withdrawals:,.2f}

ALLOCATION BY SECTOR
{"-" * 40}"""

        for sector, pct in sorted(allocation.get("by_sector", {}).items(), key=lambda x: -x[1]):
            report += f"\n  {sector:25s} {pct:6.1%}"

        report += f"""

TOP HOLDINGS
{"-" * 40}"""

        for h in allocation.get("top_holdings", [])[:5]:
            report += f"\n  {h['ticker']:10s} ${h['value']:>10,.2f} ({h['pct']:5.1%})  P&L: {h['pnl_pct']:+.1%}"

        report += f"\n\n{'=' * 60}\n"

        return report

    def positions_report(self) -> str:
        """Generate detailed positions report."""
        report = f"""
{"=" * 70}
POSITIONS DETAIL
{"=" * 70}
{"Ticker":<12} {"Shares":>10} {"Cost":>10} {"Price":>10} {"Value":>12} {"P&L":>10} {"P&L%":>8}
{"-" * 70}"""

        for holding in sorted(self.holdings.values(), key=lambda h: -h.market_value):
            report += f"""
{holding.ticker:<12} {holding.shares:>10.2f} ${holding.cost_basis:>9.2f} ${holding.current_price:>9.2f} ${holding.market_value:>11,.2f} ${holding.unrealized_pnl:>9,.2f} {holding.unrealized_pnl_pct:>7.1%}"""

        report += f"\n{'-' * 70}\n"
        return report


# =============================================================================
# Quick Helper Functions
# =============================================================================


def create_portfolio_from_holdings(
    holdings: list[dict],
    cash: float = 0.0,
) -> PortfolioTracker:
    """
    Quick helper to create a portfolio from a list of holdings.

    Args:
        holdings: List of {ticker, shares, cost_basis, current_price}
        cash: Starting cash balance

    Returns:
        Initialized PortfolioTracker
    """
    tracker = PortfolioTracker(initial_cash=cash)
    tracker.import_holdings(holdings)
    return tracker


def quick_snapshot(
    tracker: PortfolioTracker,
    prices: dict[str, float],
) -> dict:
    """
    Take a snapshot and return summary metrics.

    Returns:
        Dict with total_value, returns, and allocation
    """
    tracker.snapshot(prices)

    return {
        "total_value": tracker.get_total_value(),
        "returns": tracker.get_returns().to_dict(),
        "allocation": tracker.get_allocation(),
    }
