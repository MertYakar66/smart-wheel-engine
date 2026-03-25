"""
Master Data Pipeline

Orchestrates loading all Bloomberg data, validates it, and provides
a unified interface for the rest of the engine.

Usage:
    from data.pipeline import DataPipeline

    pipeline = DataPipeline()
    pipeline.load_all()              # Load everything
    pipeline.validate()              # Check data quality
    pipeline.status()                # Print what's loaded

    # Access data
    ohlcv = pipeline.get_ohlcv("AAPL")
    options = pipeline.get_options("AAPL")
    iv_rank = pipeline.get_iv_rank("AAPL")
    div_yield = pipeline.get_dividend_yield("AAPL")
    rfr = pipeline.get_risk_free_rate()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
import logging

import numpy as np
import pandas as pd

from .bloomberg_loader import (
    BLOOMBERG_DIR,
    load_all_ohlcv,
    load_all_options,
    load_all_earnings,
    load_all_dividends,
    load_all_iv_history,
    load_bloomberg_rates,
    load_bloomberg_fundamentals,
    compute_earnings_features,
    compute_iv_rank,
    get_annual_dividend_yield,
    get_upcoming_dividends,
    get_current_risk_free_rate,
    build_sector_map,
)

logger = logging.getLogger(__name__)


@dataclass
class DataStatus:
    """Status of loaded data."""
    ohlcv_tickers: int = 0
    ohlcv_total_days: int = 0
    options_tickers: int = 0
    options_total_contracts: int = 0
    earnings_tickers: int = 0
    earnings_total_quarters: int = 0
    dividends_tickers: int = 0
    iv_history_tickers: int = 0
    rates_loaded: bool = False
    rates_days: int = 0
    fundamentals_loaded: bool = False
    fundamentals_companies: int = 0

    def summary(self) -> str:
        """Formatted status report."""
        lines = [
            "Data Pipeline Status",
            "=" * 50,
            f"OHLCV:        {self.ohlcv_tickers} tickers, {self.ohlcv_total_days:,} total days",
            f"Options:      {self.options_tickers} tickers, {self.options_total_contracts:,} contracts",
            f"Earnings:     {self.earnings_tickers} tickers, {self.earnings_total_quarters} quarters",
            f"Dividends:    {self.dividends_tickers} tickers",
            f"IV History:   {self.iv_history_tickers} tickers",
            f"Rates:        {'Loaded' if self.rates_loaded else 'NOT LOADED (using 5% default)'} ({self.rates_days} days)",
            f"Fundamentals: {'Loaded' if self.fundamentals_loaded else 'NOT LOADED (using hardcoded sectors)'} ({self.fundamentals_companies} companies)",
        ]

        # Warnings
        warnings = []
        if self.ohlcv_tickers == 0:
            warnings.append("CRITICAL: No OHLCV data — nothing will work")
        if self.options_tickers == 0:
            warnings.append("WARNING: No options data — cannot build trade universe")
        if self.iv_history_tickers == 0:
            warnings.append("WARNING: No IV history — IV rank signals disabled")
        if self.earnings_tickers == 0:
            warnings.append("INFO: No earnings data — earnings ML model disabled")

        if warnings:
            lines.append("")
            lines.extend(warnings)

        return "\n".join(lines)


class DataPipeline:
    """
    Unified data interface for the Smart Wheel Engine.

    Loads Bloomberg CSV data, validates, and provides clean access
    methods for every module in the engine.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        tickers: Optional[List[str]] = None
    ):
        """
        Args:
            data_dir: Root Bloomberg data directory.
                      Default: data/bloomberg/
            tickers: Specific tickers to load (None = all available).
        """
        self.data_dir = Path(data_dir) if data_dir else BLOOMBERG_DIR
        self.tickers = tickers

        # Data stores
        self._ohlcv: Dict[str, pd.DataFrame] = {}
        self._options: Dict[str, pd.DataFrame] = {}
        self._earnings: Dict[str, pd.DataFrame] = {}
        self._dividends: Dict[str, pd.DataFrame] = {}
        self._iv_history: Dict[str, pd.DataFrame] = {}
        self._rates: Optional[pd.DataFrame] = None
        self._fundamentals: Optional[pd.DataFrame] = None

        # Cached computations
        self._iv_ranks: Dict[str, float] = {}
        self._div_yields: Dict[str, float] = {}
        self._sector_map: Dict[str, str] = {}
        self._risk_free_rate: Optional[float] = None

    # ─── Loading ──────────────────────────────────────────────────

    def load_all(self) -> 'DataPipeline':
        """Load all available Bloomberg data. Returns self for chaining."""
        logger.info(f"Loading all Bloomberg data from {self.data_dir}")

        self.load_ohlcv()
        self.load_options()
        self.load_earnings()
        self.load_dividends()
        self.load_iv_history()
        self.load_rates()
        self.load_fundamentals()

        # Pre-compute derived values
        self._compute_iv_ranks()
        self._compute_div_yields()
        self._compute_sector_map()

        status = self.status()
        logger.info(f"\n{status.summary()}")
        return self

    def load_ohlcv(self) -> None:
        """Load OHLCV price data."""
        ohlcv_dir = self.data_dir / "ohlcv"
        self._ohlcv = load_all_ohlcv(self.tickers, ohlcv_dir)

        # Also check legacy data_raw directory
        if not self._ohlcv:
            legacy_dir = Path("data_raw/ohlcv")
            if legacy_dir.exists():
                logger.info("Falling back to data_raw/ohlcv (legacy yfinance data)")
                self._ohlcv = load_all_ohlcv(self.tickers, legacy_dir)

    def load_options(self) -> None:
        """Load option chain data."""
        opts_dir = self.data_dir / "options"
        if not opts_dir.exists():
            return

        all_opts = load_all_options(self.tickers, data_dir=opts_dir)
        if all_opts.empty:
            return

        # Group by ticker
        for ticker, group in all_opts.groupby("ticker"):
            self._options[ticker] = group.reset_index(drop=True)

    def load_earnings(self) -> None:
        """Load earnings data."""
        earn_dir = self.data_dir / "earnings"
        if not earn_dir.exists():
            return

        all_earn = load_all_earnings(self.tickers, earn_dir)
        if all_earn.empty:
            return

        for ticker, group in all_earn.groupby("ticker"):
            self._earnings[ticker] = group.reset_index(drop=True)

    def load_dividends(self) -> None:
        """Load dividend data."""
        div_dir = self.data_dir / "dividends"
        if not div_dir.exists():
            return

        all_div = load_all_dividends(self.tickers, div_dir)
        if all_div.empty:
            return

        for ticker, group in all_div.groupby("ticker"):
            self._dividends[ticker] = group.reset_index(drop=True)

    def load_iv_history(self) -> None:
        """Load IV history data."""
        iv_dir = self.data_dir / "iv_history"
        self._iv_history = load_all_iv_history(self.tickers, iv_dir)

    def load_rates(self) -> None:
        """Load treasury rate data."""
        rates_dir = self.data_dir / "rates"
        self._rates = load_bloomberg_rates(rates_dir)
        if self._rates is not None:
            self._risk_free_rate = get_current_risk_free_rate(self._rates)

    def load_fundamentals(self) -> None:
        """Load company fundamentals."""
        fund_dir = self.data_dir / "fundamentals"
        self._fundamentals = load_bloomberg_fundamentals(fund_dir)

    # ─── Pre-computations ────────────────────────────────────────

    def _compute_iv_ranks(self) -> None:
        """Compute IV rank for all tickers with IV history."""
        for ticker, iv_df in self._iv_history.items():
            rank = compute_iv_rank(iv_df)
            if rank is not None:
                self._iv_ranks[ticker] = rank

    def _compute_div_yields(self) -> None:
        """Compute dividend yields for all tickers."""
        for ticker, div_df in self._dividends.items():
            ohlcv = self._ohlcv.get(ticker)
            if ohlcv is not None and not ohlcv.empty:
                spot = float(ohlcv.iloc[-1]["Close"])
                self._div_yields[ticker] = get_annual_dividend_yield(
                    ticker, div_df, spot
                )

    def _compute_sector_map(self) -> None:
        """Build sector map from fundamentals."""
        if self._fundamentals is not None:
            self._sector_map = build_sector_map(self._fundamentals)

    # ─── Access Methods ──────────────────────────────────────────

    def get_ohlcv(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get OHLCV DataFrame for a ticker."""
        return self._ohlcv.get(ticker)

    def get_spot_price(self, ticker: str, as_of: Optional[str] = None) -> Optional[float]:
        """
        Get spot price for a ticker.

        Args:
            ticker: Stock ticker.
            as_of: Date string (YYYY-MM-DD). None = latest.

        Returns:
            Close price or None.
        """
        df = self._ohlcv.get(ticker)
        if df is None or df.empty:
            return None

        if as_of:
            target = pd.to_datetime(as_of)
            df_sub = df[df["Date"] <= target]
            if df_sub.empty:
                return None
            return float(df_sub.iloc[-1]["Close"])

        return float(df.iloc[-1]["Close"])

    def get_all_spot_prices(self, as_of: Optional[str] = None) -> Dict[str, float]:
        """Get spot prices for all loaded tickers."""
        prices = {}
        for ticker in self._ohlcv:
            price = self.get_spot_price(ticker, as_of)
            if price is not None:
                prices[ticker] = price
        return prices

    def get_options(
        self,
        ticker: str,
        min_dte: int = 0,
        max_dte: int = 999
    ) -> Optional[pd.DataFrame]:
        """
        Get option chain for a ticker, optionally filtered by DTE.

        Returns DataFrame with columns: strike, option_type, expiration,
        bid, ask, implied_vol, open_interest, volume, delta, mid_price, etc.
        """
        df = self._options.get(ticker)
        if df is None or df.empty:
            return None

        if "expiration" in df.columns and "date" in df.columns:
            df = df.copy()
            df["dte"] = (
                pd.to_datetime(df["expiration"]) -
                pd.to_datetime(df["date"])
            ).dt.days
            df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]

        return df

    def get_all_options(self) -> pd.DataFrame:
        """Get all option chains concatenated."""
        if not self._options:
            return pd.DataFrame()
        return pd.concat(self._options.values(), ignore_index=True)

    def get_earnings(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get earnings history for a ticker."""
        return self._earnings.get(ticker)

    def get_next_earnings_date(self, ticker: str) -> Optional[date]:
        """Get the next upcoming earnings date for a ticker."""
        df = self._earnings.get(ticker)
        if df is None or df.empty:
            return None

        today = pd.Timestamp.now()
        future = df[df["earnings_date"] >= today]
        if future.empty:
            return None

        return future.iloc[0]["earnings_date"].date()

    def get_earnings_features(self, ticker: str) -> Optional[dict]:
        """
        Get computed earnings features for ml/earnings_model.py.

        Returns dict compatible with EarningsFeatures dataclass.
        """
        earnings = self._earnings.get(ticker)
        ohlcv = self._ohlcv.get(ticker)
        iv = self._iv_history.get(ticker)

        if earnings is None or ohlcv is None:
            return None

        return compute_earnings_features(ticker, earnings, ohlcv, iv)

    def get_dividends(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get dividend history for a ticker."""
        return self._dividends.get(ticker)

    def get_dividend_yield(self, ticker: str) -> float:
        """
        Get annualized dividend yield for BSM pricing.

        Returns 0.0 if no dividend data available.
        """
        return self._div_yields.get(ticker, 0.0)

    def get_upcoming_dividends_for_ticker(
        self,
        ticker: str,
        horizon_days: int = 60
    ) -> pd.DataFrame:
        """
        Get upcoming ex-dividend dates for LSM early exercise analysis.

        Returns DataFrame with ex_date, amount columns.
        """
        div_df = self._dividends.get(ticker)
        if div_df is None:
            return pd.DataFrame()
        return get_upcoming_dividends(div_df, horizon_days)

    def get_iv_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get IV history DataFrame for a ticker."""
        return self._iv_history.get(ticker)

    def get_iv_rank(self, ticker: str) -> Optional[float]:
        """
        Get IV rank (52-week percentile) for a ticker.

        Returns value in [0, 1] or None if no IV history.
        """
        return self._iv_ranks.get(ticker)

    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate.

        Returns 0.05 if no rates data loaded.
        """
        if self._risk_free_rate is not None:
            return self._risk_free_rate
        return 0.05

    def get_sector(self, ticker: str) -> Optional[str]:
        """Get GICS sector for a ticker."""
        return self._sector_map.get(ticker)

    def get_sector_map(self) -> Dict[str, str]:
        """
        Get full ticker → sector mapping.

        Falls back to hardcoded DEFAULT_SECTOR_MAP if no fundamentals loaded.
        """
        if self._sector_map:
            return self._sector_map

        # Fallback to hardcoded map
        from engine.risk_manager import DEFAULT_SECTOR_MAP
        return DEFAULT_SECTOR_MAP

    def get_daily_returns(self, ticker: str) -> Optional[np.ndarray]:
        """
        Get daily returns array for Monte Carlo inputs.

        Returns numpy array of daily percentage returns.
        """
        df = self._ohlcv.get(ticker)
        if df is None or len(df) < 2:
            return None

        returns = df["Close"].pct_change().dropna().values
        return returns

    def get_portfolio_returns(
        self,
        tickers: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Optional[np.ndarray]:
        """
        Get equal-weighted or custom-weighted portfolio returns.

        Useful for block bootstrap Monte Carlo.
        """
        if tickers is None:
            tickers = list(self._ohlcv.keys())

        returns_list = []
        valid_tickers = []
        for ticker in tickers:
            r = self.get_daily_returns(ticker)
            if r is not None and len(r) > 0:
                returns_list.append(r)
                valid_tickers.append(ticker)

        if not returns_list:
            return None

        # Align to shortest
        min_len = min(len(r) for r in returns_list)
        aligned = np.column_stack([r[-min_len:] for r in returns_list])

        if weights:
            w = np.array([weights.get(t, 1.0 / len(valid_tickers))
                         for t in valid_tickers])
            w = w / w.sum()
        else:
            w = np.ones(len(valid_tickers)) / len(valid_tickers)

        portfolio_returns = aligned @ w
        return portfolio_returns

    # ─── For trade_universe.py compatibility ─────────────────────

    def get_ohlcv_for_backtester(self) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data formatted for backtests/simulator.py.

        Returns dict of {ticker: DataFrame with 'date' and 'close' columns}.
        """
        result = {}
        for ticker, df in self._ohlcv.items():
            bt_df = pd.DataFrame({
                "date": df["Date"].dt.date,
                "close": df["Close"]
            })
            result[ticker] = bt_df
        return result

    def build_trade_universe(
        self,
        trade_date: str,
        min_dte: int = 21,
        max_dte: int = 60,
        min_open_interest: int = 100,
        min_bid: float = 0.05
    ) -> pd.DataFrame:
        """
        Build trade universe from Bloomberg option data.

        Replaces features/trade_universe.py's yfinance-based flow.
        """
        all_options = self.get_all_options()
        if all_options.empty:
            logger.warning("No options data loaded — cannot build universe")
            return pd.DataFrame()

        df = all_options.copy()

        # Add underlying price if missing
        if "underlying_price" not in df.columns or df["underlying_price"].isna().all():
            prices = self.get_all_spot_prices(trade_date)
            df["underlying_price"] = df["ticker"].map(prices)

        # Compute DTE
        if "dte" not in df.columns:
            df["dte"] = (
                pd.to_datetime(df["expiration"]) -
                pd.to_datetime(trade_date)
            ).dt.days

        # Compute moneyness
        df["moneyness_pct"] = (
            (df["strike"] / df["underlying_price"] - 1.0) * 100.0
        )

        # Mid price
        if "mid_price" not in df.columns:
            df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

        # Spread
        df["bid_ask_spread"] = df["ask"] - df["bid"]
        df["spread_pct"] = df["bid_ask_spread"] / df["mid_price"].replace(0, np.nan)

        df["date"] = trade_date

        # Filter puts
        put_mask = (
            (df["option_type"] == "P") &
            df["underlying_price"].notna() &
            df["implied_vol"].notna() &
            (df["implied_vol"] > 0) &
            (df["dte"].between(min_dte, max_dte)) &
            (df["moneyness_pct"].between(-15, 5)) &
            (df["open_interest"] >= min_open_interest) &
            (df["bid"] >= min_bid)
        )
        puts = df[put_mask].copy()
        puts["strategy_leg"] = "short_put"

        # Filter calls
        call_mask = (
            (df["option_type"] == "C") &
            df["underlying_price"].notna() &
            df["implied_vol"].notna() &
            (df["implied_vol"] > 0) &
            (df["dte"].between(min_dte, max_dte)) &
            (df["moneyness_pct"].between(0, 10)) &
            (df["open_interest"] >= min_open_interest) &
            (df["bid"] >= min_bid)
        )
        calls = df[call_mask].copy()
        calls["strategy_leg"] = "covered_call"

        universe = pd.concat([puts, calls], ignore_index=True)
        logger.info(
            f"Trade universe for {trade_date}: {len(universe)} candidates "
            f"({len(puts)} puts, {len(calls)} calls)"
        )
        return universe

    # ─── Status ──────────────────────────────────────────────────

    def status(self) -> DataStatus:
        """Get current data loading status."""
        s = DataStatus()

        s.ohlcv_tickers = len(self._ohlcv)
        s.ohlcv_total_days = sum(len(df) for df in self._ohlcv.values())

        s.options_tickers = len(self._options)
        s.options_total_contracts = sum(len(df) for df in self._options.values())

        s.earnings_tickers = len(self._earnings)
        s.earnings_total_quarters = sum(len(df) for df in self._earnings.values())

        s.dividends_tickers = len(self._dividends)

        s.iv_history_tickers = len(self._iv_history)

        s.rates_loaded = self._rates is not None and not self._rates.empty
        s.rates_days = len(self._rates) if s.rates_loaded else 0

        s.fundamentals_loaded = self._fundamentals is not None and not self._fundamentals.empty
        s.fundamentals_companies = len(self._fundamentals) if s.fundamentals_loaded else 0

        return s

    def loaded_tickers(self) -> List[str]:
        """Get list of all tickers with OHLCV data."""
        return sorted(self._ohlcv.keys())

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate all loaded data and return issues by category.

        Returns dict: category → list of issue strings.
        """
        issues: Dict[str, List[str]] = {}

        # OHLCV checks
        ohlcv_issues = []
        for ticker, df in self._ohlcv.items():
            if len(df) < 252:
                ohlcv_issues.append(
                    f"{ticker}: Only {len(df)} days (need 252 for 1 year)"
                )
            if df["Close"].isna().any():
                n_na = df["Close"].isna().sum()
                ohlcv_issues.append(f"{ticker}: {n_na} missing Close values")
        if ohlcv_issues:
            issues["ohlcv"] = ohlcv_issues

        # Options checks
        opt_issues = []
        for ticker, df in self._options.items():
            if "implied_vol" in df.columns:
                bad_iv = df["implied_vol"].isna().sum()
                if bad_iv > len(df) * 0.2:
                    opt_issues.append(
                        f"{ticker}: {bad_iv}/{len(df)} contracts missing IV"
                    )
            if "bid" in df.columns:
                zero_bid = (df["bid"] <= 0).sum()
                if zero_bid > len(df) * 0.5:
                    opt_issues.append(
                        f"{ticker}: {zero_bid}/{len(df)} contracts with zero bid"
                    )
        if opt_issues:
            issues["options"] = opt_issues

        # Earnings checks
        earn_issues = []
        for ticker, df in self._earnings.items():
            if len(df) < 4:
                earn_issues.append(
                    f"{ticker}: Only {len(df)} quarters (need 4 for features)"
                )
        if earn_issues:
            issues["earnings"] = earn_issues

        # Cross-data checks
        cross_issues = []
        opt_tickers = set(self._options.keys())
        ohlcv_tickers = set(self._ohlcv.keys())
        missing_ohlcv = opt_tickers - ohlcv_tickers
        if missing_ohlcv:
            cross_issues.append(
                f"Options exist without OHLCV for: {', '.join(sorted(missing_ohlcv))}"
            )
        if cross_issues:
            issues["cross_validation"] = cross_issues

        if not issues:
            logger.info("All data validation passed")
        else:
            total = sum(len(v) for v in issues.values())
            logger.warning(f"Data validation: {total} issues found")

        return issues
