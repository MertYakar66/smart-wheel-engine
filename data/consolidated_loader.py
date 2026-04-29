"""
Consolidated Bloomberg Data Loader

Loads Bloomberg data from consolidated CSV files (all tickers in one file).
This is the primary loader for the Smart Wheel Engine.

Data Files:
    sp500_ohlcv.csv           - Price history (OHLCV)
    sp500_vol_iv_full.csv     - IV and realized volatility history
    sp500_earnings.csv        - Earnings announcements
    sp500_dividends.csv       - Dividend history
    sp500_fundamentals.csv    - Current fundamentals snapshot
    sp500_liquidity.csv       - Volume and liquidity metrics
    sp500_vol_dvd.csv         - Volatility and dividend data
    sp500_vix_full.csv        - VIX history
    sp500_macro.csv           - Macro instrument data
    treasury_yields.csv       - Treasury yield curve
    vix_term_structure.csv    - VIX term structure
    sp500_analyst.csv         - Analyst ratings and estimates

Usage:
    from data.consolidated_loader import ConsolidatedBloombergLoader

    loader = ConsolidatedBloombergLoader()
    loader.load_all()

    # Access data
    ohlcv = loader.get_ohlcv("AAPL")
    iv_history = loader.get_iv_history("AAPL")
    fundamentals = loader.get_fundamentals("AAPL")
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default data directory
BLOOMBERG_DIR = Path("data/bloomberg")


def normalize_ticker(ticker: str) -> str:
    """
    Normalize Bloomberg ticker to simple symbol.

    Examples:
        "AAPL UW Equity" -> "AAPL"
        "AAPL UW" -> "AAPL"
        "A UN Equity" -> "A"
        "A UN" -> "A"
        "BRK/B UN Equity" -> "BRK.B"
    """
    if pd.isna(ticker):
        return ""

    ticker = str(ticker).strip()

    # Remove " Equity" suffix
    ticker = re.sub(r"\s+Equity$", "", ticker, flags=re.IGNORECASE)

    # Remove exchange suffix (UW, UN, UP, etc.)
    ticker = re.sub(r"\s+(UW|UN|UP|UA|UQ|US|UR|UV|UB|UD|UF|UH|UJ|UK|UL|UM|UY|UZ)$", "", ticker, flags=re.IGNORECASE)

    # Handle special tickers like BRK/B -> BRK.B
    ticker = ticker.replace("/", ".")

    return ticker.upper()


@dataclass
class DataStats:
    """Statistics about loaded data."""
    file_name: str
    row_count: int
    ticker_count: int
    date_range: Optional[Tuple[str, str]] = None
    columns: List[str] = field(default_factory=list)
    load_time_ms: int = 0


class ConsolidatedBloombergLoader:
    """
    Loader for consolidated Bloomberg CSV files.

    All data is stored in memory as DataFrames for fast access.
    Data is indexed by normalized ticker symbol.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = BLOOMBERG_DIR,
        auto_load: bool = False,
    ):
        self.data_dir = Path(data_dir)

        # Raw DataFrames (full files)
        self._ohlcv_df: Optional[pd.DataFrame] = None
        self._iv_history_df: Optional[pd.DataFrame] = None
        self._earnings_df: Optional[pd.DataFrame] = None
        self._dividends_df: Optional[pd.DataFrame] = None
        self._fundamentals_df: Optional[pd.DataFrame] = None
        self._liquidity_df: Optional[pd.DataFrame] = None
        self._vol_dvd_df: Optional[pd.DataFrame] = None
        self._vix_df: Optional[pd.DataFrame] = None
        self._macro_df: Optional[pd.DataFrame] = None
        self._treasury_df: Optional[pd.DataFrame] = None
        self._vix_term_df: Optional[pd.DataFrame] = None
        self._analyst_df: Optional[pd.DataFrame] = None

        # Indexed data (by ticker)
        self._ohlcv: Dict[str, pd.DataFrame] = {}
        self._iv_history: Dict[str, pd.DataFrame] = {}
        self._earnings: Dict[str, pd.DataFrame] = {}
        self._dividends: Dict[str, pd.DataFrame] = {}
        self._liquidity: Dict[str, pd.DataFrame] = {}
        self._vol_dvd: Dict[str, pd.DataFrame] = {}

        # Single DataFrames (not per-ticker)
        self._fundamentals: Dict[str, dict] = {}
        self._analyst: Dict[str, dict] = {}

        # Metadata
        self._stats: List[DataStats] = []
        self._tickers: Set[str] = set()

        if auto_load:
            self.load_all()

    def load_all(self) -> None:
        """Load all available data files."""
        logger.info(f"Loading Bloomberg data from {self.data_dir}")

        self.load_ohlcv()
        self.load_iv_history()
        self.load_earnings()
        self.load_dividends()
        self.load_fundamentals()
        self.load_liquidity()
        self.load_vol_dvd()
        self.load_vix()
        self.load_macro()
        self.load_treasury()
        self.load_vix_term_structure()
        self.load_analyst()
        self.load_index_membership()

        logger.info(f"Loaded {len(self._tickers)} unique tickers")

    def _load_csv(self, filename: str, **kwargs) -> Optional[pd.DataFrame]:
        """Load a CSV file with error handling."""
        import time
        start = time.time()

        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath, **kwargs)
            load_time = int((time.time() - start) * 1000)

            # Normalize ticker column if present
            ticker_cols = [c for c in df.columns if c.lower() == "ticker"]
            if ticker_cols:
                df["ticker_normalized"] = df[ticker_cols[0]].apply(normalize_ticker)
                self._tickers.update(df["ticker_normalized"].dropna().unique())

            # Record stats
            date_cols = [c for c in df.columns if "date" in c.lower()]
            date_range = None
            if date_cols:
                dates = pd.to_datetime(df[date_cols[0]], errors="coerce")
                if not dates.isna().all():
                    date_range = (str(dates.min().date()), str(dates.max().date()))

            self._stats.append(DataStats(
                file_name=filename,
                row_count=len(df),
                ticker_count=df["ticker_normalized"].nunique() if "ticker_normalized" in df.columns else 0,
                date_range=date_range,
                columns=list(df.columns),
                load_time_ms=load_time,
            ))

            logger.info(f"Loaded {filename}: {len(df):,} rows, {load_time}ms")
            return df

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None

    def _index_by_ticker(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> Dict[str, pd.DataFrame]:
        """Index DataFrame by normalized ticker."""
        if df is None or "ticker_normalized" not in df.columns:
            return {}

        indexed = {}
        for ticker, group in df.groupby("ticker_normalized"):
            if ticker:
                ticker_df = group.copy()
                # Parse and sort by date
                if date_col in ticker_df.columns:
                    ticker_df[date_col] = pd.to_datetime(ticker_df[date_col], errors="coerce")
                    ticker_df = ticker_df.sort_values(date_col)
                indexed[ticker] = ticker_df.reset_index(drop=True)

        return indexed

    def load_ohlcv(self) -> int:
        """Load OHLCV price data."""
        self._ohlcv_df = self._load_csv("sp500_ohlcv.csv")
        if self._ohlcv_df is not None:
            # Rename columns to standard names
            self._ohlcv_df.columns = self._ohlcv_df.columns.str.lower()
            self._ohlcv = self._index_by_ticker(self._ohlcv_df, "date")
            return len(self._ohlcv)
        return 0

    def load_iv_history(self) -> int:
        """Load IV and realized volatility history."""
        self._iv_history_df = self._load_csv("sp500_vol_iv_full.csv")
        if self._iv_history_df is not None:
            # Standardize column names
            col_map = {
                "hist_put_imp_vol": "iv_put",
                "hist_call_imp_vol": "iv_call",
                "volatility_30d": "rv_30d",
                "volatility_60d": "rv_60d",
                "volatility_90d": "rv_90d",
                "volatility_260d": "rv_260d",
            }
            self._iv_history_df = self._iv_history_df.rename(columns=col_map)
            self._iv_history_df.columns = self._iv_history_df.columns.str.lower()

            # Compute ATM IV as average of put and call
            if "iv_put" in self._iv_history_df.columns and "iv_call" in self._iv_history_df.columns:
                self._iv_history_df["atm_iv"] = (
                    self._iv_history_df["iv_put"] + self._iv_history_df["iv_call"]
                ) / 2 / 100  # Convert to decimal

            # Convert RV to decimal
            for col in ["rv_30d", "rv_60d", "rv_90d", "rv_260d"]:
                if col in self._iv_history_df.columns:
                    self._iv_history_df[col] = self._iv_history_df[col] / 100

            self._iv_history = self._index_by_ticker(self._iv_history_df, "date")
            return len(self._iv_history)
        return 0

    def load_earnings(self) -> int:
        """Load earnings announcements."""
        self._earnings_df = self._load_csv("sp500_earnings.csv")
        if self._earnings_df is not None:
            # Standardize column names
            col_map = {
                "announcement_date": "earnings_date",
                "announcement_time": "earnings_time",
                "earnings_eps": "eps_actual",
                "comparable_eps": "eps_comparable",
                "estimate_eps": "eps_estimate",
                "year/period": "fiscal_period",
            }
            self._earnings_df = self._earnings_df.rename(columns=col_map)
            self._earnings_df.columns = self._earnings_df.columns.str.lower()

            # Compute surprise
            if "eps_actual" in self._earnings_df.columns and "eps_estimate" in self._earnings_df.columns:
                self._earnings_df["eps_surprise"] = (
                    self._earnings_df["eps_actual"] - self._earnings_df["eps_estimate"]
                )
                self._earnings_df["eps_surprise_pct"] = (
                    self._earnings_df["eps_surprise"] / self._earnings_df["eps_estimate"].abs()
                ).replace([np.inf, -np.inf], np.nan)

            self._earnings = self._index_by_ticker(self._earnings_df, "earnings_date")
            return len(self._earnings)
        return 0

    def load_dividends(self) -> int:
        """Load dividend history."""
        self._dividends_df = self._load_csv("sp500_dividends.csv")
        if self._dividends_df is not None:
            self._dividends_df.columns = self._dividends_df.columns.str.lower()
            self._dividends = self._index_by_ticker(self._dividends_df, "ex_date")
            return len(self._dividends)
        return 0

    def load_fundamentals(self) -> int:
        """Load current fundamentals snapshot."""
        self._fundamentals_df = self._load_csv("sp500_fundamentals.csv")
        if self._fundamentals_df is not None:
            # Standardize column names
            col_map = {
                "30day_impvol_100.0%mny_df": "iv_30d",
                "best_pe_ratio": "pe_ratio",
                "beta_raw_overridable": "beta",
                "cur_mkt_cap": "market_cap",
                "eqy_dvd_yld_12m": "dividend_yield",
                "free_cash_flow_yield": "fcf_yield",
                "gics_industry_group_name": "industry",
                "gics_sector_name": "sector",
                "return_com_eqy": "roe",
                "tot_debt_to_tot_eqy": "debt_to_equity",
            }
            self._fundamentals_df = self._fundamentals_df.rename(columns=col_map)
            self._fundamentals_df.columns = self._fundamentals_df.columns.str.lower()

            # Index by ticker
            for _, row in self._fundamentals_df.iterrows():
                ticker = row.get("ticker_normalized", "")
                if ticker:
                    self._fundamentals[ticker] = row.to_dict()

            return len(self._fundamentals)
        return 0

    def load_liquidity(self) -> int:
        """Load liquidity data."""
        self._liquidity_df = self._load_csv("sp500_liquidity.csv")
        if self._liquidity_df is not None:
            self._liquidity_df.columns = self._liquidity_df.columns.str.lower()
            self._liquidity = self._index_by_ticker(self._liquidity_df, "date")
            return len(self._liquidity)
        return 0

    def load_vol_dvd(self) -> int:
        """Load volatility and dividend data."""
        self._vol_dvd_df = self._load_csv("sp500_vol_dvd.csv")
        if self._vol_dvd_df is not None:
            self._vol_dvd_df.columns = self._vol_dvd_df.columns.str.lower()
            self._vol_dvd = self._index_by_ticker(self._vol_dvd_df, "date")
            return len(self._vol_dvd)
        return 0

    def load_vix(self) -> int:
        """Load VIX history."""
        self._vix_df = self._load_csv("sp500_vix_full.csv")
        if self._vix_df is not None:
            self._vix_df.columns = self._vix_df.columns.str.lower()
            self._vix_df["date"] = pd.to_datetime(self._vix_df["date"], errors="coerce")
            self._vix_df = self._vix_df.sort_values("date")
            return len(self._vix_df)
        return 0

    def load_macro(self) -> int:
        """Load macro instrument data."""
        self._macro_df = self._load_csv("sp500_macro.csv")
        if self._macro_df is not None:
            self._macro_df.columns = self._macro_df.columns.str.lower()
            self._macro_df["date"] = pd.to_datetime(self._macro_df["date"], errors="coerce")
            return len(self._macro_df)
        return 0

    def load_treasury(self) -> int:
        """Load treasury yield curve."""
        self._treasury_df = self._load_csv("treasury_yields.csv")
        if self._treasury_df is not None:
            self._treasury_df.columns = self._treasury_df.columns.str.lower()
            self._treasury_df["date"] = pd.to_datetime(self._treasury_df["date"], errors="coerce")
            self._treasury_df = self._treasury_df.sort_values("date")
            return len(self._treasury_df)
        return 0

    def load_vix_term_structure(self) -> int:
        """Load VIX term structure."""
        self._vix_term_df = self._load_csv("vix_term_structure.csv")
        if self._vix_term_df is not None:
            self._vix_term_df.columns = self._vix_term_df.columns.str.lower()
            self._vix_term_df["date"] = pd.to_datetime(self._vix_term_df["date"], errors="coerce")
            self._vix_term_df = self._vix_term_df.sort_values("date")
            return len(self._vix_term_df)
        return 0

    def load_analyst(self) -> int:
        """Load analyst ratings and estimates."""
        self._analyst_df = self._load_csv("sp500_analyst.csv")
        if self._analyst_df is not None:
            self._analyst_df.columns = self._analyst_df.columns.str.lower()

            # Index by ticker
            for _, row in self._analyst_df.iterrows():
                ticker = row.get("ticker_normalized", "")
                if ticker:
                    self._analyst[ticker] = row.to_dict()

            return len(self._analyst)
        return 0

    def load_index_membership(self) -> int:
        """Load point-in-time S&P 500 membership.

        File: ``sp500_index_membership.csv`` with columns::

            member_ticker_and_exchange_code  percentage_weight  as_of_date

        We normalise the Bloomberg-style ticker into the simple symbol, parse
        ``as_of_date`` to a timestamp, and stash the result on the loader so
        callers can ask "who was in the index on 2021-06-30?".

        Without this, every backtest silently uses the *current* membership
        and inherits survivorship bias — winners stay, delisted names drop
        out, historical returns are biased upward.
        """
        self._index_membership_df = self._load_csv("sp500_index_membership.csv")
        if self._index_membership_df is None:
            self._index_membership = pd.DataFrame()
            return 0

        df = self._index_membership_df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Normalise ticker
        ticker_col = next(
            (c for c in df.columns if "member_ticker" in c or c == "ticker"),
            None,
        )
        if ticker_col is None:
            logger.warning("sp500_index_membership.csv: no member_ticker column")
            self._index_membership = pd.DataFrame()
            return 0
        df["ticker_normalized"] = df[ticker_col].apply(normalize_ticker)

        if "as_of_date" in df.columns:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
            df = df.dropna(subset=["as_of_date"])
            df = df.sort_values(["as_of_date", "ticker_normalized"])

        self._index_membership = df
        logger.info(
            "Index membership: %s rows, %s unique tickers, %s unique dates",
            len(df),
            df["ticker_normalized"].nunique(),
            df["as_of_date"].nunique() if "as_of_date" in df.columns else "?",
        )
        return len(df)

    # ==================== Accessor Methods ====================

    def get_tickers(self) -> List[str]:
        """Get all available tickers."""
        return sorted(self._tickers)

    def get_universe_as_of(
        self,
        as_of: Optional[str] = None,
        min_weight: float = 0.0,
    ) -> List[str]:
        """Return the set of tickers that were in the index on ``as_of``.

        Args:
            as_of: ISO date (YYYY-MM-DD). ``None`` = latest available snapshot.
            min_weight: Drop names below this percentage weight (0-100).
                Default 0 keeps everything; use e.g. 0.02 to restrict to the
                real active set and skip phased-out constituents that linger
                at a residual weight.

        Returns:
            Sorted list of normalised tickers. Empty list if membership data
            was not loaded.
        """
        df = getattr(self, "_index_membership", None)
        if df is None or len(df) == 0:
            return []

        if "as_of_date" in df.columns and as_of is not None:
            target = pd.to_datetime(as_of)
            # PIT semantics: the snapshot *at or before* the request date.
            available = df[df["as_of_date"] <= target]
            if available.empty:
                return []
            latest_date = available["as_of_date"].max()
            snap = available[available["as_of_date"] == latest_date]
        else:
            latest_date = df["as_of_date"].max() if "as_of_date" in df.columns else None
            snap = df[df["as_of_date"] == latest_date] if latest_date is not None else df

        if "percentage_weight" in snap.columns and min_weight > 0:
            snap = snap[snap["percentage_weight"] >= min_weight]

        out = sorted({t for t in snap["ticker_normalized"].dropna().tolist() if t})
        return out

    def was_index_member(self, ticker: str, as_of: str) -> bool:
        """True if ``ticker`` was in the index on ``as_of``, per PIT snapshot."""
        t = normalize_ticker(ticker) if " " in ticker else ticker.upper()
        return t in set(self.get_universe_as_of(as_of))

    def get_ohlcv(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a ticker."""
        ticker = normalize_ticker(ticker) if " " in ticker else ticker.upper()
        return self._ohlcv.get(ticker)

    def get_iv_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get IV and RV history for a ticker."""
        ticker = normalize_ticker(ticker) if " " in ticker else ticker.upper()
        return self._iv_history.get(ticker)

    def get_earnings(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get earnings history for a ticker."""
        ticker = normalize_ticker(ticker) if " " in ticker else ticker.upper()
        return self._earnings.get(ticker)

    def get_dividends(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get dividend history for a ticker."""
        ticker = normalize_ticker(ticker) if " " in ticker else ticker.upper()
        return self._dividends.get(ticker)

    def get_fundamentals(self, ticker: str) -> Optional[dict]:
        """Get current fundamentals for a ticker."""
        ticker = normalize_ticker(ticker) if " " in ticker else ticker.upper()
        return self._fundamentals.get(ticker)

    def get_liquidity(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get liquidity data for a ticker."""
        ticker = normalize_ticker(ticker) if " " in ticker else ticker.upper()
        return self._liquidity.get(ticker)

    def get_analyst(self, ticker: str) -> Optional[dict]:
        """Get analyst data for a ticker."""
        ticker = normalize_ticker(ticker) if " " in ticker else ticker.upper()
        return self._analyst.get(ticker)

    def get_vix(self, as_of: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get VIX history, optionally filtered to a date."""
        if self._vix_df is None:
            return None
        df = self._vix_df.copy()
        if as_of:
            df = df[df["date"] <= pd.to_datetime(as_of)]
        return df

    def get_treasury(self, as_of: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get treasury yields, optionally filtered to a date."""
        if self._treasury_df is None:
            return None
        df = self._treasury_df.copy()
        if as_of:
            df = df[df["date"] <= pd.to_datetime(as_of)]
        return df

    def get_risk_free_rate(self, tenor: str = "3m", as_of: Optional[str] = None) -> float:
        """Get risk-free rate for a tenor."""
        treasury = self.get_treasury(as_of)
        if treasury is None or treasury.empty:
            return 0.05  # Default 5%

        col = f"rate_{tenor}"
        if col not in treasury.columns:
            col = "rate_3m"  # Fallback

        rate = treasury[col].iloc[-1]
        return rate / 100 if rate > 1 else rate  # Handle percentage vs decimal

    def get_vix_term_structure(self, as_of: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get VIX term structure."""
        if self._vix_term_df is None:
            return None
        df = self._vix_term_df.copy()
        if as_of:
            df = df[df["date"] <= pd.to_datetime(as_of)]
        return df

    def get_macro(self, instrument: str, as_of: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get macro instrument data (e.g., 'us_10y')."""
        if self._macro_df is None:
            return None
        df = self._macro_df[self._macro_df["instrument"] == instrument].copy()
        if as_of:
            df = df[df["date"] <= pd.to_datetime(as_of)]
        return df

    def get_spot_price(self, ticker: str, as_of: Optional[str] = None) -> Optional[float]:
        """Get the spot price for a ticker."""
        ohlcv = self.get_ohlcv(ticker)
        if ohlcv is None or ohlcv.empty:
            return None

        if as_of:
            ohlcv = ohlcv[ohlcv["date"] <= pd.to_datetime(as_of)]

        if ohlcv.empty:
            return None

        return float(ohlcv["close"].iloc[-1])

    def get_iv_rank(self, ticker: str, window: int = 252) -> Optional[float]:
        """Compute IV rank for a ticker."""
        iv_history = self.get_iv_history(ticker)
        if iv_history is None or len(iv_history) < window:
            return None

        if "atm_iv" not in iv_history.columns:
            return None

        iv = iv_history["atm_iv"].dropna()
        if len(iv) < window:
            return None

        current_iv = iv.iloc[-1]
        historical = iv.iloc[-window:]

        rank = (historical < current_iv).sum() / len(historical) * 100
        return float(rank)

    def get_sector(self, ticker: str) -> str:
        """Get sector for a ticker."""
        fundamentals = self.get_fundamentals(ticker)
        if fundamentals is None:
            return "Unknown"
        return fundamentals.get("sector", "Unknown")

    def get_industry(self, ticker: str) -> str:
        """Get industry for a ticker."""
        fundamentals = self.get_fundamentals(ticker)
        if fundamentals is None:
            return "Unknown"
        return fundamentals.get("industry", "Unknown")

    def get_dividend_yield(self, ticker: str) -> float:
        """Get dividend yield for a ticker."""
        fundamentals = self.get_fundamentals(ticker)
        if fundamentals is None:
            return 0.0
        return fundamentals.get("dividend_yield", 0.0) / 100

    def get_beta(self, ticker: str) -> float:
        """Get beta for a ticker."""
        fundamentals = self.get_fundamentals(ticker)
        if fundamentals is None:
            return 1.0
        return fundamentals.get("beta", 1.0)

    # ==================== Utility Methods ====================

    def status(self) -> dict:
        """Get loader status."""
        return {
            "data_dir": str(self.data_dir),
            "tickers_loaded": len(self._tickers),
            "datasets": {
                "ohlcv": len(self._ohlcv),
                "iv_history": len(self._iv_history),
                "earnings": len(self._earnings),
                "dividends": len(self._dividends),
                "fundamentals": len(self._fundamentals),
                "liquidity": len(self._liquidity),
                "analyst": len(self._analyst),
                "vix": len(self._vix_df) if self._vix_df is not None else 0,
                "treasury": len(self._treasury_df) if self._treasury_df is not None else 0,
            },
            "stats": [s.__dict__ for s in self._stats],
        }

    def summary(self) -> str:
        """Get a summary of loaded data."""
        lines = [
            "Bloomberg Data Summary",
            "=" * 50,
            f"Data Directory: {self.data_dir}",
            f"Tickers Loaded: {len(self._tickers)}",
            "",
            "Datasets:",
        ]

        for stat in self._stats:
            date_str = f" ({stat.date_range[0]} to {stat.date_range[1]})" if stat.date_range else ""
            lines.append(
                f"  {stat.file_name}: {stat.row_count:,} rows, "
                f"{stat.ticker_count} tickers{date_str}"
            )

        return "\n".join(lines)


# Singleton instance
_loader: Optional[ConsolidatedBloombergLoader] = None


def get_bloomberg_loader(auto_load: bool = True) -> ConsolidatedBloombergLoader:
    """Get the default Bloomberg loader instance."""
    global _loader
    if _loader is None:
        _loader = ConsolidatedBloombergLoader(auto_load=auto_load)
    return _loader
