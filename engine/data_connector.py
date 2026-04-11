"""
Unified Data Connector for Bloomberg Market Data

Loads, normalizes, and serves Bloomberg CSV data to all engine modules.
Handles ticker normalization (Bloomberg format -> standard),
date parsing, lazy loading with caching, and provides query methods
for OHLCV, volatility/IV, events, rates, fundamentals, and screening.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bloomberg exchange suffixes to strip (exchange codes for US equities)
_EXCHANGE_SUFFIXES = {
    "UW",  # NASDAQ Global Select
    "UN",  # NYSE
    "UQ",  # NASDAQ Global Market
    "UA",  # NYSE American (AMEX)
    "UP",  # NYSE Arca
    "UR",  # NASDAQ Capital Market
    "US",  # generic US equity
}


def normalize_ticker(bbg_ticker: str) -> str:
    """Convert Bloomberg ticker to standard symbol.

    Examples:
        'AAPL UW Equity' -> 'AAPL'
        'A UN'           -> 'A'
        'BRK/B UN Equity'-> 'BRK/B'
        'AAPL'           -> 'AAPL'  (already standard)
    """
    if not isinstance(bbg_ticker, str):
        return str(bbg_ticker)

    ticker = bbg_ticker.strip()

    # Strip trailing " Equity"
    if ticker.endswith(" Equity"):
        ticker = ticker[: -len(" Equity")].strip()

    # Strip exchange suffix (last token if it matches known suffixes)
    parts = ticker.rsplit(" ", 1)
    if len(parts) == 2 and parts[1] in _EXCHANGE_SUFFIXES:
        ticker = parts[0]

    return ticker


class MarketDataConnector:
    """Unified data layer for Smart Wheel Engine.

    Loads, normalizes, and serves Bloomberg data to all engine modules.
    Handles ticker normalization (Bloomberg format -> standard),
    date parsing, and provides query methods.

    All date parameters accept ``"YYYY-MM-DD"`` strings.  Internally
    they are converted to ``pd.Timestamp`` for filtering.

    DataFrames are lazy-loaded on first access and cached for the
    lifetime of the connector instance.  If a CSV file is missing,
    methods return empty DataFrames or ``None`` rather than raising.
    """

    # CSV file names expected under ``data_dir``
    _FILES = {
        "ohlcv": "sp500_ohlcv.csv",
        "vol_iv": "sp500_vol_iv_full.csv",
        "dividends": "sp500_dividends.csv",
        "earnings": "sp500_earnings.csv",
        "treasury": "treasury_yields.csv",
        "vix": "vix_term_structure.csv",
        "fundamentals": "sp500_fundamentals.csv",
        "credit_risk": "sp500_credit_risk.csv",
        "liquidity": "sp500_liquidity.csv",
    }

    def __init__(self, data_dir: str = "data/bloomberg") -> None:
        self._data_dir = Path(data_dir)
        # Cache for loaded DataFrames – keys match ``_FILES`` keys
        self._cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, key: str) -> pd.DataFrame:
        """Lazy-load a CSV by key, normalize tickers, cache the result."""
        if key in self._cache:
            return self._cache[key]

        path = self._data_dir / self._FILES[key]
        if not path.exists():
            logger.warning("Data file not found: %s", path)
            self._cache[key] = pd.DataFrame()
            return self._cache[key]

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            logger.exception("Failed to read %s", path)
            self._cache[key] = pd.DataFrame()
            return self._cache[key]

        # Normalize ticker column if present
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].apply(normalize_ticker)

        # Parse common date columns
        for col in (
            "date",
            "ex_date",
            "declared_date",
            "record_date",
            "payable_date",
            "announcement_date",
        ):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        self._cache[key] = df
        logger.info("Loaded %s: %d rows from %s", key, len(df), path)
        return df

    @staticmethod
    def _to_ts(value: str | None) -> pd.Timestamp | None:
        """Convert an optional date string to Timestamp."""
        if value is None:
            return None
        return pd.Timestamp(value)

    @staticmethod
    def _filter_dates(
        df: pd.DataFrame,
        date_col: str,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        """Filter a DataFrame by a date range (inclusive)."""
        if df.empty:
            return df
        out = df
        if start is not None:
            out = out[out[date_col] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out[date_col] <= pd.Timestamp(end)]
        return out

    @staticmethod
    def _filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Filter DataFrame to a single (already-normalized) ticker."""
        if df.empty or "ticker" not in df.columns:
            return df
        return df[df["ticker"] == ticker]

    # ------------------------------------------------------------------
    # Ticker normalization (public static)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_ticker(bbg_ticker: str) -> str:
        """Convert ``'AAPL UW Equity'`` or ``'AAPL UW'`` -> ``'AAPL'``."""
        return normalize_ticker(bbg_ticker)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV data for *ticker*.

        Returns a DataFrame with a ``DatetimeIndex`` named ``date`` and
        columns ``open, high, low, close, volume``.
        """
        df = self._load("ohlcv")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "date", start_date, end_date)
        if df.empty:
            return df
        # Bloomberg CSV columns are mislabeled in source data:
        #   CSV "open" actually contains HIGH prices
        #   CSV "high" actually contains CLOSE prices
        #   CSV "close" actually contains OPEN prices
        #   CSV "low" is correct
        df = df.rename(
            columns={
                "open": "high",
                "high": "close",
                "close": "open",
            }
        )
        out = (
            df[["date", "open", "high", "low", "close", "volume"]]
            .sort_values("date")
            .set_index("date")
        )
        return out

    # ------------------------------------------------------------------
    # Volatility & IV
    # ------------------------------------------------------------------

    def get_iv_history(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get historical IV data for *ticker*.

        Columns: ``hist_put_imp_vol, hist_call_imp_vol,
        volatility_30d, volatility_60d, volatility_90d, volatility_260d``.
        """
        df = self._load("vol_iv")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "date", start_date, end_date)
        if df.empty:
            return df
        value_cols = [
            "date",
            "hist_put_imp_vol",
            "hist_call_imp_vol",
            "volatility_30d",
            "volatility_60d",
            "volatility_90d",
            "volatility_260d",
        ]
        present = [c for c in value_cols if c in df.columns]
        return df[present].sort_values("date").set_index("date")

    def _iv_series(self, ticker: str, as_of: str | None, lookback_days: int) -> pd.Series:
        """Return the 30-day IV series for *ticker* up to *as_of*."""
        end = as_of
        if as_of is not None:
            start = str(pd.Timestamp(as_of) - pd.Timedelta(days=int(lookback_days * 1.6)))[:10]
        else:
            start = None
        iv = self.get_iv_history(ticker, start_date=start, end_date=end)
        if iv.empty or "hist_put_imp_vol" not in iv.columns:
            return pd.Series(dtype=float)
        # Use average of put and call IV as the composite IV
        series = (iv["hist_put_imp_vol"] + iv["hist_call_imp_vol"]) / 2.0
        series = series.dropna()
        # Trim to exact lookback window
        if as_of is not None:
            cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days)
            series = series[series.index >= cutoff]
        else:
            series = series.iloc[-lookback_days:]
        return series

    def get_iv_rank(
        self,
        ticker: str,
        as_of: str | None = None,
        lookback_days: int = 252,
    ) -> float:
        """IV Rank: ``(current - min) / (max - min)`` over *lookback_days*.

        Returns a value in ``[0, 1]`` or ``NaN`` if insufficient data.
        """
        series = self._iv_series(ticker, as_of, lookback_days)
        if series.empty:
            return float("nan")
        current = series.iloc[-1]
        lo, hi = series.min(), series.max()
        if hi == lo:
            return 0.5
        return float((current - lo) / (hi - lo))

    def get_iv_percentile(
        self,
        ticker: str,
        as_of: str | None = None,
        lookback_days: int = 252,
    ) -> float:
        """IV Percentile: fraction of days with IV below the current level.

        Returns a value in ``[0, 1]`` or ``NaN`` if insufficient data.
        """
        series = self._iv_series(ticker, as_of, lookback_days)
        if series.empty:
            return float("nan")
        current = series.iloc[-1]
        return float((series < current).sum() / len(series))

    def get_vol_risk_premium(self, ticker: str, as_of: str | None = None) -> float:
        """Volatility risk premium: IV - RV (30-day windows).

        Positive values indicate that implied vol exceeds realized vol,
        which is the typical premium harvested by option sellers.
        """
        end = as_of
        iv = self.get_iv_history(ticker, end_date=end)
        if iv.empty:
            return float("nan")
        last = iv.iloc[-1]
        impl = np.nanmean(
            [last.get("hist_put_imp_vol", np.nan), last.get("hist_call_imp_vol", np.nan)]
        )
        realized = last.get("volatility_30d", np.nan)
        if np.isnan(impl) or np.isnan(realized):
            return float("nan")
        return float(impl - realized)

    # ------------------------------------------------------------------
    # Events – Earnings
    # ------------------------------------------------------------------

    def get_earnings(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get earnings announcements for *ticker*.

        Columns: ``year_period, announcement_date, announcement_time,
        earnings_eps, comparable_eps, estimate_eps``.
        """
        df = self._load("earnings")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "announcement_date", start_date, end_date)
        if df.empty:
            return df
        rename = {}
        if "year/period" in df.columns:
            rename["year/period"] = "year_period"
        out = df.rename(columns=rename).sort_values("announcement_date")
        cols = [
            c
            for c in [
                "year_period",
                "announcement_date",
                "announcement_time",
                "earnings_eps",
                "comparable_eps",
                "estimate_eps",
            ]
            if c in out.columns
        ]
        return out[cols].reset_index(drop=True)

    def get_next_earnings(self, ticker: str, as_of: str | None = None) -> dict | None:
        """Return the next upcoming earnings event after *as_of*.

        Returns a dict with keys ``announcement_date``,
        ``announcement_time``, ``estimate_eps``, and ``year_period``
        or ``None`` if no future earnings are found.
        """
        ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
        df = self._load("earnings")
        df = self._filter_ticker(df, ticker)
        if df.empty or "announcement_date" not in df.columns:
            return None
        future = df[df["announcement_date"] > ref].sort_values("announcement_date")
        if future.empty:
            return None
        row = future.iloc[0]
        return {
            "announcement_date": row["announcement_date"],
            "announcement_time": row.get("announcement_time"),
            "estimate_eps": row.get("estimate_eps"),
            "year_period": row.get("year/period", row.get("year_period")),
        }

    # ------------------------------------------------------------------
    # Events – Dividends
    # ------------------------------------------------------------------

    def get_dividends(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get dividend history for *ticker*.

        Filtering uses the ``ex_date`` column.  Columns returned:
        ``declared_date, ex_date, record_date, payable_date,
        dividend_amount, dividend_frequency, dividend_type``.
        """
        df = self._load("dividends")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "ex_date", start_date, end_date)
        if df.empty:
            return df
        cols = [
            c
            for c in [
                "declared_date",
                "ex_date",
                "record_date",
                "payable_date",
                "dividend_amount",
                "dividend_frequency",
                "dividend_type",
            ]
            if c in df.columns
        ]
        return df[cols].sort_values("ex_date").reset_index(drop=True)

    def get_next_dividend(self, ticker: str, as_of: str | None = None) -> dict | None:
        """Return the next upcoming ex-dividend event after *as_of*.

        Returns a dict with ``ex_date``, ``dividend_amount``,
        ``payable_date``, ``dividend_frequency`` or ``None``.
        """
        ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
        df = self._load("dividends")
        df = self._filter_ticker(df, ticker)
        if df.empty or "ex_date" not in df.columns:
            return None
        future = df[df["ex_date"] > ref].sort_values("ex_date")
        if future.empty:
            return None
        row = future.iloc[0]
        return {
            "ex_date": row["ex_date"],
            "dividend_amount": row.get("dividend_amount"),
            "payable_date": row.get("payable_date"),
            "dividend_frequency": row.get("dividend_frequency"),
        }

    # ------------------------------------------------------------------
    # Rates
    # ------------------------------------------------------------------

    def get_risk_free_rate(self, as_of: str | None = None, tenor: str = "rate_3m") -> float:
        """Get risk-free rate from treasury yields.

        *tenor* must be one of ``rate_3m``, ``rate_6m``, ``rate_2y``,
        ``rate_10y``.  Returns the rate as a percentage (e.g. 4.5 means
        4.5%) or ``NaN`` if data is unavailable.
        """
        df = self._load("treasury")
        if df.empty or tenor not in df.columns:
            return float("nan")
        df = df.dropna(subset=[tenor])
        if as_of is not None:
            df = df[df["date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return float("nan")
        return float(df.sort_values("date").iloc[-1][tenor])

    # ------------------------------------------------------------------
    # VIX
    # ------------------------------------------------------------------

    def get_vix(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get VIX and term-structure history.

        Columns: ``vix, vix_3m, vix_6m`` with ``DatetimeIndex``.
        """
        df = self._load("vix")
        df = self._filter_dates(df, "date", start_date, end_date)
        if df.empty:
            return df
        cols = [c for c in ["date", "vix", "vix_3m", "vix_6m"] if c in df.columns]
        return df[cols].sort_values("date").set_index("date")

    def get_vix_regime(self, as_of: str | None = None) -> dict:
        """Characterize the VIX regime as of a given date.

        Returns a dict with:
        - ``vix``: current VIX level
        - ``vix_percentile``: percentile over full history (0-1)
        - ``term_structure``: ``'contango'`` or ``'backwardation'``
        - ``vix_3m``, ``vix_6m``: longer-dated VIX values
        """
        df = self._load("vix")
        if df.empty or "vix" not in df.columns:
            return {
                "vix": float("nan"),
                "vix_percentile": float("nan"),
                "term_structure": "unknown",
                "vix_3m": float("nan"),
                "vix_6m": float("nan"),
            }

        if as_of is not None:
            df = df[df["date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return {
                "vix": float("nan"),
                "vix_percentile": float("nan"),
                "term_structure": "unknown",
                "vix_3m": float("nan"),
                "vix_6m": float("nan"),
            }

        df = df.sort_values("date")
        last = df.iloc[-1]
        vix_val = float(last["vix"])
        all_vix = df["vix"].dropna()
        pct = float((all_vix < vix_val).sum() / len(all_vix)) if len(all_vix) > 0 else float("nan")
        vix_3m = float(last.get("vix_3m", np.nan))
        vix_6m = float(last.get("vix_6m", np.nan))

        if np.isnan(vix_3m):
            ts = "unknown"
        elif vix_val < vix_3m:
            ts = "contango"
        elif vix_val > vix_3m:
            ts = "backwardation"
        else:
            ts = "flat"

        return {
            "vix": vix_val,
            "vix_percentile": pct,
            "term_structure": ts,
            "vix_3m": vix_3m,
            "vix_6m": vix_6m,
        }

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals(self, ticker: str) -> dict | None:
        """Get a fundamentals snapshot for *ticker*.

        Returns a dict with keys such as ``pe_ratio``, ``beta``,
        ``market_cap``, ``sector``, ``industry_group``,
        ``dividend_yield``, ``fcf_yield``, ``roe``,
        ``debt_to_equity``, ``volatility_30d``, ``implied_vol_atm``,
        or ``None`` if the ticker is not found.
        """
        df = self._load("fundamentals")
        if df.empty:
            return None
        row = df[df["ticker"] == ticker]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "ticker": ticker,
            "pe_ratio": r.get("pe_ratio"),
            "best_pe_ratio": r.get("best_pe_ratio"),
            "beta": r.get("beta_raw_overridable"),
            "market_cap": r.get("cur_mkt_cap"),
            "dividend_yield": r.get("eqy_dvd_yld_12m"),
            "fcf_yield": r.get("free_cash_flow_yield"),
            "roe": r.get("return_com_eqy"),
            "debt_to_equity": r.get("tot_debt_to_tot_eqy"),
            "sector": r.get("gics_sector_name"),
            "industry_group": r.get("gics_industry_group_name"),
            "volatility_30d": r.get("volatility_30d"),
            "implied_vol_atm": r.get("30day_impvol_100.0%mny_df"),
        }

    def get_credit_risk(self, ticker: str) -> dict | None:
        """Get credit risk metrics for *ticker*.

        Returns a dict with ``altman_z_score``,
        ``interest_coverage_ratio``, ``sp_rating`` or ``None``.
        """
        df = self._load("credit_risk")
        if df.empty:
            return None
        row = df[df["ticker"] == ticker]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "ticker": ticker,
            "altman_z_score": r.get("altman_z_score"),
            "interest_coverage_ratio": r.get("interest_coverage_ratio"),
            "sp_rating": r.get("rtg_sp_lt_lc_issuer_credit"),
        }

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(self) -> list[str]:
        """Get all available tickers (normalized, deduplicated, sorted)."""
        tickers: set[str] = set()
        for key in ("ohlcv", "fundamentals", "vol_iv"):
            df = self._load(key)
            if not df.empty and "ticker" in df.columns:
                tickers.update(df["ticker"].dropna().unique())
        return sorted(tickers)

    def screen_universe(
        self,
        min_market_cap: float = 0,
        max_pe: float | None = None,
        sectors: list[str] | None = None,
        min_iv_rank: float | None = None,
        max_beta: float | None = None,
    ) -> pd.DataFrame:
        """Screen the universe by fundamental and volatility criteria.

        Returns a DataFrame with one row per ticker that passes all
        filters.  Columns include ``ticker``, ``pe_ratio``, ``beta``,
        ``market_cap``, ``sector``, ``iv_rank`` (if requested).
        """
        df = self._load("fundamentals")
        if df.empty:
            return pd.DataFrame()

        result = df.copy()

        # Market cap filter
        if "cur_mkt_cap" in result.columns:
            result = result[result["cur_mkt_cap"].fillna(0) >= min_market_cap]

        # P/E filter
        if max_pe is not None and "pe_ratio" in result.columns:
            result = result[result["pe_ratio"].fillna(float("inf")) <= max_pe]

        # Beta filter
        if max_beta is not None and "beta_raw_overridable" in result.columns:
            result = result[result["beta_raw_overridable"].fillna(float("inf")) <= max_beta]

        # Sector filter
        if sectors is not None and "gics_sector_name" in result.columns:
            sectors_lower = {s.lower() for s in sectors}
            result = result[result["gics_sector_name"].str.lower().fillna("").isin(sectors_lower)]

        # Build output frame with friendly column names
        out_cols = {
            "ticker": "ticker",
            "pe_ratio": "pe_ratio",
            "beta_raw_overridable": "beta",
            "cur_mkt_cap": "market_cap",
            "gics_sector_name": "sector",
            "eqy_dvd_yld_12m": "dividend_yield",
            "volatility_30d": "volatility_30d",
        }
        out = result.rename(columns={k: v for k, v in out_cols.items() if k in result.columns})
        keep = [v for v in out_cols.values() if v in out.columns]
        out = out[keep].reset_index(drop=True)

        # IV rank filter (computed on-the-fly; potentially expensive)
        if min_iv_rank is not None:
            iv_ranks = []
            for t in out["ticker"]:
                iv_ranks.append(self.get_iv_rank(t))
            out["iv_rank"] = iv_ranks
            out = out[out["iv_rank"].fillna(-1) >= min_iv_rank]
        elif "iv_rank" not in out.columns:
            # Don't add iv_rank column if not filtering by it
            pass

        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Liquidity (bonus – available data)
    # ------------------------------------------------------------------

    def get_liquidity(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get liquidity metrics (avg volume, turnover, shares out).

        Returns DataFrame with ``DatetimeIndex`` and columns
        ``avg_vol_30d, turnover, shares_out``.
        """
        df = self._load("liquidity")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "date", start_date, end_date)
        if df.empty:
            return df
        cols = [c for c in ["date", "avg_vol_30d", "turnover", "shares_out"] if c in df.columns]
        return df[cols].sort_values("date").set_index("date")
