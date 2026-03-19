"""
Bloomberg Live Data Connector

Provides REAL-TIME data from Bloomberg Terminal for live trading.
Complements bloomberg_loader.py which handles historical CSV exports.

Supports two modes:
1. Direct API (blpapi) - requires Bloomberg Python SDK
2. COM/Excel automation - works via Bloomberg Excel Add-in

Use Cases:
- Live price monitoring during trading hours
- Real-time option chain updates
- Intraday IV tracking
- Live position P&L calculation

For historical data (backtesting), use bloomberg_loader.py instead.
"""

import os
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import blpapi (direct Bloomberg API)
try:
    import blpapi
    HAS_BLPAPI = True
except ImportError:
    HAS_BLPAPI = False

# Try to import win32com for Excel COM automation
try:
    import win32com.client
    HAS_WIN32COM = True
except ImportError:
    HAS_WIN32COM = False


@dataclass
class OptionQuote:
    """Single option quote from Bloomberg."""
    ticker: str
    underlying: str
    strike: float
    expiry: date
    option_type: str  # 'C' or 'P'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else self.last


@dataclass
class StockQuote:
    """Stock quote from Bloomberg."""
    ticker: str
    last: float
    bid: float
    ask: float
    volume: int
    high: float
    low: float
    open: float
    prev_close: float
    change_pct: float
    iv_30d: Optional[float] = None  # 30-day implied volatility
    hv_30d: Optional[float] = None  # 30-day historical volatility

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else self.last


class BloombergError(Exception):
    """Bloomberg connection or data error."""
    pass


class BloombergConnector:
    """
    Bloomberg data connector supporting both direct API and Excel COM automation.

    Usage:
        # Initialize (auto-detects best available method)
        bbg = BloombergConnector()

        # Get stock quote
        quote = bbg.get_quote("AAPL US Equity")

        # Get option chain
        chain = bbg.get_option_chain("AAPL US Equity", expiry_date)

        # Get historical data
        hist = bbg.get_historical("AAPL US Equity", "2024-01-01", "2024-12-31")
    """

    def __init__(self, mode: str = "auto"):
        """
        Initialize Bloomberg connector.

        Args:
            mode: "auto" (detect best), "api" (direct blpapi), "excel" (COM automation)
        """
        self.mode = mode
        self._session = None
        self._excel = None
        self._workbook = None

        if mode == "auto":
            if HAS_BLPAPI:
                self.mode = "api"
            elif HAS_WIN32COM:
                self.mode = "excel"
            else:
                raise BloombergError(
                    "No Bloomberg interface available. "
                    "Install blpapi or win32com (pywin32)."
                )

        self._connect()

    def _connect(self):
        """Establish connection to Bloomberg."""
        if self.mode == "api":
            self._connect_api()
        elif self.mode == "excel":
            self._connect_excel()

    def _connect_api(self):
        """Connect via direct Bloomberg API."""
        if not HAS_BLPAPI:
            raise BloombergError("blpapi not installed")

        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost("localhost")
        sessionOptions.setServerPort(8194)

        self._session = blpapi.Session(sessionOptions)
        if not self._session.start():
            raise BloombergError("Failed to start Bloomberg session")

        if not self._session.openService("//blp/refdata"):
            raise BloombergError("Failed to open Bloomberg reference data service")

    def _connect_excel(self):
        """Connect via Excel COM automation."""
        if not HAS_WIN32COM:
            raise BloombergError("win32com not installed (pip install pywin32)")

        try:
            # Try to connect to existing Excel instance
            self._excel = win32com.client.GetActiveObject("Excel.Application")
        except:
            # Start new Excel instance
            self._excel = win32com.client.Dispatch("Excel.Application")
            self._excel.Visible = True

        # Create a hidden workbook for Bloomberg queries
        self._workbook = self._excel.Workbooks.Add()
        self._sheet = self._workbook.Sheets(1)

    def get_quote(self, security: str) -> StockQuote:
        """
        Get real-time quote for a security.

        Args:
            security: Bloomberg security identifier (e.g., "AAPL US Equity")

        Returns:
            StockQuote object
        """
        if self.mode == "api":
            return self._get_quote_api(security)
        else:
            return self._get_quote_excel(security)

    def _get_quote_api(self, security: str) -> StockQuote:
        """Get quote via direct API."""
        refDataService = self._session.getService("//blp/refdata")
        request = refDataService.createRequest("ReferenceDataRequest")
        request.append("securities", security)

        fields = [
            "PX_LAST", "PX_BID", "PX_ASK", "VOLUME",
            "PX_HIGH", "PX_LOW", "PX_OPEN", "PREV_CLOSE_VALUE_ADJ",
            "CHG_PCT_1D", "IVOL_30D", "VOLATILITY_30D"
        ]
        for field in fields:
            request.append("fields", field)

        self._session.sendRequest(request)

        data = {}
        while True:
            ev = self._session.nextEvent()
            for msg in ev:
                if msg.hasElement("securityData"):
                    secData = msg.getElement("securityData")
                    for sec in secData.values():
                        fieldData = sec.getElement("fieldData")
                        for field in fields:
                            if fieldData.hasElement(field):
                                data[field] = fieldData.getElementValue(field)
            if ev.eventType() == blpapi.Event.RESPONSE:
                break

        return StockQuote(
            ticker=security,
            last=data.get("PX_LAST", 0),
            bid=data.get("PX_BID", 0),
            ask=data.get("PX_ASK", 0),
            volume=int(data.get("VOLUME", 0)),
            high=data.get("PX_HIGH", 0),
            low=data.get("PX_LOW", 0),
            open=data.get("PX_OPEN", 0),
            prev_close=data.get("PREV_CLOSE_VALUE_ADJ", 0),
            change_pct=data.get("CHG_PCT_1D", 0),
            iv_30d=data.get("IVOL_30D"),
            hv_30d=data.get("VOLATILITY_30D")
        )

    def _get_quote_excel(self, security: str) -> StockQuote:
        """Get quote via Excel BDP formulas."""
        fields = {
            "A1": ("PX_LAST", "last"),
            "A2": ("PX_BID", "bid"),
            "A3": ("PX_ASK", "ask"),
            "A4": ("VOLUME", "volume"),
            "A5": ("PX_HIGH", "high"),
            "A6": ("PX_LOW", "low"),
            "A7": ("PX_OPEN", "open"),
            "A8": ("PREV_CLOSE_VALUE_ADJ", "prev_close"),
            "A9": ("CHG_PCT_1D", "change_pct"),
            "A10": ("IVOL_30D", "iv_30d"),
            "A11": ("VOLATILITY_30D", "hv_30d"),
        }

        # Set BDP formulas
        for cell, (field, _) in fields.items():
            self._sheet.Range(cell).Formula = f'=BDP("{security}","{field}")'

        # Wait for data to populate
        self._wait_for_data("A1")

        # Read values
        data = {}
        for cell, (_, name) in fields.items():
            val = self._sheet.Range(cell).Value
            if val is not None and not isinstance(val, str):
                data[name] = val
            else:
                data[name] = 0 if name != "iv_30d" and name != "hv_30d" else None

        return StockQuote(
            ticker=security,
            last=float(data["last"]),
            bid=float(data["bid"]),
            ask=float(data["ask"]),
            volume=int(data["volume"]),
            high=float(data["high"]),
            low=float(data["low"]),
            open=float(data["open"]),
            prev_close=float(data["prev_close"]),
            change_pct=float(data["change_pct"]),
            iv_30d=data["iv_30d"],
            hv_30d=data["hv_30d"]
        )

    def _wait_for_data(self, cell: str, timeout: float = 10.0):
        """Wait for Bloomberg data to populate in Excel cell."""
        start = time.time()
        while time.time() - start < timeout:
            val = self._sheet.Range(cell).Value
            if val is not None and val != "#N/A Requesting Data...":
                return
            time.sleep(0.5)
        raise BloombergError(f"Timeout waiting for Bloomberg data in {cell}")

    def get_multiple_quotes(self, securities: List[str]) -> Dict[str, StockQuote]:
        """
        Get quotes for multiple securities efficiently.

        Args:
            securities: List of Bloomberg security identifiers

        Returns:
            Dict mapping security to StockQuote
        """
        quotes = {}
        for sec in securities:
            try:
                quotes[sec] = self.get_quote(sec)
            except Exception as e:
                print(f"Warning: Failed to get quote for {sec}: {e}")
        return quotes

    def get_historical(
        self,
        security: str,
        start_date: Union[str, date],
        end_date: Union[str, date],
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get historical data (OHLCV) for a security.

        Args:
            security: Bloomberg security identifier
            start_date: Start date (YYYY-MM-DD or date object)
            end_date: End date (YYYY-MM-DD or date object)
            fields: List of fields (default: OHLCV)

        Returns:
            DataFrame with historical data
        """
        if fields is None:
            fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "VOLUME"]

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        if self.mode == "api":
            return self._get_historical_api(security, start_date, end_date, fields)
        else:
            return self._get_historical_excel(security, start_date, end_date, fields)

    def _get_historical_api(
        self,
        security: str,
        start_date: date,
        end_date: date,
        fields: List[str]
    ) -> pd.DataFrame:
        """Get historical data via direct API."""
        refDataService = self._session.getService("//blp/refdata")
        request = refDataService.createRequest("HistoricalDataRequest")
        request.append("securities", security)

        for field in fields:
            request.append("fields", field)

        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")

        self._session.sendRequest(request)

        data = []
        while True:
            ev = self._session.nextEvent()
            for msg in ev:
                if msg.hasElement("securityData"):
                    secData = msg.getElement("securityData")
                    fieldDataArray = secData.getElement("fieldData")
                    for fd in fieldDataArray.values():
                        row = {"date": fd.getElementValue("date")}
                        for field in fields:
                            if fd.hasElement(field):
                                row[field] = fd.getElementValue(field)
                        data.append(row)
            if ev.eventType() == blpapi.Event.RESPONSE:
                break

        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            # Rename columns to standard names
            rename_map = {
                "PX_OPEN": "open",
                "PX_HIGH": "high",
                "PX_LOW": "low",
                "PX_LAST": "close",
                "VOLUME": "volume"
            }
            df = df.rename(columns=rename_map)
        return df

    def _get_historical_excel(
        self,
        security: str,
        start_date: date,
        end_date: date,
        fields: List[str]
    ) -> pd.DataFrame:
        """Get historical data via Excel BDH formula."""
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        fields_str = ",".join([f'"{f}"' for f in fields])

        # BDH formula
        formula = f'=BDH("{security}",{{{fields_str}}},"{start_str}","{end_str}")'
        self._sheet.Range("D1").Formula = formula

        # Wait for data
        self._wait_for_data("D1", timeout=30.0)

        # Find data range
        used_range = self._sheet.UsedRange
        data = used_range.Value

        if data is None:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=["date"] + fields)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Rename columns
        rename_map = {
            "PX_OPEN": "open",
            "PX_HIGH": "high",
            "PX_LOW": "low",
            "PX_LAST": "close",
            "VOLUME": "volume"
        }
        df = df.rename(columns=rename_map)

        # Clear the sheet for next query
        self._sheet.Cells.Clear()

        return df

    def get_option_chain(
        self,
        underlying: str,
        expiry: Optional[date] = None,
        strikes: Optional[List[float]] = None,
        option_type: Optional[str] = None  # 'C', 'P', or None for both
    ) -> pd.DataFrame:
        """
        Get option chain for an underlying.

        Args:
            underlying: Underlying security (e.g., "AAPL US Equity")
            expiry: Specific expiry date (None for all available)
            strikes: Specific strikes (None for all available)
            option_type: 'C' for calls, 'P' for puts, None for both

        Returns:
            DataFrame with option chain data
        """
        # Get the ticker root (e.g., "AAPL" from "AAPL US Equity")
        ticker_root = underlying.split()[0]

        if self.mode == "api":
            return self._get_option_chain_api(ticker_root, expiry, strikes, option_type)
        else:
            return self._get_option_chain_excel(ticker_root, expiry, strikes, option_type)

    def _get_option_chain_api(
        self,
        ticker_root: str,
        expiry: Optional[date],
        strikes: Optional[List[float]],
        option_type: Optional[str]
    ) -> pd.DataFrame:
        """Get option chain via direct API using BDS."""
        refDataService = self._session.getService("//blp/refdata")

        # First get list of options using OPT_CHAIN
        request = refDataService.createRequest("ReferenceDataRequest")
        request.append("securities", f"{ticker_root} US Equity")
        request.append("fields", "OPT_CHAIN")

        self._session.sendRequest(request)

        option_tickers = []
        while True:
            ev = self._session.nextEvent()
            for msg in ev:
                if msg.hasElement("securityData"):
                    secData = msg.getElement("securityData")
                    for sec in secData.values():
                        fieldData = sec.getElement("fieldData")
                        if fieldData.hasElement("OPT_CHAIN"):
                            chain = fieldData.getElement("OPT_CHAIN")
                            for opt in chain.values():
                                option_tickers.append(opt.getElementValue("Security Description"))
            if ev.eventType() == blpapi.Event.RESPONSE:
                break

        if not option_tickers:
            return pd.DataFrame()

        # Now get data for each option
        return self._get_option_data(option_tickers, expiry, strikes, option_type)

    def _get_option_chain_excel(
        self,
        ticker_root: str,
        expiry: Optional[date],
        strikes: Optional[List[float]],
        option_type: Optional[str]
    ) -> pd.DataFrame:
        """Get option chain via Excel BDS formula."""
        # Get option chain using BDS
        formula = f'=BDS("{ticker_root} US Equity","OPT_CHAIN")'
        self._sheet.Range("A1").Formula = formula

        # Wait for data
        self._wait_for_data("A1", timeout=30.0)

        # Read option tickers
        col = self._sheet.Range("A:A")
        option_tickers = []
        row = 1
        while True:
            val = self._sheet.Cells(row, 1).Value
            if val is None or val == "":
                break
            if isinstance(val, str) and val.strip():
                option_tickers.append(val.strip())
            row += 1

        # Clear sheet
        self._sheet.Cells.Clear()

        if not option_tickers:
            return pd.DataFrame()

        # Get data for each option
        return self._get_option_data_excel(option_tickers, expiry, strikes, option_type)

    def _get_option_data_excel(
        self,
        option_tickers: List[str],
        expiry: Optional[date],
        strikes: Optional[List[float]],
        option_type: Optional[str]
    ) -> pd.DataFrame:
        """Get detailed option data for a list of options via Excel."""
        fields = [
            "OPT_STRIKE_PX", "OPT_EXPIRE_DT", "OPT_PUT_CALL",
            "PX_BID", "PX_ASK", "PX_LAST", "VOLUME", "OPEN_INT",
            "IVOL_MID", "OPT_DELTA", "OPT_GAMMA", "OPT_THETA", "OPT_VEGA"
        ]

        data = []
        batch_size = 20  # Process in batches to avoid Excel limits

        for i in range(0, len(option_tickers), batch_size):
            batch = option_tickers[i:i + batch_size]

            for j, ticker in enumerate(batch):
                row = j + 1
                for k, field in enumerate(fields):
                    col = k + 1
                    self._sheet.Cells(row, col).Formula = f'=BDP("{ticker}","{field}")'

            # Wait for data
            time.sleep(2)  # Give Bloomberg time to fetch
            self._wait_for_data("A1", timeout=15.0)

            # Read values
            for j, ticker in enumerate(batch):
                row = j + 1
                row_data = {"ticker": ticker}
                for k, field in enumerate(fields):
                    col = k + 1
                    val = self._sheet.Cells(row, col).Value
                    row_data[field] = val
                data.append(row_data)

            # Clear for next batch
            self._sheet.Cells.Clear()

        df = pd.DataFrame(data)

        if df.empty:
            return df

        # Clean and filter
        df = df.rename(columns={
            "OPT_STRIKE_PX": "strike",
            "OPT_EXPIRE_DT": "expiry",
            "OPT_PUT_CALL": "type",
            "PX_BID": "bid",
            "PX_ASK": "ask",
            "PX_LAST": "last",
            "VOLUME": "volume",
            "OPEN_INT": "open_interest",
            "IVOL_MID": "iv",
            "OPT_DELTA": "delta",
            "OPT_GAMMA": "gamma",
            "OPT_THETA": "theta",
            "OPT_VEGA": "vega"
        })

        # Convert expiry to date
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date

        # Filter by expiry if specified
        if expiry:
            df = df[df["expiry"] == expiry]

        # Filter by strikes if specified
        if strikes:
            df = df[df["strike"].isin(strikes)]

        # Filter by type if specified
        if option_type:
            df = df[df["type"] == option_type]

        return df.sort_values(["expiry", "type", "strike"])

    def get_iv_surface(
        self,
        underlying: str,
        expiries: Optional[List[date]] = None,
        deltas: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Get implied volatility surface for an underlying.

        Args:
            underlying: Underlying security
            expiries: List of expiry dates (None for standard tenors)
            deltas: List of deltas (None for standard: 25P, 50, 25C)

        Returns:
            DataFrame with IV surface (rows=expiry, cols=delta/strike)
        """
        # Get option chain first
        chain = self.get_option_chain(underlying)

        if chain.empty:
            return pd.DataFrame()

        # Pivot to create surface
        # Group by expiry and create strike/IV mapping
        surface_data = []

        for exp in chain["expiry"].unique():
            exp_data = chain[chain["expiry"] == exp]

            for _, row in exp_data.iterrows():
                surface_data.append({
                    "expiry": exp,
                    "strike": row["strike"],
                    "type": row["type"],
                    "iv": row["iv"],
                    "delta": row["delta"]
                })

        return pd.DataFrame(surface_data)

    def get_earnings_date(self, security: str) -> Optional[date]:
        """Get next earnings date for a security."""
        if self.mode == "excel":
            self._sheet.Range("A1").Formula = f'=BDP("{security}","EXPECTED_REPORT_DT")'
            self._wait_for_data("A1")
            val = self._sheet.Range("A1").Value
            self._sheet.Cells.Clear()
            if val:
                return pd.to_datetime(val).date()
        return None

    def get_dividend_info(self, security: str) -> Dict:
        """Get dividend information for a security."""
        fields = ["DVD_SH_LAST", "DVD_FREQ", "EQY_DVD_YLD_IND", "DVD_EX_DT"]

        if self.mode == "excel":
            data = {}
            for i, field in enumerate(fields):
                self._sheet.Range(f"A{i+1}").Formula = f'=BDP("{security}","{field}")'

            self._wait_for_data("A1")

            for i, field in enumerate(fields):
                data[field] = self._sheet.Range(f"A{i+1}").Value

            self._sheet.Cells.Clear()

            return {
                "dividend_amount": data.get("DVD_SH_LAST"),
                "frequency": data.get("DVD_FREQ"),
                "yield": data.get("EQY_DVD_YLD_IND"),
                "ex_date": pd.to_datetime(data.get("DVD_EX_DT")).date() if data.get("DVD_EX_DT") else None
            }
        return {}

    def close(self):
        """Close Bloomberg connection and cleanup."""
        if self.mode == "api" and self._session:
            self._session.stop()
        elif self.mode == "excel" and self._workbook:
            try:
                self._workbook.Close(SaveChanges=False)
            except:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions for quick data access
def get_price(ticker: str) -> float:
    """Quick function to get last price."""
    with BloombergConnector() as bbg:
        quote = bbg.get_quote(f"{ticker} US Equity")
        return quote.last


def get_iv(ticker: str) -> float:
    """Quick function to get 30-day IV."""
    with BloombergConnector() as bbg:
        quote = bbg.get_quote(f"{ticker} US Equity")
        return quote.iv_30d or 0.0


def download_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Download OHLCV data from Bloomberg.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Optional path to save CSV

    Returns:
        DataFrame with OHLCV data
    """
    with BloombergConnector() as bbg:
        df = bbg.get_historical(f"{ticker} US Equity", start_date, end_date)

        if output_path:
            df.to_csv(output_path)

        return df


# ─────────────────────────────────────────────────────────────────────
# Integration with bloomberg_loader.py - Save live data to CSV files
# ─────────────────────────────────────────────────────────────────────

BLOOMBERG_DIR = Path("data/bloomberg")


def refresh_ohlcv(tickers: List[str], start_date: str = "2019-01-01") -> int:
    """
    Download/refresh OHLCV data and save to data/bloomberg/ohlcv/.

    This updates the CSV files that bloomberg_loader.py reads.

    Args:
        tickers: List of stock tickers
        start_date: Start date for historical data

    Returns:
        Number of tickers successfully updated
    """
    output_dir = BLOOMBERG_DIR / "ohlcv"
    output_dir.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().strftime("%Y-%m-%d")
    success_count = 0

    with BloombergConnector() as bbg:
        for ticker in tickers:
            try:
                logger.info(f"Downloading OHLCV for {ticker}...")
                df = bbg.get_historical(f"{ticker} US Equity", start_date, end_date)
                if not df.empty:
                    output_path = output_dir / f"{ticker}.csv"
                    df.to_csv(output_path)
                    logger.info(f"Saved {len(df)} rows to {output_path}")
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")

    return success_count


def refresh_option_chain(tickers: List[str]) -> int:
    """
    Download option chains and save to data/bloomberg/options/.

    Args:
        tickers: List of stock tickers

    Returns:
        Number of tickers successfully updated
    """
    output_dir = BLOOMBERG_DIR / "options"
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    success_count = 0

    with BloombergConnector() as bbg:
        for ticker in tickers:
            try:
                logger.info(f"Downloading option chain for {ticker}...")
                df = bbg.get_option_chain(f"{ticker} US Equity")
                if not df.empty:
                    output_path = output_dir / f"{today}_{ticker}.csv"
                    df.to_csv(output_path, index=False)
                    logger.info(f"Saved {len(df)} contracts to {output_path}")
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to download options for {ticker}: {e}")

    return success_count


def get_live_quotes(tickers: List[str]) -> pd.DataFrame:
    """
    Get live quotes for multiple tickers.

    Args:
        tickers: List of stock tickers (e.g., ["AAPL", "MSFT"])

    Returns:
        DataFrame with columns: ticker, last, bid, ask, volume, change_pct, iv_30d
    """
    data = []
    with BloombergConnector() as bbg:
        for ticker in tickers:
            try:
                quote = bbg.get_quote(f"{ticker} US Equity")
                data.append({
                    "ticker": ticker,
                    "last": quote.last,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "volume": quote.volume,
                    "change_pct": quote.change_pct,
                    "iv_30d": quote.iv_30d,
                    "hv_30d": quote.hv_30d
                })
            except Exception as e:
                logger.warning(f"Failed to get quote for {ticker}: {e}")

    return pd.DataFrame(data)


def get_live_iv_rank(ticker: str, lookback_days: int = 252) -> Optional[float]:
    """
    Calculate live IV rank using current IV and historical data.

    Combines live Bloomberg data with historical bloomberg_loader data.

    Args:
        ticker: Stock ticker
        lookback_days: Number of days for percentile calculation

    Returns:
        IV rank (0 to 1) or None if insufficient data
    """
    from .bloomberg_loader import load_bloomberg_iv_history

    # Get historical IV data
    iv_df = load_bloomberg_iv_history(ticker)
    if iv_df is None or iv_df.empty:
        return None

    # Get live IV
    with BloombergConnector() as bbg:
        quote = bbg.get_quote(f"{ticker} US Equity")
        current_iv = quote.iv_30d

    if current_iv is None:
        return None

    # Calculate rank
    col = "iv_atm_30d"
    if col not in iv_df.columns:
        return None

    series = iv_df[col].dropna().tail(lookback_days)
    if len(series) < 20:
        return None

    low = series.min()
    high = series.max()

    if high <= low:
        return 0.5

    return float((current_iv - low) / (high - low))


def get_wheel_candidates(
    tickers: List[str],
    min_iv_rank: float = 0.30,
    min_price: float = 20.0,
    max_price: float = 500.0
) -> pd.DataFrame:
    """
    Screen stocks for wheel strategy candidates.

    Args:
        tickers: List of tickers to screen
        min_iv_rank: Minimum IV rank (0-1)
        min_price: Minimum stock price
        max_price: Maximum stock price

    Returns:
        DataFrame of candidates sorted by IV rank
    """
    candidates = []

    with BloombergConnector() as bbg:
        for ticker in tickers:
            try:
                quote = bbg.get_quote(f"{ticker} US Equity")

                # Price filter
                if not (min_price <= quote.last <= max_price):
                    continue

                # Get IV rank
                iv_rank = get_live_iv_rank(ticker)
                if iv_rank is None or iv_rank < min_iv_rank:
                    continue

                # Get earnings date
                earnings_date = bbg.get_earnings_date(f"{ticker} US Equity")
                days_to_earnings = None
                if earnings_date:
                    days_to_earnings = (earnings_date - date.today()).days

                candidates.append({
                    "ticker": ticker,
                    "price": quote.last,
                    "iv_30d": quote.iv_30d,
                    "iv_rank": iv_rank,
                    "hv_30d": quote.hv_30d,
                    "days_to_earnings": days_to_earnings,
                    "change_pct": quote.change_pct
                })

            except Exception as e:
                logger.warning(f"Error screening {ticker}: {e}")

    df = pd.DataFrame(candidates)
    if not df.empty:
        df = df.sort_values("iv_rank", ascending=False)

    return df


# ─────────────────────────────────────────────────────────────────────
# Check Bloomberg availability
# ─────────────────────────────────────────────────────────────────────

def check_bloomberg_available() -> Dict[str, bool]:
    """
    Check which Bloomberg interfaces are available.

    Returns:
        Dict with 'blpapi' and 'excel' availability
    """
    return {
        "blpapi": HAS_BLPAPI,
        "excel_com": HAS_WIN32COM,
        "any_available": HAS_BLPAPI or HAS_WIN32COM
    }


def test_connection() -> bool:
    """
    Test Bloomberg connection by fetching a simple quote.

    Returns:
        True if connection successful
    """
    try:
        with BloombergConnector() as bbg:
            quote = bbg.get_quote("SPY US Equity")
            return quote.last > 0
    except Exception as e:
        logger.error(f"Bloomberg connection test failed: {e}")
        return False
