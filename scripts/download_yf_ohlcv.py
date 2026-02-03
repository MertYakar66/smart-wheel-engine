"""
Download OHLCV data from yfinance with proper header cleanup.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DIR = Path("data_raw/ohlcv")


def load_tickers() -> list[str]:
    """Read tickers from the S&P500 constituents CSV."""
    df = pd.read_csv("data_raw/sp500_constituents_current.csv")
    return df["ticker"].dropna().tolist()


def download_ohlcv(ticker: str):
    """
    Download daily OHLCV for a single ticker from 2010-01-01 to today.
    Properly handles yfinance multi-index headers.
    """
    try:
        logger.info(f"Fetching {ticker}")
        df = yf.download(
            ticker,
            start="2010-01-01",
            progress=False,
            auto_adjust=False,
        )

        if df.empty:
            logger.warning(f"No OHLCV for {ticker}")
            return

        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-index: take first level (Open, High, etc.)
            df.columns = df.columns.get_level_values(0)

        # Reset index to make Date a column
        df = df.reset_index()

        # Ensure Date column exists and is valid
        if "Date" not in df.columns:
            raise RuntimeError(f"Downloaded OHLCV for {ticker} has no 'Date' column.")

        # Clean any rows where Date is NaN or invalid
        df = df.dropna(subset=["Date"])

        # Remove any rows where numeric columns contain the ticker name (corruption)
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN, then drop
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=["Open", "Close"])

        # Normalize column names
        rename_map = {
            "Adj Close": "Adj_Close",
        }
        df = df.rename(columns=rename_map)

        # Keep clean column set
        cols_order = ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]
        cols_present = [c for c in cols_order if c in df.columns]
        df = df[cols_present]

        # Ensure Date is datetime
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        outpath = RAW_DIR / f"{ticker}.csv"
        df.to_csv(outpath, index=False)
        logger.info(f"Saved OHLCV for {ticker} -> {outpath} ({len(df)} rows)")

    except Exception as e:
        logger.error(f"{ticker}: {e}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_tickers()
    # Limit for testing - remove this limit for production
    tickers = tickers[:5]
    logger.info(f"Downloading OHLCV for: {tickers}")

    for t in tickers:
        download_ohlcv(t)


if __name__ == "__main__":
    main()
