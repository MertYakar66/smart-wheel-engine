"""
Bloomberg Export Processor

Processes CSVs exported from Bloomberg Excel macro and:
1. Cleans sparse data (especially earnings)
2. Renames columns to engine-standard names
3. Validates data integrity
4. Copies to data/bloomberg/{category}/ directories

USAGE:
    python scripts/process_bloomberg_exports.py --input C:/BloombergExport --output data/bloomberg

The input directory should contain files named:
    {TICKER}_ohlcv.csv
    {TICKER}_iv.csv
    {TICKER}_earnings.csv
    {TICKER}_dividends.csv
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Column mappings: Bloomberg raw -> engine standard
OHLCV_COLUMNS = {
    "Date": "Date",
    "PX_OPEN": "Open",
    "PX_HIGH": "High",
    "PX_LOW": "Low",
    "PX_LAST": "Close",
    "PX_VOLUME": "Volume",
}

IV_COLUMNS = {
    "Date": "date",
    "30DAY_IMPVOL_100.0%MNY_DF": "iv_atm_30d",
}

EARNINGS_COLUMNS = {
    "Date": "earnings_date",
    "IS_EPS": "eps_actual",
    "BEST_EPS": "eps_estimate",
}

DIVIDENDS_COLUMNS = {
    "Date": "date",
    "EQY_DVD_YLD_IND": "dividend_yield",
}


def parse_bloomberg_date(series: pd.Series) -> pd.Series:
    """Parse Bloomberg date formats."""
    return pd.to_datetime(series, format="mixed", errors="coerce")


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric, handling Bloomberg #N/A values."""
    # Replace Bloomberg error strings
    series = series.replace(["#N/A", "#N/A N/A", "#N/A Requesting Data...", ""], np.nan)
    return pd.to_numeric(series, errors="coerce")


def process_ohlcv(filepath: Path, ticker: str) -> pd.DataFrame | None:
    """Process OHLCV export."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return None

    if df.empty:
        return None

    # Rename columns
    df = df.rename(columns={k: v for k, v in OHLCV_COLUMNS.items() if k in df.columns})

    # Parse dates
    if "Date" in df.columns:
        df["Date"] = parse_bloomberg_date(df["Date"])
        df = df.dropna(subset=["Date"])

    # Clean numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Drop rows with missing Close
    df = df.dropna(subset=["Close"])

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info(
        f"  OHLCV {ticker}: {len(df)} rows ({df['Date'].min().date()} to {df['Date'].max().date()})"
    )

    return df


def process_iv(filepath: Path, ticker: str) -> pd.DataFrame | None:
    """Process IV history export."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return None

    if df.empty:
        return None

    # Rename columns
    df = df.rename(columns={k: v for k, v in IV_COLUMNS.items() if k in df.columns})

    # Parse dates
    if "date" in df.columns:
        df["date"] = parse_bloomberg_date(df["date"])
        df = df.dropna(subset=["date"])

    # Clean IV values
    if "iv_atm_30d" in df.columns:
        df["iv_atm_30d"] = clean_numeric(df["iv_atm_30d"])

        # Bloomberg returns IV as percentage (e.g., 25.4 for 25.4%)
        # Normalize to decimal (0.254)
        if df["iv_atm_30d"].median() > 1.0:
            df["iv_atm_30d"] = df["iv_atm_30d"] / 100.0

    df = df.dropna(subset=["iv_atm_30d"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ticker"] = ticker

    logger.info(f"  IV {ticker}: {len(df)} rows")

    return df


def process_earnings(filepath: Path, ticker: str) -> pd.DataFrame | None:
    """
    Process earnings export with sparse data handling.

    Bloomberg earnings data is often sparse:
    - Some quarters have EPS actual but no estimate
    - Some quarters have estimate but no actual (future)
    - Some quarters have #N/A for both

    Strategy:
    1. Keep rows that have at least one valid value
    2. Forward-fill estimates for rows missing estimates
    3. Compute surprise where both values exist
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return None

    if df.empty:
        return None

    # Rename columns
    df = df.rename(columns={k: v for k, v in EARNINGS_COLUMNS.items() if k in df.columns})

    # Parse dates
    if "earnings_date" in df.columns:
        df["earnings_date"] = parse_bloomberg_date(df["earnings_date"])
        df = df.dropna(subset=["earnings_date"])

    # Clean numeric columns
    for col in ["eps_actual", "eps_estimate"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Keep rows with at least one non-null EPS value
    has_any_eps = (
        df["eps_actual"].notna() | df["eps_estimate"].notna()
        if "eps_actual" in df.columns and "eps_estimate" in df.columns
        else df.notna().any(axis=1)
    )
    df = df[has_any_eps].copy()

    if df.empty:
        logger.warning(f"  Earnings {ticker}: No valid data after cleaning")
        return None

    # Compute surprise where both values exist
    if "eps_actual" in df.columns and "eps_estimate" in df.columns:
        mask = df["eps_actual"].notna() & df["eps_estimate"].notna()
        df.loc[mask, "eps_surprise"] = df.loc[mask, "eps_actual"] - df.loc[mask, "eps_estimate"]

        # Surprise percentage (handle zero estimate)
        with np.errstate(divide="ignore", invalid="ignore"):
            df.loc[mask, "surprise_pct"] = (
                df.loc[mask, "eps_surprise"] / df.loc[mask, "eps_estimate"].replace(0, np.nan) * 100
            )

    df = df.sort_values("earnings_date").reset_index(drop=True)
    df["ticker"] = ticker

    # Report quality
    actual_count = df["eps_actual"].notna().sum() if "eps_actual" in df.columns else 0
    estimate_count = df["eps_estimate"].notna().sum() if "eps_estimate" in df.columns else 0

    logger.info(
        f"  Earnings {ticker}: {len(df)} rows "
        f"(actuals: {actual_count}, estimates: {estimate_count})"
    )

    return df


def process_dividends(filepath: Path, ticker: str) -> pd.DataFrame | None:
    """Process dividend yield proxy export."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return None

    if df.empty:
        return None

    # Rename columns
    df = df.rename(columns={k: v for k, v in DIVIDENDS_COLUMNS.items() if k in df.columns})

    # Parse dates
    if "date" in df.columns:
        df["date"] = parse_bloomberg_date(df["date"])
        df = df.dropna(subset=["date"])

    # Clean numeric
    if "dividend_yield" in df.columns:
        df["dividend_yield"] = clean_numeric(df["dividend_yield"])

        # Bloomberg returns yield as percentage (e.g., 1.5 for 1.5%)
        # Normalize to decimal (0.015)
        if df["dividend_yield"].median() > 0.5:
            df["dividend_yield"] = df["dividend_yield"] / 100.0

    df = df.dropna(subset=["dividend_yield"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ticker"] = ticker

    logger.info(f"  Dividends {ticker}: {len(df)} rows")

    return df


def discover_tickers(input_dir: Path) -> list[str]:
    """Find all unique tickers in the input directory."""
    tickers = set()
    pattern = re.compile(r"^([A-Z0-9]+)_(ohlcv|iv|earnings|dividends)\.csv$", re.IGNORECASE)

    for f in input_dir.glob("*.csv"):
        match = pattern.match(f.name)
        if match:
            tickers.add(match.group(1).upper())

    return sorted(tickers)


def process_ticker(ticker: str, input_dir: Path, output_dir: Path) -> dict[str, bool]:
    """Process all data types for a single ticker."""
    results = {}

    logger.info(f"Processing {ticker}...")

    # OHLCV
    ohlcv_file = input_dir / f"{ticker}_ohlcv.csv"
    if ohlcv_file.exists():
        df = process_ohlcv(ohlcv_file, ticker)
        if df is not None and not df.empty:
            out_path = output_dir / "ohlcv" / f"{ticker}.csv"
            df.to_csv(out_path, index=False)
            results["ohlcv"] = True
        else:
            results["ohlcv"] = False
    else:
        results["ohlcv"] = False

    # IV History
    iv_file = input_dir / f"{ticker}_iv.csv"
    if iv_file.exists():
        df = process_iv(iv_file, ticker)
        if df is not None and not df.empty:
            out_path = output_dir / "iv_history" / f"{ticker}.csv"
            df.to_csv(out_path, index=False)
            results["iv"] = True
        else:
            results["iv"] = False
    else:
        results["iv"] = False

    # Earnings
    earnings_file = input_dir / f"{ticker}_earnings.csv"
    if earnings_file.exists():
        df = process_earnings(earnings_file, ticker)
        if df is not None and not df.empty:
            out_path = output_dir / "earnings" / f"{ticker}.csv"
            df.to_csv(out_path, index=False)
            results["earnings"] = True
        else:
            results["earnings"] = False
    else:
        results["earnings"] = False

    # Dividends
    div_file = input_dir / f"{ticker}_dividends.csv"
    if div_file.exists():
        df = process_dividends(div_file, ticker)
        if df is not None and not df.empty:
            out_path = output_dir / "dividends" / f"{ticker}.csv"
            df.to_csv(out_path, index=False)
            results["dividends"] = True
        else:
            results["dividends"] = False
    else:
        results["dividends"] = False

    return results


def main():
    parser = argparse.ArgumentParser(description="Process Bloomberg Excel exports")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="C:/BloombergExport",
        help="Input directory containing Bloomberg CSV exports",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/bloomberg",
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--ticker",
        "-t",
        type=str,
        default=None,
        help="Process single ticker only",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output subdirectories
    for subdir in ["ohlcv", "iv_history", "earnings", "dividends"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Discover or use specified ticker
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = discover_tickers(input_dir)

    if not tickers:
        logger.error(f"No Bloomberg CSV files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(tickers)} tickers: {', '.join(tickers)}")

    # Process all tickers
    summary = {"ohlcv": 0, "iv": 0, "earnings": 0, "dividends": 0}

    for ticker in tickers:
        results = process_ticker(ticker, input_dir, output_dir)
        for key, success in results.items():
            if success:
                summary[key] += 1

    # Print summary
    logger.info("=" * 50)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Tickers processed: {len(tickers)}")
    logger.info(f"OHLCV files:       {summary['ohlcv']}/{len(tickers)}")
    logger.info(f"IV files:          {summary['iv']}/{len(tickers)}")
    logger.info(f"Earnings files:    {summary['earnings']}/{len(tickers)}")
    logger.info(f"Dividend files:    {summary['dividends']}/{len(tickers)}")
    logger.info(f"Output directory:  {output_dir.absolute()}")


if __name__ == "__main__":
    main()
