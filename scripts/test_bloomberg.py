#!/usr/bin/env python3
"""
Bloomberg Connection Test & Quick Data Extraction

Run this script to:
1. Test Bloomberg connectivity
2. Extract sample data for a few tickers
3. Verify data pipeline is working

Usage:
    python scripts/test_bloomberg.py
    python scripts/test_bloomberg.py --ticker AAPL
    python scripts/test_bloomberg.py --extract-all
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    deps = {
        "pandas": False,
        "numpy": False,
        "win32com (pywin32)": False,
        "blpapi": False,
    }

    try:
        import pandas
        deps["pandas"] = True
    except ImportError:
        pass

    try:
        import numpy
        deps["numpy"] = True
    except ImportError:
        pass

    try:
        import win32com.client
        deps["win32com (pywin32)"] = True
    except ImportError:
        pass

    try:
        import blpapi
        deps["blpapi"] = True
    except ImportError:
        pass

    print("\nDependencies:")
    for dep, installed in deps.items():
        status = "OK" if installed else "NOT INSTALLED"
        print(f"  {dep}: {status}")

    # Check which Bloomberg interface is available
    if deps["blpapi"]:
        print("\nBloomberg interface: Direct API (blpapi)")
    elif deps["win32com (pywin32)"]:
        print("\nBloomberg interface: Excel COM automation")
    else:
        print("\nWARNING: No Bloomberg interface available!")
        print("Install pywin32: pip install pywin32")

    return deps


def test_connection():
    """Test Bloomberg connection."""
    print("\n" + "=" * 50)
    print("Testing Bloomberg Connection")
    print("=" * 50)

    try:
        from data.bloomberg import test_connection, check_bloomberg_available

        available = check_bloomberg_available()
        print(f"\nBloomberg API available: {available['blpapi']}")
        print(f"Excel COM available: {available['excel_com']}")

        if not available['any_available']:
            print("\nNo Bloomberg interface available.")
            print("Make sure Bloomberg Terminal is running and Excel Add-in is loaded.")
            return False

        print("\nTesting connection (fetching SPY quote)...")
        if test_connection():
            print("SUCCESS: Bloomberg connection working!")
            return True
        else:
            print("FAILED: Could not connect to Bloomberg")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def get_sample_quote(ticker: str = "AAPL"):
    """Get a sample quote from Bloomberg."""
    print(f"\n" + "=" * 50)
    print(f"Fetching Quote for {ticker}")
    print("=" * 50)

    try:
        from data.bloomberg import BloombergConnector

        with BloombergConnector() as bbg:
            quote = bbg.get_quote(f"{ticker} US Equity")

            print(f"\n{ticker} Quote:")
            print(f"  Last Price:  ${quote.last:.2f}")
            print(f"  Bid/Ask:     ${quote.bid:.2f} / ${quote.ask:.2f}")
            print(f"  Change:      {quote.change_pct:+.2f}%")
            print(f"  Volume:      {quote.volume:,}")
            print(f"  Day Range:   ${quote.low:.2f} - ${quote.high:.2f}")

            if quote.iv_30d:
                print(f"  30D IV:      {quote.iv_30d:.1%}")
            if quote.hv_30d:
                print(f"  30D HV:      {quote.hv_30d:.1%}")

            return quote

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def get_sample_historical(ticker: str = "AAPL"):
    """Get sample historical data."""
    print(f"\n" + "=" * 50)
    print(f"Fetching Historical Data for {ticker}")
    print("=" * 50)

    try:
        from data.bloomberg import BloombergConnector

        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today().replace(day=1)).strftime("%Y-%m-%d")

        with BloombergConnector() as bbg:
            df = bbg.get_historical(f"{ticker} US Equity", start_date, end_date)

            if df.empty:
                print("No data returned")
                return None

            print(f"\nReceived {len(df)} days of data:")
            print(df.tail(5).to_string())
            return df

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def extract_sample_data():
    """Extract sample data for a few tickers and save to CSV."""
    print("\n" + "=" * 50)
    print("Extracting Sample Data")
    print("=" * 50)

    tickers = ["AAPL", "MSFT", "GOOGL"]

    try:
        from data.bloomberg import refresh_ohlcv, get_live_quotes

        # Get live quotes
        print("\nFetching live quotes...")
        quotes_df = get_live_quotes(tickers)
        if not quotes_df.empty:
            print(quotes_df.to_string(index=False))

        # Download historical data
        print("\nDownloading historical data (last 30 days)...")
        start_date = (date.today().replace(day=1) -
                      (date.today().replace(day=1) - date.today().replace(day=1).replace(month=date.today().month - 1))).strftime("%Y-%m-%d")

        count = refresh_ohlcv(tickers, start_date=start_date)
        print(f"Successfully downloaded data for {count}/{len(tickers)} tickers")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def check_existing_data():
    """Check what Bloomberg data already exists."""
    print("\n" + "=" * 50)
    print("Checking Existing Bloomberg Data")
    print("=" * 50)

    from pathlib import Path

    bloomberg_dir = Path("data/bloomberg")

    if not bloomberg_dir.exists():
        print("\nNo data/bloomberg directory found.")
        print("Run data extraction first.")
        return

    categories = ["ohlcv", "options", "iv_history", "earnings", "dividends", "rates", "fundamentals"]

    for cat in categories:
        cat_dir = bloomberg_dir / cat
        if cat_dir.exists():
            files = list(cat_dir.glob("*.csv"))
            print(f"\n{cat}/ : {len(files)} files")
            if files:
                # Show first few
                for f in files[:3]:
                    print(f"  - {f.name}")
                if len(files) > 3:
                    print(f"  ... and {len(files) - 3} more")
        else:
            print(f"\n{cat}/ : (directory not found)")


def main():
    parser = argparse.ArgumentParser(description="Test Bloomberg connection and extract data")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker to test")
    parser.add_argument("--extract-all", action="store_true", help="Extract sample data for multiple tickers")
    parser.add_argument("--check-data", action="store_true", help="Check existing data files")
    args = parser.parse_args()

    print("=" * 50)
    print("Smart Wheel Engine - Bloomberg Test")
    print("=" * 50)

    # Check dependencies
    deps = check_dependencies()

    # Check existing data
    if args.check_data:
        check_existing_data()
        return

    # Need at least one Bloomberg interface
    if not deps["win32com (pywin32)"] and not deps["blpapi"]:
        print("\nCannot test Bloomberg without an interface.")
        print("\nOptions:")
        print("1. Install pywin32: pip install pywin32")
        print("2. Install Bloomberg Python API (blpapi)")
        print("3. Use the VBA script in scripts/bloomberg_excel_extractor.bas")
        return

    # Test connection
    if not test_connection():
        print("\nBloomberg connection failed.")
        print("Make sure:")
        print("1. Bloomberg Terminal is running")
        print("2. You are logged in")
        print("3. Excel Add-in is loaded (for COM mode)")
        return

    # Get sample quote
    get_sample_quote(args.ticker)

    # Get historical data
    get_sample_historical(args.ticker)

    # Extract all if requested
    if args.extract_all:
        extract_sample_data()

    print("\n" + "=" * 50)
    print("Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
