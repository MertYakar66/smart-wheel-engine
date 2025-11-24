import os
from datetime import datetime
import time

import pandas as pd
import yfinance as yf

CONSTITUENTS_PATH = "data_raw/sp500_constituents_current.csv"
OUTPUT_DIR = "data_raw/ohlcv"

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tickers() -> list:
    df = pd.read_csv(CONSTITUENTS_PATH)
    tickers = df["ticker"].dropna().unique().tolist()
    return tickers


def download_single_ticker(ticker: str):
    """
    Download OHLCV for a single ticker and save to CSV.
    """
    filepath = os.path.join(OUTPUT_DIR, f"{ticker}.csv")

    # Skip if already downloaded (idempotent)
    if os.path.exists(filepath):
        print(f"[SKIP] {ticker} already exists.")
        return

    try:
        print(f"[DOWNLOADING] {ticker} from {START_DATE} to {END_DATE}")
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if data.empty:
            print(f"[WARN] No data for {ticker}")
            return

        data.reset_index(inplace=True)
        data.to_csv(filepath, index=False)
        print(f"[OK] Saved {ticker} to {filepath}")
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")


def main():
    ensure_output_dir()
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers.")

    for i, ticker in enumerate(tickers, start=1):
        download_single_ticker(ticker)
        time.sleep(0.2)

    print("Done.")


if __name__ == "__main__":
    main()
