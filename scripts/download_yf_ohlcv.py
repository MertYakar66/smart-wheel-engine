import pandas as pd
import yfinance as yf
from pathlib import Path

RAW_DIR = Path("data_raw/ohlcv")


def load_tickers() -> list[str]:
    """
    Read tickers from the S&P500 constituents CSV.
    """
    df = pd.read_csv("data_raw/sp500_constituents_current.csv")
    return df["ticker"].dropna().tolist()


def download_ohlcv(ticker: str):
    """
    Download daily OHLCV for a single ticker from 2010-01-01 to today.
    Force a proper Date column and normalize column names.
    """
    try:
        print(f"[FETCH] {ticker}")
        df = yf.download(
            ticker,
            start="2010-01-01",
            progress=False,
            auto_adjust=False,   # keep raw OHLCV, don't auto-adjust
        )

        if df.empty:
            print(f"[WARN] No OHLCV for {ticker}")
            return

        # Ensure Date is a normal column instead of an index
        df.reset_index(inplace=True)

        if "Date" not in df.columns:
            raise RuntimeError(f"Downloaded OHLCV for {ticker} has no 'Date' column.")

        # Normalize column names
        rename_map = {
            "Adj Close": "Adj_Close",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
        df = df.rename(columns=rename_map)

        # Optional: drop any weird columns like 'Price' if present
        for junk_col in ["Price"]:
            if junk_col in df.columns:
                df.drop(columns=[junk_col], inplace=True)

        # Keep a clean, predictable set of columns
        cols_order = ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]
        cols_present = [c for c in cols_order if c in df.columns]
        df = df[cols_present]

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        outpath = RAW_DIR / f"{ticker}.csv"
        df.to_csv(outpath, index=False)
        print(f"[OK] Saved OHLCV for {ticker} -> {outpath}")

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_tickers()
    # match options script: only first 5 for now
    tickers = tickers[:5]
    print(f"[INFO] Downloading OHLCV for: {tickers}")

    for t in tickers:
        download_ohlcv(t)


if __name__ == "__main__":
    main()

