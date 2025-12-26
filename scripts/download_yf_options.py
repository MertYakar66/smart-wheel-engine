import os
from datetime import datetime

import pandas as pd
import yfinance as yf

CONSTITUENTS_PATH = "data_raw/sp500_constituents_current.csv"
OUTPUT_DIR = "data_raw/yfinance/options"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tickers(limit: int | None = None) -> list[str]:
    """
    Load tickers from the S&P500 constituents file.
    Optionally limit to the first N tickers for testing.
    """
    df = pd.read_csv(CONSTITUENTS_PATH)
    tickers = df["ticker"].dropna().unique().tolist()
    if limit is not None:
        tickers = tickers[:limit]
    return tickers


def fetch_option_chain_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Fetches the *current* full option chain for a given ticker from yfinance
    and returns a normalized DataFrame with calls and puts combined.
    """
    yf_ticker = yf.Ticker(ticker)

    try:
        expirations = yf_ticker.options
    except Exception as e:
        print(f"[ERROR] Failed to fetch expirations for {ticker}: {e}")
        return pd.DataFrame()

    if not expirations:
        print(f"[WARN] No option expirations for {ticker}")
        return pd.DataFrame()

    all_rows = []

    for exp in expirations:
        try:
            chain = yf_ticker.option_chain(exp)
        except Exception as e:
            print(f"[WARN] Failed to fetch chain for {ticker} @ {exp}: {e}")
            continue

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        if not calls.empty:
            calls["option_type"] = "C"
        if not puts.empty:
            puts["option_type"] = "P"

        # Normalize columns
        for df_side in [calls, puts]:
            if df_side.empty:
                continue

            df_side["ticker"] = ticker
            df_side["expiration"] = exp

            # Some yfinance versions use 'impliedVolatility'; normalize name
            if "impliedVolatility" in df_side.columns:
                df_side.rename(columns={"impliedVolatility": "implied_vol"}, inplace=True)
            elif "implied_vol" not in df_side.columns:
                df_side["implied_vol"] = None

            # Optional: keep only relevant columns
            keep_cols = [
                "ticker",
                "expiration",
                "option_type",
                "strike",
                "bid",
                "ask",
                "lastPrice" if "lastPrice" in df_side.columns else "last",
                "implied_vol",
                "volume",
                "openInterest",
            ]
            # Filter existing columns only
            keep_cols = [c for c in keep_cols if c in df_side.columns]

            # Take a real copy to avoid SettingWithCopyWarning
            df_side = df_side.loc[:, keep_cols].copy()

            # Standardize names
            df_side.rename(
                columns={
                    "lastPrice": "last",
                    "openInterest": "open_interest",
                },
                inplace=True,
            )


            all_rows.append(df_side)

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.insert(0, "date", datetime.today().strftime("%Y-%m-%d"))
    return df_all


def save_chain(df: pd.DataFrame, date_str: str, ticker: str):
    """
    Save a single ticker's option chain for a given date.
    """
    if df.empty:
        print(f"[WARN] No options data to save for {ticker}")
        return

    filename = f"{date_str}_{ticker}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"[OK] Saved {ticker} options to {filepath}")


def main():
    ensure_output_dir()
    today_str = datetime.today().strftime("%Y-%m-%d")

    # For now, limit to a small subset to keep it fast.
    # You can increase this number later.
    tickers = load_tickers(limit=5)
    print(f"Loaded {len(tickers)} tickers: {tickers}")

    for ticker in tickers:
        print(f"[FETCH] {ticker}")
        df_chain = fetch_option_chain_for_ticker(ticker)
        save_chain(df_chain, today_str, ticker)

    print("Done.")


if __name__ == "__main__":
    main()
