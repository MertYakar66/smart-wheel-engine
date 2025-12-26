import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_PATH = "data_raw/sp500_constituents_current.csv"


def fetch_sp500_constituents() -> pd.DataFrame:
    """
    Fetch the current list of S&P 500 constituents from Wikipedia.

    - Uses requests with a browser-like User-Agent to avoid HTTP 403.
    - Searches all tables on the page to find one with a ticker column
      (containing 'Symbol' or 'Ticker' in its name).
    - Normalizes that column to 'ticker' and cleans up the values for yfinance.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(WIKI_URL, headers=headers, timeout=20)
    resp.raise_for_status()

    # NOTE: pandas warns about passing literal HTML; that's fine for now.
    tables = pd.read_html(resp.text)

    ticker_col_name = None
    df = None

    # Find the first table that has a column containing 'Symbol' or 'Ticker'
    for t in tables:
        col_strings = [str(c) for c in t.columns]
        match = [c for c in col_strings if "Symbol" in c or "Ticker" in c]
        if match:
            df = t
            ticker_col_name = match[0]
            break

    if df is None or ticker_col_name is None:
        raise RuntimeError("Could not find a table with a Symbol/Ticker column on the S&P 500 page.")

    # Rename that column to 'ticker'
    df = df.rename(columns={ticker_col_name: "ticker"})

    # Clean ticker values for yfinance (BRK.B -> BRK-B, etc.)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    return df


def main():
    df = fetch_sp500_constituents()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} tickers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


