import os
from datetime import datetime

import pandas as pd


RAW_OHLCV_DIR = "data_raw/ohlcv"
RAW_YF_OPTIONS_DIR = "data_raw/yfinance/options"
OUTPUT_DIR = "data_processed/trade_universe"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_spot_price(ticker: str, trade_date: str) -> float | None:
    """
    Load underlying close price for a given ticker and trade_date (YYYY-MM-DD).

    If there is no exact match for trade_date in the OHLCV file
    (e.g. trade_date is a weekend/holiday), use the most recent
    close ON OR BEFORE trade_date.

    Returns None if there is no data up to that date.
    """
    ohlcv_path = os.path.join(RAW_OHLCV_DIR, f"{ticker}.csv")
    if not os.path.exists(ohlcv_path):
        return None

    df = pd.read_csv(ohlcv_path, parse_dates=["Date"])
    if df.empty:
        return None

    df = df.sort_values("Date")
    target = pd.to_datetime(trade_date)

    # Keep all rows up to the target date, pick the last one
    df_sub = df[df["Date"] <= target]
    if df_sub.empty:
        return None

    return float(df_sub.iloc[-1]["Close"])



def load_option_snapshot_for_date(trade_date: str) -> pd.DataFrame:
    """
    Load all yfinance option snapshot files for a given trade_date
    (e.g. '2025-11-22') and concatenate into one DataFrame.
    """
    frames = []
    for fname in os.listdir(RAW_YF_OPTIONS_DIR):
        if not fname.endswith(".csv"):
            continue
        if not fname.startswith(trade_date):
            continue

        path = os.path.join(RAW_YF_OPTIONS_DIR, fname)

        df = pd.read_csv(path)

        # Force all option rows to use the trade_date we requested
        df["date"] = trade_date

        frames.append(df)


    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute DTE, moneyness, mid price, etc.
    Assumes columns:
      - date, ticker, expiration, option_type, strike, bid, ask, implied_vol,
        volume, open_interest
    """
    if df.empty:
        return df

    # Parse dates
    df["trade_date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # DTE
    df["dte"] = (df["expiration"] - df["trade_date"]).dt.days

    # Mid price
    df["mid_price"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
    df.loc[
        (df["mid_price"] <= 0) & df["bid"].notna() & df["ask"].notna(),
        "mid_price"
    ] = df["bid"]

    # Spot price per ticker & trade date
    df["trade_date_str"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    spots = {}
    underlying_prices = []

    for _, row in df[["ticker", "trade_date_str"]].drop_duplicates().iterrows():
        key = (row["ticker"], row["trade_date_str"])
        price = load_spot_price(row["ticker"], row["trade_date_str"])
        spots[key] = price

    for _, row in df.iterrows():
        key = (row["ticker"], row["trade_date_str"])
        underlying_prices.append(spots.get(key))

    df["underlying_price"] = underlying_prices

    # Moneyness
    df["moneyness_pct"] = (
        df["strike"] / df["underlying_price"] - 1.0
    ) * 100.0

    return df


def filter_short_put_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    TEMP: relaxed filter to ensure we see some rows for debugging.
    Later we will tighten DTE, moneyness, liquidity, etc.
    """
    mask = (
        (df["option_type"] == "P")
        & df["underlying_price"].notna()
        & df["dte"].between(1, 120)
        & df["moneyness_pct"].between(-30, 30)
    )
    return df.loc[mask].copy()


def filter_covered_call_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    TEMP: relaxed filter to ensure we see some rows for debugging.
    Later we will tighten DTE, moneyness, liquidity, etc.
    """
    mask = (
        (df["option_type"] == "C")
        & df["underlying_price"].notna()
        & df["dte"].between(1, 120)
        & df["moneyness_pct"].between(-30, 30)
    )
    return df.loc[mask].copy()


def build_trade_universe_for_date(trade_date: str) -> pd.DataFrame:
    """
    Main entry point for Block 3.
    Given a trade_date 'YYYY-MM-DD', loads all option snapshots for that date,
    merges underlying prices, computes basic fields and filters to candidate
    short puts and covered calls.
    """
    snap = load_option_snapshot_for_date(trade_date)
    if snap.empty:
        print(f"[WARN] No option snapshots for {trade_date}")
        return snap

    # Ensure required columns exist; if some are missing, fill with NaN
    required_cols = [
        "date",
        "ticker",
        "expiration",
        "option_type",
        "strike",
        "bid",
        "ask",
        "implied_vol",
        "volume",
        "open_interest",
    ]
    for col in required_cols:
        if col not in snap.columns:
            snap[col] = pd.NA

    df = snap[required_cols].copy()

    # IMPORTANT: many snapshots don't store a 'date' column,
    # only the filename encodes it. Force it from the argument.
    df["date"] = trade_date

    # Compute dte, moneyness, mid_price, underlying_price
    df = add_basic_fields(df)

    # Normalize option_type to 'P' / 'C' regardless of source format
    df["option_type"] = (
        df["option_type"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str[0]  # 'PUT' -> 'P', 'CALL' -> 'C'
    )

    puts = filter_short_put_candidates(df)
    calls = filter_covered_call_candidates(df)

    universe = pd.concat(
        [
            puts.assign(strategy_leg="short_put"),
            calls.assign(strategy_leg="covered_call"),
        ],
        ignore_index=True,
    )

    # Placeholders for future labels
    universe["future_pnl"] = pd.NA
    universe["win_flag"] = pd.NA
    universe["assignment_flag"] = pd.NA

    return universe



def save_trade_universe(df: pd.DataFrame, trade_date: str):
    ensure_output_dir()
    if df.empty:
        print(f"[WARN] Empty trade universe for {trade_date}, nothing saved.")
        return

    fname = f"{trade_date}_trade_universe.csv"
    path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=False)
    print(f"[OK] Saved trade universe for {trade_date} to {path}")


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format. If not given, defaults to today.",
    )
    args = parser.parse_args()

    # Use provided date or default to today
    trade_date = args.date or datetime.today().strftime("%Y-%m-%d")
    print(f"[INFO] Building trade universe for {trade_date}")

    universe = build_trade_universe_for_date(trade_date)
    save_trade_universe(universe, trade_date)


