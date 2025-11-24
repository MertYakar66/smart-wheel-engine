import os
from pathlib import Path

import pandas as pd


RAW_OHLCV_DIR = Path("data_raw/ohlcv")
TRADE_UNIVERSE_DIR = Path("data_processed/trade_universe")
LABELS_DIR = Path("data_processed/labels")


def ensure_labels_dir():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


def load_ohlcv_for_ticker(ticker: str) -> pd.DataFrame | None:
    """
    Load OHLCV for a ticker. Returns None if not found or empty.
    """
    path = RAW_OHLCV_DIR / f"{ticker}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["Date"])
    if df.empty:
        return None

    df = df.sort_values("Date")
    return df


def get_close_on_or_before(df_ohlcv: pd.DataFrame, target_date: str) -> float | None:
    """
    Given an OHLCV DataFrame and a target date (YYYY-MM-DD),
    return the close on or before that date.

    If target_date is after the last available OHLCV date,
    we return None to avoid look-ahead bias (we don't know expiry yet).
    """
    if df_ohlcv is None or df_ohlcv.empty:
        return None

    target = pd.to_datetime(target_date)
    max_date = df_ohlcv["Date"].max()

    # If we don't have data up to the expiry date, we cannot label this trade
    if max_date < target:
        return None

    df_sub = df_ohlcv[df_ohlcv["Date"] <= target]
    if df_sub.empty:
        return None

    return float(df_sub.iloc[-1]["Close"])


def label_trades_for_date(trade_date: str) -> pd.DataFrame:
    """
    Load the trade universe for a given trade_date (YYYY-MM-DD),
    compute expiry-based P&L and win/assignment flags,
    and return a labels DataFrame.
    """
    universe_path = TRADE_UNIVERSE_DIR / f"{trade_date}_trade_universe.csv"
    if not universe_path.exists():
        print(f"[WARN] Trade universe file not found for {trade_date}: {universe_path}")
        return pd.DataFrame()

    df = pd.read_csv(universe_path)
    if df.empty:
        print(f"[WARN] Trade universe is empty for {trade_date}")
        return pd.DataFrame()

    # Ensure required columns exist
    required_cols = [
        "date",
        "ticker",
        "expiration",
        "option_type",
        "strike",
        "mid_price",
        "underlying_price",
        "strategy_leg",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in trade universe.")

    # Preload OHLCV for all tickers in the universe
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    ohlcv_map: dict[str, pd.DataFrame | None] = {}
    for t in tickers:
        ohlcv_map[t] = load_ohlcv_for_ticker(t)

    labels = []

    for idx, row in df.iterrows():
        ticker = str(row["ticker"])
        option_type = str(row["option_type"]).strip().upper()
        strike = float(row["strike"])
        mid_price = float(row["mid_price"])
        entry_underlying = float(row["underlying_price"])
        strategy_leg = str(row.get("strategy_leg", ""))

        trade_date_str = str(row["date"])
        expiry_str = str(row["expiration"])

        df_ohlcv = ohlcv_map.get(ticker)
        expiry_close = get_close_on_or_before(df_ohlcv, expiry_str)

        # If we don't have data up to expiry, skip this trade (cannot label yet)
        if expiry_close is None:
            continue

        credit = mid_price * 100.0  # 1 contract, 100 shares
        future_pnl = None
        win_flag = None
        assignment_flag = None

        if option_type.startswith("P"):  # short put
            intrinsic = max(0.0, (strike - expiry_close) * 100.0)
            future_pnl = credit - intrinsic
            win_flag = 1 if future_pnl >= 0.0 else 0
            assignment_flag = 1 if expiry_close < strike else 0

        elif option_type.startswith("C"):  # short call (covered or not)
            intrinsic = max(0.0, (expiry_close - strike) * 100.0)
            future_pnl = credit - intrinsic
            win_flag = 1 if future_pnl >= 0.0 else 0
            assignment_flag = 1 if expiry_close > strike else 0

        else:
            # Unknown option type, skip
            continue

        labels.append(
            {
                "trade_date": trade_date_str,
                "ticker": ticker,
                "expiration": expiry_str,
                "option_type": option_type,
                "strategy_leg": strategy_leg,
                "strike": strike,
                "mid_price": mid_price,
                "underlying_entry_price": entry_underlying,
                "underlying_expiry_price": expiry_close,
                "future_pnl": future_pnl,
                "win_flag": win_flag,
                "assignment_flag": assignment_flag,
            }
        )

    if not labels:
        print(f"[WARN] No labelable trades for {trade_date} (likely expiries in the future).")
        return pd.DataFrame()

    df_labels = pd.DataFrame(labels)
    return df_labels


def save_labels(df_labels: pd.DataFrame, trade_date: str):
    ensure_labels_dir()
    if df_labels.empty:
        print(f"[WARN] Empty labels for {trade_date}, nothing saved.")
        return

    out_path = LABELS_DIR / f"{trade_date}_labels_expiry.csv"
    df_labels.to_csv(out_path, index=False)
    print(f"[OK] Saved expiry labels for {trade_date} to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Trade date in YYYY-MM-DD format, must match trade universe file.",
    )
    args = parser.parse_args()

    trade_date = args.date
    print(f"[INFO] Labelling trades for {trade_date} (expiry-based)")

    df_labels = label_trades_for_date(trade_date)
    save_labels(df_labels, trade_date)
