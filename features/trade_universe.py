"""
Trade Universe Generator

Builds candidate trade datasets from option snapshots and spot prices.
Applies data validation, liquidity filters, and IV normalization.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_validation import validate_and_normalize_iv
from utils.metadata import save_with_metadata

logger = logging.getLogger(__name__)

RAW_OHLCV_DIR = Path("data_raw/ohlcv")
RAW_YF_OPTIONS_DIR = Path("data_raw/yfinance/options")
BLOOMBERG_OHLCV_DIR = Path("data/bloomberg/ohlcv")
BLOOMBERG_OPTIONS_DIR = Path("data/bloomberg/options")
OUTPUT_DIR = Path("data_processed/trade_universe")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_spot_price(ticker: str, trade_date: str) -> float | None:
    """
    Load underlying close price for a given ticker and trade_date.

    Checks Bloomberg data first, then falls back to yfinance data.
    Uses "on or before" logic to handle weekends/holidays.

    Args:
        ticker: Stock ticker
        trade_date: Date string (YYYY-MM-DD)

    Returns:
        Close price or None if not found
    """
    # Try Bloomberg first, then yfinance
    ohlcv_path = BLOOMBERG_OHLCV_DIR / f"{ticker}.csv"
    if not ohlcv_path.exists():
        ohlcv_path = RAW_OHLCV_DIR / f"{ticker}.csv"
    if not ohlcv_path.exists():
        return None

    try:
        df = pd.read_csv(ohlcv_path, parse_dates=["Date"])
    except Exception as e:
        logger.warning(f"Error loading OHLCV for {ticker}: {e}")
        return None

    if df.empty:
        return None

    # Handle Bloomberg column names (PX_LAST → Close)
    col_map = {"PX_LAST": "Close", "PX_OPEN": "Open", "PX_HIGH": "High", "PX_LOW": "Low"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure numeric Close column
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])

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
    Load option snapshot files for a given trade_date.

    Checks Bloomberg data first, then falls back to yfinance data.

    Args:
        trade_date: Date string (YYYY-MM-DD)

    Returns:
        Concatenated DataFrame of all options
    """
    frames = []

    # Bloomberg column mapping
    bbg_col_map = {
        "OPT_STRIKE_PX": "strike",
        "OPT_PUT_CALL": "option_type",
        "OPT_EXPIRE_DT": "expiration",
        "BID": "bid",
        "ASK": "ask",
        "IVOL_MID": "implied_vol",
        "OPT_IMPLIED_VOLATILITY_MID": "implied_vol",
        "OPEN_INT": "open_interest",
        "VOLUME": "volume",
        "OPT_UNDL_PX": "underlying_price",
        "PX_LAST": "last",
    }

    # Try Bloomberg options directory first
    for opts_dir in [BLOOMBERG_OPTIONS_DIR, RAW_YF_OPTIONS_DIR]:
        if not opts_dir.exists():
            continue

        for fname in os.listdir(opts_dir):
            if not fname.endswith(".csv"):
                continue
            # Accept both dated (2025-01-15_AAPL.csv) and undated (AAPL.csv)
            if trade_date and not fname.startswith(trade_date):
                # For undated Bloomberg files, load them regardless
                if "_" in fname or opts_dir == RAW_YF_OPTIONS_DIR:
                    continue

            path = opts_dir / fname

            try:
                df = pd.read_csv(path)
                # Rename Bloomberg columns to engine format
                df = df.rename(columns={
                    k: v for k, v in bbg_col_map.items() if k in df.columns
                })
                df["date"] = trade_date
                # Extract ticker from filename
                if "ticker" not in df.columns:
                    stem = fname.replace(".csv", "")
                    ticker = stem.split("_")[-1] if "_" in stem else stem
                    df["ticker"] = ticker
                frames.append(df)
            except Exception as e:
                logger.warning(f"Error loading {fname}: {e}")
                continue

        if frames:
            break  # Use first source that has data

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute DTE, moneyness, mid price, and validate IV.

    Args:
        df: Raw options DataFrame

    Returns:
        DataFrame with computed fields
    """
    if df.empty:
        return df

    df = df.copy()

    # Parse dates
    df["trade_date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # DTE
    df["dte"] = (df["expiration"] - df["trade_date"]).dt.days

    # Mid price with validation
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0)
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0)
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

    # Handle zero mid (use ask if bid is 0)
    zero_mid = (df["mid_price"] <= 0) & (df["ask"] > 0)
    df.loc[zero_mid, "mid_price"] = df.loc[zero_mid, "ask"] / 2

    # Calculate actual spread
    df["bid_ask_spread"] = df["ask"] - df["bid"]
    df["spread_pct"] = df["bid_ask_spread"] / df["mid_price"].replace(0, np.nan)

    # Spot price per ticker & trade date
    df["trade_date_str"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    spots = {}
    unique_pairs = df[["ticker", "trade_date_str"]].drop_duplicates()

    for _, row in unique_pairs.iterrows():
        key = (row["ticker"], row["trade_date_str"])
        price = load_spot_price(row["ticker"], row["trade_date_str"])
        spots[key] = price

    df["underlying_price"] = df.apply(
        lambda r: spots.get((r["ticker"], r["trade_date_str"])), axis=1
    )

    # Moneyness
    df["moneyness_pct"] = (df["strike"] / df["underlying_price"] - 1.0) * 100.0

    # Validate and normalize IV
    df["implied_vol_raw"] = df["implied_vol"]
    normalized_ivs = []
    iv_warnings = []

    for idx, iv in df["implied_vol"].items():
        norm_iv, warning = validate_and_normalize_iv(iv)
        normalized_ivs.append(norm_iv)
        if warning and norm_iv is not None:
            iv_warnings.append((idx, warning))

    df["implied_vol"] = normalized_ivs

    if iv_warnings:
        logger.info(f"IV normalization: {len(iv_warnings)} values adjusted")

    # Ensure numeric open_interest
    if "open_interest" in df.columns:
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0).astype(int)
    else:
        df["open_interest"] = 0

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    else:
        df["volume"] = 0

    return df


def filter_short_put_candidates(
    df: pd.DataFrame,
    min_dte: int = 21,
    max_dte: int = 60,
    min_moneyness: float = -15.0,
    max_moneyness: float = 5.0,
    min_open_interest: int = 100,
    min_bid: float = 0.05,
    max_spread_pct: float = 0.50
) -> pd.DataFrame:
    """
    Filter for short put candidates with liquidity and moneyness requirements.

    Args:
        df: Options DataFrame with computed fields
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration
        min_moneyness: Minimum moneyness % (negative = OTM puts)
        max_moneyness: Maximum moneyness % (positive = ITM puts)
        min_open_interest: Minimum open interest
        min_bid: Minimum bid price
        max_spread_pct: Maximum spread as % of mid

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    mask = (
        (df["option_type"].str.upper().str.startswith("P"))
        & df["underlying_price"].notna()
        & df["implied_vol"].notna()
        & (df["implied_vol"] > 0)
        & df["dte"].between(min_dte, max_dte)
        & df["moneyness_pct"].between(min_moneyness, max_moneyness)
        & (df["open_interest"] >= min_open_interest)
        & (df["bid"] >= min_bid)
        & (df["spread_pct"] <= max_spread_pct)
    )

    filtered = df.loc[mask].copy()
    logger.info(f"Short put filter: {len(filtered)}/{len(df)} options passed")
    return filtered


def filter_covered_call_candidates(
    df: pd.DataFrame,
    min_dte: int = 21,
    max_dte: int = 45,
    min_moneyness: float = 0.0,
    max_moneyness: float = 10.0,
    min_open_interest: int = 100,
    min_bid: float = 0.05,
    max_spread_pct: float = 0.50
) -> pd.DataFrame:
    """
    Filter for covered call candidates (OTM calls only).

    Args:
        df: Options DataFrame with computed fields
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration
        min_moneyness: Minimum moneyness % (0 = ATM)
        max_moneyness: Maximum moneyness % (10 = 10% OTM)
        min_open_interest: Minimum open interest
        min_bid: Minimum bid price
        max_spread_pct: Maximum spread as % of mid

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    mask = (
        (df["option_type"].str.upper().str.startswith("C"))
        & df["underlying_price"].notna()
        & df["implied_vol"].notna()
        & (df["implied_vol"] > 0)
        & df["dte"].between(min_dte, max_dte)
        & df["moneyness_pct"].between(min_moneyness, max_moneyness)
        & (df["open_interest"] >= min_open_interest)
        & (df["bid"] >= min_bid)
        & (df["spread_pct"] <= max_spread_pct)
    )

    filtered = df.loc[mask].copy()
    logger.info(f"Covered call filter: {len(filtered)}/{len(df)} options passed")
    return filtered


def build_trade_universe_for_date(
    trade_date: str,
    put_filters: dict = None,
    call_filters: dict = None,
    apply_strict_liquidity: bool = True
) -> pd.DataFrame:
    """
    Build trade universe for a given date.

    Args:
        trade_date: Date string (YYYY-MM-DD)
        put_filters: Override filters for puts
        call_filters: Override filters for calls
        apply_strict_liquidity: Apply strict liquidity requirements

    Returns:
        DataFrame with candidate trades
    """
    logger.info(f"Building trade universe for {trade_date}")

    snap = load_option_snapshot_for_date(trade_date)
    if snap.empty:
        logger.warning(f"No option snapshots for {trade_date}")
        return snap

    # Ensure required columns exist
    required_cols = [
        "date", "ticker", "expiration", "option_type", "strike",
        "bid", "ask", "implied_vol", "volume", "open_interest",
    ]
    for col in required_cols:
        if col not in snap.columns:
            snap[col] = pd.NA

    df = snap[required_cols].copy()
    df["date"] = trade_date

    # Compute DTE, moneyness, mid_price, underlying_price
    df = add_basic_fields(df)

    # Normalize option_type
    df["option_type"] = (
        df["option_type"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str[0]
    )

    # Apply filters
    put_params = put_filters or {}
    call_params = call_filters or {}

    if apply_strict_liquidity:
        puts = filter_short_put_candidates(df, **put_params)
        calls = filter_covered_call_candidates(df, **call_params)
    else:
        # Relaxed filters for debugging
        puts = df[
            (df["option_type"] == "P") &
            df["underlying_price"].notna() &
            df["dte"].between(1, 120) &
            df["moneyness_pct"].between(-30, 30)
        ].copy()
        calls = df[
            (df["option_type"] == "C") &
            df["underlying_price"].notna() &
            df["dte"].between(1, 120) &
            df["moneyness_pct"].between(-30, 30)
        ].copy()

    universe = pd.concat(
        [
            puts.assign(strategy_leg="short_put"),
            calls.assign(strategy_leg="covered_call"),
        ],
        ignore_index=True,
    )

    # Placeholders for labels
    universe["future_pnl"] = pd.NA
    universe["win_flag"] = pd.NA
    universe["assignment_flag"] = pd.NA

    logger.info(f"Trade universe: {len(universe)} candidates ({len(puts)} puts, {len(calls)} calls)")
    return universe


def save_trade_universe(df: pd.DataFrame, trade_date: str, config: dict = None):
    """
    Save trade universe with metadata.

    Args:
        df: Trade universe DataFrame
        trade_date: Date string
        config: Configuration parameters used
    """
    ensure_output_dir()

    if df.empty:
        logger.warning(f"Empty trade universe for {trade_date}, nothing saved.")
        return

    fname = f"{trade_date}_trade_universe.csv"
    path = OUTPUT_DIR / fname

    # Save with metadata sidecar
    save_with_metadata(
        df=df,
        filepath=str(path),
        config=config or {"date": trade_date},
        data_start=trade_date,
        data_end=trade_date
    )

    logger.info(f"Saved trade universe for {trade_date} to {path}")


if __name__ == "__main__":
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format. If not given, defaults to today.",
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Use relaxed liquidity filters (for debugging)"
    )
    args = parser.parse_args()

    trade_date = args.date or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Building trade universe for {trade_date}")

    universe = build_trade_universe_for_date(
        trade_date,
        apply_strict_liquidity=not args.relaxed
    )
    save_trade_universe(universe, trade_date)
