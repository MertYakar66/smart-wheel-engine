"""
Trade Outcome Labeler

Creates training labels for trades using the SAME valuation logic as the simulator.
This ensures ML models learn the correct objective function.

CRITICAL: This module imports from engine.shared_valuation to ensure consistency
with the backtest simulator. Any changes to exit logic must be made there.
"""

import sys
from pathlib import Path
import logging

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.shared_valuation import simulate_option_trade, TradeOutcome
from utils.metadata import save_with_metadata
from utils.dates import normalize_date

logger = logging.getLogger(__name__)

RAW_OHLCV_DIR = Path("data_raw/ohlcv")
BLOOMBERG_OHLCV_DIR = Path("data/bloomberg/ohlcv")
TRADE_UNIVERSE_DIR = Path("data_processed/trade_universe")
LABELS_DIR = Path("data_processed/labels")


def ensure_labels_dir():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


def load_ohlcv_for_ticker(ticker: str) -> pd.DataFrame | None:
    """
    Load OHLCV for a ticker with data cleaning.

    Checks Bloomberg data first, then falls back to yfinance data.

    Args:
        ticker: Stock ticker

    Returns:
        Clean OHLCV DataFrame or None
    """
    # Try Bloomberg first, then yfinance
    path = BLOOMBERG_OHLCV_DIR / f"{ticker}.csv"
    if not path.exists():
        path = RAW_OHLCV_DIR / f"{ticker}.csv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception as e:
        logger.warning(f"Error loading OHLCV for {ticker}: {e}")
        return None

    if df.empty:
        return None

    # Handle Bloomberg column names (PX_LAST → Close)
    col_map = {"PX_LAST": "Close", "PX_OPEN": "Open", "PX_HIGH": "High", "PX_LOW": "Low"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Clean data
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def label_trades_with_simulation(
    trade_date: str,
    profit_target_pct: float = 0.60,
    stop_loss_multiple: float = 2.0,
    risk_free_rate: float = 0.04
) -> pd.DataFrame:
    """
    Label trades using full simulation (matches simulator behavior exactly).

    This simulates each trade through its lifecycle, applying the same
    exit rules as the backtest simulator.

    Args:
        trade_date: Trade date string (YYYY-MM-DD)
        profit_target_pct: Profit target (default 60%)
        stop_loss_multiple: Stop loss multiple (default 2x)
        risk_free_rate: Risk-free rate

    Returns:
        DataFrame with simulation-based labels
    """
    universe_path = TRADE_UNIVERSE_DIR / f"{trade_date}_trade_universe.csv"
    if not universe_path.exists():
        logger.warning(f"Trade universe file not found for {trade_date}")
        return pd.DataFrame()

    df = pd.read_csv(universe_path)
    if df.empty:
        logger.warning(f"Trade universe is empty for {trade_date}")
        return pd.DataFrame()

    # Required columns
    required_cols = [
        "date", "ticker", "expiration", "option_type", "strike",
        "mid_price", "underlying_price", "strategy_leg", "implied_vol",
        "bid", "ask"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return pd.DataFrame()

    # Load OHLCV for all tickers
    tickers = df["ticker"].dropna().unique().tolist()
    ohlcv_map = {t: load_ohlcv_for_ticker(t) for t in tickers}

    labels = []
    skipped = 0

    for idx, row in df.iterrows():
        ticker = str(row["ticker"])
        option_type_raw = str(row["option_type"]).strip().upper()

        # Normalize option type
        if option_type_raw.startswith("P"):
            option_type = "put"
        elif option_type_raw.startswith("C"):
            option_type = "call"
        else:
            skipped += 1
            continue

        ohlcv_df = ohlcv_map.get(ticker)
        if ohlcv_df is None or ohlcv_df.empty:
            skipped += 1
            continue

        # Parse dates
        entry_date = normalize_date(row["date"])
        expiration_date = normalize_date(row["expiration"])

        if entry_date is None or expiration_date is None:
            skipped += 1
            continue

        # Get trade parameters
        strike = float(row["strike"])
        mid_price = float(row["mid_price"])
        iv = row.get("implied_vol")
        bid = row.get("bid")
        ask = row.get("ask")

        # Skip if IV is invalid
        if pd.isna(iv) or iv <= 0:
            skipped += 1
            continue

        # Simulate the trade
        outcome = simulate_option_trade(
            option_type=option_type,
            strike=strike,
            entry_premium=mid_price,
            entry_date=entry_date,
            expiration_date=expiration_date,
            entry_iv=float(iv),
            ohlcv_df=ohlcv_df,
            risk_free_rate=risk_free_rate,
            profit_target_pct=profit_target_pct,
            stop_loss_multiple=stop_loss_multiple,
            entry_bid=float(bid) if pd.notna(bid) else None,
            entry_ask=float(ask) if pd.notna(ask) else None
        )

        if outcome is None:
            skipped += 1
            continue

        # Build label record
        labels.append({
            # Identification
            "trade_date": str(entry_date),
            "ticker": ticker,
            "expiration": str(expiration_date),
            "option_type": option_type_raw[0],
            "strategy_leg": row.get("strategy_leg", ""),
            "strike": strike,
            "mid_price": mid_price,

            # Entry conditions
            "underlying_entry_price": float(row["underlying_price"]),
            "entry_iv": float(iv),

            # Exit details (from simulation)
            "exit_date": str(outcome.exit_date),
            "exit_reason": outcome.exit_reason,
            "exit_price": outcome.exit_price,
            "days_held": outcome.days_held,
            "underlying_exit_price": outcome.underlying_price_at_exit,

            # P&L breakdown (consistent with simulator)
            "gross_pnl": outcome.gross_pnl,
            "entry_costs": outcome.entry_costs,
            "exit_costs": outcome.exit_costs,
            "assignment_costs": outcome.assignment_costs,
            "net_pnl": outcome.net_pnl,

            # Binary flags
            "win_flag": 1 if outcome.net_pnl >= 0 else 0,
            "assignment_flag": 1 if outcome.was_assigned else 0,

            # Risk metrics
            "max_profit_during_hold": outcome.max_profit_reached,
            "max_loss_during_hold": outcome.max_loss_reached,
        })

    if skipped > 0:
        logger.info(f"Skipped {skipped} trades (missing data or invalid)")

    if not labels:
        logger.warning(f"No labelable trades for {trade_date}")
        return pd.DataFrame()

    return pd.DataFrame(labels)


def label_trades_expiry_only(trade_date: str) -> pd.DataFrame:
    """
    Label trades using expiry-only logic (simpler, doesn't match simulator).

    WARNING: This does NOT match the simulator's exit behavior.
    Use label_trades_with_simulation() for ML training.

    This function is kept for comparison/debugging purposes.

    Args:
        trade_date: Trade date string

    Returns:
        DataFrame with expiry-based labels
    """
    universe_path = TRADE_UNIVERSE_DIR / f"{trade_date}_trade_universe.csv"
    if not universe_path.exists():
        logger.warning(f"Trade universe file not found for {trade_date}")
        return pd.DataFrame()

    df = pd.read_csv(universe_path)
    if df.empty:
        logger.warning(f"Trade universe is empty for {trade_date}")
        return pd.DataFrame()

    # Load OHLCV
    tickers = df["ticker"].dropna().unique().tolist()
    ohlcv_map = {t: load_ohlcv_for_ticker(t) for t in tickers}

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

        ohlcv_df = ohlcv_map.get(ticker)
        if ohlcv_df is None:
            continue

        # Get expiry price
        target = pd.to_datetime(expiry_str)
        if ohlcv_df["Date"].max() < target:
            continue  # Don't have data yet

        df_sub = ohlcv_df[ohlcv_df["Date"] <= target]
        if df_sub.empty:
            continue

        expiry_close = float(df_sub.iloc[-1]["Close"])

        # Calculate P&L at expiry (no early exit)
        credit = mid_price * 100.0

        if option_type.startswith("P"):
            intrinsic = max(0.0, (strike - expiry_close) * 100.0)
            future_pnl = credit - intrinsic
            assignment_flag = 1 if expiry_close < strike else 0
        elif option_type.startswith("C"):
            intrinsic = max(0.0, (expiry_close - strike) * 100.0)
            future_pnl = credit - intrinsic
            assignment_flag = 1 if expiry_close > strike else 0
        else:
            continue

        labels.append({
            "trade_date": trade_date_str,
            "ticker": ticker,
            "expiration": expiry_str,
            "option_type": option_type[0],
            "strategy_leg": strategy_leg,
            "strike": strike,
            "mid_price": mid_price,
            "underlying_entry_price": entry_underlying,
            "underlying_expiry_price": expiry_close,
            "future_pnl": future_pnl,
            "win_flag": 1 if future_pnl >= 0 else 0,
            "assignment_flag": assignment_flag,
            "label_method": "expiry_only"
        })

    return pd.DataFrame(labels) if labels else pd.DataFrame()


def save_labels(df_labels: pd.DataFrame, trade_date: str, label_method: str = "simulation"):
    """
    Save labels with metadata.

    Args:
        df_labels: Labels DataFrame
        trade_date: Trade date
        label_method: 'simulation' or 'expiry_only'
    """
    ensure_labels_dir()

    if df_labels.empty:
        logger.warning(f"Empty labels for {trade_date}, nothing saved.")
        return

    fname = f"{trade_date}_labels_{label_method}.csv"
    path = LABELS_DIR / fname

    config = {
        "trade_date": trade_date,
        "label_method": label_method,
        "profit_target_pct": 0.60,
        "stop_loss_multiple": 2.0,
    }

    save_with_metadata(
        df=df_labels,
        filepath=str(path),
        config=config,
        data_start=trade_date,
        data_end=trade_date
    )

    logger.info(f"Saved {label_method} labels for {trade_date} to {path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Trade date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="simulation",
        choices=["simulation", "expiry_only"],
        help="Labeling method (default: simulation)"
    )
    args = parser.parse_args()

    logger.info(f"Labelling trades for {args.date} using {args.method} method")

    if args.method == "simulation":
        df_labels = label_trades_with_simulation(args.date)
    else:
        df_labels = label_trades_expiry_only(args.date)

    save_labels(df_labels, args.date, args.method)
