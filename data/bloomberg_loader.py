"""
Bloomberg Data Ingestion Module

Parses CSV files exported from Bloomberg Excel Add-In (BDH/BDP/BDS)
and normalizes them into the engine's internal formats.

Data categories handled:
1. OHLCV daily price history
2. Option chains (strikes, bids, asks, IV, Greeks, OI)
3. Earnings dates + historical surprises
4. Dividend schedule (ex-dates, amounts, yields)
5. IV history (daily ATM IV for IV rank calculation)
6. Risk-free rates (Treasury yields)
7. Fundamentals (market cap, GICS sector)

Expected file locations:
    data/bloomberg/ohlcv/          - Daily prices per ticker
    data/bloomberg/options/        - Option chain snapshots
    data/bloomberg/earnings/       - Earnings data per ticker
    data/bloomberg/dividends/      - Dividend data per ticker
    data/bloomberg/iv_history/     - Daily IV per ticker
    data/bloomberg/rates/          - Treasury yield curve
    data/bloomberg/fundamentals/   - Company fundamentals
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
import logging

import numpy as np
import pandas as pd

from utils.data_validation import (
    validate_and_normalize_iv,
    validate_ohlcv_data,
    validate_option_data,
)

logger = logging.getLogger(__name__)

# ─── Base directory ───────────────────────────────────────────────────
BLOOMBERG_DIR = Path("data/bloomberg")


# ─────────────────────────────────────────────────────────────────────
# Column mappings: Bloomberg field name → engine field name
# ─────────────────────────────────────────────────────────────────────

OHLCV_COLUMN_MAP = {
    # Bloomberg BDH fields → engine columns
    "Date": "Date",
    "DATE": "Date",
    "date": "Date",
    "PX_OPEN": "Open",
    "PX_HIGH": "High",
    "PX_LOW": "Low",
    "PX_LAST": "Close",
    "PX_CLOSE": "Close",
    "PX_VOLUME": "Volume",
    "VOLUME": "Volume",
    "EQY_WEIGHTED_AVG_PX": "VWAP",
    # Already-named columns (if user renames in Excel)
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Volume": "Volume",
}

OPTIONS_COLUMN_MAP = {
    # Bloomberg option chain fields → engine columns
    "OPT_STRIKE_PX": "strike",
    "Strike": "strike",
    "strike": "strike",
    "OPT_PUT_CALL": "option_type",
    "Put/Call": "option_type",
    "option_type": "option_type",
    "OPT_EXPIRE_DT": "expiration",
    "Expiration": "expiration",
    "expiration": "expiration",
    "BID": "bid",
    "Bid": "bid",
    "bid": "bid",
    "ASK": "ask",
    "Ask": "ask",
    "ask": "ask",
    "PX_LAST": "last",
    "Last": "last",
    "last": "last",
    "IVOL_MID": "implied_vol",
    "OPT_IMPLIED_VOLATILITY_MID": "implied_vol",
    "IV_Mid": "implied_vol",
    "implied_vol": "implied_vol",
    "OPEN_INT": "open_interest",
    "Open_Int": "open_interest",
    "open_interest": "open_interest",
    "VOLUME": "volume",
    "Volume": "volume",
    "volume": "volume",
    "OPT_DELTA": "delta",
    "Delta": "delta",
    "OPT_GAMMA": "gamma",
    "Gamma": "gamma",
    "OPT_THETA": "theta",
    "Theta": "theta",
    "OPT_VEGA": "vega",
    "Vega": "vega",
    "OPT_UNDL_PX": "underlying_price",
    "Underlying": "underlying_price",
    "underlying_price": "underlying_price",
}

EARNINGS_COLUMN_MAP = {
    "Date": "earnings_date",
    "date": "earnings_date",
    "EXPECTED_REPORT_DT": "earnings_date",
    "ANNOUNCEMENT_DT": "earnings_date",
    "IS_EPS": "eps_actual",
    "EPS_Actual": "eps_actual",
    "BEST_EPS_MEDIAN": "eps_estimate",
    "EPS_Estimate": "eps_estimate",
    "EARN_EST_EPS_SURPRISE": "eps_surprise",
    "EPS_Surprise": "eps_surprise",
    "EARN_EST_EPS_SURPRISE_PCT": "surprise_pct",
    "Surprise_Pct": "surprise_pct",
    "BMO_AMC": "timing",
    "EARNING_ANNOUNCEMENT_TIMING": "timing",
    "Timing": "timing",
    "PX_LAST": "close_price",
    "Close": "close_price",
    "IVOL_MID": "iv_pre_earnings",
    "IV_Pre": "iv_pre_earnings",
}

DIVIDEND_COLUMN_MAP = {
    "DVD_EX_DT": "ex_date",
    "Ex_Date": "ex_date",
    "ex_date": "ex_date",
    "DVD_RECORD_DT": "record_date",
    "Record_Date": "record_date",
    "DVD_PAY_DT": "pay_date",
    "Pay_Date": "pay_date",
    "DVD_SH_LAST": "amount",
    "Amount": "amount",
    "amount": "amount",
    "DVD_FREQ": "frequency",
    "Frequency": "frequency",
    "EQY_DVD_YLD_IND": "dividend_yield",
    "Div_Yield": "dividend_yield",
}

IV_HISTORY_COLUMN_MAP = {
    "Date": "date",
    "DATE": "date",
    "date": "date",
    "30DAY_IMPVOL_100.0%MNY_DF": "iv_atm_30d",
    "IVOL_30D": "iv_atm_30d",
    "IV_30D": "iv_atm_30d",
    "iv_atm_30d": "iv_atm_30d",
    "60DAY_IMPVOL_100.0%MNY_DF": "iv_atm_60d",
    "IVOL_60D": "iv_atm_60d",
    "IV_60D": "iv_atm_60d",
    "iv_atm_60d": "iv_atm_60d",
    "90DAY_IMPVOL_100.0%MNY_DF": "iv_atm_90d",
    "IV_90D": "iv_atm_90d",
    "iv_atm_90d": "iv_atm_90d",
    "30DAY_IMPVOL_90.0%MNY_DF": "iv_25d_put",
    "IV_25D_Put": "iv_25d_put",
    "30DAY_IMPVOL_110.0%MNY_DF": "iv_25d_call",
    "IV_25D_Call": "iv_25d_call",
    "HIST_PUT_IMP_VOL": "iv_put",
    "HIST_CALL_IMP_VOL": "iv_call",
    "20DAY_HV": "rv_20d",
    "HV_20D": "rv_20d",
    "rv_20d": "rv_20d",
    "60DAY_HV": "rv_60d",
    "HV_60D": "rv_60d",
    "rv_60d": "rv_60d",
}

RATES_COLUMN_MAP = {
    "Date": "date",
    "DATE": "date",
    "date": "date",
    "PX_LAST": "yield",
    "Yield": "yield",
}


# ─────────────────────────────────────────────────────────────────────
# Core parsing utilities
# ─────────────────────────────────────────────────────────────────────

def _rename_columns(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Rename columns using mapping, keeping unmapped columns as-is."""
    rename_dict = {}
    for col in df.columns:
        col_stripped = str(col).strip()
        if col_stripped in column_map:
            rename_dict[col] = column_map[col_stripped]
    return df.rename(columns=rename_dict)


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Parse date column handling Bloomberg's common date formats."""
    # Bloomberg uses: MM/DD/YYYY, YYYY-MM-DD, YYYYMMDD
    return pd.to_datetime(series, format="mixed", errors="coerce")


def _detect_header_rows(filepath: str) -> int:
    """
    Detect how many header rows to skip in Bloomberg Excel exports.

    Bloomberg BDH exports often have:
    - Row 0: security name
    - Row 1: field names
    - Row 2+: data

    Returns number of rows to skip.
    """
    try:
        with open(filepath, "r") as f:
            lines = [f.readline() for _ in range(5)]
    except Exception:
        return 0

    # Check if first row looks like a header (contains "Equity" or ticker)
    if lines and ("Equity" in lines[0] or "US Equity" in lines[0]):
        return 1

    return 0


# ─────────────────────────────────────────────────────────────────────
# 1. OHLCV Loader
# ─────────────────────────────────────────────────────────────────────

def load_bloomberg_ohlcv(
    ticker: str,
    data_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Load and normalize Bloomberg OHLCV data for a single ticker.

    Expected Bloomberg BDH formula:
        =BDH("AAPL US Equity","PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME",
              "20190101","20260101","Dir=V")

    Saved as: data/bloomberg/ohlcv/AAPL.csv

    Args:
        ticker: Stock ticker (e.g., "AAPL").
        data_dir: Override data directory.

    Returns:
        Normalized DataFrame with columns: Date, Open, High, Low, Close, Volume
        or None if file not found / invalid.
    """
    base_dir = data_dir or (BLOOMBERG_DIR / "ohlcv")
    filepath = base_dir / f"{ticker}.csv"

    if not filepath.exists():
        logger.warning(f"OHLCV file not found: {filepath}")
        return None

    try:
        skip = _detect_header_rows(str(filepath))
        df = pd.read_csv(filepath, skiprows=skip)
    except Exception as e:
        logger.error(f"Error reading OHLCV for {ticker}: {e}")
        return None

    if df.empty:
        return None

    # Rename columns
    df = _rename_columns(df, OHLCV_COLUMN_MAP)

    # Ensure required columns exist
    required = ["Date", "Close"]
    for col in required:
        if col not in df.columns:
            # Try to find it case-insensitively
            for c in df.columns:
                if c.lower() == col.lower():
                    df = df.rename(columns={c: col})
                    break
            else:
                logger.error(f"Missing required column '{col}' in {filepath}")
                return None

    # Parse dates
    df["Date"] = _parse_date_column(df["Date"])
    df = df.dropna(subset=["Date"])

    # Ensure numeric price columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing OHLV from Close if needed
    if "Open" not in df.columns:
        df["Open"] = df["Close"]
    if "High" not in df.columns:
        df["High"] = df["Close"]
    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.dropna(subset=["Close"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Validate
    result = validate_ohlcv_data(df)
    if result.issues:
        for issue in result.issues:
            logger.info(f"OHLCV {ticker}: {issue.message}")

    logger.info(f"Loaded OHLCV for {ticker}: {len(df)} days "
                f"({df['Date'].min().date()} to {df['Date'].max().date()})")
    return result.valid_df if len(result.valid_df) > 0 else df


def load_all_ohlcv(
    tickers: Optional[List[str]] = None,
    data_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV for all tickers (or specified list).

    Returns:
        Dict mapping ticker → OHLCV DataFrame.
    """
    base_dir = data_dir or (BLOOMBERG_DIR / "ohlcv")
    if not base_dir.exists():
        logger.warning(f"OHLCV directory not found: {base_dir}")
        return {}

    if tickers is None:
        tickers = [f.stem for f in base_dir.glob("*.csv")]

    result = {}
    for ticker in tickers:
        df = load_bloomberg_ohlcv(ticker, data_dir)
        if df is not None and not df.empty:
            result[ticker] = df

    logger.info(f"Loaded OHLCV for {len(result)}/{len(tickers)} tickers")
    return result


# ─────────────────────────────────────────────────────────────────────
# 2. Option Chain Loader
# ─────────────────────────────────────────────────────────────────────

def load_bloomberg_options(
    ticker: str,
    snapshot_date: Optional[str] = None,
    data_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Load and normalize Bloomberg option chain data.

    Expected Bloomberg BDS formula (for option chain):
        =BDS("AAPL US Equity","OPT_CHAIN",
             "OPTION_CHAIN_OVERRIDE=B","STRIKE_SPACING_OVERRIDE=5")

    Then for each option in the chain, use BDP for fields:
        =BDP(optionTicker,"BID,ASK,PX_LAST,OPT_IMPLIED_VOLATILITY_MID,
              OPEN_INT,VOLUME,OPT_DELTA,OPT_GAMMA,OPT_THETA,OPT_VEGA,
              OPT_STRIKE_PX,OPT_EXPIRE_DT,OPT_PUT_CALL,OPT_UNDL_PX")

    Saved as: data/bloomberg/options/AAPL.csv
    or dated: data/bloomberg/options/2025-01-15_AAPL.csv

    Args:
        ticker: Stock ticker.
        snapshot_date: Optional date string (YYYY-MM-DD) for dated files.
        data_dir: Override data directory.

    Returns:
        Normalized DataFrame with option chain data.
    """
    base_dir = data_dir or (BLOOMBERG_DIR / "options")

    # Try dated file first, then undated
    if snapshot_date:
        filepath = base_dir / f"{snapshot_date}_{ticker}.csv"
        if not filepath.exists():
            filepath = base_dir / f"{ticker}.csv"
    else:
        filepath = base_dir / f"{ticker}.csv"

    if not filepath.exists():
        # Try finding any dated file for this ticker
        candidates = sorted(base_dir.glob(f"*_{ticker}.csv"), reverse=True)
        if candidates:
            filepath = candidates[0]
        else:
            logger.warning(f"Options file not found for {ticker}")
            return None

    try:
        skip = _detect_header_rows(str(filepath))
        df = pd.read_csv(filepath, skiprows=skip)
    except Exception as e:
        logger.error(f"Error reading options for {ticker}: {e}")
        return None

    if df.empty:
        return None

    # Rename columns
    df = _rename_columns(df, OPTIONS_COLUMN_MAP)

    # Add ticker if not present
    if "ticker" not in df.columns:
        df["ticker"] = ticker

    # Parse expiration dates
    if "expiration" in df.columns:
        df["expiration"] = _parse_date_column(df["expiration"])

    # Normalize option_type: Bloomberg uses "Call"/"Put" or "C"/"P"
    if "option_type" in df.columns:
        df["option_type"] = (
            df["option_type"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str[0]
        )

    # Ensure numeric columns
    numeric_cols = [
        "strike", "bid", "ask", "last", "implied_vol",
        "open_interest", "volume", "delta", "gamma", "theta",
        "vega", "underlying_price"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize IV (Bloomberg may give as percentage or decimal)
    if "implied_vol" in df.columns:
        normalized_ivs = []
        for iv in df["implied_vol"]:
            norm_iv, _ = validate_and_normalize_iv(iv)
            normalized_ivs.append(norm_iv)
        df["implied_vol"] = normalized_ivs

    # Fill missing fields
    if "bid" not in df.columns:
        df["bid"] = 0.0
    if "ask" not in df.columns:
        df["ask"] = 0.0
    if "open_interest" not in df.columns:
        df["open_interest"] = 0
    if "volume" not in df.columns:
        df["volume"] = 0

    # Compute mid price
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

    # Add snapshot date
    if snapshot_date:
        df["date"] = snapshot_date
    elif "date" not in df.columns:
        # Try to extract from filename
        fname = filepath.stem
        if len(fname) >= 10 and fname[4] == "-":
            df["date"] = fname[:10]
        else:
            df["date"] = datetime.now().strftime("%Y-%m-%d")

    df = df.dropna(subset=["strike"])

    logger.info(f"Loaded options for {ticker}: {len(df)} contracts")
    return df


def load_all_options(
    tickers: Optional[List[str]] = None,
    snapshot_date: Optional[str] = None,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load option chains for all tickers and concatenate.

    Returns:
        Single DataFrame with all option chains.
    """
    base_dir = data_dir or (BLOOMBERG_DIR / "options")
    if not base_dir.exists():
        logger.warning(f"Options directory not found: {base_dir}")
        return pd.DataFrame()

    if tickers is None:
        # Extract tickers from filenames
        tickers = set()
        for f in base_dir.glob("*.csv"):
            name = f.stem
            # Handle dated files: 2025-01-15_AAPL -> AAPL
            if len(name) > 10 and name[4] == "-" and name[7] == "-":
                tickers.add(name[11:])
            else:
                tickers.add(name)
        tickers = list(tickers)

    frames = []
    for ticker in tickers:
        df = load_bloomberg_options(ticker, snapshot_date, data_dir)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────
# 3. Earnings Loader
# ─────────────────────────────────────────────────────────────────────

def load_bloomberg_earnings(
    ticker: str,
    data_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Load earnings history for a single ticker.

    Expected Bloomberg BDH formula:
        =BDH("AAPL US Equity","IS_EPS,BEST_EPS_MEDIAN,
              EARN_EST_EPS_SURPRISE,EARN_EST_EPS_SURPRISE_PCT",
              "20190101","20260101","Dir=V","Per=Q")

    For earnings dates:
        =BDS("AAPL US Equity","ERN_ANN_DT_AND_PER",
             "EARN_ANN_DT_TIME_HIST_WITH_EPS=Y")

    Saved as: data/bloomberg/earnings/AAPL.csv

    Expected columns after normalization:
        earnings_date, eps_actual, eps_estimate, eps_surprise,
        surprise_pct, timing (BMO/AMC)

    Args:
        ticker: Stock ticker.
        data_dir: Override data directory.

    Returns:
        DataFrame with earnings history or None.
    """
    base_dir = data_dir or (BLOOMBERG_DIR / "earnings")
    filepath = base_dir / f"{ticker}.csv"

    if not filepath.exists():
        return None

    try:
        skip = _detect_header_rows(str(filepath))
        df = pd.read_csv(filepath, skiprows=skip)
    except Exception as e:
        logger.error(f"Error reading earnings for {ticker}: {e}")
        return None

    if df.empty:
        return None

    df = _rename_columns(df, EARNINGS_COLUMN_MAP)

    # Parse dates
    if "earnings_date" in df.columns:
        df["earnings_date"] = _parse_date_column(df["earnings_date"])
        df = df.dropna(subset=["earnings_date"])
        df = df.sort_values("earnings_date").reset_index(drop=True)

    # Numeric columns
    for col in ["eps_actual", "eps_estimate", "eps_surprise",
                "surprise_pct", "close_price", "iv_pre_earnings"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute surprise if missing
    if ("eps_surprise" not in df.columns and
            "eps_actual" in df.columns and "eps_estimate" in df.columns):
        df["eps_surprise"] = df["eps_actual"] - df["eps_estimate"]

    if ("surprise_pct" not in df.columns and
            "eps_surprise" in df.columns and "eps_estimate" in df.columns):
        df["surprise_pct"] = (
            df["eps_surprise"] / df["eps_estimate"].replace(0, np.nan) * 100
        )

    # Normalize timing (BMO = Before Market Open, AMC = After Market Close)
    if "timing" in df.columns:
        df["timing"] = (
            df["timing"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        df["is_pre_market"] = df["timing"].isin(["BMO", "BF-MKT", "BEFORE"])
    else:
        df["is_pre_market"] = True  # Default assumption

    df["ticker"] = ticker
    logger.info(f"Loaded earnings for {ticker}: {len(df)} quarters")
    return df


def load_all_earnings(
    tickers: Optional[List[str]] = None,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load earnings for all tickers."""
    base_dir = data_dir or (BLOOMBERG_DIR / "earnings")
    if not base_dir.exists():
        return pd.DataFrame()

    if tickers is None:
        tickers = [f.stem for f in base_dir.glob("*.csv")]

    frames = []
    for ticker in tickers:
        df = load_bloomberg_earnings(ticker, data_dir)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def compute_earnings_features(
    ticker: str,
    earnings_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    iv_df: Optional[pd.DataFrame] = None,
    target_date: Optional[date] = None
) -> Optional[dict]:
    """
    Compute EarningsFeatures dict from raw Bloomberg data.

    Bridges between Bloomberg CSV data and ml/earnings_model.py input.

    Args:
        ticker: Stock ticker.
        earnings_df: Earnings history for this ticker.
        ohlcv_df: OHLCV history for this ticker.
        iv_df: IV history for this ticker (optional).
        target_date: Next earnings date to predict for (optional).

    Returns:
        Dict compatible with EarningsFeatures dataclass, or None.
    """
    if earnings_df.empty or len(earnings_df) < 4:
        return None

    # Sort by date
    edf = earnings_df.sort_values("earnings_date").copy()

    # Historical move calculation: absolute % change around earnings
    moves = []
    for _, row in edf.iterrows():
        edate = pd.to_datetime(row["earnings_date"])
        # Find price day before and day after
        before = ohlcv_df[ohlcv_df["Date"] < edate]
        after = ohlcv_df[ohlcv_df["Date"] > edate]

        if before.empty or after.empty:
            continue

        pre_price = before.iloc[-1]["Close"]
        post_price = after.iloc[0]["Close"]
        move = abs(post_price / pre_price - 1)
        moves.append(move)

    if len(moves) < 2:
        return None

    # Last 8 quarters (or fewer)
    recent_moves = moves[-8:]

    # IV metrics
    iv_rank = 0.5
    iv_30d = 0.25
    iv_7d = 0.25
    if iv_df is not None and "iv_atm_30d" in iv_df.columns:
        iv_series = iv_df["iv_atm_30d"].dropna()
        if len(iv_series) > 20:
            current_iv = iv_series.iloc[-1]
            iv_30d = current_iv
            iv_min = iv_series.min()
            iv_max = iv_series.max()
            iv_rank = (current_iv - iv_min) / (iv_max - iv_min) if iv_max > iv_min else 0.5

    # Surprise stats
    if "surprise_pct" in edf.columns:
        surprise_series = edf["surprise_pct"].dropna()
        beat_rate = float((surprise_series > 0).mean()) if len(surprise_series) > 0 else 0.5
        avg_surprise = float(surprise_series.abs().mean()) if len(surprise_series) > 0 else 5.0
    else:
        beat_rate = 0.5
        avg_surprise = 5.0

    # Timing
    is_pre = True
    if "is_pre_market" in edf.columns:
        is_pre = bool(edf.iloc[-1]["is_pre_market"])

    # Determine next earnings date
    if target_date is None:
        target_date = edf.iloc[-1]["earnings_date"]
    days_to = max(0, (pd.to_datetime(target_date) - pd.Timestamp.now()).days)

    return {
        "symbol": ticker,
        "earnings_date": target_date,
        "implied_move": np.mean(recent_moves),  # Proxy until live straddle
        "historical_avg_move": float(np.mean(recent_moves)),
        "historical_max_move": float(np.max(recent_moves)),
        "implied_vs_realized_ratio": 1.0,  # Requires live straddle pricing
        "iv_rank_52w": iv_rank,
        "iv_30d": iv_30d,
        "iv_7d": iv_7d,
        "iv_term_slope": (iv_30d - iv_7d) / iv_7d if iv_7d > 0 else 0,
        "avg_iv_crush_pct": float(np.mean(recent_moves)) * 0.6,  # Rough proxy
        "earnings_beat_rate": beat_rate,
        "avg_surprise_magnitude": avg_surprise,
        "vix_level": 20.0,  # Will be filled from rates/VIX data
        "vix_percentile": 0.5,
        "sector_iv_rank": iv_rank,
        "days_to_earnings": days_to,
        "is_pre_market": is_pre,
    }


# ─────────────────────────────────────────────────────────────────────
# 4. Dividend Loader
# ─────────────────────────────────────────────────────────────────────

def load_bloomberg_dividends(
    ticker: str,
    data_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Load dividend history for a single ticker.

    Expected Bloomberg BDS formula:
        =BDS("AAPL US Equity","DVD_HIST_ALL",
             "DVD_START_DT=20190101","DVD_END_DT=20260101")

    Or BDH for yield:
        =BDH("AAPL US Equity","EQY_DVD_YLD_IND,DVD_SH_LAST",
              "20190101","20260101","Dir=V","Per=Q")

    Saved as: data/bloomberg/dividends/AAPL.csv

    Returns:
        DataFrame with columns: ex_date, amount, [record_date, pay_date,
        frequency, dividend_yield]
    """
    base_dir = data_dir or (BLOOMBERG_DIR / "dividends")
    filepath = base_dir / f"{ticker}.csv"

    if not filepath.exists():
        return None

    try:
        skip = _detect_header_rows(str(filepath))
        df = pd.read_csv(filepath, skiprows=skip)
    except Exception as e:
        logger.error(f"Error reading dividends for {ticker}: {e}")
        return None

    if df.empty:
        return None

    df = _rename_columns(df, DIVIDEND_COLUMN_MAP)

    # Parse dates
    for date_col in ["ex_date", "record_date", "pay_date"]:
        if date_col in df.columns:
            df[date_col] = _parse_date_column(df[date_col])

    # Ensure ex_date exists
    if "ex_date" not in df.columns:
        # Try to find a date column
        for col in df.columns:
            if "date" in col.lower() or "dt" in col.lower():
                df["ex_date"] = _parse_date_column(df[col])
                break
        else:
            logger.error(f"No date column found in dividends for {ticker}")
            return None

    df = df.dropna(subset=["ex_date"])

    # Numeric amount
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if "dividend_yield" in df.columns:
        df["dividend_yield"] = pd.to_numeric(df["dividend_yield"], errors="coerce")

    df["ticker"] = ticker
    df = df.sort_values("ex_date").reset_index(drop=True)

    logger.info(f"Loaded dividends for {ticker}: {len(df)} payments")
    return df


def load_all_dividends(
    tickers: Optional[List[str]] = None,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load dividends for all tickers."""
    base_dir = data_dir or (BLOOMBERG_DIR / "dividends")
    if not base_dir.exists():
        return pd.DataFrame()

    if tickers is None:
        tickers = [f.stem for f in base_dir.glob("*.csv")]

    frames = []
    for ticker in tickers:
        df = load_bloomberg_dividends(ticker, data_dir)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def get_annual_dividend_yield(
    ticker: str,
    div_df: pd.DataFrame,
    spot_price: float,
    lookback_months: int = 12
) -> float:
    """
    Compute annualized dividend yield from historical payments.

    Used to feed continuous dividend yield q into BSM pricing.

    Args:
        ticker: Stock ticker.
        div_df: Dividend DataFrame for this ticker.
        spot_price: Current stock price.
        lookback_months: How far back to sum dividends.

    Returns:
        Annualized dividend yield (decimal, e.g., 0.015 for 1.5%).
    """
    if div_df.empty or spot_price <= 0:
        return 0.0

    cutoff = pd.Timestamp.now() - pd.DateOffset(months=lookback_months)
    recent = div_df[div_df["ex_date"] >= cutoff]

    if recent.empty or "amount" not in recent.columns:
        return 0.0

    total_div = recent["amount"].sum()
    annualized = total_div * (12 / lookback_months)
    return float(annualized / spot_price)


def get_upcoming_dividends(
    div_df: pd.DataFrame,
    horizon_days: int = 60
) -> pd.DataFrame:
    """
    Get upcoming ex-dividend dates within horizon.

    Used by LSM pricer to identify early exercise risk for calls.
    """
    if div_df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now()
    cutoff = today + pd.Timedelta(days=horizon_days)

    upcoming = div_df[
        (div_df["ex_date"] >= today) &
        (div_df["ex_date"] <= cutoff)
    ].copy()

    return upcoming


# ─────────────────────────────────────────────────────────────────────
# 5. IV History Loader
# ─────────────────────────────────────────────────────────────────────

def load_bloomberg_iv_history(
    ticker: str,
    data_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Load historical implied volatility data.

    Expected Bloomberg BDH formula:
        =BDH("AAPL US Equity",
              "30DAY_IMPVOL_100.0%MNY_DF,60DAY_IMPVOL_100.0%MNY_DF,
               30DAY_IMPVOL_90.0%MNY_DF,30DAY_IMPVOL_110.0%MNY_DF,
               20DAY_HV,60DAY_HV",
              "20190101","20260101","Dir=V")

    Saved as: data/bloomberg/iv_history/AAPL.csv

    Returns:
        DataFrame with columns: date, iv_atm_30d, iv_atm_60d,
        iv_25d_put, iv_25d_call, rv_20d, rv_60d
    """
    base_dir = data_dir or (BLOOMBERG_DIR / "iv_history")
    filepath = base_dir / f"{ticker}.csv"

    if not filepath.exists():
        return None

    try:
        skip = _detect_header_rows(str(filepath))
        df = pd.read_csv(filepath, skiprows=skip)
    except Exception as e:
        logger.error(f"Error reading IV history for {ticker}: {e}")
        return None

    if df.empty:
        return None

    df = _rename_columns(df, IV_HISTORY_COLUMN_MAP)

    # Parse dates
    if "date" in df.columns:
        df["date"] = _parse_date_column(df["date"])
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Normalize IV values (Bloomberg gives as percentage, e.g., 25.4 for 25.4%)
    iv_cols = [c for c in df.columns if c.startswith("iv_") or c.startswith("rv_")]
    for col in iv_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Bloomberg typically gives IV as percentage (25.4 not 0.254)
        if col in df.columns and df[col].median() > 1.0:
            df[col] = df[col] / 100.0

    df["ticker"] = ticker
    logger.info(f"Loaded IV history for {ticker}: {len(df)} days")
    return df


def load_all_iv_history(
    tickers: Optional[List[str]] = None,
    data_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """Load IV history for all tickers. Returns dict: ticker → DataFrame."""
    base_dir = data_dir or (BLOOMBERG_DIR / "iv_history")
    if not base_dir.exists():
        return {}

    if tickers is None:
        tickers = [f.stem for f in base_dir.glob("*.csv")]

    result = {}
    for ticker in tickers:
        df = load_bloomberg_iv_history(ticker, data_dir)
        if df is not None and not df.empty:
            result[ticker] = df

    logger.info(f"Loaded IV history for {len(result)}/{len(tickers)} tickers")
    return result


def compute_iv_rank(
    iv_df: pd.DataFrame,
    lookback_days: int = 252
) -> Optional[float]:
    """
    Compute IV rank (percentile) from IV history.

    IV Rank = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV)

    Args:
        iv_df: IV history DataFrame with 'iv_atm_30d' column.
        lookback_days: Number of trading days for lookback.

    Returns:
        IV rank [0, 1] or None if insufficient data.
    """
    if iv_df is None or iv_df.empty:
        return None

    col = "iv_atm_30d"
    if col not in iv_df.columns:
        # Try alternative columns
        for alt in ["iv_atm_60d", "iv_put", "iv_call"]:
            if alt in iv_df.columns:
                col = alt
                break
        else:
            return None

    series = iv_df[col].dropna().tail(lookback_days)
    if len(series) < 20:
        return None

    current = series.iloc[-1]
    low = series.min()
    high = series.max()

    if high <= low:
        return 0.5

    return float((current - low) / (high - low))


# ─────────────────────────────────────────────────────────────────────
# 6. Risk-Free Rate Loader
# ─────────────────────────────────────────────────────────────────────

def load_bloomberg_rates(
    data_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Load Treasury yield curve history.

    Expected Bloomberg BDH formula:
        =BDH("USGG3M Index,USGG6M Index,USGG2YR Index,USGG10YR Index",
              "PX_LAST","20190101","20260101","Dir=V")

    Or simpler single-series:
        =BDH("USGG3M Index","PX_LAST","20190101","20260101","Dir=V")

    Saved as: data/bloomberg/rates/treasury_yields.csv

    Returns:
        DataFrame with columns: date, rate_3m, rate_6m, rate_2y, rate_10y
    """
    base_dir = data_dir or (BLOOMBERG_DIR / "rates")

    # Try standard filename
    filepath = base_dir / "treasury_yields.csv"
    if not filepath.exists():
        # Try any CSV in the rates directory
        candidates = list(base_dir.glob("*.csv"))
        if not candidates:
            return None
        filepath = candidates[0]

    try:
        skip = _detect_header_rows(str(filepath))
        df = pd.read_csv(filepath, skiprows=skip)
    except Exception as e:
        logger.error(f"Error reading rates: {e}")
        return None

    if df.empty:
        return None

    # Rename date column
    df = _rename_columns(df, RATES_COLUMN_MAP)

    if "date" in df.columns:
        df["date"] = _parse_date_column(df["date"])
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Rename rate columns based on common Bloomberg names
    rate_renames = {
        "USGG3M Index": "rate_3m",
        "USGG6M Index": "rate_6m",
        "USGG2YR Index": "rate_2y",
        "USGG10YR Index": "rate_10y",
        "3M": "rate_3m",
        "6M": "rate_6m",
        "2Y": "rate_2y",
        "10Y": "rate_10y",
    }
    df = df.rename(columns={
        c: rate_renames[c] for c in df.columns if c in rate_renames
    })

    # If single yield column, name it
    if "yield" in df.columns and "rate_3m" not in df.columns:
        df["rate_3m"] = df["yield"]

    # Convert from percentage to decimal
    for col in df.columns:
        if col.startswith("rate_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].median() > 1.0:
                df[col] = df[col] / 100.0

    logger.info(f"Loaded rates: {len(df)} days")
    return df


def get_current_risk_free_rate(
    rates_df: Optional[pd.DataFrame] = None,
    tenor: str = "rate_3m"
) -> float:
    """
    Get the most recent risk-free rate.

    Falls back to config default (0.05) if no data.

    Args:
        rates_df: Rates DataFrame.
        tenor: Which tenor to use (rate_3m, rate_6m, rate_2y, rate_10y).

    Returns:
        Risk-free rate as decimal (e.g., 0.0525).
    """
    if rates_df is None or rates_df.empty:
        return 0.05  # Default

    if tenor not in rates_df.columns:
        # Try any rate column
        rate_cols = [c for c in rates_df.columns if c.startswith("rate_")]
        if not rate_cols:
            return 0.05
        tenor = rate_cols[0]

    latest = rates_df[tenor].dropna()
    if latest.empty:
        return 0.05

    return float(latest.iloc[-1])


# ─────────────────────────────────────────────────────────────────────
# 7. Fundamentals Loader
# ─────────────────────────────────────────────────────────────────────

def load_bloomberg_fundamentals(
    data_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Load company fundamentals (sector, market cap, etc.).

    Expected Bloomberg BDP formula (batch for all SP500):
        =BDP("AAPL US Equity,MSFT US Equity,...",
             "CUR_MKT_CAP,GICS_SECTOR_NAME,GICS_INDUSTRY_GROUP_NAME,
              EQY_DVD_YLD_IND,PE_RATIO,TRAIL_12M_EPS")

    Saved as: data/bloomberg/fundamentals/sp500_fundamentals.csv

    Returns:
        DataFrame with columns: ticker, market_cap, gics_sector,
        gics_industry, dividend_yield, pe_ratio, eps_ttm
    """
    base_dir = data_dir or (BLOOMBERG_DIR / "fundamentals")

    filepath = base_dir / "sp500_fundamentals.csv"
    if not filepath.exists():
        candidates = list(base_dir.glob("*.csv"))
        if not candidates:
            return None
        filepath = candidates[0]

    try:
        skip = _detect_header_rows(str(filepath))
        df = pd.read_csv(filepath, skiprows=skip)
    except Exception as e:
        logger.error(f"Error reading fundamentals: {e}")
        return None

    if df.empty:
        return None

    # Rename common Bloomberg columns
    fund_map = {
        "Security": "ticker",
        "Ticker": "ticker",
        "ticker": "ticker",
        "CUR_MKT_CAP": "market_cap",
        "Market_Cap": "market_cap",
        "GICS_SECTOR_NAME": "gics_sector",
        "GICS_Sector": "gics_sector",
        "Sector": "gics_sector",
        "GICS_INDUSTRY_GROUP_NAME": "gics_industry",
        "Industry": "gics_industry",
        "EQY_DVD_YLD_IND": "dividend_yield",
        "Div_Yield": "dividend_yield",
        "PE_RATIO": "pe_ratio",
        "PE": "pe_ratio",
        "TRAIL_12M_EPS": "eps_ttm",
        "EPS": "eps_ttm",
    }
    df = _rename_columns(df, fund_map)

    # Clean ticker (remove " US Equity" suffix if present)
    if "ticker" in df.columns:
        df["ticker"] = (
            df["ticker"]
            .astype(str)
            .str.replace(" US Equity", "", regex=False)
            .str.strip()
        )

    # Numeric columns
    for col in ["market_cap", "dividend_yield", "pe_ratio", "eps_ttm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dividend yield: convert percentage to decimal
    if "dividend_yield" in df.columns and df["dividend_yield"].median() > 1.0:
        df["dividend_yield"] = df["dividend_yield"] / 100.0

    logger.info(f"Loaded fundamentals for {len(df)} companies")
    return df


def build_sector_map(fundamentals_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build ticker → GICS sector mapping from Bloomberg fundamentals.

    Replaces the hardcoded DEFAULT_SECTOR_MAP in risk_manager.py.
    """
    if fundamentals_df is None or fundamentals_df.empty:
        return {}

    if "ticker" not in fundamentals_df.columns or "gics_sector" not in fundamentals_df.columns:
        return {}

    sector_map = {}
    for _, row in fundamentals_df.iterrows():
        ticker = str(row["ticker"]).strip()
        sector = str(row["gics_sector"]).strip()
        if ticker and sector and sector != "nan":
            sector_map[ticker] = sector

    return sector_map
