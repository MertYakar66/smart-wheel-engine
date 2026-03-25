"""
Import and process Bloomberg OHLCV data for wheel strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.technical import TechnicalFeatures
from src.features.volatility import VolatilityFeatures


def load_bloomberg_csv(filepath: str) -> pd.DataFrame:
    """
    Load Bloomberg OHLCV export.

    Expected columns: date, ticker, open, high, low, close, volume
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.columns = df.columns.str.lower().str.strip()

    # Ensure proper column names
    required = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean ticker names (remove " Equity" suffix if present)
    df['ticker'] = df['ticker'].str.replace(' Equity', '', regex=False)
    df['ticker'] = df['ticker'].str.replace(' UN', '', regex=False)
    df['ticker'] = df['ticker'].str.replace(' UW', '', regex=False)
    df['ticker'] = df['ticker'].str.strip()

    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    return df


def compute_features_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical and volatility features for a single ticker.
    """
    tech = TechnicalFeatures()
    vol = VolatilityFeatures()

    # Technical features
    result = tech.compute_all(df)

    # Volatility features
    log_returns = np.log(result['close'] / result['close'].shift(1))
    result['realized_vol_20'] = vol.realized_volatility_close(log_returns, window=20)
    result['realized_vol_60'] = vol.realized_volatility_close(log_returns, window=60)
    result['parkinson_vol_20'] = vol.realized_volatility_parkinson(
        result['high'], result['low'], window=20
    )
    result['vol_ratio'] = result['realized_vol_20'] / result['realized_vol_60']

    # IV rank proxy (using realized vol percentile as proxy until we have IV data)
    result['rv_rank_252'] = result['realized_vol_20'].rolling(252).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )

    # Trend features
    result['trend_20d'] = result['close'].pct_change(20)
    result['trend_60d'] = result['close'].pct_change(60)
    result['above_sma_20'] = (result['close'] > result['close'].rolling(20).mean()).astype(int)
    result['above_sma_50'] = (result['close'] > result['close'].rolling(50).mean()).astype(int)
    result['above_sma_200'] = (result['close'] > result['close'].rolling(200).mean()).astype(int)

    # Drawdown from 52-week high
    rolling_max = result['close'].rolling(252).max()
    result['drawdown_52w'] = (result['close'] - rolling_max) / rolling_max

    return result


def process_bloomberg_data(
    input_path: str,
    output_path: Optional[str] = None,
    min_history_days: int = 252
) -> pd.DataFrame:
    """
    Full pipeline: load Bloomberg data, compute features, save.

    Args:
        input_path: Path to Bloomberg CSV
        output_path: Optional output path for processed data
        min_history_days: Minimum trading days required per ticker

    Returns:
        Processed DataFrame with all features
    """
    print(f"Loading data from {input_path}...")
    raw_df = load_bloomberg_csv(input_path)
    print(f"  Loaded {len(raw_df):,} rows, {raw_df['ticker'].nunique()} tickers")

    # Process each ticker
    processed_dfs = []
    tickers = raw_df['ticker'].unique()

    print(f"Computing features for {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(tickers)}...")

        ticker_df = raw_df[raw_df['ticker'] == ticker].copy()

        # Skip tickers with insufficient history
        if len(ticker_df) < min_history_days:
            continue

        ticker_df = compute_features_per_ticker(ticker_df)
        processed_dfs.append(ticker_df)

    result = pd.concat(processed_dfs, ignore_index=True)

    # Drop rows with NaN features (from rolling calculations)
    feature_cols = [c for c in result.columns if c not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
    initial_rows = len(result)
    result = result.dropna(subset=feature_cols)
    print(f"  Dropped {initial_rows - len(result):,} rows with NaN features")

    print(f"Final: {len(result):,} rows, {result['ticker'].nunique()} tickers")

    if output_path:
        result.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Bloomberg CSV path")
    parser.add_argument("-o", "--output", help="Output parquet path", default="data/sp500_features.parquet")
    args = parser.parse_args()

    process_bloomberg_data(args.input, args.output)
