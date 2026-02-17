"""
Data Module - Bloomberg Data Ingestion and Pipeline

Provides loaders for all Bloomberg data categories:
- OHLCV price history
- Option chains with Greeks
- Earnings dates and surprises
- Dividend schedule and yields
- IV history (for IV rank)
- Treasury yield curve
- Company fundamentals (GICS sectors)
"""

from .bloomberg_loader import (
    # OHLCV
    load_bloomberg_ohlcv,
    load_all_ohlcv,
    # Options
    load_bloomberg_options,
    load_all_options,
    # Earnings
    load_bloomberg_earnings,
    load_all_earnings,
    compute_earnings_features,
    # Dividends
    load_bloomberg_dividends,
    load_all_dividends,
    get_annual_dividend_yield,
    get_upcoming_dividends,
    # IV History
    load_bloomberg_iv_history,
    load_all_iv_history,
    compute_iv_rank,
    # Rates
    load_bloomberg_rates,
    get_current_risk_free_rate,
    # Fundamentals
    load_bloomberg_fundamentals,
    build_sector_map,
    # Constants
    BLOOMBERG_DIR,
)
