"""
Data Module - Bloomberg Data Ingestion and Pipeline

Provides two complementary interfaces to Bloomberg:

1. CSV Loaders (bloomberg_loader.py):
   - Load historical data from exported CSV files
   - Used for backtesting and analysis
   - Data extracted via Bloomberg Excel formulas

2. Live Connector (bloomberg.py):
   - Real-time data via API or Excel COM
   - Used for live trading and monitoring
   - Requires Bloomberg Terminal connection

Data categories:
- OHLCV price history
- Option chains with Greeks
- Earnings dates and surprises
- Dividend schedule and yields
- IV history (for IV rank)
- Treasury yield curve
- Company fundamentals (GICS sectors)
"""

# CSV-based loaders for historical data
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

# Live data connector
from .bloomberg import (
    BloombergConnector,
    BloombergError,
    StockQuote,
    OptionQuote,
    # Quick access functions
    get_price,
    get_iv,
    download_ohlcv,
    # Batch operations
    refresh_ohlcv,
    refresh_option_chain,
    get_live_quotes,
    get_live_iv_rank,
    # Screening
    get_wheel_candidates,
    # Diagnostics
    check_bloomberg_available,
    test_connection,
)

# Bloomberg CSV import and feature processing
from .bloomberg_import import (
    load_bloomberg_csv,
    compute_features_per_ticker,
    process_bloomberg_data,
)

# Master data pipeline
from .pipeline import DataPipeline, DataStatus

# Feature engineering pipeline
from .feature_pipeline import (
    FeaturePipeline,
    ComputeResult,
    PipelineResult,
    compute_features,
)

# Feature store
from .feature_store import (
    FeatureStore,
    FeatureCategory,
    FeatureMetadata,
    get_feature_store,
)

# Data quality framework
from .quality import (
    DataQualityFramework,
    ValidationResult,
    DataContract,
    validate_ohlcv,
    validate_options,
)

# Pipeline orchestrator
from .orchestrator import (
    PipelineOrchestrator,
    PipelineRun,
    StageType,
    TaskStatus,
    run_pipeline,
)

# Observability
from .observability import (
    metrics,
    trace,
    logger,
    setup_logging,
    timed,
    traced,
)
