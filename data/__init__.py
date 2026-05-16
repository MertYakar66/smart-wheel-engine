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
# Live data connector
from .bloomberg import (
    BloombergConnector,
    BloombergError,
    OptionQuote,
    StockQuote,
    # Diagnostics
    check_bloomberg_available,
    download_ohlcv,
    get_iv,
    get_live_iv_rank,
    get_live_quotes,
    # Quick access functions
    get_price,
    # Screening
    get_wheel_candidates,
    # Batch operations
    refresh_ohlcv,
    refresh_option_chain,
    test_connection,
)

# Bloomberg CSV import and feature processing
from .bloomberg_import import (
    compute_features_per_ticker,
    load_bloomberg_csv,
    process_bloomberg_data,
)
from .bloomberg_loader import (
    # Constants
    BLOOMBERG_DIR,
    build_sector_map,
    compute_earnings_features,
    compute_iv_rank,
    get_annual_dividend_yield,
    get_current_risk_free_rate,
    get_upcoming_dividends,
    load_all_dividends,
    load_all_earnings,
    load_all_iv_history,
    load_all_ohlcv,
    load_all_options,
    # Dividends
    load_bloomberg_dividends,
    # Earnings
    load_bloomberg_earnings,
    # Fundamentals
    load_bloomberg_fundamentals,
    # IV History
    load_bloomberg_iv_history,
    # OHLCV
    load_bloomberg_ohlcv,
    # Options
    load_bloomberg_options,
    # Rates
    load_bloomberg_rates,
)

# Consolidated Bloomberg loader
from .consolidated_loader import (
    ConsolidatedBloombergLoader,
    get_bloomberg_loader,
    normalize_ticker,
)

# Feature engineering pipeline
from .feature_pipeline import (
    ComputeResult,
    FeaturePipeline,
    PipelineResult,
    compute_features,
)

# Feature store
from .feature_store import (
    FeatureCategory,
    FeatureMetadata,
    FeatureStore,
    get_feature_store,
)

# Observability
from .observability import (
    logger,
    metrics,
    setup_logging,
    timed,
    trace,
    traced,
)

# Pipeline orchestrator
from .orchestrator import (
    PipelineOrchestrator,
    PipelineRun,
    StageType,
    TaskStatus,
    run_pipeline,
)

# Master data pipeline
from .pipeline import DataPipeline, DataStatus

# Data quality framework
from .quality import (
    DataContract,
    DataQualityFramework,
    ValidationResult,
    validate_ohlcv,
    validate_options,
)
