# Smart Wheel Engine - System Architecture

## Overview

The Smart Wheel Engine is a professional-grade algorithmic trading system designed for optimal execution of the wheel strategy (cash-secured puts + covered calls) on S&P 500 constituents.

---

## System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SMART WHEEL ENGINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │    DATA     │   │   FEATURE   │   │    MODEL    │   │   SIGNAL    │     │
│  │  INGESTION  │──▶│   ENGINE    │──▶│   LAYER     │──▶│  GENERATOR  │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│         │                                                      │            │
│         │                                                      ▼            │
│         │          ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│         │          │    RISK     │◀──│  PORTFOLIO  │◀──│   BACKTEST  │     │
│         │          │   MANAGER   │   │  OPTIMIZER  │   │   ENGINE    │     │
│         │          └─────────────┘   └─────────────┘   └─────────────┘     │
│         │                 │                                                 │
│         │                 ▼                                                 │
│         │          ┌─────────────┐   ┌─────────────┐                       │
│         └─────────▶│  EXECUTION  │──▶│  REPORTING  │                       │
│                    │   ENGINE    │   │  DASHBOARD  │                       │
│                    └─────────────┘   └─────────────┘                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. Data Ingestion (`src/data/`)

**Purpose:** Fetch, validate, and store market data

```
src/data/
├── __init__.py
├── bloomberg_client.py      # Bloomberg API wrapper
├── ingest_ohlcv.py          # Price data ingestion
├── ingest_options.py        # Options data ingestion
├── ingest_fundamentals.py   # Fundamental data ingestion
├── ingest_macro.py          # Macro/index data ingestion
├── validators.py            # Data quality checks
├── parquet_writer.py        # Partitioned parquet output
└── scheduler.py             # Daily refresh orchestration
```

**Key Classes:**

```python
class DataIngestion:
    """Main orchestrator for data ingestion pipeline."""

    def ingest_daily(self, date: date) -> IngestResult:
        """Run full daily ingestion pipeline."""

    def validate(self, data: pd.DataFrame, schema: Schema) -> ValidationResult:
        """Validate data against schema."""

    def backfill(self, start: date, end: date, datasets: list[str]) -> None:
        """Backfill historical data."""
```

---

### 2. Feature Engine (`src/features/`)

**Purpose:** Transform raw data into ML-ready features

```
src/features/
├── __init__.py
├── technical.py             # Price-based features
├── volatility.py            # Realized vol, IV features
├── options.py               # Options-specific features
├── fundamental.py           # Fundamental features
├── sentiment.py             # News/sentiment features
├── macro.py                 # Macro regime features
├── composite.py             # Combined feature sets
├── feature_store.py         # Feature persistence
└── feature_registry.py      # Feature metadata
```

**Key Feature Groups:**

```python
# Technical Features
TECHNICAL_FEATURES = [
    "return_1d", "return_5d", "return_21d",
    "ma_ratio_20", "ma_ratio_50", "ma_ratio_200",
    "rsi_14", "macd_signal", "bollinger_position",
    "atr_14", "volume_ratio_20",
]

# Volatility Features
VOLATILITY_FEATURES = [
    "rv_21d", "rv_63d",
    "rv_parkinson_21d", "rv_garman_klass_21d",
    "iv_30d", "iv_60d", "iv_90d",
    "iv_rv_spread", "iv_rank", "iv_percentile",
    "iv_term_slope", "iv_skew",
]

# Options Flow Features
OPTIONS_FEATURES = [
    "put_call_volume_ratio", "put_call_oi_ratio",
    "oi_change_call", "oi_change_put",
    "unusual_volume_flag", "iv_crush_expected",
]

# Fundamental Features
FUNDAMENTAL_FEATURES = [
    "pe_zscore", "pb_zscore", "ev_ebitda_zscore",
    "roe_rank", "margin_stability",
    "earnings_surprise_ma", "guidance_trend",
]

# Macro/Regime Features
REGIME_FEATURES = [
    "vix_level", "vix_term_structure",
    "yield_curve_slope", "credit_spread",
    "spx_momentum", "sector_rotation_score",
]
```

---

### 3. Model Layer (`src/models/`)

**Purpose:** ML models for signal generation

```
src/models/
├── __init__.py
├── base.py                  # Base model interface
├── entry_model.py           # CSP entry timing
├── strike_model.py          # Strike selection
├── exit_model.py            # Position exit timing
├── regime_model.py          # Market regime classifier
├── ensemble.py              # Model combination
├── training.py              # Training pipeline
├── evaluation.py            # Model evaluation
└── registry.py              # Model versioning
```

**Model Specifications:**

```python
class EntryModel(BaseModel):
    """
    Predicts optimal entry points for cash-secured puts.

    Target: Forward N-day return > premium collected
    Features: Volatility, flow, technical, fundamental
    Algorithm: LightGBM with Bayesian hyperparameter tuning
    """

class StrikeModel(BaseModel):
    """
    Selects optimal strike price for CSP/CC.

    Target: Maximize risk-adjusted premium
    Features: IV surface, delta, support levels
    Algorithm: Quantile regression
    """

class RegimeModel(BaseModel):
    """
    Classifies market regime (bull/bear/sideways/crisis).

    Target: Regime label
    Features: Macro, VIX, breadth, momentum
    Algorithm: Hidden Markov Model + Random Forest
    """
```

---

### 4. Signal Generator (`src/signals/`)

**Purpose:** Convert model outputs to actionable signals

```
src/signals/
├── __init__.py
├── screener.py              # Stock screening
├── ranker.py                # Opportunity ranking
├── signal_generator.py      # Trade signal creation
├── filters.py               # Signal filters
└── output.py                # Signal formatting
```

**Signal Schema:**

```python
@dataclass
class TradeSignal:
    timestamp: datetime
    ticker: str
    signal_type: Literal["CSP_ENTRY", "CSP_EXIT", "CC_ENTRY", "CC_EXIT"]
    direction: Literal["OPEN", "CLOSE", "ROLL"]

    # Option details
    strike: float
    expiration: date
    delta: float
    premium: float

    # Model confidence
    entry_score: float
    regime_score: float
    composite_score: float

    # Risk metrics
    max_loss: float
    probability_profit: float
    expected_return: float

    # Sizing
    position_size: int
    capital_required: float
    portfolio_weight: float
```

---

### 5. Risk Manager (`src/risk/`)

**Purpose:** Position sizing, exposure limits, risk monitoring

```
src/risk/
├── __init__.py
├── position_sizer.py        # Kelly/optimal f sizing
├── exposure_monitor.py      # Sector/factor exposure
├── correlation_risk.py      # Correlation-based limits
├── drawdown_control.py      # Drawdown management
├── margin_calculator.py     # Margin requirements
├── var_calculator.py        # Value at Risk
└── stress_test.py           # Scenario analysis
```

**Risk Parameters:**

```python
class RiskConfig:
    # Position limits
    max_position_pct: float = 0.05          # Max 5% per position
    max_sector_pct: float = 0.25            # Max 25% per sector
    max_correlated_pct: float = 0.30        # Max 30% highly correlated

    # Drawdown controls
    max_portfolio_drawdown: float = 0.15    # 15% portfolio drawdown limit
    max_position_drawdown: float = 0.20     # 20% position drawdown limit

    # Margin safety
    margin_buffer: float = 0.30             # 30% margin buffer

    # Volatility scaling
    target_volatility: float = 0.12         # 12% target annual vol
    vol_scaling_enabled: bool = True

    # Regime adjustments
    crisis_mode_scaling: float = 0.50       # 50% size in crisis
    high_vix_threshold: float = 30.0
```

---

### 6. Portfolio Optimizer (`src/portfolio/`)

**Purpose:** Optimal portfolio construction

```
src/portfolio/
├── __init__.py
├── optimizer.py             # Mean-variance optimization
├── constraints.py           # Portfolio constraints
├── rebalancer.py            # Rebalancing logic
├── tax_optimizer.py         # Tax-loss harvesting
└── cash_management.py       # Cash allocation
```

---

### 7. Backtest Engine (`src/backtest/`)

**Purpose:** Historical strategy simulation

```
src/backtest/
├── __init__.py
├── engine.py                # Core backtest loop
├── data_handler.py          # Point-in-time data
├── execution_sim.py         # Execution simulation
├── slippage.py              # Slippage models
├── commission.py            # Commission models
├── metrics.py               # Performance metrics
├── report_generator.py      # Backtest reports
└── walk_forward.py          # Walk-forward validation
```

**Backtest Requirements:**

1. **Point-in-time accuracy** - No lookahead bias
2. **Survivorship-bias free** - Use historical membership
3. **Realistic execution** - Slippage, commissions, market impact
4. **Options-specific** - Early assignment, dividend risk

---

### 8. Execution Engine (`src/execution/`)

**Purpose:** Order management and execution

```
src/execution/
├── __init__.py
├── order_manager.py         # Order lifecycle
├── broker_interface.py      # Broker API (IBKR)
├── smart_router.py          # Order routing
├── fill_tracker.py          # Fill monitoring
└── execution_analytics.py   # Execution quality
```

---

### 9. Reporting Dashboard (`src/reporting/`)

**Purpose:** Performance monitoring and visualization

```
src/reporting/
├── __init__.py
├── performance.py           # Performance analytics
├── risk_report.py           # Risk dashboard
├── attribution.py           # Return attribution
├── visualizations.py        # Charts/graphs
└── alerts.py                # Alert system
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Data Storage | Parquet + DuckDB |
| ML Framework | LightGBM, scikit-learn, PyTorch |
| Backtesting | Custom (vectorized) |
| Visualization | Plotly, Streamlit |
| Orchestration | Prefect |
| Monitoring | Prometheus + Grafana |
| Broker API | Interactive Brokers TWS |

---

## Data Flow

```
┌─────────────┐
│  Bloomberg  │
│  Terminal   │
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐
│    Raw      │───▶│  Processed  │
│   Parquet   │    │   Parquet   │
└─────────────┘    └──────┬──────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ Technical │   │ Options   │   │   Macro   │
    │ Features  │   │ Features  │   │ Features  │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                   ┌─────────────┐
                   │   Feature   │
                   │    Store    │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │    Model    │
                   │  Inference  │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Signals   │
                   └──────┬──────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │   Risk    │   │ Portfolio │   │ Execution │
    │  Manager  │   │ Optimizer │   │  Engine   │
    └───────────┘   └───────────┘   └───────────┘
```

---

## Development Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Project setup and configuration
- [ ] Data ingestion pipeline
- [ ] Basic feature engineering
- [ ] Data validation framework

### Phase 2: Core Models (Weeks 3-4)
- [ ] Entry model development
- [ ] Strike selection model
- [ ] Regime classification
- [ ] Model evaluation framework

### Phase 3: Backtest Engine (Weeks 5-6)
- [ ] Core backtest loop
- [ ] Options-specific simulation
- [ ] Performance metrics
- [ ] Walk-forward validation

### Phase 4: Risk & Portfolio (Weeks 7-8)
- [ ] Risk management system
- [ ] Portfolio optimization
- [ ] Position sizing
- [ ] Exposure monitoring

### Phase 5: Execution & Monitoring (Weeks 9-10)
- [ ] Broker integration
- [ ] Order management
- [ ] Performance dashboard
- [ ] Alerting system

### Phase 6: Production (Weeks 11-12)
- [ ] Paper trading
- [ ] System hardening
- [ ] Documentation
- [ ] Go-live

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Annual Return | 15-25% |
| Sharpe Ratio | > 1.5 |
| Max Drawdown | < 15% |
| Win Rate | > 70% |
| Average Win/Loss | > 2.0 |
| Correlation to SPX | < 0.5 |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-19 | Initial architecture |
