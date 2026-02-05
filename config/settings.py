"""
Configuration Management System

Centralized configuration for the Wheel Strategy Engine:
- Strategy parameters
- Risk limits
- Execution settings
- Data paths
- Environment-specific overrides

Configuration hierarchy:
1. Default values (this file)
2. Environment variables
3. Config file overrides (YAML/JSON)
4. Runtime overrides
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import os
import json
import yaml
from enum import Enum


class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    BACKTEST = "backtest"
    PAPER = "paper"
    PRODUCTION = "production"


@dataclass
class StrategyConfig:
    """Core strategy parameters."""
    # Target selection
    target_delta: float = 0.30          # Target put delta (0.20-0.40)
    min_delta: float = 0.15             # Minimum acceptable delta
    max_delta: float = 0.45             # Maximum acceptable delta

    target_dte_min: int = 25            # Minimum days to expiration
    target_dte_max: int = 45            # Maximum days to expiration
    target_dte_ideal: int = 35          # Ideal DTE for new positions

    # Exit rules
    profit_target_pct: float = 0.50     # Close at 50% profit
    stop_loss_pct: float = 2.00         # Stop at 200% loss (2x premium)
    dte_exit_threshold: int = 5         # Close positions at 5 DTE

    # Position management
    max_positions: int = 10             # Maximum concurrent positions
    max_single_allocation: float = 0.15 # Max 15% per position
    min_premium_collected: float = 0.50 # Minimum $0.50 premium

    # Covered call settings
    covered_call_delta: float = 0.30    # Target call delta
    covered_call_dte_min: int = 20
    covered_call_dte_max: int = 45

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if not 0.10 <= self.target_delta <= 0.50:
            errors.append(f"target_delta {self.target_delta} out of range [0.10, 0.50]")
        if self.min_delta >= self.max_delta:
            errors.append("min_delta must be less than max_delta")
        if self.target_dte_min >= self.target_dte_max:
            errors.append("target_dte_min must be less than target_dte_max")
        if not 0 < self.profit_target_pct < 1:
            errors.append(f"profit_target_pct {self.profit_target_pct} should be between 0 and 1")
        if not 1 < self.stop_loss_pct < 10:
            errors.append(f"stop_loss_pct {self.stop_loss_pct} should be between 1 and 10")
        return errors


@dataclass
class RiskConfig:
    """Risk management parameters."""
    # Position limits
    max_portfolio_delta: float = 0.50   # Max net delta as % of portfolio
    max_portfolio_gamma: float = 0.10   # Max gamma exposure
    max_portfolio_vega: float = 0.20    # Max vega exposure

    # Loss limits
    max_daily_loss_pct: float = 0.03    # 3% daily loss limit
    max_weekly_loss_pct: float = 0.07   # 7% weekly loss limit
    max_drawdown_pct: float = 0.20      # 20% max drawdown

    # VaR limits
    var_confidence: float = 0.95
    max_var_pct: float = 0.05           # Max 5% daily VaR

    # Concentration
    max_sector_exposure: float = 0.40   # Max 40% in single sector
    max_correlation_cluster: float = 0.35  # Max in correlated group

    # Margin safety
    margin_utilization_target: float = 0.60  # Target 60% margin use
    margin_utilization_max: float = 0.80     # Max 80% margin use

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if not 0 < self.max_drawdown_pct < 1:
            errors.append("max_drawdown_pct must be between 0 and 1")
        if self.margin_utilization_target >= self.margin_utilization_max:
            errors.append("margin_utilization_target must be less than max")
        return errors


@dataclass
class ExecutionConfig:
    """Trade execution parameters."""
    # Slippage assumptions
    default_slippage_pct: float = 0.02  # 2% of mid price
    max_spread_pct: float = 0.20        # Reject if spread > 20%

    # Commission structure
    commission_per_contract: float = 0.65
    assignment_fee: float = 0.00
    exercise_fee: float = 0.00

    # Liquidity filters
    min_open_interest: int = 100
    min_bid_price: float = 0.05
    min_volume: int = 10

    # Order types
    use_limit_orders: bool = True
    limit_offset_pct: float = 0.01      # 1% better than mid

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if self.default_slippage_pct < 0:
            errors.append("slippage cannot be negative")
        if self.commission_per_contract < 0:
            errors.append("commission cannot be negative")
        return errors


@dataclass
class DataConfig:
    """Data paths and settings."""
    # Directories
    data_dir: str = "data"
    ohlcv_dir: str = "data/ohlcv"
    options_dir: str = "data/options"
    output_dir: str = "output"

    # File patterns
    ohlcv_pattern: str = "{symbol}_ohlcv.parquet"
    options_pattern: str = "{symbol}_options.parquet"

    # Data quality
    max_missing_days: int = 5           # Max consecutive missing days
    min_history_days: int = 252         # Minimum 1 year history

    # Symbols
    universe_file: str = "data/universe.csv"
    default_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "JPM", "BAC", "WFC", "GS", "MS",
        "XOM", "CVX", "COP", "SLB", "EOG"
    ])

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if self.min_history_days < 30:
            errors.append("min_history_days should be at least 30")
        return errors


@dataclass
class BacktestConfig:
    """Backtesting parameters."""
    # Capital
    initial_capital: float = 100000.0
    allow_margin: bool = True

    # Period
    start_date: Optional[str] = None    # YYYY-MM-DD or None for all
    end_date: Optional[str] = None

    # Validation
    use_walk_forward: bool = True
    walk_forward_folds: int = 5
    embargo_days: int = 5

    # Simulation
    use_realistic_fills: bool = True
    random_seed: Optional[int] = 42

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if self.initial_capital < 10000:
            errors.append("initial_capital should be at least $10,000")
        return errors


@dataclass
class Config:
    """Master configuration container."""
    environment: Environment = Environment.DEVELOPMENT
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Market assumptions
    risk_free_rate: float = 0.05        # 5% risk-free rate
    default_dividend_yield: float = 0.02 # 2% default dividend

    def validate(self) -> List[str]:
        """Validate all configuration sections."""
        errors = []
        errors.extend(self.strategy.validate())
        errors.extend(self.risk.validate())
        errors.extend(self.execution.validate())
        errors.extend(self.data.validate())
        errors.extend(self.backtest.validate())
        return errors

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'environment': self.environment.value,
            'strategy': asdict(self.strategy),
            'risk': asdict(self.risk),
            'execution': asdict(self.execution),
            'data': asdict(self.data),
            'backtest': asdict(self.backtest),
            'risk_free_rate': self.risk_free_rate,
            'default_dividend_yield': self.default_dividend_yield
        }

    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        data = self.to_dict()
        path = Path(filepath)

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from file."""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(filepath) as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath) as f:
                data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Config':
        """Create config from dictionary."""
        config = cls()

        if 'environment' in data:
            config.environment = Environment(data['environment'])

        if 'strategy' in data:
            for key, value in data['strategy'].items():
                if hasattr(config.strategy, key):
                    setattr(config.strategy, key, value)

        if 'risk' in data:
            for key, value in data['risk'].items():
                if hasattr(config.risk, key):
                    setattr(config.risk, key, value)

        if 'execution' in data:
            for key, value in data['execution'].items():
                if hasattr(config.execution, key):
                    setattr(config.execution, key, value)

        if 'data' in data:
            for key, value in data['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        if 'backtest' in data:
            for key, value in data['backtest'].items():
                if hasattr(config.backtest, key):
                    setattr(config.backtest, key, value)

        if 'risk_free_rate' in data:
            config.risk_free_rate = data['risk_free_rate']
        if 'default_dividend_yield' in data:
            config.default_dividend_yield = data['default_dividend_yield']

        return config

    @classmethod
    def from_environment(cls) -> 'Config':
        """Create config from environment variables."""
        config = cls()

        # Environment
        env = os.getenv('WHEEL_ENV', 'development')
        config.environment = Environment(env)

        # Strategy overrides
        if os.getenv('WHEEL_TARGET_DELTA'):
            config.strategy.target_delta = float(os.getenv('WHEEL_TARGET_DELTA'))
        if os.getenv('WHEEL_MAX_POSITIONS'):
            config.strategy.max_positions = int(os.getenv('WHEEL_MAX_POSITIONS'))

        # Risk overrides
        if os.getenv('WHEEL_MAX_DRAWDOWN'):
            config.risk.max_drawdown_pct = float(os.getenv('WHEEL_MAX_DRAWDOWN'))

        # Data overrides
        if os.getenv('WHEEL_DATA_DIR'):
            config.data.data_dir = os.getenv('WHEEL_DATA_DIR')

        # Backtest overrides
        if os.getenv('WHEEL_INITIAL_CAPITAL'):
            config.backtest.initial_capital = float(os.getenv('WHEEL_INITIAL_CAPITAL'))

        return config


class ConfigManager:
    """
    Configuration management with layered overrides.

    Hierarchy: defaults < config file < environment < runtime
    """

    _instance: Optional['ConfigManager'] = None
    _config: Optional[Config] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_config(cls) -> Config:
        """Get current configuration."""
        if cls._config is None:
            cls._config = Config()
        return cls._config

    @classmethod
    def load_config(
        cls,
        config_file: Optional[str] = None,
        use_env: bool = True
    ) -> Config:
        """
        Load configuration with all overrides.

        Args:
            config_file: Path to config file (YAML or JSON)
            use_env: Whether to apply environment variable overrides
        """
        # Start with defaults
        config = Config()

        # Load from file if provided
        if config_file and Path(config_file).exists():
            config = Config.load(config_file)

        # Apply environment overrides
        if use_env:
            env_config = Config.from_environment()
            # Merge environment overrides
            if os.getenv('WHEEL_ENV'):
                config.environment = env_config.environment
            if os.getenv('WHEEL_TARGET_DELTA'):
                config.strategy.target_delta = env_config.strategy.target_delta

        # Validate
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {errors}")

        cls._config = config
        return config

    @classmethod
    def update(cls, **kwargs) -> Config:
        """Update configuration at runtime."""
        config = cls.get_config()

        for key, value in kwargs.items():
            if '.' in key:
                # Nested key like 'strategy.target_delta'
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            elif hasattr(config, key):
                setattr(config, key, value)

        # Re-validate
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration errors after update: {errors}")

        return config


# Preset configurations
def get_conservative_config() -> Config:
    """Conservative configuration for risk-averse trading."""
    config = Config()
    config.strategy.target_delta = 0.20
    config.strategy.profit_target_pct = 0.40
    config.strategy.max_positions = 5
    config.risk.max_drawdown_pct = 0.10
    config.risk.margin_utilization_max = 0.50
    return config


def get_aggressive_config() -> Config:
    """Aggressive configuration for higher returns."""
    config = Config()
    config.strategy.target_delta = 0.40
    config.strategy.profit_target_pct = 0.60
    config.strategy.max_positions = 15
    config.risk.max_drawdown_pct = 0.30
    config.risk.margin_utilization_max = 0.90
    return config


def get_backtest_config(
    start_date: str,
    end_date: str,
    initial_capital: float = 100000
) -> Config:
    """Configuration for backtesting."""
    config = Config()
    config.environment = Environment.BACKTEST
    config.backtest.start_date = start_date
    config.backtest.end_date = end_date
    config.backtest.initial_capital = initial_capital
    config.backtest.use_walk_forward = True
    return config
