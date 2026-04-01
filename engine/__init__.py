"""
Smart Wheel Engine - Core Trading Components

This module provides the core trading engine components:
- WheelTracker: Position lifecycle management
- Option pricing with Black-Scholes and Greeks
- Transaction cost modeling
- Shared valuation for consistent labeling
- Performance metrics calculation
- Risk management and position sizing
- Market regime detection
- Event calendar for earnings/dividends/FOMC
- Stress testing and scenario analysis
- Signal generation framework
- Monte Carlo simulation (bootstrap, jump diffusion, LSM)
"""

from .event_calendar import (
    EventCalendar,
    EventCalendarBuilder,
    EventImpact,
    EventRiskFilter,
    EventType,
    MarketEvent,
    build_default_calendar,
)
from .monte_carlo import (
    BlockBootstrap,
    BootstrapResult,
    JumpDiffusionParams,
    JumpDiffusionResult,
    JumpDiffusionSimulator,
    LSMPricer,
    LSMResult,
    price_american_option,
    run_bagholder_analysis,
    run_bootstrap_analysis,
)
from .option_pricer import (
    black_scholes_all_greeks,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_rho,
    black_scholes_theta,
    black_scholes_vega,
    estimate_option_price_from_iv,
    vectorized_bs_all_greeks,
    vectorized_bs_delta,
    vectorized_bs_price,
)
from .performance_metrics import (
    PerformanceReport,
    calculate_performance_report,
    generate_monthly_returns,
    generate_trade_analysis,
)
from .regime_detector import (
    RegimeDetector,
    RegimeState,
    TrendRegime,
    VolatilityRegime,
    VolTermStructure,
    calculate_regime_signals,
)
from .risk_manager import (
    DEFAULT_SECTOR_MAP,
    HierarchicalRiskParity,
    PortfolioGreeks,
    PositionSizingMethod,
    RiskLimits,
    RiskManager,
    RiskMetrics,
    SectorExposure,
    SectorExposureManager,
    calculate_hrp_weights,
    calculate_kelly_fraction,
    calculate_optimal_contracts,
    optimize_position_weights,
)
from .shared_valuation import TradeOutcome, simulate_option_trade, simulate_wheel_cycle
from .signal_context import (
    build_batch_entry_contexts,
    build_entry_context,
    build_exit_context,
    evaluate_wheel_opportunities,
)
from .signals import (
    CompositeSignal,
    DTESignal,
    EventFilterSignal,
    IVRankSignal,
    ProfitTargetSignal,
    Signal,
    SignalAggregator,
    SignalStrength,
    SignalType,
    StopLossSignal,
    TrendSignal,
    create_default_aggregator,
)
from .stress_testing import (
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SCENARIOS,
    Scenario,
    ScenarioResult,
    ScenarioType,
    StressTester,
    StressTestReport,
    calculate_max_loss,
    quick_stress_test,
)
from .transaction_costs import (
    calculate_actual_spread,
    calculate_assignment_costs,
    calculate_assignment_fee,
    calculate_commission,
    calculate_reg_t_margin_short_put,
    calculate_slippage,
    calculate_total_entry_cost,
    calculate_total_exit_cost,
    estimate_round_trip_cost,
)
from .volatility_surface import (
    SplineVolSurface,
    SVICalibrator,
    SVIParams,
    VolatilitySurface,
    VolatilitySurfaceBuilder,
    create_constant_surface,
    estimate_iv_for_delta,
    surface_to_dataframe,
)
from .wheel_tracker import PositionState, WheelPosition, WheelTracker

__all__ = [
    # Core
    'WheelTracker', 'WheelPosition', 'PositionState',

    # Pricing
    'black_scholes_price', 'black_scholes_delta', 'black_scholes_gamma',
    'black_scholes_theta', 'black_scholes_vega', 'black_scholes_rho',
    'black_scholes_all_greeks', 'estimate_option_price_from_iv',
    'vectorized_bs_price', 'vectorized_bs_delta', 'vectorized_bs_all_greeks',

    # Costs
    'calculate_commission', 'calculate_slippage', 'calculate_assignment_fee',
    'calculate_total_entry_cost', 'calculate_total_exit_cost',
    'calculate_assignment_costs', 'calculate_reg_t_margin_short_put',
    'calculate_actual_spread', 'estimate_round_trip_cost',

    # Valuation
    'simulate_option_trade', 'simulate_wheel_cycle', 'TradeOutcome',

    # Performance
    'calculate_performance_report', 'generate_trade_analysis',
    'generate_monthly_returns', 'PerformanceReport',

    # Risk
    'RiskManager', 'RiskLimits', 'RiskMetrics', 'PortfolioGreeks',
    'PositionSizingMethod', 'calculate_kelly_fraction', 'calculate_optimal_contracts',
    'SectorExposureManager', 'SectorExposure', 'HierarchicalRiskParity',
    'calculate_hrp_weights', 'optimize_position_weights', 'DEFAULT_SECTOR_MAP',

    # Volatility Surface
    'VolatilitySurface', 'VolatilitySurfaceBuilder', 'SVIParams', 'SVICalibrator',
    'SplineVolSurface', 'create_constant_surface', 'estimate_iv_for_delta',
    'surface_to_dataframe',

    # Regime
    'RegimeDetector', 'RegimeState', 'VolatilityRegime', 'TrendRegime',
    'VolTermStructure', 'calculate_regime_signals',

    # Events
    'EventCalendar', 'EventCalendarBuilder', 'EventRiskFilter',
    'MarketEvent', 'EventType', 'EventImpact', 'build_default_calendar',

    # Stress Testing
    'StressTester', 'Scenario', 'ScenarioResult', 'StressTestReport',
    'ScenarioType', 'quick_stress_test', 'calculate_max_loss',
    'HISTORICAL_SCENARIOS', 'HYPOTHETICAL_SCENARIOS',

    # Signals
    'SignalAggregator', 'Signal', 'CompositeSignal', 'SignalType',
    'SignalStrength', 'IVRankSignal', 'TrendSignal', 'ProfitTargetSignal',
    'StopLossSignal', 'DTESignal', 'EventFilterSignal', 'create_default_aggregator',

    # Signal Context (Bloomberg integration)
    'build_entry_context', 'build_exit_context',
    'build_batch_entry_contexts', 'evaluate_wheel_opportunities',

    # Monte Carlo
    'BlockBootstrap', 'BootstrapResult',
    'JumpDiffusionSimulator', 'JumpDiffusionParams', 'JumpDiffusionResult',
    'LSMPricer', 'LSMResult',
    'run_bootstrap_analysis', 'run_bagholder_analysis', 'price_american_option'
]
