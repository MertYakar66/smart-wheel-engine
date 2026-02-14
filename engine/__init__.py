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
"""

from .wheel_tracker import WheelTracker, WheelPosition, PositionState
from .option_pricer import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_theta,
    black_scholes_vega,
    black_scholes_rho,
    black_scholes_all_greeks,
    estimate_option_price_from_iv,
    vectorized_bs_price,
    vectorized_bs_delta,
    vectorized_bs_all_greeks
)
from .transaction_costs import (
    calculate_commission,
    calculate_slippage,
    calculate_assignment_fee,
    calculate_total_entry_cost,
    calculate_total_exit_cost,
    calculate_assignment_costs,
    calculate_reg_t_margin_short_put,
    calculate_actual_spread,
    estimate_round_trip_cost
)
from .shared_valuation import (
    simulate_option_trade,
    simulate_wheel_cycle,
    TradeOutcome,
    calculate_actual_spread as valuation_spread
)
from .performance_metrics import (
    calculate_performance_report,
    generate_trade_analysis,
    generate_monthly_returns,
    PerformanceReport
)
from .risk_manager import (
    RiskManager,
    RiskLimits,
    RiskMetrics,
    PortfolioGreeks,
    PositionSizingMethod,
    calculate_kelly_fraction,
    calculate_optimal_contracts,
    SectorExposureManager,
    SectorExposure,
    HierarchicalRiskParity,
    calculate_hrp_weights,
    optimize_position_weights,
    DEFAULT_SECTOR_MAP
)
from .volatility_surface import (
    VolatilitySurface,
    VolatilitySurfaceBuilder,
    SVIParams,
    SVICalibrator,
    SplineVolSurface,
    create_constant_surface,
    estimate_iv_for_delta,
    surface_to_dataframe
)
from .regime_detector import (
    RegimeDetector,
    RegimeState,
    VolatilityRegime,
    TrendRegime,
    VolTermStructure,
    calculate_regime_signals
)
from .event_calendar import (
    EventCalendar,
    EventCalendarBuilder,
    EventRiskFilter,
    MarketEvent,
    EventType,
    EventImpact,
    build_default_calendar
)
from .stress_testing import (
    StressTester,
    Scenario,
    ScenarioResult,
    StressTestReport,
    ScenarioType,
    quick_stress_test,
    calculate_max_loss,
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SCENARIOS
)
from .signals import (
    SignalAggregator,
    Signal,
    CompositeSignal,
    SignalType,
    SignalStrength,
    IVRankSignal,
    TrendSignal,
    ProfitTargetSignal,
    StopLossSignal,
    DTESignal,
    EventFilterSignal,
    create_default_aggregator
)

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
    'StopLossSignal', 'DTESignal', 'EventFilterSignal', 'create_default_aggregator'
]
