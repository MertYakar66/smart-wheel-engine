"""
Smart Wheel Engine - Core Trading Components

This module provides the core trading engine components:
- WheelTracker: Position lifecycle management
- Option pricing with Black-Scholes and Greeks
- Transaction cost modeling
- Shared valuation for consistent labeling
- Performance metrics calculation
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
