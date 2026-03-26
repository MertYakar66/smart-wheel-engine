"""
Smart Wheel Engine Dashboard Module

Provides professional quantitative trading interfaces for:
- Option pricing (European & American)
- Greeks analysis (1st, 2nd, 3rd order)
- Portfolio risk management (VaR, CVaR)
- Position sizing (Kelly criterion)
- Stress testing

Quick Start:
    from dashboard import QuantDashboard, OptionInput, Position

    # Create dashboard
    dash = QuantDashboard()

    # Price an option
    opt = OptionInput(spot=150, strike=145, dte=30, volatility=0.28)
    result = dash.price_european(opt)

    # Run interactive CLI
    dash.run()

For quick calculations without dashboard:
    from dashboard import quick_price, quick_greeks

    price = quick_price(150, 145, 30, volatility=0.28)
    greeks = quick_greeks(150, 145, 30, volatility=0.28)
"""

from .quant_dashboard import (
    QuantDashboard,
    OptionInput,
    Position,
    PortfolioInput,
    quick_price,
    quick_greeks,
)

__all__ = [
    'QuantDashboard',
    'OptionInput',
    'Position',
    'PortfolioInput',
    'quick_price',
    'quick_greeks',
]

__version__ = '2.0.0'
