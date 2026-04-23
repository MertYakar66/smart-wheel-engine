"""
External free-data adapters for the smart-wheel engine.

Each adapter wraps a public feed and returns pandas objects with stable
column names. All adapters have identical fallback semantics: any
network error returns an empty DataFrame/dict so upstream callers can
degrade gracefully without crashing.

Adapters
--------
- fred_adapter:  FRED economic series (Fed Funds, CPI, yield curve, HY OAS)
- cboe_adapter:  VIX-family and SKEW/MOVE indices via CBOE/Yahoo
- yfinance_adapter: DXY, oil, gold, BTC and any Yahoo-accessible ticker
- edgar_adapter: SEC EDGAR Form 4 (insider trades), short-interest
"""

from .fred_adapter import FREDAdapter
from .cboe_adapter import CBOEAdapter
from .yfinance_adapter import YFinanceAdapter
from .edgar_adapter import EDGARAdapter

__all__ = ["FREDAdapter", "CBOEAdapter", "YFinanceAdapter", "EDGARAdapter"]
