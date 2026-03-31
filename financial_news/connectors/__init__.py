"""
Official Source Connectors for Macro + SP500 Event Intelligence System

Connectors for Tier 1 official sources:
- FedConnector: Federal Reserve (FOMC statements, minutes, speeches)
- SECEdgarConnector: SEC EDGAR filings (8-K, 10-Q, 10-K) with rate limiting
- EIAConnector: Energy Information Administration (Weekly Petroleum)
- BLSConnector: Bureau of Labor Statistics (CPI, Employment)
- BEAConnector: Bureau of Economic Analysis (GDP, PCE)

All connectors inherit from BaseConnector which provides:
- Rate limiting with token bucket
- Exponential backoff retry
- Request logging
- Error handling
"""

from .base import BaseConnector, RateLimiter
from .fed import FedConnector
from .sec_edgar import SECEdgarConnector
from .eia import EIAConnector

__all__ = [
    "BaseConnector",
    "RateLimiter",
    "FedConnector",
    "SECEdgarConnector",
    "EIAConnector",
]
