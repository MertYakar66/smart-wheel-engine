"""
Official Source Connectors for Macro + SP500 Event Intelligence System

Tier 1 - Official Sources:
- FedConnector: Federal Reserve (FOMC statements, minutes, speeches)
- SECEdgarConnector: SEC EDGAR filings (8-K, 10-Q, 10-K) with rate limiting
- EIAConnector: Energy Information Administration (Weekly Petroleum)
- BLSConnector: Bureau of Labor Statistics (CPI, Employment)
- BEAConnector: Bureau of Economic Analysis (GDP, PCE)

Tier 3 - Discovery Sources:
- DiscoveryConnector: Google News, CNBC, Reuters, Yahoo Finance RSS
- CorroborationEngine: Validate discovery signals with Tier 1 sources

All connectors inherit from BaseConnector which provides:
- Rate limiting with token bucket
- Exponential backoff retry
- Request logging
- Error handling
"""

from .base import BaseConnector, RateLimiter
from .discovery import DISCOVERY_SOURCES, CorroborationEngine, DiscoveryConnector
from .eia import EIAConnector
from .fed import FedConnector
from .sec_edgar import SECEdgarConnector

__all__ = [
    # Base
    "BaseConnector",
    "RateLimiter",
    # Tier 1 - Official
    "FedConnector",
    "SECEdgarConnector",
    "EIAConnector",
    # Tier 3 - Discovery
    "DiscoveryConnector",
    "CorroborationEngine",
    "DISCOVERY_SOURCES",
]
