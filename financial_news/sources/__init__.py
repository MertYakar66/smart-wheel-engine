"""
Data Source Connectors for Financial News Platform

Supports:
- GDELT (global news monitoring)
- SEC EDGAR (US regulatory filings)
- RSS feeds (official sources)
"""

from .base import BaseSourceFetcher
from .gdelt import GDELTFetcher
from .rss_feeds import RSSFetcher
from .sec_edgar import SECEdgarFetcher

__all__ = [
    "BaseSourceFetcher",
    "GDELTFetcher",
    "SECEdgarFetcher",
    "RSSFetcher",
]
