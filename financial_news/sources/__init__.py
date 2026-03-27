"""
Data Source Connectors for Financial News Platform

Supports:
- GDELT (global news monitoring)
- SEC EDGAR (US regulatory filings)
- RSS feeds (official sources)
"""

from .gdelt import GDELTFetcher
from .sec_edgar import SECEdgarFetcher
from .rss_feeds import RSSFetcher
from .base import BaseSourceFetcher

__all__ = [
    "BaseSourceFetcher",
    "GDELTFetcher",
    "SECEdgarFetcher",
    "RSSFetcher",
]
