"""
Base News Scraper

Defines common interface and data structures for news scrapers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SourceType(Enum):
    """Type of news source."""

    OFFICIAL = "official"  # Fed, SEC, company releases
    MAINSTREAM = "mainstream"  # Bloomberg, Reuters, CNBC
    AGGREGATOR = "aggregator"  # Finviz, MarketWatch
    SOCIAL = "social"  # Twitter/X, Reddit
    RSS = "rss"
    UNKNOWN = "unknown"


class NewsCategory(Enum):
    """News categories aligned with trading relevance."""

    FED = "fed"
    EARNINGS = "earnings"
    SP500 = "sp500_events"
    OIL = "oil"
    GEOPOLITICS = "geopolitics"
    MACRO = "macro"
    CRYPTO = "crypto"
    TECH = "tech"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    BREAKING = "breaking"
    OTHER = "other"


@dataclass
class NewsItem:
    """
    A scraped news item.

    Represents raw news before verification.
    """

    headline: str
    source_name: str
    source_url: str
    source_type: SourceType = SourceType.UNKNOWN

    # Optional fields
    snippet: str = ""
    published_at: datetime | None = None
    scraped_at: datetime = field(default_factory=datetime.utcnow)

    # Extracted metadata
    tickers: list[str] = field(default_factory=list)
    category: NewsCategory = NewsCategory.OTHER

    # Quality indicators
    credibility_score: float = 0.5  # 0-1 scale

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "source_type": self.source_type.value,
            "snippet": self.snippet,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "scraped_at": self.scraped_at.isoformat(),
            "tickers": self.tickers,
            "category": self.category.value,
            "credibility_score": self.credibility_score,
        }


class NewsScraper(ABC):
    """
    Abstract base class for news scrapers.

    Each scraper implements fetching from a specific source type.
    """

    SOURCE_NAME: str = "Unknown"
    SOURCE_TYPE: SourceType = SourceType.UNKNOWN
    CREDIBILITY: float = 0.5

    @abstractmethod
    async def fetch(
        self,
        categories: list[str] | None = None,
        tickers: list[str] | None = None,
        max_items: int = 50,
        hours_back: int = 12,
    ) -> list[NewsItem]:
        """
        Fetch news items from the source.

        Args:
            categories: Filter by categories
            tickers: Filter by ticker symbols
            max_items: Maximum items to return
            hours_back: How far back to look

        Returns:
            List of NewsItem objects
        """
        pass

    def categorize_headline(self, headline: str) -> NewsCategory:
        """Categorize a headline based on keywords."""
        headline_lower = headline.lower()

        # Fed / Monetary policy
        if any(
            kw in headline_lower
            for kw in [
                "fed",
                "fomc",
                "powell",
                "rate cut",
                "rate hike",
                "interest rate",
                "monetary policy",
                "inflation",
            ]
        ):
            return NewsCategory.FED

        # Earnings
        if any(
            kw in headline_lower
            for kw in [
                "earnings",
                "quarterly",
                "revenue",
                "profit",
                "eps",
                "guidance",
                "beat",
                "miss",
                "outlook",
            ]
        ):
            return NewsCategory.EARNINGS

        # Oil / Energy
        if any(
            kw in headline_lower
            for kw in ["oil", "crude", "opec", "energy", "natural gas", "petroleum", "brent", "wti"]
        ):
            return NewsCategory.OIL

        # Geopolitics
        if any(
            kw in headline_lower
            for kw in [
                "china",
                "russia",
                "ukraine",
                "war",
                "sanctions",
                "trade war",
                "tariff",
                "geopolit",
                "conflict",
            ]
        ):
            return NewsCategory.GEOPOLITICS

        # Macro
        if any(
            kw in headline_lower
            for kw in [
                "gdp",
                "employment",
                "jobs",
                "unemployment",
                "cpi",
                "pce",
                "retail sales",
                "housing",
                "consumer",
            ]
        ):
            return NewsCategory.MACRO

        # Crypto
        if any(
            kw in headline_lower for kw in ["bitcoin", "crypto", "ethereum", "btc", "blockchain"]
        ):
            return NewsCategory.CRYPTO

        return NewsCategory.OTHER

    def extract_tickers(self, text: str) -> list[str]:
        """Extract stock ticker symbols from text."""
        import re

        # Common ticker pattern: 1-5 uppercase letters
        # Exclude common words that look like tickers
        exclude = {
            "CEO",
            "CFO",
            "IPO",
            "ETF",
            "NYSE",
            "SEC",
            "FED",
            "GDP",
            "CPI",
            "PPI",
            "PMI",
            "ISM",
            "USA",
            "FDA",
            "THE",
            "AND",
            "FOR",
            "ARE",
            "BUT",
            "NOT",
            "YOU",
        }

        pattern = r"\b([A-Z]{1,5})\b"
        matches = re.findall(pattern, text)

        # Filter and deduplicate
        tickers = []
        for match in matches:
            if match not in exclude and match not in tickers:
                # Basic validation: common ticker patterns
                if len(match) >= 2 or match in ["X", "V", "F", "T"]:
                    tickers.append(match)

        return tickers[:10]  # Limit to 10 tickers
