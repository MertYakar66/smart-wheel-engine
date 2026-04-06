"""
News Scrapers

Fetches financial news from various sources without APIs.
Uses browser automation and RSS parsing.

Sources:
- Bloomberg, Reuters, CNBC, WSJ (via RSS/scraping)
- Fed, SEC, Treasury (official RSS feeds)
- Finviz, MarketWatch (headlines)

Note: BrowserNewsScraper and NewsAggregator require playwright.
"""

# Core types (no optional dependency)
from news_pipeline.scrapers.base import (
    NewsCategory,
    NewsItem,
    NewsScraper,
    SourceType,
)

__all__ = [
    "NewsScraper",
    "NewsItem",
    "NewsCategory",
    "SourceType",
    "RSSNewsScraper",
    "BrowserNewsScraper",
    "NewsAggregator",
]


def __getattr__(name: str):
    """Lazy import scrapers that require optional dependencies (aiohttp, playwright)."""
    if name == "RSSNewsScraper":
        from news_pipeline.scrapers.rss_scraper import RSSNewsScraper

        return RSSNewsScraper
    if name == "BrowserNewsScraper":
        from news_pipeline.scrapers.browser_scraper import BrowserNewsScraper

        return BrowserNewsScraper
    if name == "NewsAggregator":
        from news_pipeline.scrapers.aggregator import NewsAggregator

        return NewsAggregator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
