"""
News Scrapers

Fetches financial news from various sources without APIs.
Uses browser automation and RSS parsing.

Sources:
- Bloomberg, Reuters, CNBC, WSJ (via RSS/scraping)
- Fed, SEC, Treasury (official RSS feeds)
- Finviz, MarketWatch (headlines)
"""

from news_pipeline.scrapers.aggregator import NewsAggregator
from news_pipeline.scrapers.base import NewsItem, NewsScraper
from news_pipeline.scrapers.browser_scraper import BrowserNewsScraper
from news_pipeline.scrapers.rss_scraper import RSSNewsScraper

__all__ = [
    "NewsScraper",
    "NewsItem",
    "RSSNewsScraper",
    "BrowserNewsScraper",
    "NewsAggregator",
]
