"""
RSS News Scraper

Fetches news from RSS feeds - zero cost, no rate limits.
Covers major financial news sources and official feeds.
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import aiohttp

from news_pipeline.scrapers.base import (
    NewsItem,
    NewsScraper,
    SourceType,
)

logger = logging.getLogger(__name__)


@dataclass
class RSSFeed:
    """RSS feed configuration."""

    name: str
    url: str
    source_type: SourceType
    credibility: float
    categories: list[str] | None = None


# Pre-configured financial RSS feeds
FINANCIAL_RSS_FEEDS = [
    # Official sources (highest credibility)
    RSSFeed(
        name="Federal Reserve",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
        source_type=SourceType.OFFICIAL,
        credibility=1.0,
        categories=["fed", "macro"],
    ),
    RSSFeed(
        name="SEC EDGAR Filings",
        url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=40&output=atom",
        source_type=SourceType.OFFICIAL,
        credibility=1.0,
        categories=["earnings", "sp500_events"],
    ),
    RSSFeed(
        name="Treasury Direct",
        url="https://www.treasurydirect.gov/rss/news.xml",
        source_type=SourceType.OFFICIAL,
        credibility=1.0,
        categories=["macro", "fed"],
    ),
    # Mainstream media
    RSSFeed(
        name="Reuters Business",
        url="https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
        source_type=SourceType.MAINSTREAM,
        credibility=0.9,
    ),
    RSSFeed(
        name="CNBC Top News",
        url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
        source_type=SourceType.MAINSTREAM,
        credibility=0.8,
    ),
    RSSFeed(
        name="MarketWatch",
        url="https://feeds.marketwatch.com/marketwatch/topstories/",
        source_type=SourceType.MAINSTREAM,
        credibility=0.8,
    ),
    RSSFeed(
        name="Bloomberg Markets",
        url="https://feeds.bloomberg.com/markets/news.rss",
        source_type=SourceType.MAINSTREAM,
        credibility=0.9,
    ),
    # Aggregators
    RSSFeed(
        name="Yahoo Finance",
        url="https://finance.yahoo.com/rss/topstories",
        source_type=SourceType.AGGREGATOR,
        credibility=0.7,
    ),
    RSSFeed(
        name="Seeking Alpha",
        url="https://seekingalpha.com/market_currents.xml",
        source_type=SourceType.AGGREGATOR,
        credibility=0.6,
    ),
]


class RSSNewsScraper(NewsScraper):
    """
    Scrapes news from RSS feeds.

    Zero cost, reliable, no rate limits.
    Aggregates from multiple financial RSS feeds.
    """

    SOURCE_NAME = "RSS Aggregator"
    SOURCE_TYPE = SourceType.RSS

    def __init__(self, feeds: list[RSSFeed] | None = None):
        """
        Initialize RSS scraper.

        Args:
            feeds: Custom list of feeds (uses defaults if None)
        """
        self.feeds = feeds or FINANCIAL_RSS_FEEDS

    async def fetch(
        self,
        categories: list[str] | None = None,
        tickers: list[str] | None = None,
        max_items: int = 50,
        hours_back: int = 12,
    ) -> list[NewsItem]:
        """Fetch news from all configured RSS feeds."""
        all_items = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        # Filter feeds by category if specified
        feeds_to_fetch = self.feeds
        if categories:
            feeds_to_fetch = [
                f
                for f in self.feeds
                if not f.categories or any(c in f.categories for c in categories)
            ]

        # Fetch all feeds in parallel
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_feed(session, feed, cutoff_time) for feed in feeds_to_fetch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Feed fetch failed: {result}")

        # Filter by tickers if specified
        if tickers:
            tickers_upper = [t.upper() for t in tickers]
            all_items = [
                item
                for item in all_items
                if any(t in tickers_upper for t in item.tickers)
                or any(t in item.headline.upper() for t in tickers_upper)
            ]

        # Sort by publish time (newest first) and limit
        all_items.sort(
            key=lambda x: x.published_at or x.scraped_at,
            reverse=True,
        )

        logger.info(
            f"[RSS] Fetched {len(all_items[:max_items])} items from {len(feeds_to_fetch)} feeds"
        )
        return all_items[:max_items]

    async def _fetch_feed(
        self,
        session: aiohttp.ClientSession,
        feed: RSSFeed,
        cutoff_time: datetime,
    ) -> list[NewsItem]:
        """Fetch and parse a single RSS feed."""
        items = []

        try:
            async with session.get(feed.url, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"[RSS] {feed.name}: HTTP {response.status}")
                    return []

                content = await response.text()
                items = self._parse_rss(content, feed, cutoff_time)

                logger.debug(f"[RSS] {feed.name}: {len(items)} items")

        except TimeoutError:
            logger.warning(f"[RSS] {feed.name}: Timeout")
        except Exception as e:
            logger.warning(f"[RSS] {feed.name}: {e}")

        return items

    def _parse_rss(
        self,
        content: str,
        feed: RSSFeed,
        cutoff_time: datetime,
    ) -> list[NewsItem]:
        """Parse RSS/Atom XML content."""
        items = []

        try:
            root = ET.fromstring(content)

            # Handle RSS 2.0
            for item in root.findall(".//item"):
                news_item = self._parse_rss_item(item, feed)
                if news_item and (
                    not news_item.published_at or news_item.published_at > cutoff_time
                ):
                    items.append(news_item)

            # Handle Atom
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                news_item = self._parse_atom_entry(entry, feed)
                if news_item and (
                    not news_item.published_at or news_item.published_at > cutoff_time
                ):
                    items.append(news_item)

        except ET.ParseError as e:
            logger.warning(f"[RSS] XML parse error for {feed.name}: {e}")

        return items

    def _parse_rss_item(self, item: ET.Element, feed: RSSFeed) -> NewsItem | None:
        """Parse RSS 2.0 item element."""
        title = item.findtext("title", "")
        link = item.findtext("link", "")
        description = item.findtext("description", "")
        pub_date_str = item.findtext("pubDate", "")

        if not title or not link:
            return None

        # Parse publication date
        pub_date = None
        if pub_date_str:
            pub_date = self._parse_date(pub_date_str)

        return NewsItem(
            headline=title.strip(),
            source_name=feed.name,
            source_url=link.strip(),
            source_type=feed.source_type,
            snippet=description[:500] if description else "",
            published_at=pub_date,
            category=self.categorize_headline(title),
            tickers=self.extract_tickers(f"{title} {description}"),
            credibility_score=feed.credibility,
        )

    def _parse_atom_entry(self, entry: ET.Element, feed: RSSFeed) -> NewsItem | None:
        """Parse Atom entry element."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        title = entry.findtext("atom:title", "", ns)
        link_elem = entry.find("atom:link", ns)
        link = link_elem.get("href", "") if link_elem is not None else ""
        summary = entry.findtext("atom:summary", "", ns)
        updated = entry.findtext("atom:updated", "", ns)

        if not title or not link:
            return None

        pub_date = None
        if updated:
            pub_date = self._parse_date(updated)

        return NewsItem(
            headline=title.strip(),
            source_name=feed.name,
            source_url=link.strip(),
            source_type=feed.source_type,
            snippet=summary[:500] if summary else "",
            published_at=pub_date,
            category=self.categorize_headline(title),
            tickers=self.extract_tickers(f"{title} {summary}"),
            credibility_score=feed.credibility,
        )

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse various date formats."""
        from email.utils import parsedate_to_datetime

        try:
            # Try RFC 2822 format (common in RSS)
            return parsedate_to_datetime(date_str)
        except (TypeError, ValueError):
            pass

        try:
            # Try ISO 8601 format (common in Atom)
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        return None
