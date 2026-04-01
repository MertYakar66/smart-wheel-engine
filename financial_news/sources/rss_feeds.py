"""
RSS Feed Fetcher - Official Sources

Fetches news from official RSS/Atom feeds:
- Central banks (Fed, ECB, BoE, BoJ)
- Government agencies
- Corporate investor relations
- Regulatory bodies

These are "clean" from a rights standpoint and high signal.
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from financial_news.models import Article, ArticleSource, Category, TopicCategory
from financial_news.sources.base import BaseSourceFetcher

logger = logging.getLogger(__name__)


@dataclass
class FeedConfig:
    """Configuration for an RSS/Atom feed"""
    name: str
    url: str
    source_type: str  # central_bank, government, corporate, etc.
    topics: list[TopicCategory]
    region: str = "US"
    language: str = "en"


# Pre-configured feeds for financial news
DEFAULT_FEEDS: list[FeedConfig] = [
    # Central Banks
    FeedConfig(
        name="Federal Reserve Press Releases",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
        source_type="central_bank",
        topics=[TopicCategory.CENTRAL_BANKS, TopicCategory.MACRO_RATES],
        region="US",
    ),
    FeedConfig(
        name="Federal Reserve Speeches",
        url="https://www.federalreserve.gov/feeds/speeches.xml",
        source_type="central_bank",
        topics=[TopicCategory.CENTRAL_BANKS, TopicCategory.MACRO_RATES],
        region="US",
    ),
    FeedConfig(
        name="ECB Press Releases",
        url="https://www.ecb.europa.eu/rss/press.html",
        source_type="central_bank",
        topics=[TopicCategory.CENTRAL_BANKS, TopicCategory.MACRO_RATES],
        region="EU",
    ),
    FeedConfig(
        name="Bank of England News",
        url="https://www.bankofengland.co.uk/rss/news",
        source_type="central_bank",
        topics=[TopicCategory.CENTRAL_BANKS, TopicCategory.MACRO_RATES],
        region="GB",
    ),

    # US Government / Economic Data
    FeedConfig(
        name="Bureau of Labor Statistics",
        url="https://www.bls.gov/feed/bls_latest.rss",
        source_type="government",
        topics=[TopicCategory.MACRO_EMPLOYMENT, TopicCategory.MACRO_INFLATION],
        region="US",
    ),
    FeedConfig(
        name="Treasury Department",
        url="https://home.treasury.gov/system/files/136/treasury-news.xml",
        source_type="government",
        topics=[TopicCategory.MACRO_RATES, TopicCategory.FIXED_INCOME],
        region="US",
    ),

    # Regulatory
    FeedConfig(
        name="SEC Press Releases",
        url="https://www.sec.gov/news/pressreleases.rss",
        source_type="regulatory",
        topics=[TopicCategory.REGULATION],
        region="US",
    ),
    FeedConfig(
        name="CFTC Press Releases",
        url="https://www.cftc.gov/PressRoom/PressReleases/RSS",
        source_type="regulatory",
        topics=[TopicCategory.REGULATION, TopicCategory.COMMODITIES_OIL],
        region="US",
    ),

    # Energy / Commodities
    FeedConfig(
        name="EIA (Energy Information)",
        url="https://www.eia.gov/rss/todayinenergy.xml",
        source_type="government",
        topics=[TopicCategory.COMMODITIES_OIL],
        region="US",
    ),
    FeedConfig(
        name="OPEC News",
        url="https://www.opec.org/opec_web/en/press_room/28.htm",  # May need scraping
        source_type="organization",
        topics=[TopicCategory.COMMODITIES_OIL],
        region="INT",
    ),

    # International
    FeedConfig(
        name="IMF News",
        url="https://www.imf.org/en/News/rss",
        source_type="organization",
        topics=[TopicCategory.MACRO_RATES, TopicCategory.EMERGING_MARKETS],
        region="INT",
    ),
    FeedConfig(
        name="World Bank News",
        url="https://www.worldbank.org/en/news/all/rss.xml",
        source_type="organization",
        topics=[TopicCategory.EMERGING_MARKETS],
        region="INT",
    ),
]


class RSSFetcher(BaseSourceFetcher):
    """
    Fetches news from RSS/Atom feeds.

    Supports both RSS 2.0 and Atom formats.
    """

    def __init__(
        self,
        feeds: list[FeedConfig] | None = None,
        rate_limit_per_second: float = 5.0,
    ):
        """
        Initialize RSS fetcher.

        Args:
            feeds: List of feed configurations (defaults to DEFAULT_FEEDS)
            rate_limit_per_second: Request rate limit
        """
        super().__init__(rate_limit_per_second=rate_limit_per_second)
        self.feeds = feeds or DEFAULT_FEEDS
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
                headers={
                    "User-Agent": "SmartWheelEngine/1.0 Financial News Aggregator",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if at least one feed is accessible"""
        for feed in self.feeds[:3]:  # Check first 3 feeds
            try:
                client = await self._get_client()
                response = await client.head(feed.url)
                if response.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    async def fetch_articles(
        self,
        category: Category,
        start_time: datetime,
        end_time: datetime,
        max_results: int = 100,
    ) -> list[Article]:
        """
        Fetch articles from RSS feeds matching category.

        Filters feeds by topic overlap and region.
        """
        articles = []

        # Find relevant feeds for this category
        relevant_feeds = self._get_relevant_feeds(category)

        # Fetch from each feed concurrently
        tasks = [
            self._fetch_feed(feed, start_time, end_time)
            for feed in relevant_feeds
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                articles.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Feed fetch error: {result}")

        # Sort by date and limit
        articles.sort(key=lambda a: a.published_at_utc, reverse=True)

        logger.info(
            f"RSS: Fetched {len(articles)} articles for '{category.name}' "
            f"from {len(relevant_feeds)} feeds"
        )

        return articles[:max_results]

    def _get_relevant_feeds(self, category: Category) -> list[FeedConfig]:
        """Find feeds that match category configuration"""
        relevant = []

        for feed in self.feeds:
            # Check topic overlap
            feed_topics_set = set(feed.topics)
            category_topics_set = set(category.topics)

            if feed_topics_set & category_topics_set:
                relevant.append(feed)
                continue

            # Check region overlap
            if category.regions and feed.region in category.regions:
                relevant.append(feed)
                continue

        return relevant

    async def _fetch_feed(
        self,
        feed: FeedConfig,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Article]:
        """Fetch and parse a single RSS/Atom feed"""
        await self._rate_limit()

        try:
            client = await self._get_client()
            response = await client.get(feed.url)
            response.raise_for_status()

            # Parse based on content type
            content = response.text
            if "<feed" in content[:500]:
                articles = self._parse_atom(content, feed)
            else:
                articles = self._parse_rss(content, feed)

            # Filter by date
            articles = [
                a for a in articles
                if start_time <= a.published_at_utc <= end_time
            ]

            return articles

        except Exception as e:
            logger.warning(f"Error fetching feed {feed.name}: {e}")
            return []

    def _parse_rss(self, content: str, feed: FeedConfig) -> list[Article]:
        """Parse RSS 2.0 feed"""
        articles = []

        try:
            root = ET.fromstring(content)

            for item in root.findall(".//item"):
                try:
                    title = item.findtext("title", "")
                    link = item.findtext("link", "")
                    description = item.findtext("description", "")
                    pub_date_str = item.findtext("pubDate", "")

                    if not link:
                        continue

                    # Parse date (RSS date format varies)
                    pub_date = self._parse_rss_date(pub_date_str)

                    url = self.canonicalize_url(link)
                    article_id = Article.generate_id(url, pub_date, "rss")

                    article = Article(
                        article_id=article_id,
                        canonical_url=url,
                        source=ArticleSource.RSS,
                        source_name=feed.name,
                        published_at_utc=pub_date,
                        title=title,
                        snippet=self._clean_html(description)[:300],
                        language=feed.language,
                        country=feed.region,
                        topics=feed.topics.copy(),
                        retrieval_provider="rss",
                    )

                    articles.append(article)

                except Exception as e:
                    logger.debug(f"Error parsing RSS item: {e}")
                    continue

        except ET.ParseError as e:
            logger.warning(f"RSS parse error for {feed.name}: {e}")

        return articles

    def _parse_atom(self, content: str, feed: FeedConfig) -> list[Article]:
        """Parse Atom feed"""
        articles = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        try:
            root = ET.fromstring(content)

            for entry in root.findall(".//atom:entry", ns):
                try:
                    title = entry.findtext("atom:title", "", ns)

                    # Get link (may have multiple)
                    link_elem = entry.find("atom:link[@rel='alternate']", ns)
                    if link_elem is None:
                        link_elem = entry.find("atom:link", ns)
                    link = link_elem.get("href", "") if link_elem is not None else ""

                    summary = entry.findtext("atom:summary", "", ns)
                    if not summary:
                        summary = entry.findtext("atom:content", "", ns)

                    updated = entry.findtext("atom:updated", "", ns)
                    published = entry.findtext("atom:published", updated, ns)

                    if not link:
                        continue

                    pub_date = self._parse_iso_date(published)
                    url = self.canonicalize_url(link)
                    article_id = Article.generate_id(url, pub_date, "rss")

                    article = Article(
                        article_id=article_id,
                        canonical_url=url,
                        source=ArticleSource.RSS,
                        source_name=feed.name,
                        published_at_utc=pub_date,
                        title=title,
                        snippet=self._clean_html(summary)[:300],
                        language=feed.language,
                        country=feed.region,
                        topics=feed.topics.copy(),
                        retrieval_provider="rss",
                    )

                    articles.append(article)

                except Exception as e:
                    logger.debug(f"Error parsing Atom entry: {e}")
                    continue

        except ET.ParseError as e:
            logger.warning(f"Atom parse error for {feed.name}: {e}")

        return articles

    def _parse_rss_date(self, date_str: str) -> datetime:
        """Parse various RSS date formats"""
        from email.utils import parsedate_to_datetime

        if not date_str:
            return datetime.utcnow()

        try:
            # RFC 2822 format (standard RSS)
            return parsedate_to_datetime(date_str)
        except Exception:
            pass

        # Try ISO format
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            pass

        return datetime.utcnow()

    def _parse_iso_date(self, date_str: str) -> datetime:
        """Parse ISO 8601 date"""
        if not date_str:
            return datetime.utcnow()

        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.utcnow()

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        import re
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()

    def add_feed(self, feed: FeedConfig) -> None:
        """Add a new feed to monitor"""
        self.feeds.append(feed)

    def remove_feed(self, feed_name: str) -> bool:
        """Remove a feed by name"""
        for i, feed in enumerate(self.feeds):
            if feed.name == feed_name:
                self.feeds.pop(i)
                return True
        return False

    def list_feeds(self) -> list[dict[str, Any]]:
        """List all configured feeds"""
        return [
            {
                "name": f.name,
                "url": f.url,
                "source_type": f.source_type,
                "topics": [t.value for t in f.topics],
                "region": f.region,
            }
            for f in self.feeds
        ]
