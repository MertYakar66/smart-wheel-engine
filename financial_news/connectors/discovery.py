"""
Tier 3 Discovery Connector - Premium News Signal Detection

Uses headline-only discovery from:
- Google News RSS (free, public)
- CNBC RSS (free, public)
- Reuters RSS (free, public)
- Yahoo Finance RSS (free, public)

These are SIGNAL GENERATORS, not truth sources.

Pattern:
1. Detect headline + metadata (title, publisher, URL, timestamp)
2. Open internal "story ticket"
3. Corroborate with Tier 1/2 sources (Fed, SEC, EIA, etc.)
4. Generate original analysis from corroborated facts
5. Link to original publisher for full reading

IMPORTANT: We only capture metadata. We do NOT:
- Store full article text
- Reconstruct paywalled content
- Substitute for subscription journalism
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from xml.etree import ElementTree as ET
from urllib.parse import quote_plus
import logging

from financial_news.schema import (
    Article, Entity, Source, SourceType, SourceProvider, CategoryType, EntityType,
)
from .base import BaseConnector

logger = logging.getLogger(__name__)


# Discovery source definitions
DISCOVERY_SOURCES = {
    "google_news": Source(
        source_id="google_news",
        name="Google News",
        provider=SourceProvider.RSS_FEED,
        source_type=SourceType.AGGREGATOR,
        priority_weight=0.4,  # Lower priority - discovery only
        rate_limit_per_second=2.0,
        base_url="https://news.google.com",
        default_categories=[],  # Determined by content
    ),
    "cnbc": Source(
        source_id="cnbc",
        name="CNBC",
        provider=SourceProvider.RSS_FEED,
        source_type=SourceType.LICENSED,
        priority_weight=0.6,
        rate_limit_per_second=2.0,
        base_url="https://www.cnbc.com",
        default_categories=[CategoryType.SP500_CORPORATE, CategoryType.FED_RATES],
    ),
    "reuters": Source(
        source_id="reuters",
        name="Reuters",
        provider=SourceProvider.RSS_FEED,
        source_type=SourceType.LICENSED,
        priority_weight=0.6,
        rate_limit_per_second=2.0,
        base_url="https://www.reuters.com",
        default_categories=[],
    ),
    "yahoo_finance": Source(
        source_id="yahoo_finance",
        name="Yahoo Finance",
        provider=SourceProvider.RSS_FEED,
        source_type=SourceType.AGGREGATOR,
        priority_weight=0.4,
        rate_limit_per_second=2.0,
        base_url="https://finance.yahoo.com",
        default_categories=[CategoryType.SP500_CORPORATE],
    ),
}


class DiscoveryConnector(BaseConnector):
    """
    Tier 3 Discovery Connector.

    Captures headlines and metadata from premium news sources.
    Used for story discovery, not as primary truth source.

    Output: "Story tickets" that need corroboration from Tier 1 sources.
    """

    # RSS feed URLs
    FEEDS = {
        # Google News - topic-specific
        "google_business": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
        "google_markets": "https://news.google.com/rss/topics/CAAqIggKIhxDQkFTRHdvSkwyMHZNR2RtY0hNekVnSmxiaWdBUAE",

        # CNBC
        "cnbc_top": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "cnbc_markets": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        "cnbc_economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",

        # Yahoo Finance
        "yahoo_top": "https://finance.yahoo.com/news/rssindex",

        # Reuters (if available)
        "reuters_business": "https://www.reutersagency.com/feed/?best-topics=business-finance",
    }

    # Keywords that indicate market-relevant stories
    MARKET_KEYWORDS = {
        # Fed/Rates
        "fed", "federal reserve", "fomc", "powell", "rate hike", "rate cut",
        "interest rate", "treasury yield", "bond",

        # Inflation
        "inflation", "cpi", "pce", "prices",

        # Labor
        "jobs", "employment", "unemployment", "payroll", "labor",

        # Growth
        "gdp", "recession", "growth", "retail sales", "housing",

        # Oil/Energy
        "oil", "crude", "opec", "energy", "gas prices", "petroleum",

        # Geopolitics
        "sanctions", "tariff", "trade war", "china", "russia",

        # Corporate
        "earnings", "revenue", "guidance", "merger", "acquisition",
        "ceo", "layoffs", "buyback", "dividend",

        # Market
        "stock", "market", "s&p", "nasdaq", "dow", "rally", "selloff",
    }

    def __init__(self, source_id: str = "google_news"):
        source = DISCOVERY_SOURCES.get(source_id, DISCOVERY_SOURCES["google_news"])
        super().__init__(source)

        self._seen_urls: Set[str] = set()

    async def fetch_latest(
        self,
        since: Optional[datetime] = None,
        limit: int = 50,
        feeds: Optional[List[str]] = None,
    ) -> List[Article]:
        """
        Fetch latest headlines from discovery feeds.

        Returns Article objects with:
        - title (headline)
        - canonical_url (link to original)
        - published_at
        - snippet (if available)
        - metadata_json with publisher info

        These are DISCOVERY SIGNALS, not complete articles.
        """
        articles = []

        # Default to all feeds
        if feeds is None:
            feeds = list(self.FEEDS.keys())

        for feed_name in feeds:
            feed_url = self.FEEDS.get(feed_name)
            if not feed_url:
                continue

            try:
                feed_articles = await self._fetch_rss_feed(feed_url, feed_name, since)
                articles.extend(feed_articles)
            except Exception as e:
                logger.warning(f"Error fetching {feed_name}: {e}")

        # Deduplicate by URL
        unique = []
        for article in articles:
            if article.canonical_url not in self._seen_urls:
                self._seen_urls.add(article.canonical_url)
                unique.append(article)

        # Filter to market-relevant headlines
        relevant = [a for a in unique if self._is_market_relevant(a.title)]

        # Sort by date and limit
        relevant.sort(key=lambda a: a.published_at, reverse=True)

        logger.info(f"Discovery: {len(relevant)} market-relevant headlines from {len(feeds)} feeds")
        return relevant[:limit]

    async def _fetch_rss_feed(
        self,
        feed_url: str,
        feed_name: str,
        since: Optional[datetime],
    ) -> List[Article]:
        """Fetch and parse an RSS feed."""
        result = await self.fetch(feed_url)

        if not result.success:
            logger.debug(f"Failed to fetch {feed_name}: {result.error}")
            return []

        articles = []
        try:
            root = ET.fromstring(result.content)

            # Handle both RSS and Atom formats
            items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")

            for item in items:
                try:
                    article = self._parse_feed_item(item, feed_name, since)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.debug(f"Error parsing item: {e}")

        except ET.ParseError as e:
            logger.debug(f"XML parse error for {feed_name}: {e}")

        return articles

    def _parse_feed_item(
        self,
        item: ET.Element,
        feed_name: str,
        since: Optional[datetime],
    ) -> Optional[Article]:
        """Parse a single RSS/Atom item."""
        # RSS format
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        description = item.findtext("description", "")
        pub_date_str = item.findtext("pubDate", "")

        # Atom format fallback
        if not title:
            title = item.findtext("{http://www.w3.org/2005/Atom}title", "").strip()
        if not link:
            link_elem = item.find("{http://www.w3.org/2005/Atom}link")
            link = link_elem.get("href", "") if link_elem is not None else ""
        if not pub_date_str:
            pub_date_str = item.findtext("{http://www.w3.org/2005/Atom}published", "")

        if not title or not link:
            return None

        # Parse date
        pub_date = self._parse_date(pub_date_str)
        if pub_date is None:
            pub_date = datetime.utcnow()

        # Filter by since
        if since and pub_date < since:
            return None

        # Extract publisher from URL or feed name
        publisher = self._extract_publisher(link, feed_name)

        # Clean description
        snippet = self._clean_html(description)[:300] if description else None

        # Create article (discovery signal)
        return Article(
            article_id=Article.generate_id(link, f"discovery_{feed_name}"),
            source_id=f"discovery_{feed_name}",
            canonical_url=link,
            title=title,
            snippet=snippet,
            published_at=pub_date,
            ingested_at=datetime.utcnow(),
            categories=[],  # To be determined by classifier
            tickers=self._extract_tickers(title),
            entities=self._extract_entities(title),
            metadata_json={
                "discovery_source": feed_name,
                "publisher": publisher,
                "requires_corroboration": True,
                "tier": 3,
            },
        )

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue
        return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags."""
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()

    def _extract_publisher(self, url: str, feed_name: str) -> str:
        """Extract publisher from URL."""
        publishers = {
            "cnbc.com": "CNBC",
            "reuters.com": "Reuters",
            "nytimes.com": "New York Times",
            "wsj.com": "Wall Street Journal",
            "ft.com": "Financial Times",
            "bloomberg.com": "Bloomberg",
            "yahoo.com": "Yahoo Finance",
            "marketwatch.com": "MarketWatch",
            "fool.com": "Motley Fool",
            "seekingalpha.com": "Seeking Alpha",
        }
        for domain, name in publishers.items():
            if domain in url:
                return name
        return feed_name

    def _is_market_relevant(self, title: str) -> bool:
        """Check if headline is market-relevant."""
        title_lower = title.lower()
        return any(kw in title_lower for kw in self.MARKET_KEYWORDS)

    def _extract_tickers(self, title: str) -> List[str]:
        """Extract stock tickers from title."""
        # Match patterns like $AAPL, (AAPL), AAPL:
        patterns = [
            r'\$([A-Z]{1,5})\b',
            r'\(([A-Z]{1,5})\)',
            r'\b([A-Z]{2,5}):\s',
        ]
        tickers = []
        for pattern in patterns:
            matches = re.findall(pattern, title)
            tickers.extend(matches)

        # Also check for known company names
        company_tickers = {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "amazon": "AMZN", "meta": "META", "nvidia": "NVDA",
            "tesla": "TSLA", "netflix": "NFLX", "jpmorgan": "JPM",
        }
        title_lower = title.lower()
        for company, ticker in company_tickers.items():
            if company in title_lower:
                tickers.append(ticker)

        return list(set(tickers))[:5]

    def _extract_entities(self, title: str) -> List[Entity]:
        """Extract entities from title."""
        entities = []

        # Fed-related
        if any(kw in title.lower() for kw in ["fed", "fomc", "powell"]):
            entities.append(Entity(
                entity_id="fed",
                entity_type=EntityType.CENTRAL_BANK,
                value="Federal Reserve",
            ))

        # OPEC
        if "opec" in title.lower():
            entities.append(Entity(
                entity_id="opec",
                entity_type=EntityType.COMPANY,
                value="OPEC",
            ))

        return entities

    async def search_topic(
        self,
        topic: str,
        limit: int = 20,
    ) -> List[Article]:
        """
        Search Google News for a specific topic.

        Useful for targeted discovery when a story breaks.
        """
        # Google News search RSS
        encoded_topic = quote_plus(topic)
        search_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl=en-US&gl=US&ceid=US:en"

        result = await self.fetch(search_url)
        if not result.success:
            logger.warning(f"Topic search failed for '{topic}': {result.error}")
            return []

        articles = []
        try:
            root = ET.fromstring(result.content)
            for item in root.findall(".//item"):
                article = self._parse_feed_item(item, f"google_search_{topic}", None)
                if article:
                    articles.append(article)
        except Exception as e:
            logger.warning(f"Error parsing search results: {e}")

        return articles[:limit]

    async def health_check(self) -> bool:
        """Check if discovery feeds are accessible."""
        result = await self.fetch(self.FEEDS.get("cnbc_top", ""))
        return result.success


class CorroborationEngine:
    """
    Corroborate discovery signals with Tier 1 sources.

    Takes a discovery headline and finds supporting evidence from:
    - Fed releases
    - SEC filings
    - EIA data
    - Company IR pages
    """

    def __init__(self, connectors: Dict[str, BaseConnector]):
        """
        Args:
            connectors: Dict of source_id -> connector for Tier 1/2 sources
        """
        self.connectors = connectors

    async def corroborate(
        self,
        discovery_article: Article,
        max_sources: int = 3,
    ) -> Dict[str, any]:
        """
        Find corroborating evidence for a discovery signal.

        Returns:
            Dict with:
            - corroborated: bool
            - confidence: float (0-1)
            - supporting_articles: List[Article]
            - official_sources: List[str]
        """
        result = {
            "corroborated": False,
            "confidence": 0.0,
            "supporting_articles": [],
            "official_sources": [],
        }

        tickers = discovery_article.tickers
        title_lower = discovery_article.title.lower()

        # Determine which Tier 1 sources to check
        sources_to_check = []

        if any(kw in title_lower for kw in ["fed", "fomc", "rate", "powell"]):
            sources_to_check.append("fed")

        if any(kw in title_lower for kw in ["oil", "crude", "petroleum", "eia"]):
            sources_to_check.append("eia")

        if any(kw in title_lower for kw in ["earnings", "filing", "sec", "8-k"]) or tickers:
            sources_to_check.append("sec_edgar")

        # Fetch from each source
        for source_id in sources_to_check[:max_sources]:
            connector = self.connectors.get(source_id)
            if not connector:
                continue

            try:
                # Fetch recent articles
                since = datetime.utcnow() - timedelta(hours=24)
                articles = await connector.fetch_latest(since=since, limit=20)

                # Check for related articles
                for article in articles:
                    if self._is_related(discovery_article, article):
                        result["supporting_articles"].append(article)
                        result["official_sources"].append(source_id)

            except Exception as e:
                logger.warning(f"Error checking {source_id}: {e}")

        # Calculate confidence
        if result["supporting_articles"]:
            result["corroborated"] = True
            # More sources = higher confidence
            result["confidence"] = min(1.0, len(result["official_sources"]) * 0.4)

        return result

    def _is_related(self, discovery: Article, official: Article) -> bool:
        """Check if an official article relates to the discovery."""
        # Check ticker overlap
        if discovery.tickers and official.tickers:
            if set(discovery.tickers) & set(official.tickers):
                return True

        # Check title keyword overlap
        discovery_words = set(discovery.title.lower().split())
        official_words = set(official.title.lower().split())

        # Remove common words
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or"}
        discovery_words -= stopwords
        official_words -= stopwords

        overlap = len(discovery_words & official_words)
        return overlap >= 3
