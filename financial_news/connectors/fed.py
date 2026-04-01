"""
Federal Reserve Connector

Fetches:
- FOMC statements and press releases
- FOMC minutes
- Fed speeches
- Monetary policy reports

Source: https://www.federalreserve.gov

Data is parsed from the Fed's public RSS feeds and HTML pages.
"""

import logging
import re
from datetime import datetime
from xml.etree import ElementTree as ET

from financial_news.schema import (
    DEFAULT_SOURCES,
    Article,
    CategoryType,
    Entity,
    EntityType,
    Source,
)

from .base import BaseConnector

logger = logging.getLogger(__name__)


class FedConnector(BaseConnector):
    """
    Connector for Federal Reserve news and releases.

    Fetches from:
    - RSS feed for press releases
    - FOMC calendar page for statements/minutes
    - Speeches page for Fed official speeches
    """

    # Fed RSS feeds
    PRESS_RELEASES_RSS = "https://www.federalreserve.gov/feeds/press_all.xml"
    MONETARY_POLICY_RSS = "https://www.federalreserve.gov/feeds/press_monetary.xml"

    # Fed pages
    FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    SPEECHES_URL = "https://www.federalreserve.gov/newsevents/speeches.htm"

    def __init__(self, source: Source | None = None):
        if source is None:
            source = next(s for s in DEFAULT_SOURCES if s.source_id == "fed")
        super().__init__(source)

    async def fetch_latest(
        self,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[Article]:
        """
        Fetch latest Fed releases.

        Combines:
        - Press releases from RSS
        - FOMC statements and minutes
        """
        articles = []

        # Fetch from RSS feeds
        rss_articles = await self._fetch_rss_feed(self.MONETARY_POLICY_RSS, since, limit)
        articles.extend(rss_articles)

        # Fetch from press releases RSS (broader)
        press_articles = await self._fetch_rss_feed(self.PRESS_RELEASES_RSS, since, limit // 2)
        # Filter to avoid duplicates
        seen_urls = {a.canonical_url for a in articles}
        for article in press_articles:
            if article.canonical_url not in seen_urls:
                articles.append(article)
                seen_urls.add(article.canonical_url)

        # Sort by date and limit
        articles.sort(key=lambda a: a.published_at, reverse=True)
        return articles[:limit]

    async def _fetch_rss_feed(
        self,
        feed_url: str,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[Article]:
        """Parse Fed RSS feed."""
        result = await self.fetch(feed_url)

        if not result.success:
            logger.warning(f"Failed to fetch Fed RSS: {result.error}")
            return []

        articles = []
        try:
            root = ET.fromstring(result.content)

            # RSS 2.0 format
            for item in root.findall(".//item"):
                try:
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    description = item.findtext("description", "").strip()
                    pub_date_str = item.findtext("pubDate", "")

                    if not title or not link:
                        continue

                    # Parse date (RFC 822 format)
                    pub_date = self._parse_rss_date(pub_date_str)
                    if pub_date is None:
                        pub_date = datetime.utcnow()

                    # Filter by since
                    if since and pub_date < since:
                        continue

                    # Create article
                    article = Article(
                        article_id=Article.generate_id(link, self.source.source_id),
                        source_id=self.source.source_id,
                        canonical_url=link,
                        title=title,
                        snippet=self._clean_html(description)[:500] if description else None,
                        published_at=pub_date,
                        ingested_at=datetime.utcnow(),
                        categories=[CategoryType.FED_RATES],
                        entities=self._extract_fed_entities(title, description),
                    )
                    articles.append(article)

                    if len(articles) >= limit:
                        break

                except Exception as e:
                    logger.warning(f"Error parsing RSS item: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"Error parsing Fed RSS XML: {e}")

        return articles

    def _parse_rss_date(self, date_str: str) -> datetime | None:
        """Parse RFC 822 date from RSS."""
        # Example: "Mon, 15 Jan 2026 14:00:00 -0500"
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                # Convert to UTC if timezone aware
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue
        return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _extract_fed_entities(self, title: str, description: str) -> list[Entity]:
        """Extract Fed-related entities from text."""
        entities = []
        text = f"{title} {description}".lower()

        # Central bank
        entities.append(
            Entity(
                entity_id="fed_central_bank",
                entity_type=EntityType.CENTRAL_BANK,
                value="Federal Reserve",
                confidence=1.0,
            )
        )

        # Fed officials
        fed_officials = {
            "powell": "Jerome Powell",
            "jefferson": "Philip Jefferson",
            "barr": "Michael Barr",
            "waller": "Christopher Waller",
            "bowman": "Michelle Bowman",
            "cook": "Lisa Cook",
            "kugler": "Adriana Kugler",
        }
        for key, name in fed_officials.items():
            if key in text:
                entities.append(
                    Entity(
                        entity_id=f"fed_official_{key}",
                        entity_type=EntityType.PERSON,
                        value=name,
                        confidence=0.9,
                    )
                )

        # Macro indicators mentioned
        indicators = {
            "inflation": "Inflation",
            "employment": "Employment",
            "unemployment": "Unemployment",
            "gdp": "GDP",
            "interest rate": "Interest Rates",
            "federal funds": "Federal Funds Rate",
        }
        for key, name in indicators.items():
            if key in text:
                entities.append(
                    Entity(
                        entity_id=f"macro_{key.replace(' ', '_')}",
                        entity_type=EntityType.MACRO_INDICATOR,
                        value=name,
                        confidence=0.8,
                    )
                )

        return entities

    async def fetch_fomc_statements(self, year: int = 2026) -> list[Article]:
        """
        Fetch FOMC statements for a specific year.

        Parses the FOMC calendar page to find statement links.
        """
        result = await self.fetch(self.FOMC_CALENDAR_URL)

        if not result.success:
            logger.warning(f"Failed to fetch FOMC calendar: {result.error}")
            return []

        articles = []
        # Look for statement links in the HTML
        # Pattern: /newsevents/pressreleases/monetary20260318a.htm
        pattern = rf"/newsevents/pressreleases/monetary{year}\d{{4}}[a-z]?\.htm"
        matches = re.findall(pattern, result.content)

        for match in set(matches):
            url = f"https://www.federalreserve.gov{match}"

            # Extract date from URL
            date_match = re.search(r"monetary(\d{8})", match)
            if date_match:
                date_str = date_match.group(1)
                try:
                    pub_date = datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    pub_date = datetime.utcnow()
            else:
                pub_date = datetime.utcnow()

            article = Article(
                article_id=Article.generate_id(url, self.source.source_id),
                source_id=self.source.source_id,
                canonical_url=url,
                title=f"FOMC Statement - {pub_date.strftime('%B %d, %Y')}",
                published_at=pub_date,
                ingested_at=datetime.utcnow(),
                categories=[CategoryType.FED_RATES],
                release_type="FOMC_STATEMENT",
                entities=[
                    Entity(
                        entity_id="fomc",
                        entity_type=EntityType.CENTRAL_BANK,
                        value="Federal Open Market Committee",
                        confidence=1.0,
                    )
                ],
            )
            articles.append(article)

        return articles

    async def health_check(self) -> bool:
        """Check if Fed RSS is accessible."""
        result = await self.fetch(self.MONETARY_POLICY_RSS)
        return result.success and result.status_code == 200
