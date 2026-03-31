"""
EIA Connector - Energy Information Administration

Fetches:
- Weekly Petroleum Status Report (WPSR)
- Crude oil inventories
- Refinery utilization
- Gasoline and distillate stocks
- Natural gas storage

Source: https://www.eia.gov/

The Weekly Petroleum Status Report is released every Wednesday at 10:30 AM ET.
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging

from financial_news.schema import (
    Article, Entity, Source, CategoryType, EntityType,
    DEFAULT_SOURCES,
)
from .base import BaseConnector

logger = logging.getLogger(__name__)


class EIAConnector(BaseConnector):
    """
    Connector for EIA energy data and reports.

    Primary data:
    - Weekly Petroleum Status Report
    - Natural Gas Storage Report

    Uses EIA's RSS feeds and API.
    """

    # EIA RSS feeds
    PETROLEUM_RSS = "https://www.eia.gov/petroleum/supply/weekly/rss/petroleum_weekly.xml"
    TODAY_IN_ENERGY_RSS = "https://www.eia.gov/rss/todayinenergy.xml"

    # EIA API (requires API key for some endpoints)
    API_BASE = "https://api.eia.gov/v2"

    # EIA pages
    WPSR_PAGE = "https://www.eia.gov/petroleum/supply/weekly/"
    NATURAL_GAS_PAGE = "https://www.eia.gov/naturalgas/storage/"

    # Key data series
    PETROLEUM_SERIES = {
        "WCESTUS1": "U.S. Ending Stocks of Crude Oil",
        "WGTSTUS1": "U.S. Ending Stocks of Total Gasoline",
        "WDISTUS1": "U.S. Ending Stocks of Distillate Fuel Oil",
        "WCRFPUS2": "U.S. Refinery Utilization Rate",
        "WCRIMUS2": "U.S. Crude Oil Imports",
        "WCREXUS2": "U.S. Crude Oil Exports",
    }

    def __init__(
        self,
        source: Optional[Source] = None,
        api_key: Optional[str] = None,
    ):
        if source is None:
            source = next(s for s in DEFAULT_SOURCES if s.source_id == "eia")
        super().__init__(source)

        self.api_key = api_key  # Optional, needed for some API endpoints

    async def fetch_latest(
        self,
        since: Optional[datetime] = None,
        limit: int = 30,
    ) -> List[Article]:
        """
        Fetch latest EIA releases.

        Combines:
        - Weekly Petroleum Status Report
        - Today in Energy articles
        """
        articles = []

        # Fetch petroleum RSS
        petroleum_articles = await self._fetch_petroleum_rss(since, limit)
        articles.extend(petroleum_articles)

        # Fetch Today in Energy for broader energy news
        tie_articles = await self._fetch_today_in_energy(since, limit // 2)
        seen_urls = {a.canonical_url for a in articles}
        for article in tie_articles:
            if article.canonical_url not in seen_urls:
                articles.append(article)
                seen_urls.add(article.canonical_url)

        # Sort by date and limit
        articles.sort(key=lambda a: a.published_at, reverse=True)
        return articles[:limit]

    async def _fetch_petroleum_rss(
        self,
        since: Optional[datetime] = None,
        limit: int = 20,
    ) -> List[Article]:
        """Fetch Weekly Petroleum Status Report RSS."""
        result = await self.fetch(self.PETROLEUM_RSS)

        if not result.success:
            logger.warning(f"Failed to fetch EIA petroleum RSS: {result.error}")
            return []

        articles = []
        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(result.content)

            for item in root.findall(".//item"):
                try:
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    description = item.findtext("description", "").strip()
                    pub_date_str = item.findtext("pubDate", "")

                    if not title or not link:
                        continue

                    # Parse date
                    pub_date = self._parse_rss_date(pub_date_str)
                    if pub_date is None:
                        pub_date = datetime.utcnow()

                    # Filter by since
                    if since and pub_date < since:
                        continue

                    # Determine release type
                    release_type = self._classify_petroleum_release(title)

                    # Extract entities
                    entities = self._extract_energy_entities(title, description)

                    article = Article(
                        article_id=Article.generate_id(link, self.source.source_id),
                        source_id=self.source.source_id,
                        canonical_url=link,
                        title=title,
                        snippet=self._clean_text(description)[:500] if description else None,
                        published_at=pub_date,
                        ingested_at=datetime.utcnow(),
                        release_type=release_type,
                        categories=[CategoryType.OIL_ENERGY],
                        entities=entities,
                        tickers=self._get_energy_tickers(),
                    )
                    articles.append(article)

                    if len(articles) >= limit:
                        break

                except Exception as e:
                    logger.debug(f"Error parsing EIA RSS item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing EIA RSS: {e}")

        return articles

    async def _fetch_today_in_energy(
        self,
        since: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[Article]:
        """Fetch Today in Energy articles."""
        result = await self.fetch(self.TODAY_IN_ENERGY_RSS)

        if not result.success:
            logger.warning(f"Failed to fetch Today in Energy RSS: {result.error}")
            return []

        articles = []
        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(result.content)

            for item in root.findall(".//item"):
                try:
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    description = item.findtext("description", "").strip()
                    pub_date_str = item.findtext("pubDate", "")

                    if not title or not link:
                        continue

                    pub_date = self._parse_rss_date(pub_date_str)
                    if pub_date is None:
                        pub_date = datetime.utcnow()

                    if since and pub_date < since:
                        continue

                    # Only include energy-market-relevant articles
                    if not self._is_market_relevant(title, description):
                        continue

                    article = Article(
                        article_id=Article.generate_id(link, self.source.source_id),
                        source_id=self.source.source_id,
                        canonical_url=link,
                        title=title,
                        snippet=self._clean_text(description)[:500] if description else None,
                        published_at=pub_date,
                        ingested_at=datetime.utcnow(),
                        categories=[CategoryType.OIL_ENERGY],
                        entities=self._extract_energy_entities(title, description),
                    )
                    articles.append(article)

                    if len(articles) >= limit:
                        break

                except Exception as e:
                    logger.debug(f"Error parsing Today in Energy item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing Today in Energy RSS: {e}")

        return articles

    def _parse_rss_date(self, date_str: str) -> Optional[datetime]:
        """Parse RSS date formats."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
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

    def _clean_text(self, text: str) -> str:
        """Clean HTML tags from text."""
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()

    def _classify_petroleum_release(self, title: str) -> str:
        """Classify petroleum release type."""
        title_lower = title.lower()

        if "weekly petroleum status report" in title_lower or "wpsr" in title_lower:
            return "WPSR"
        elif "crude oil" in title_lower and "inventory" in title_lower:
            return "CRUDE_INVENTORY"
        elif "gasoline" in title_lower:
            return "GASOLINE"
        elif "distillate" in title_lower:
            return "DISTILLATE"
        elif "refinery" in title_lower:
            return "REFINERY"
        elif "natural gas" in title_lower:
            return "NATURAL_GAS"
        else:
            return "PETROLEUM_OTHER"

    def _extract_energy_entities(self, title: str, description: str) -> List[Entity]:
        """Extract energy-related entities."""
        entities = []
        text = f"{title} {description}".lower()

        # Commodities
        commodities = {
            "crude oil": "Crude Oil",
            "brent": "Brent Crude",
            "wti": "WTI Crude",
            "natural gas": "Natural Gas",
            "gasoline": "Gasoline",
            "distillate": "Distillate",
            "heating oil": "Heating Oil",
        }
        for key, name in commodities.items():
            if key in text:
                entities.append(Entity(
                    entity_id=f"commodity_{key.replace(' ', '_')}",
                    entity_type=EntityType.COMMODITY,
                    value=name,
                    confidence=0.9,
                ))

        # Organizations
        orgs = {
            "opec": "OPEC",
            "eia": "EIA",
            "iea": "IEA",
        }
        for key, name in orgs.items():
            if key in text:
                entities.append(Entity(
                    entity_id=f"org_{key}",
                    entity_type=EntityType.COMPANY,
                    value=name,
                    confidence=0.9,
                ))

        # Countries (major oil producers)
        countries = {
            "saudi": "Saudi Arabia",
            "russia": "Russia",
            "iran": "Iran",
            "iraq": "Iraq",
            "venezuela": "Venezuela",
            "libya": "Libya",
            "nigeria": "Nigeria",
        }
        for key, name in countries.items():
            if key in text:
                entities.append(Entity(
                    entity_id=f"country_{key}",
                    entity_type=EntityType.COUNTRY,
                    value=name,
                    confidence=0.8,
                ))

        return entities

    def _get_energy_tickers(self) -> List[str]:
        """Get relevant energy sector tickers."""
        return ["XLE", "USO", "XOP", "OIH", "CVX", "XOM", "COP", "SLB"]

    def _is_market_relevant(self, title: str, description: str) -> bool:
        """Check if article is market-relevant."""
        text = f"{title} {description}".lower()

        market_keywords = [
            "price", "inventory", "production", "refinery", "export", "import",
            "supply", "demand", "opec", "crude", "gasoline", "barrel",
            "shortage", "surplus", "sanctions", "disruption",
        ]

        return any(kw in text for kw in market_keywords)

    async def fetch_wpsr_data(self) -> Dict[str, Any]:
        """
        Fetch latest Weekly Petroleum Status Report data.

        Returns structured data about inventories, production, etc.
        Requires API key for full data access.
        """
        if not self.api_key:
            logger.warning("EIA API key not configured, limited data available")
            return {}

        # Example API call for crude inventories
        url = f"{self.API_BASE}/petroleum/sum/sndw/data/"
        params = {
            "api_key": self.api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": "WCESTUS1",  # Crude inventories
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": "10",
        }

        success, data = await self.fetch_json(url, params=params)

        if not success:
            logger.warning(f"Failed to fetch WPSR data: {data}")
            return {}

        return data

    async def health_check(self) -> bool:
        """Check if EIA is accessible."""
        result = await self.fetch(self.PETROLEUM_RSS)
        return result.success and result.status_code == 200
