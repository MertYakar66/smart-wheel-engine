"""
GDELT (Global Database of Events, Language, and Tone) Fetcher

GDELT provides:
- Global news coverage (65+ languages)
- 15-minute update cycle
- Free access for research/non-commercial use
- Precise time windows via STARTDATETIME/ENDDATETIME

API Endpoints:
- DOC API: Full-text article search (recent content)
- GEO API: Geographic filtering
- Article List API: Metadata + URLs

Note: GDELT has rolling window constraints on some endpoints.
"""

import logging
from datetime import datetime
from typing import Any

import httpx

from financial_news.models import Article, ArticleSource, Category
from financial_news.sources.base import BaseSourceFetcher

logger = logging.getLogger(__name__)


class GDELTFetcher(BaseSourceFetcher):
    """
    Fetches news from GDELT's DOC 2.0 API.

    GDELT is excellent for:
    - Broad, multilingual coverage
    - Fast discovery of breaking news
    - Geographic filtering
    - Tone/sentiment data
    """

    # GDELT API endpoints
    DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    GEO_API_URL = "https://api.gdeltproject.org/api/v2/geo/geo"

    def __init__(self, rate_limit_per_second: float = 2.0):
        """
        Initialize GDELT fetcher.

        GDELT doesn't have strict rate limits but recommends
        reasonable request rates for heavy users.
        """
        super().__init__(rate_limit_per_second=rate_limit_per_second)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if GDELT API is accessible"""
        try:
            client = await self._get_client()
            response = await client.get(
                self.DOC_API_URL,
                params={"query": "test", "mode": "artlist", "maxrecords": 1, "format": "json"},
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"GDELT health check failed: {e}")
            return False

    async def fetch_articles(
        self,
        category: Category,
        start_time: datetime,
        end_time: datetime,
        max_results: int = 100,
    ) -> list[Article]:
        """
        Fetch articles from GDELT matching category within time window.

        Args:
            category: Category configuration
            start_time: Start of time window
            end_time: End of time window
            max_results: Maximum articles to return

        Returns:
            List of Article objects
        """
        await self._rate_limit()

        # Build GDELT query
        query = self.build_query(category)

        # Add time filters (GDELT format: YYYYMMDDHHMMSS)
        start_str = start_time.strftime("%Y%m%d%H%M%S")
        end_str = end_time.strftime("%Y%m%d%H%M%S")

        params = {
            "query": query,
            "mode": "artlist",  # Article list mode
            "maxrecords": min(max_results, 250),  # GDELT max is 250
            "format": "json",
            "startdatetime": start_str,
            "enddatetime": end_str,
            "sort": "DateDesc",  # Most recent first
        }

        # Add language filter
        if category.languages:
            params["sourcelang"] = category.languages[0].lower()

        # Add country filter
        if category.regions:
            params["sourcecountry"] = category.regions[0]

        try:
            client = await self._get_client()
            response = await client.get(self.DOC_API_URL, params=params)
            response.raise_for_status()

            data = response.json()
            articles = self._parse_response(data, category)

            logger.info(
                f"GDELT: Fetched {len(articles)} articles for '{category.name}' "
                f"({start_time.isoformat()} - {end_time.isoformat()})"
            )

            return articles

        except httpx.HTTPStatusError as e:
            logger.error(f"GDELT HTTP error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"GDELT fetch error: {e}")
            return []

    def _parse_response(self, data: dict[str, Any], category: Category) -> list[Article]:
        """Parse GDELT API response into Article objects"""
        articles = []

        # GDELT returns articles in 'articles' array
        article_list = data.get("articles", [])

        for item in article_list:
            try:
                # Parse publication date
                date_str = item.get("seendate", "")
                if date_str:
                    # GDELT format: YYYYMMDDTHHMMSSZ
                    pub_date = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ")
                else:
                    pub_date = datetime.utcnow()

                # Build canonical URL
                url = self.canonicalize_url(item.get("url", ""))
                if not url:
                    continue

                # Generate article ID
                article_id = Article.generate_id(url=url, published_at=pub_date, source="gdelt")

                # Extract source name from domain
                domain = item.get("domain", "unknown")
                source_name = item.get("sourcecountry", domain)

                # Create Article
                article = Article(
                    article_id=article_id,
                    canonical_url=url,
                    source=ArticleSource.GDELT,
                    source_name=source_name,
                    published_at_utc=pub_date,
                    title=item.get("title", ""),
                    snippet=item.get("title", "")[
                        :300
                    ],  # GDELT doesn't provide snippets, use title
                    language=item.get("language", "en"),
                    country=item.get("sourcecountry", "US"),
                    retrieval_provider="gdelt",
                )

                # Extract entities from GDELT data if available
                if "socialimage" in item:
                    # GDELT sometimes embeds entity data
                    pass

                # Assign topics from category
                article.topics = category.topics.copy()

                articles.append(article)

            except Exception as e:
                logger.warning(f"Error parsing GDELT article: {e}")
                continue

        return articles

    async def fetch_trending_themes(self, max_themes: int = 20) -> list[dict[str, Any]]:
        """
        Fetch currently trending themes from GDELT.

        Useful for discovering breaking news topics.
        """
        await self._rate_limit()

        params = {
            "query": "*",
            "mode": "timelinetone",
            "format": "json",
        }

        try:
            client = await self._get_client()
            response = await client.get(self.DOC_API_URL, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"GDELT trending fetch error: {e}")
            return []

    async def fetch_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 100,
        max_results: int = 50,
    ) -> list[Article]:
        """
        Fetch articles related to a geographic location.

        Useful for regional news coverage.
        """
        await self._rate_limit()

        params = {
            "query": f"near:{latitude},{longitude},{radius_km}km",
            "mode": "artlist",
            "maxrecords": min(max_results, 250),
            "format": "json",
            "sort": "DateDesc",
        }

        try:
            client = await self._get_client()
            response = await client.get(self.GEO_API_URL, params=params)
            response.raise_for_status()

            data = response.json()
            # Create a dummy category for parsing
            dummy_category = Category(
                category_id="geo", name="Geographic", description="Location-based search"
            )
            return self._parse_response(data, dummy_category)

        except Exception as e:
            logger.error(f"GDELT geo fetch error: {e}")
            return []
