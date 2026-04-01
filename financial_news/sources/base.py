"""
Base class for news source fetchers
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from financial_news.models import Article, Category

logger = logging.getLogger(__name__)


class BaseSourceFetcher(ABC):
    """Abstract base class for all news source fetchers"""

    def __init__(self, rate_limit_per_second: float = 1.0):
        self.rate_limit_per_second = rate_limit_per_second
        self._last_request_time: float | None = None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        if self._last_request_time is not None:
            elapsed = asyncio.get_event_loop().time() - self._last_request_time
            wait_time = (1.0 / self.rate_limit_per_second) - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        self._last_request_time = asyncio.get_event_loop().time()

    @abstractmethod
    async def fetch_articles(
        self,
        category: Category,
        start_time: datetime,
        end_time: datetime,
        max_results: int = 100,
    ) -> list[Article]:
        """
        Fetch articles matching a category within a time window.

        Args:
            category: Category with query configuration
            start_time: Start of time window
            end_time: End of time window
            max_results: Maximum articles to return

        Returns:
            List of Article objects
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the source is available"""
        pass

    def build_query(self, category: Category) -> str:
        """Build search query from category configuration"""
        query_parts = []

        # Add keywords
        if category.keywords:
            keyword_query = " OR ".join(f'"{kw}"' for kw in category.keywords)
            query_parts.append(f"({keyword_query})")

        # Add entity names
        if category.entities:
            entity_query = " OR ".join(f'"{e}"' for e in category.entities)
            query_parts.append(f"({entity_query})")

        # Add tickers (often mentioned in financial news)
        if category.tickers:
            ticker_query = " OR ".join(category.tickers)
            query_parts.append(f"({ticker_query})")

        # Combine with AND
        query = (
            " AND ".join(query_parts)
            if len(query_parts) > 1
            else (query_parts[0] if query_parts else "*")
        )

        # Add exclusions
        if category.exclusions:
            exclusion_query = " ".join(f'-"{ex}"' for ex in category.exclusions)
            query = f"({query}) {exclusion_query}"

        return query

    @staticmethod
    def canonicalize_url(url: str) -> str:
        """Remove tracking parameters and normalize URL"""
        from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

        # Common tracking parameters to remove
        tracking_params = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "fbclid",
            "gclid",
            "ref",
            "source",
            "mc_cid",
            "mc_eid",
        }

        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)

        # Remove tracking parameters
        filtered_params = {
            k: v for k, v in query_params.items() if k.lower() not in tracking_params
        }

        # Rebuild URL
        new_query = urlencode(filtered_params, doseq=True)
        cleaned = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                "",  # Remove fragment
            )
        )

        return cleaned
