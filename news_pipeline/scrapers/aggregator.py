"""
News Aggregator

Combines multiple scrapers and deduplicates results.
Single entry point for all news fetching.
"""

import asyncio
import logging
from datetime import datetime
from hashlib import md5

from news_pipeline.scrapers.base import NewsItem, SourceType
from news_pipeline.scrapers.browser_scraper import BrowserNewsScraper
from news_pipeline.scrapers.rss_scraper import RSSNewsScraper

logger = logging.getLogger(__name__)


class NewsAggregator:
    """
    Aggregates news from multiple scrapers.

    Handles:
    - Parallel fetching from all sources
    - Deduplication by headline similarity
    - Ranking by source credibility
    - Category and ticker filtering
    """

    def __init__(
        self,
        use_rss: bool = True,
        use_browser: bool = False,  # Disabled by default (resource-intensive)
    ):
        """
        Initialize aggregator.

        Args:
            use_rss: Enable RSS scraping (recommended)
            use_browser: Enable browser scraping (optional)
        """
        self.scrapers = []

        if use_rss:
            self.scrapers.append(RSSNewsScraper())

        if use_browser:
            self.scrapers.append(BrowserNewsScraper())

    async def fetch_news(
        self,
        categories: list[str] | None = None,
        tickers: list[str] | None = None,
        max_items: int = 50,
        hours_back: int = 12,
        deduplicate: bool = True,
    ) -> list[NewsItem]:
        """
        Fetch news from all configured sources.

        Args:
            categories: Filter by categories
            tickers: Filter by tickers
            max_items: Maximum items to return
            hours_back: How far back to look
            deduplicate: Remove similar headlines

        Returns:
            List of NewsItem sorted by credibility and recency
        """
        logger.info(
            f"[Aggregator] Fetching news: categories={categories}, "
            f"tickers={tickers}, hours_back={hours_back}"
        )

        all_items = []

        # Fetch from all scrapers in parallel
        tasks = [
            scraper.fetch(
                categories=categories,
                tickers=tickers,
                max_items=max_items * 2,  # Fetch extra for dedup
                hours_back=hours_back,
            )
            for scraper in self.scrapers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"[Aggregator] Scraper failed: {result}")

        logger.info(f"[Aggregator] Raw items: {len(all_items)}")

        # Deduplicate
        if deduplicate:
            all_items = self._deduplicate(all_items)
            logger.info(f"[Aggregator] After dedup: {len(all_items)}")

        # Sort by credibility and recency
        all_items = self._rank_items(all_items)

        return all_items[:max_items]

    def _deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        """Remove duplicate or very similar headlines."""
        seen_hashes = set()
        unique_items = []

        for item in items:
            # Create hash from normalized headline
            normalized = self._normalize_headline(item.headline)
            headline_hash = md5(normalized.encode()).hexdigest()[:16]

            if headline_hash not in seen_hashes:
                seen_hashes.add(headline_hash)
                unique_items.append(item)

        return unique_items

    def _normalize_headline(self, headline: str) -> str:
        """Normalize headline for comparison."""
        import re

        # Lowercase
        text = headline.lower()

        # Remove common prefixes/suffixes
        prefixes = ["breaking:", "update:", "exclusive:", "report:"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :]

        # Remove punctuation and extra spaces
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "in", "on"}
        words = [w for w in text.split() if w not in stopwords]

        # Take first 8 significant words
        return " ".join(words[:8])

    def _rank_items(self, items: list[NewsItem]) -> list[NewsItem]:
        """Rank items by credibility and recency."""
        now = datetime.utcnow()

        def score(item: NewsItem) -> float:
            # Base score from credibility
            s = item.credibility_score * 100

            # Boost for recency (decay over hours)
            if item.published_at:
                age_hours = (now - item.published_at).total_seconds() / 3600
                recency_boost = max(0, 50 - age_hours * 2)
                s += recency_boost

            # Boost for official sources
            if item.source_type == SourceType.OFFICIAL:
                s += 30

            # Boost for having tickers
            if item.tickers:
                s += len(item.tickers) * 5

            return s

        return sorted(items, key=score, reverse=True)

    async def close(self) -> None:
        """Close all scrapers."""
        for scraper in self.scrapers:
            if hasattr(scraper, "close"):
                await scraper.close()
