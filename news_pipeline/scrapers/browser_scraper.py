"""
Browser-Based News Scraper

Scrapes news from websites that don't have RSS feeds.
Uses Playwright for JavaScript-rendered content.
"""

import logging
from dataclasses import dataclass

from playwright.async_api import Browser, async_playwright

from news_pipeline.scrapers.base import (
    NewsItem,
    NewsScraper,
    SourceType,
)

logger = logging.getLogger(__name__)


@dataclass
class ScrapingTarget:
    """Configuration for scraping a news website."""

    name: str
    url: str
    source_type: SourceType
    credibility: float

    # CSS selectors
    headline_selector: str
    link_selector: str | None = None
    snippet_selector: str | None = None
    time_selector: str | None = None

    # Pagination
    load_more_selector: str | None = None
    max_pages: int = 1


# Pre-configured scraping targets
SCRAPING_TARGETS = [
    ScrapingTarget(
        name="Finviz News",
        url="https://finviz.com/news.ashx",
        source_type=SourceType.AGGREGATOR,
        credibility=0.7,
        headline_selector=".nn-tab-link",
        link_selector=".nn-tab-link",
    ),
    ScrapingTarget(
        name="Benzinga",
        url="https://www.benzinga.com/news",
        source_type=SourceType.MAINSTREAM,
        credibility=0.7,
        headline_selector=".content-headline a",
        link_selector=".content-headline a",
    ),
    ScrapingTarget(
        name="Investing.com News",
        url="https://www.investing.com/news/stock-market-news",
        source_type=SourceType.AGGREGATOR,
        credibility=0.6,
        headline_selector=".textDiv a",
        link_selector=".textDiv a",
        snippet_selector=".textDiv p",
    ),
]


class BrowserNewsScraper(NewsScraper):
    """
    Scrapes news from websites using browser automation.

    For sites without RSS that require JavaScript rendering.
    More resource-intensive but catches more sources.
    """

    SOURCE_NAME = "Browser Scraper"
    SOURCE_TYPE = SourceType.AGGREGATOR

    def __init__(
        self,
        targets: list[ScrapingTarget] | None = None,
        headless: bool = True,
    ):
        """
        Initialize browser scraper.

        Args:
            targets: Custom scraping targets (uses defaults if None)
            headless: Run browser without GUI
        """
        self.targets = targets or SCRAPING_TARGETS
        self.headless = headless
        self._playwright = None
        self._browser: Browser | None = None

    async def initialize(self) -> None:
        """Initialize browser instance."""
        if self._browser:
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
        )
        logger.info("[BrowserScraper] Initialized")

    async def close(self) -> None:
        """Close browser instance."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._playwright = None

    async def fetch(
        self,
        categories: list[str] | None = None,
        tickers: list[str] | None = None,
        max_items: int = 50,
        hours_back: int = 12,
    ) -> list[NewsItem]:
        """Fetch news from all configured targets."""
        await self.initialize()

        all_items = []

        # Scrape each target
        for target in self.targets:
            try:
                items = await self._scrape_target(target)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"[BrowserScraper] {target.name} failed: {e}")

        # Filter by tickers if specified
        if tickers:
            tickers_upper = [t.upper() for t in tickers]
            all_items = [
                item
                for item in all_items
                if any(t in tickers_upper for t in item.tickers)
                or any(t in item.headline.upper() for t in tickers_upper)
            ]

        # Sort and limit
        all_items.sort(
            key=lambda x: x.published_at or x.scraped_at,
            reverse=True,
        )

        logger.info(f"[BrowserScraper] Fetched {len(all_items[:max_items])} items")
        return all_items[:max_items]

    async def _scrape_target(self, target: ScrapingTarget) -> list[NewsItem]:
        """Scrape a single target website."""
        items = []

        page = await self._browser.new_page()

        try:
            await page.goto(target.url, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=10000)

            # Find all headlines
            headline_elements = await page.query_selector_all(target.headline_selector)

            for elem in headline_elements[:30]:  # Limit per target
                try:
                    headline = await elem.inner_text()
                    headline = headline.strip()

                    if not headline or len(headline) < 10:
                        continue

                    # Get link
                    link = ""
                    if target.link_selector:
                        link_elem = (
                            elem
                            if target.link_selector == target.headline_selector
                            else await elem.query_selector(target.link_selector)
                        )
                        if link_elem:
                            link = await link_elem.get_attribute("href") or ""

                    # Make absolute URL
                    if link and not link.startswith("http"):
                        from urllib.parse import urljoin

                        link = urljoin(target.url, link)

                    items.append(
                        NewsItem(
                            headline=headline,
                            source_name=target.name,
                            source_url=link or target.url,
                            source_type=target.source_type,
                            category=self.categorize_headline(headline),
                            tickers=self.extract_tickers(headline),
                            credibility_score=target.credibility,
                        )
                    )

                except Exception as e:
                    logger.debug(f"[BrowserScraper] Element parse error: {e}")
                    continue

            logger.debug(f"[BrowserScraper] {target.name}: {len(items)} items")

        except Exception as e:
            logger.warning(f"[BrowserScraper] {target.name}: {e}")

        finally:
            await page.close()

        return items
