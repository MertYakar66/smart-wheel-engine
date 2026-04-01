"""
Grok Provider - Discovery Layer

Uses xAI's Grok API for web search and news discovery.
Grok excels at real-time web search which makes it ideal for news discovery.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

from .base import DiscoveryProvider

logger = logging.getLogger(__name__)

# xAI API endpoint
XAI_API_BASE = "https://api.x.ai/v1"


class GrokProvider(DiscoveryProvider):
    """
    Grok-powered news discovery provider.

    Uses xAI's Grok model with web search capabilities to find
    relevant financial news for specified tickers and categories.
    """

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        self.client = None
        self.model = "grok-2"  # Latest Grok model

    @property
    def name(self) -> str:
        return "Grok"

    async def initialize(self) -> None:
        """Initialize the xAI client."""
        if not self.api_key:
            raise ValueError("XAI_API_KEY not found in environment")

        try:
            # Use OpenAI-compatible client for xAI
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=XAI_API_BASE,
            )
            self._initialized = True
            logger.info(f"[{self.name}] Initialized with model {self.model}")
        except ImportError:
            raise ImportError("openai package required for Grok provider")

    async def health_check(self) -> bool:
        """Check if Grok API is available."""
        if not self.client:
            return False

        try:
            # Simple test call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            self._log_error("health_check", e)
            return False

    async def discover_news(
        self,
        tickers: list[str],
        categories: list[str],
        time_window: str,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Discover financial news using Grok's web search.

        Args:
            tickers: Stock tickers to monitor
            categories: News categories (sp500_events, oil, geopolitics, etc.)
            time_window: Time window for search
            max_results: Maximum stories to return

        Returns:
            List of candidate story dictionaries
        """
        if not self._initialized:
            await self.initialize()

        self._log_request(
            "discover_news",
            {
                "tickers": tickers,
                "categories": categories,
                "time_window": time_window,
            },
        )

        # Build the discovery prompt
        prompt = self._build_discovery_prompt(tickers, categories, time_window, max_results)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.3,  # Lower for more factual responses
                max_tokens=4000,
            )

            content = response.choices[0].message.content
            stories = self._parse_discovery_response(content)
            self._log_response("discover_news", f"Found {len(stories)} candidates")
            return stories

        except Exception as e:
            self._log_error("discover_news", e)
            raise

    def _get_system_prompt(self) -> str:
        """System prompt for Grok discovery."""
        return """You are a financial news discovery agent. Your task is to search the web for recent financial news stories that are relevant to specific stocks and market themes.

For each story you find, extract:
1. The exact headline
2. The source publication name
3. The source URL
4. The publication timestamp (if available)
5. Related stock tickers (if any)
6. The primary category it belongs to
7. A brief snippet or summary

Output your findings as a JSON array. Each story should be an object with these fields:
- headline: string
- source_name: string
- source_url: string
- published_at: ISO timestamp string or null
- tickers: array of ticker symbols
- category: string (one of: sp500_events, oil, geopolitics, fed, inflation, labor, earnings, macro, crypto, tech)
- snippet: string (2-3 sentences)

Focus on breaking news and material developments that would affect trading decisions. Prioritize:
- Earnings announcements and guidance changes
- Fed/central bank policy signals
- Macro data releases (CPI, NFP, GDP)
- Oil/energy supply disruptions
- Geopolitical events with market transmission
- Major M&A announcements
- Regulatory actions affecting specific stocks

Return ONLY the JSON array, no other text."""

    def _build_discovery_prompt(
        self,
        tickers: list[str],
        categories: list[str],
        time_window: str,
        max_results: int,
    ) -> str:
        """Build the user prompt for news discovery."""
        time_desc = self._time_window_to_description(time_window)

        ticker_str = ", ".join(tickers) if tickers else "major S&P 500 stocks"
        category_str = ", ".join(categories) if categories else "all market-relevant categories"

        return f"""Search for the most important financial news from {time_desc}.

Focus on:
- Tickers: {ticker_str}
- Categories: {category_str}

Find up to {max_results} relevant stories. Prioritize stories that:
1. Have clear market implications
2. Are from reputable financial sources
3. Contain specific facts (numbers, dates, names)
4. Are breaking or recently updated

Return the results as a JSON array."""

    def _time_window_to_description(self, time_window: str) -> str:
        """Convert time window to natural language."""
        mapping = {
            "overnight": "the past 12 hours (overnight through this morning)",
            "last_6h": "the past 6 hours",
            "last_3h": "the past 3 hours",
            "last_1h": "the past hour",
            "today": "today",
            "yesterday": "yesterday",
        }
        return mapping.get(time_window, f"the {time_window}")

    def _parse_discovery_response(self, content: str) -> list[dict[str, Any]]:
        """Parse Grok's response into story dictionaries."""
        try:
            # Try to extract JSON from the response
            content = content.strip()

            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            stories = json.loads(content.strip())

            if not isinstance(stories, list):
                logger.warning("Grok response was not a list, wrapping")
                stories = [stories]

            # Normalize each story
            normalized = []
            for story in stories:
                if isinstance(story, dict) and "headline" in story:
                    normalized.append(
                        {
                            "headline": story.get("headline", ""),
                            "source_name": story.get("source_name", "Unknown"),
                            "source_url": story.get("source_url", ""),
                            "published_at": story.get("published_at"),
                            "tickers": story.get("tickers", []),
                            "category": story.get("category", ""),
                            "snippet": story.get("snippet", ""),
                            "discovered_at": datetime.utcnow().isoformat(),
                        }
                    )

            return normalized

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Grok response as JSON: {e}")
            logger.debug(f"Raw response: {content[:500]}")
            return []
