"""
Grok Discovery Provider

Uses xAI's Grok model for real-time financial news discovery.
Grok excels at breaking news detection through live web search.

Features:
- Real-time web search for financial news
- Breaking news detection
- Source credibility assessment
- Ticker and category filtering
"""

import json
import logging

from news_pipeline.config import ProviderConfig
from news_pipeline.providers.base import DiscoveryProvider

logger = logging.getLogger(__name__)


class GrokProvider(DiscoveryProvider):
    """
    Grok-powered news discovery.

    Uses xAI's Grok model with web search capabilities to find
    breaking financial news in real-time.
    """

    def __init__(self, config: ProviderConfig | None = None):
        """Initialize Grok provider."""
        if config is None:
            from news_pipeline.config import PipelineConfig

            config = PipelineConfig().grok
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize the xAI client."""
        if self._initialized:
            return

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            self._initialized = True
            logger.info(f"[{self.name}] Initialized successfully")

        except ImportError:
            logger.error(f"[{self.name}] openai package required")
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Initialization failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if Grok API is accessible."""
        if not self._initialized:
            return False

        try:
            # Simple completion to verify connectivity
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            logger.warning(f"[{self.name}] Health check failed: {e}")
            return False

    async def discover_news(
        self,
        tickers: list[str],
        categories: list[str],
        time_window: str,
        max_results: int = 50,
    ) -> list[dict]:
        """
        Discover financial news via Grok web search.

        Args:
            tickers: Stock symbols to monitor
            categories: News categories to include
            time_window: Time range for news
            max_results: Maximum stories to return

        Returns:
            List of candidate story dictionaries
        """
        if not self._initialized:
            await self.initialize()

        logger.info(
            f"[{self.name}] Discovering news: "
            f"tickers={tickers}, categories={categories}, window={time_window}"
        )

        # Build search prompt
        prompt = self._build_discovery_prompt(tickers, categories, time_window, max_results)

        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
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
                temperature=0.3,
                max_tokens=4000,
            )

            content = response.choices[0].message.content
            stories = self._parse_discovery_response(content)

            logger.info(f"[{self.name}] Discovered {len(stories)} candidates")
            return stories[:max_results]

        except Exception as e:
            logger.error(f"[{self.name}] Discovery failed: {e}")
            return []

    def _get_system_prompt(self) -> str:
        """Get system prompt for Grok."""
        return """You are a financial news discovery agent. Your task is to search the web
for breaking financial news and market-moving events.

SEARCH PRIORITIES:
1. Official sources (Fed, SEC, company press releases)
2. Major financial news outlets (Bloomberg, Reuters, WSJ, CNBC)
3. Verified social media from official accounts

OUTPUT FORMAT:
Return a JSON array of news stories. Each story must have:
- headline: The main headline
- source_name: Name of the source
- source_url: URL to the article
- published_at: ISO timestamp if available
- tickers: Array of related stock symbols
- category: Category (fed, earnings, oil, geopolitics, etc.)
- snippet: 1-2 sentence summary
- source_type: "official", "mainstream", "social", or "unknown"
- relevance_score: 0.0-1.0 based on market impact potential

Only include stories that are:
- Recent and relevant to the time window
- From credible sources
- Related to financial markets or specified tickers
- Factual (not opinion pieces)"""

    def _build_discovery_prompt(
        self,
        tickers: list[str],
        categories: list[str],
        time_window: str,
        max_results: int,
    ) -> str:
        """Build the discovery prompt."""
        # Map time windows to descriptions
        window_map = {
            "overnight": "the last 12 hours (overnight session)",
            "last_1h": "the last hour",
            "last_3h": "the last 3 hours",
            "last_6h": "the last 6 hours",
            "today": "today's trading session",
            "yesterday": "yesterday",
        }
        time_desc = window_map.get(time_window, time_window)

        # Build category descriptions
        category_map = {
            "fed": "Federal Reserve policy, FOMC, interest rates",
            "earnings": "Corporate earnings, guidance, quarterly reports",
            "sp500_events": "S&P 500 companies major announcements",
            "oil": "Oil prices, OPEC, energy markets",
            "geopolitics": "Geopolitical events affecting markets",
            "crypto": "Cryptocurrency, Bitcoin, digital assets",
            "macro": "Macroeconomic data, GDP, employment",
        }
        category_descs = [category_map.get(c, c) for c in categories]

        prompt = f"""Search for breaking financial news from {time_desc}.

CATEGORIES TO MONITOR:
{chr(10).join(f"- {c}" for c in category_descs)}
"""

        if tickers:
            prompt += f"""
PRIORITY TICKERS:
{", ".join(tickers)}
"""

        prompt += f"""
Return up to {max_results} stories as a JSON array.
Focus on market-moving news that traders need to know.
Prioritize official sources and breaking developments."""

        return prompt

    def _parse_discovery_response(self, content: str) -> list[dict]:
        """Parse Grok's response into story dictionaries."""
        if not content:
            return []

        try:
            # Try to extract JSON from response
            content = content.strip()

            # Handle markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            stories = json.loads(content)

            if not isinstance(stories, list):
                stories = [stories]

            # Validate and clean stories
            valid_stories = []
            for story in stories:
                if self._validate_story(story):
                    # Ensure required fields
                    story.setdefault("tickers", [])
                    story.setdefault("category", "other")
                    story.setdefault("source_type", "unknown")
                    story.setdefault("relevance_score", 0.5)
                    valid_stories.append(story)

            return valid_stories

        except json.JSONDecodeError as e:
            logger.warning(f"[{self.name}] JSON parse failed: {e}")
            return []

    def _validate_story(self, story: dict) -> bool:
        """Validate a story dictionary has required fields."""
        required = ["headline", "source_name", "source_url"]
        return all(story.get(field) for field in required)
