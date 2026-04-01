"""
Claude Editorial Provider

Uses Anthropic's Claude for final editorial polish.
Claude excels at nuanced analysis and the "why it matters" perspective.

Features:
- Editorial polish and refinement
- "Why it matters" market analysis
- Trading considerations
- Priority and breaking news classification
"""

import json
import logging

from news_pipeline.config import ProviderConfig
from news_pipeline.providers.base import EditorialProvider

logger = logging.getLogger(__name__)


class ClaudeProvider(EditorialProvider):
    """
    Claude-powered editorial polish.

    Uses Anthropic's Claude to add the final editorial layer,
    including "why it matters" analysis and market implications.
    """

    def __init__(self, config: ProviderConfig | None = None):
        """Initialize Claude provider."""
        if config is None:
            from news_pipeline.config import PipelineConfig

            config = PipelineConfig().claude
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if self._initialized:
            return

        try:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.config.api_key)
            self._initialized = True
            logger.info(f"[{self.name}] Initialized successfully")

        except ImportError:
            logger.error(f"[{self.name}] anthropic package required")
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Initialization failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        if not self._initialized:
            return False

        try:
            response = await self._client.messages.create(
                model=self.config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return len(response.content) > 0
        except Exception as e:
            logger.warning(f"[{self.name}] Health check failed: {e}")
            return False

    async def finalize_story(
        self,
        story_id: str,
        title: str,
        what_happened: str,
        bullet_points: list[str],
        affected_assets: list[str],
        category: str,
        confidence: int,
    ) -> dict:
        """
        Finalize story with editorial polish.

        Args:
            story_id: Unique story identifier
            title: Story title
            what_happened: Summary of what occurred
            bullet_points: Key points
            affected_assets: Related tickers/assets
            category: Story category
            confidence: Verification confidence (0-10)

        Returns:
            Finalized story with "why it matters"
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"[{self.name}] Finalizing story {story_id}")

        prompt = self._build_editorial_prompt(
            title, what_happened, bullet_points, affected_assets, category, confidence
        )

        try:
            response = await self._client.messages.create(
                model=self.config.model,
                max_tokens=2000,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            result = self._parse_editorial_response(content)

            # Preserve original data if not overridden
            result.setdefault("title", title)
            result.setdefault("what_happened", what_happened)
            result.setdefault("bullet_points", bullet_points)
            result.setdefault("affected_assets", affected_assets)

            logger.info(f"[{self.name}] Finalized: priority={result.get('priority', 5)}")
            return result

        except Exception as e:
            logger.error(f"[{self.name}] Finalization failed: {e}")
            return self._fallback_finalize(
                title, what_happened, bullet_points, affected_assets, category
            )

    def _get_system_prompt(self) -> str:
        """Get system prompt for Claude."""
        return """You are a senior financial news editor at a major publication.
Your role is to add the final editorial polish to verified financial news stories.

YOUR EXPERTISE:
- Deep understanding of financial markets
- Ability to connect news to broader market implications
- Clear, authoritative writing style
- Trader-focused perspective

KEY RESPONSIBILITIES:
1. Write compelling "why it matters" analysis
2. Identify market implications and trading considerations
3. Assess story priority and urgency
4. Ensure professional quality

TONE:
- Authoritative but accessible
- Factual, not sensational
- Forward-looking where appropriate
- Trader-focused (what does this mean for positions?)

Always maintain journalistic integrity and never speculate beyond the facts."""

    def _build_editorial_prompt(
        self,
        title: str,
        what_happened: str,
        bullet_points: list[str],
        affected_assets: list[str],
        category: str,
        confidence: int,
    ) -> str:
        """Build the editorial prompt."""
        bullets_text = "\n".join(f"- {bp}" for bp in bullet_points)

        return f"""Add editorial polish to this verified financial news story:

TITLE: {title}

WHAT HAPPENED:
{what_happened}

KEY POINTS:
{bullets_text}

AFFECTED ASSETS: {", ".join(affected_assets)}
CATEGORY: {category}
VERIFICATION CONFIDENCE: {confidence}/10

TASKS:
1. Polish the title if needed (keep under 80 chars)
2. Write "why it matters" (2-3 sentences explaining market significance)
3. Add market implications (how might markets react?)
4. Include trading considerations (what should traders watch for?)
5. Assess priority (1-10) and if this is breaking news
6. Suggest relevant tags

RETURN JSON FORMAT:
{{
    "title": "polished title",
    "what_happened": "refined summary",
    "why_it_matters": "2-3 sentences on market significance",
    "bullet_points": ["refined point 1", "refined point 2"],
    "affected_assets": ["TICKER1", "TICKER2"],
    "market_implications": "how markets might react",
    "trading_considerations": "what traders should watch",
    "is_breaking": true/false,
    "priority": 1-10,
    "tags": ["tag1", "tag2"]
}}

PRIORITY SCALE:
- 9-10: Major market-moving event (Fed decision, major earnings miss)
- 7-8: Significant news (sector-moving, notable company event)
- 5-6: Relevant news (worth knowing, limited immediate impact)
- 3-4: Background context
- 1-2: Minor update"""

    def _parse_editorial_response(self, content: str) -> dict:
        """Parse Claude's editorial response."""
        if not content:
            return {}

        try:
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

            result = json.loads(content)

            # Validate and normalize
            result.setdefault("why_it_matters", "")
            result.setdefault("market_implications", None)
            result.setdefault("trading_considerations", None)
            result.setdefault("is_breaking", False)
            result.setdefault("priority", 5)
            result.setdefault("tags", [])

            # Ensure priority is 1-10
            result["priority"] = max(1, min(10, int(result["priority"])))

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[{self.name}] JSON parse failed: {e}")
            return {}

    def _fallback_finalize(
        self,
        title: str,
        what_happened: str,
        bullet_points: list[str],
        affected_assets: list[str],
        category: str,
    ) -> dict:
        """Fallback finalization when API fails."""
        # Generate basic "why it matters" based on category
        category_implications = {
            "fed": "Federal Reserve actions directly impact interest rates and market liquidity.",
            "earnings": "Earnings results affect stock valuations and sector sentiment.",
            "oil": "Energy prices ripple through transportation, manufacturing, and inflation expectations.",
            "geopolitics": "Geopolitical developments create uncertainty and affect global trade.",
            "sp500_events": "S&P 500 company news affects index performance and related ETFs.",
        }

        return {
            "title": title,
            "what_happened": what_happened,
            "why_it_matters": category_implications.get(
                category, "This development may impact related securities and market sentiment."
            ),
            "bullet_points": bullet_points,
            "affected_assets": affected_assets,
            "market_implications": None,
            "trading_considerations": None,
            "is_breaking": False,
            "priority": 5,
            "tags": [category],
        }
