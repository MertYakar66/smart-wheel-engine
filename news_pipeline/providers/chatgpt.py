"""
ChatGPT Formatting Provider

Uses OpenAI's GPT-4 for clear, structured story formatting.
ChatGPT excels at transforming raw facts into readable content.

Features:
- Clean, professional formatting
- Bullet point extraction
- Asset impact analysis
- Time sensitivity classification
"""

import json
import logging

from news_pipeline.config import ProviderConfig
from news_pipeline.providers.base import FormattingProvider

logger = logging.getLogger(__name__)


class ChatGPTProvider(FormattingProvider):
    """
    ChatGPT-powered story formatting.

    Uses OpenAI's GPT-4 to transform verified facts into
    clear, structured, professionally formatted content.
    """

    def __init__(self, config: ProviderConfig | None = None):
        """Initialize ChatGPT provider."""
        if config is None:
            from news_pipeline.config import PipelineConfig

            config = PipelineConfig().chatgpt
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.config.api_key)
            self._initialized = True
            logger.info(f"[{self.name}] Initialized successfully")

        except ImportError:
            logger.error(f"[{self.name}] openai package required")
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Initialization failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self._initialized:
            return False

        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            logger.warning(f"[{self.name}] Health check failed: {e}")
            return False

    async def format_story(
        self,
        story_id: str,
        verified_facts: list[str],
        what_happened: str,
        affected_assets: list[str],
        category: str,
    ) -> dict:
        """
        Format verified story into structured content.

        Args:
            story_id: Unique story identifier
            verified_facts: List of verified facts
            what_happened: Summary of what occurred
            affected_assets: Related tickers/assets
            category: Story category

        Returns:
            Formatted story dictionary
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"[{self.name}] Formatting story {story_id}")

        prompt = self._build_formatting_prompt(
            verified_facts, what_happened, affected_assets, category
        )

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
                temperature=0.4,
                max_tokens=1500,
            )

            content = response.choices[0].message.content
            result = self._parse_formatting_response(content)

            logger.info(f"[{self.name}] Formatted: {result.get('title', 'unknown')[:50]}")
            return result

        except Exception as e:
            logger.error(f"[{self.name}] Formatting failed: {e}")
            return self._fallback_format(verified_facts, what_happened, affected_assets, category)

    def _get_system_prompt(self) -> str:
        """Get system prompt for ChatGPT."""
        return """You are a financial news editor. Your task is to transform
verified facts into clear, professional news content.

STYLE GUIDELINES:
- Write in active voice, present tense for recent events
- Be concise and factual - no speculation
- Use financial terminology appropriately
- Headlines should be informative, not clickbait
- Bullet points should be scannable and actionable

OUTPUT QUALITY:
- Title: Clear, informative (under 80 characters)
- What happened: 1-2 sentences, factual summary
- Bullet points: 3-5 key points, each under 100 characters
- Affected assets: Normalized ticker symbols

Maintain professional financial journalism standards."""

    def _build_formatting_prompt(
        self,
        verified_facts: list[str],
        what_happened: str,
        affected_assets: list[str],
        category: str,
    ) -> str:
        """Build the formatting prompt."""
        facts_text = "\n".join(f"- {fact}" for fact in verified_facts)

        return f"""Format this verified financial news into structured content:

VERIFIED FACTS:
{facts_text}

SUMMARY: {what_happened}

AFFECTED ASSETS: {", ".join(affected_assets)}

CATEGORY: {category}

RETURN JSON FORMAT:
{{
    "title": "Clear, informative headline",
    "what_happened": "1-2 sentence factual summary",
    "bullet_points": [
        "Key point 1",
        "Key point 2",
        "Key point 3"
    ],
    "affected_assets": ["TICKER1", "TICKER2"],
    "related_tickers": ["other related tickers"],
    "sector_impact": "brief sector analysis or null",
    "time_sensitivity": "urgent" | "normal" | "background"
}}

TIME SENSITIVITY:
- urgent: Breaking news, immediate market impact
- normal: Important but not time-critical
- background: Context/analysis, no immediate action needed"""

    def _parse_formatting_response(self, content: str) -> dict:
        """Parse ChatGPT's formatting response."""
        if not content:
            return self._empty_result()

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

            # Validate required fields
            result.setdefault("title", "Untitled Story")
            result.setdefault("what_happened", "")
            result.setdefault("bullet_points", [])
            result.setdefault("affected_assets", [])
            result.setdefault("related_tickers", [])
            result.setdefault("sector_impact", None)
            result.setdefault("time_sensitivity", "normal")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[{self.name}] JSON parse failed: {e}")
            return self._empty_result()

    def _fallback_format(
        self,
        verified_facts: list[str],
        what_happened: str,
        affected_assets: list[str],
        category: str,
    ) -> dict:
        """Fallback formatting when API fails."""
        return {
            "title": what_happened[:80] if what_happened else "Breaking News",
            "what_happened": what_happened,
            "bullet_points": verified_facts[:5],
            "affected_assets": affected_assets,
            "related_tickers": [],
            "sector_impact": None,
            "time_sensitivity": "normal",
        }

    def _empty_result(self) -> dict:
        """Return empty result structure."""
        return {
            "title": "",
            "what_happened": "",
            "bullet_points": [],
            "affected_assets": [],
            "related_tickers": [],
            "sector_impact": None,
            "time_sensitivity": "normal",
        }
