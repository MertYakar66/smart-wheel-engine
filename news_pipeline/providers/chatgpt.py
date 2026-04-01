"""
ChatGPT Provider - Formatting Layer

Uses OpenAI's GPT models for structuring verified stories into feed format.
ChatGPT excels at transforming raw facts into clean, structured content.
"""

import json
import logging
import os
from typing import Any

from .base import FormattingProvider

logger = logging.getLogger(__name__)


class ChatGPTProvider(FormattingProvider):
    """
    ChatGPT-powered formatting provider.

    Uses OpenAI's GPT model to transform verified facts
    into a structured, feed-ready format.
    """

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        self.model = "gpt-4o"  # Latest GPT-4 model

    @property
    def name(self) -> str:
        return "ChatGPT"

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=self.api_key)
            self._initialized = True
            logger.info(f"[{self.name}] Initialized with model {self.model}")
        except ImportError:
            raise ImportError("openai package required for ChatGPT provider")

    async def health_check(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.client:
            return False

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            self._log_error("health_check", e)
            return False

    async def format_story(
        self,
        story_id: str,
        verified_facts: list[str],
        verification_confidence: int,
        affected_assets: list[str],
        category: str,
    ) -> dict[str, Any]:
        """
        Format verified facts into a structured story.

        Args:
            story_id: Story identifier
            verified_facts: List of verified facts
            verification_confidence: Confidence score (0-10)
            affected_assets: List of affected assets
            category: Story category

        Returns:
            Formatted story dictionary
        """
        if not self._initialized:
            await self.initialize()

        self._log_request(
            "format_story",
            {
                "story_id": story_id,
                "facts_count": len(verified_facts),
                "confidence": verification_confidence,
            },
        )

        prompt = self._build_formatting_prompt(
            verified_facts, verification_confidence, affected_assets, category
        )

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
                temperature=0.4,  # Slightly creative but still structured
                max_tokens=1500,
            )

            content = response.choices[0].message.content
            result = self._parse_formatting_response(content, story_id, verification_confidence)
            self._log_response("format_story", f"Title: {result.get('title', '')[:50]}")
            return result

        except Exception as e:
            self._log_error("format_story", e)
            raise

    def _get_system_prompt(self) -> str:
        """System prompt for formatting."""
        return """You are a financial news editor. Your task is to transform verified facts into a clean, structured news item for a professional trading feed.

Your output should be:
- Concise and factual
- Free of speculation or opinion
- Written for traders who need quick, actionable information
- Professional in tone

Output your formatted story as a JSON object with these fields:
- title: A clear, descriptive headline (max 80 characters)
- what_happened: A 2-3 sentence summary of the key facts
- bullet_points: An array of 3-5 key takeaways
- affected_assets: Array of tickers/assets most directly impacted

Focus on:
1. What specific event or announcement occurred
2. Who/what is involved
3. When it happened or was announced
4. What the immediate market implications might be

Do NOT include:
- Speculative price targets
- Your own trading recommendations
- Excessive hedging language
- Information not in the verified facts

Return ONLY the JSON object."""

    def _build_formatting_prompt(
        self,
        verified_facts: list[str],
        verification_confidence: int,
        affected_assets: list[str],
        category: str,
    ) -> str:
        """Build the formatting prompt."""
        facts_str = "\n".join(f"- {fact}" for fact in verified_facts)
        assets_str = ", ".join(affected_assets) if affected_assets else "general market"

        return f"""Format the following verified facts into a structured news item.

VERIFIED FACTS:
{facts_str}

CONTEXT:
- Verification Confidence: {verification_confidence}/10
- Category: {category}
- Affected Assets: {assets_str}

Transform these facts into a clean, professional news item for our trading feed.
Return the result as a JSON object."""

    def _parse_formatting_response(
        self,
        content: str,
        story_id: str,
        confidence: int,
    ) -> dict[str, Any]:
        """Parse ChatGPT's formatting response."""
        try:
            content = content.strip()

            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())

            # Normalize the result
            return {
                "story_id": story_id,
                "title": result.get("title", "Untitled Story"),
                "what_happened": result.get("what_happened", ""),
                "bullet_points": result.get("bullet_points", []),
                "affected_assets": result.get("affected_assets", []),
                "confidence": confidence,
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ChatGPT response as JSON: {e}")
            logger.debug(f"Raw response: {content[:500]}")
            return {
                "story_id": story_id,
                "title": "Parse Error",
                "what_happened": content[:200] if content else "",
                "bullet_points": [],
                "affected_assets": [],
                "confidence": confidence,
            }
