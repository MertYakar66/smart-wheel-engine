"""
Claude Provider - Editorial Layer

Uses Anthropic's Claude for final editorial polish and "why it matters" generation.
Claude excels at nuanced analysis and clear, professional writing.
"""

import json
import logging
import os
from typing import Any

from .base import EditorialProvider

logger = logging.getLogger(__name__)


class ClaudeProvider(EditorialProvider):
    """
    Claude-powered editorial provider.

    Uses Anthropic's Claude model for final editorial polish,
    adding "why it matters" context and refining the language.
    """

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        self.model = "claude-sonnet-4-20250514"  # Latest Claude model

    @property
    def name(self) -> str:
        return "Claude"

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        try:
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=self.api_key)
            self._initialized = True
            logger.info(f"[{self.name}] Initialized with model {self.model}")
        except ImportError:
            raise ImportError("anthropic package required for Claude provider")

    async def health_check(self) -> bool:
        """Check if Anthropic API is available."""
        if not self.client:
            return False

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return response.content[0].text is not None
        except Exception as e:
            self._log_error("health_check", e)
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
    ) -> dict[str, Any]:
        """
        Finalize a story with editorial polish.

        Args:
            story_id: Story identifier
            title: Draft title
            what_happened: Draft summary
            bullet_points: Draft bullet points
            affected_assets: List of affected assets
            category: Story category
            confidence: Verification confidence

        Returns:
            Finalized story with "why it matters" section
        """
        if not self._initialized:
            await self.initialize()

        self._log_request(
            "finalize_story",
            {
                "story_id": story_id,
                "title": title[:50],
            },
        )

        prompt = self._build_editorial_prompt(
            title, what_happened, bullet_points, affected_assets, category, confidence
        )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            result = self._parse_editorial_response(content, story_id, confidence)
            self._log_response("finalize_story", f"Final title: {result.get('title', '')[:50]}")
            return result

        except Exception as e:
            self._log_error("finalize_story", e)
            raise

    def _get_system_prompt(self) -> str:
        """System prompt for editorial refinement."""
        return """You are a senior financial editor. Your task is to refine news items for a professional trading intelligence feed.

Your editorial responsibilities:
1. Tighten the language for maximum clarity
2. Add a "why it matters" section explaining market implications
3. Ensure professional, objective tone
4. Improve the headline if needed
5. Refine bullet points for clarity and impact

The "why it matters" section should:
- Explain the mechanism by which this affects markets
- Identify potential second-order effects
- Note relevant context (sector trends, macro backdrop)
- Be 2-3 sentences maximum
- Avoid speculation about price movements

Output your finalized story as a JSON object with these fields:
- title: The refined headline (max 80 characters)
- what_happened: Refined summary (2-3 sentences)
- why_it_matters: Market implications (2-3 sentences)
- bullet_points: Array of 3-5 refined takeaways
- affected_assets: Array of primary affected tickers/assets

Return ONLY the JSON object."""

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
        bullets_str = "\n".join(f"- {bp}" for bp in bullet_points)
        assets_str = ", ".join(affected_assets) if affected_assets else "general market"

        return f"""Refine this news item for our trading intelligence feed.

DRAFT CONTENT:
Title: {title}

What Happened:
{what_happened}

Key Points:
{bullets_str}

CONTEXT:
- Category: {category}
- Affected Assets: {assets_str}
- Verification Confidence: {confidence}/10

Please:
1. Refine the title for clarity and impact
2. Tighten the "what happened" summary
3. Add a "why it matters" section explaining market implications
4. Polish the bullet points
5. Confirm or adjust the affected assets list

Return the finalized story as a JSON object."""

    def _parse_editorial_response(
        self,
        content: str,
        story_id: str,
        confidence: int,
    ) -> dict[str, Any]:
        """Parse Claude's editorial response."""
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
                "why_it_matters": result.get("why_it_matters", ""),
                "bullet_points": result.get("bullet_points", []),
                "affected_assets": result.get("affected_assets", []),
                "verification_confidence": confidence,
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.debug(f"Raw response: {content[:500]}")
            return {
                "story_id": story_id,
                "title": "Parse Error",
                "what_happened": "",
                "why_it_matters": "",
                "bullet_points": [],
                "affected_assets": [],
                "verification_confidence": confidence,
            }
