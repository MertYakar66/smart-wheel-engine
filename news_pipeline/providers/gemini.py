"""
Gemini Provider - Verification Layer

Uses Google's Gemini API with Grounding (Google Search) for cross-source verification.
Gemini's grounding capability allows it to verify facts against live web search results.
"""

import json
import logging
import os
from typing import Any

from .base import VerificationProvider

logger = logging.getLogger(__name__)


class GeminiProvider(VerificationProvider):
    """
    Gemini-powered verification provider.

    Uses Google's Gemini model with grounding to verify news stories
    by cross-checking them against multiple web sources.
    """

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        self.model = "gemini-2.0-flash"  # Fast model with grounding support

    @property
    def name(self) -> str:
        return "Gemini"

    async def initialize(self) -> None:
        """Initialize the Google Generative AI client."""
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.client = genai
            self._initialized = True
            logger.info(f"[{self.name}] Initialized with model {self.model}")
        except ImportError:
            raise ImportError("google-generativeai package required for Gemini provider")

    async def health_check(self) -> bool:
        """Check if Gemini API is available."""
        if not self.client:
            return False

        try:
            model = self.client.GenerativeModel(self.model)
            response = model.generate_content("ping")
            return response.text is not None
        except Exception as e:
            self._log_error("health_check", e)
            return False

    async def verify_story(
        self,
        headline: str,
        source_url: str,
        tickers: list[str],
        category: str,
    ) -> dict[str, Any]:
        """
        Verify a story using Gemini with Google Search grounding.

        Args:
            headline: Story headline to verify
            source_url: Original source URL
            tickers: Related tickers
            category: Story category

        Returns:
            Verification result with confidence and evidence
        """
        if not self._initialized:
            await self.initialize()

        self._log_request(
            "verify_story",
            {
                "headline": headline[:100],
                "tickers": tickers,
            },
        )

        prompt = self._build_verification_prompt(headline, source_url, tickers, category)

        try:
            # Configure grounding with Google Search
            model = self.client.GenerativeModel(
                self.model,
                tools="google_search_retrieval",  # Enable web search grounding
            )

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,  # Low temperature for factual verification
                    "max_output_tokens": 2000,
                },
            )

            result = self._parse_verification_response(response.text)
            self._log_response("verify_story", f"Confidence: {result.get('confidence', 0)}/10")
            return result

        except Exception as e:
            self._log_error("verify_story", e)
            # Return unverified result on error
            return {
                "status": "error",
                "confidence": 0,
                "verified_facts": [],
                "contradictions": [],
                "evidence": [],
                "error": str(e),
            }

    def _build_verification_prompt(
        self,
        headline: str,
        source_url: str,
        tickers: list[str],
        category: str,
    ) -> str:
        """Build the verification prompt."""
        ticker_str = ", ".join(tickers) if tickers else "general market"

        return f"""You are a financial news verification agent. Your task is to verify the following news story by checking it against multiple web sources.

STORY TO VERIFY:
Headline: {headline}
Original Source: {source_url}
Related Assets: {ticker_str}
Category: {category}

VERIFICATION REQUIREMENTS:
1. Search for corroborating reports from other sources
2. Check if the facts are consistent across sources
3. Look for official or primary source confirmation
4. Identify any contradictions between sources
5. Assess the credibility of the original claim

Return your verification as a JSON object with these fields:
- status: "verified" | "partial" | "unverified" | "contradicted"
- confidence: integer 0-10 (0=no evidence, 10=multiple official confirmations)
- verified_facts: array of specific facts that were confirmed
- contradictions: array of contradicting information found
- evidence: array of objects, each with:
  - source_name: name of corroborating source
  - source_url: URL of the source
  - evidence_type: "corroboration" | "official" | "contradiction"
  - summary: brief description of what this source says
  - weight: float 0-1 indicating evidence strength
- verification_notes: string explaining your confidence assessment

CONFIDENCE SCALE:
0-2: No corroboration found, cannot verify
3-4: Weak corroboration, single secondary source
5-6: Moderate corroboration, multiple sources but no official confirmation
7-8: Strong corroboration, multiple sources including reputable outlets
9-10: Very strong, official/primary source confirmation available

Return ONLY the JSON object, no other text."""

    def _parse_verification_response(self, content: str) -> dict[str, Any]:
        """Parse Gemini's verification response."""
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
                "status": result.get("status", "unverified"),
                "confidence": min(10, max(0, int(result.get("confidence", 0)))),
                "verified_facts": result.get("verified_facts", []),
                "contradictions": result.get("contradictions", []),
                "evidence": result.get("evidence", []),
                "verification_notes": result.get("verification_notes", ""),
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.debug(f"Raw response: {content[:500]}")
            return {
                "status": "error",
                "confidence": 0,
                "verified_facts": [],
                "contradictions": [],
                "evidence": [],
                "verification_notes": f"Parse error: {e}",
            }
