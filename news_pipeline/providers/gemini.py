"""
Gemini Verification Provider

Uses Google's Gemini model with grounding for fact verification.
Gemini's search grounding provides authoritative source verification.

Features:
- Cross-source verification with Google Search grounding
- Confidence scoring (0-10)
- Evidence collection
- Contradiction detection
"""

import json
import logging

from news_pipeline.config import ProviderConfig
from news_pipeline.providers.base import VerificationProvider

logger = logging.getLogger(__name__)


class GeminiProvider(VerificationProvider):
    """
    Gemini-powered story verification.

    Uses Google's Gemini model with search grounding to verify
    stories against multiple authoritative sources.
    """

    def __init__(self, config: ProviderConfig | None = None):
        """Initialize Gemini provider."""
        if config is None:
            from news_pipeline.config import PipelineConfig

            config = PipelineConfig().gemini
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize the Gemini client."""
        if self._initialized:
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.config.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_output_tokens": 2000,
                },
            )
            self._genai = genai
            self._initialized = True
            logger.info(f"[{self.name}] Initialized successfully")

        except ImportError:
            logger.error(f"[{self.name}] google-generativeai package required")
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Initialization failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        if not self._initialized:
            return False

        try:
            response = self._client.generate_content("ping")
            return response.text is not None
        except Exception as e:
            logger.warning(f"[{self.name}] Health check failed: {e}")
            return False

    async def verify_story(
        self,
        headline: str,
        source_url: str,
        tickers: list[str],
        category: str,
    ) -> dict:
        """
        Verify a story using Gemini with search grounding.

        Args:
            headline: Story headline to verify
            source_url: Original source URL
            tickers: Related stock symbols
            category: Story category

        Returns:
            Verification result dictionary
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"[{self.name}] Verifying: {headline[:60]}...")

        prompt = self._build_verification_prompt(headline, source_url, tickers, category)

        try:
            # Use grounding with Google Search for verification
            response = self._client.generate_content(
                prompt,
                tools=[self._genai.Tool(google_search=self._genai.GoogleSearch())],
            )

            result = self._parse_verification_response(response.text)

            logger.info(
                f"[{self.name}] Verification complete: confidence={result.get('confidence', 0)}/10"
            )
            return result

        except Exception as e:
            logger.error(f"[{self.name}] Verification failed: {e}")
            return {
                "status": "error",
                "confidence": 0,
                "verified_facts": [],
                "contradictions": [],
                "verification_notes": f"Verification failed: {e}",
            }

    def _build_verification_prompt(
        self,
        headline: str,
        source_url: str,
        tickers: list[str],
        category: str,
    ) -> str:
        """Build the verification prompt."""
        return f"""You are a financial news fact-checker. Verify this headline using search:

HEADLINE: "{headline}"
SOURCE: {source_url}
TICKERS: {", ".join(tickers) if tickers else "None specified"}
CATEGORY: {category}

VERIFICATION TASKS:
1. Search for corroborating reports from other credible sources
2. Check official sources (company websites, SEC filings, Fed statements)
3. Identify any contradicting information
4. Assess the credibility of sources found

RETURN JSON FORMAT:
{{
    "status": "verified" | "partial" | "unverified" | "contradicted",
    "confidence": 0-10,
    "verified_facts": ["fact 1", "fact 2"],
    "what_happened": "1-2 sentence factual summary",
    "corroborating_sources": ["source 1", "source 2"],
    "contradictions": ["any conflicting info found"],
    "sources_checked": number,
    "verification_notes": "brief explanation of verification process"
}}

CONFIDENCE SCALE:
- 9-10: Multiple official sources confirm
- 7-8: Major news outlets corroborate
- 5-6: Single credible source, no contradictions
- 3-4: Limited verification, some uncertainty
- 0-2: Cannot verify or found contradictions

Be rigorous. Only mark as "verified" with 7+ confidence if multiple credible sources confirm."""

    def _parse_verification_response(self, content: str) -> dict:
        """Parse Gemini's verification response."""
        if not content:
            return self._default_result()

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
            result.setdefault("status", "unverified")
            result.setdefault("confidence", 0)
            result.setdefault("verified_facts", [])
            result.setdefault("what_happened", "")
            result.setdefault("contradictions", [])
            result.setdefault("sources_checked", 0)
            result.setdefault("verification_notes", "")

            # Ensure confidence is integer 0-10
            result["confidence"] = max(0, min(10, int(result["confidence"])))

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[{self.name}] JSON parse failed: {e}")
            return self._default_result()

    def _default_result(self) -> dict:
        """Return default verification result on error."""
        return {
            "status": "error",
            "confidence": 0,
            "verified_facts": [],
            "what_happened": "",
            "contradictions": [],
            "sources_checked": 0,
            "verification_notes": "Failed to parse verification response",
        }
