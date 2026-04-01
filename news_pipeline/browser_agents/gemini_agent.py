"""
Gemini Browser Agent

Interacts with gemini.google.com via browser automation.
Uses your paid Gemini Advanced subscription - no API costs.

Features:
- Persistent Google login state
- Handles streaming responses
- Google Search grounding
- Automatic retry on rate limits
"""

import asyncio
import logging

from news_pipeline.browser_agents.base import (
    BrowserModelSession,
    ModelType,
)

logger = logging.getLogger(__name__)


class GeminiBrowserAgent(BrowserModelSession):
    """
    Browser automation for gemini.google.com.

    Uses Gemini Advanced subscription via browser.
    Leverages Google's search grounding for verification.
    """

    MODEL_TYPE = ModelType.GEMINI
    BASE_URL = "https://gemini.google.com/app"
    RESPONSE_TIMEOUT = 120000

    # Selectors for Gemini interface
    SELECTORS = {
        # Input area
        "prompt_input": 'div[contenteditable="true"].ql-editor',
        "prompt_input_alt": 'textarea[aria-label*="Enter"]',
        "send_button": 'button[aria-label="Send message"]',
        # Response area
        "response_container": ".response-container",
        "response_text": ".model-response-text",
        "streaming_indicator": ".loading-indicator",
        # Navigation
        "new_chat_button": 'button[aria-label="New chat"]',
        # Auth indicators
        "user_avatar": 'img[aria-label*="Account"]',
        "sign_in_button": 'a[href*="accounts.google.com"]',
    }

    async def _check_authenticated(self) -> bool:
        """Check if user is logged into Gemini."""
        try:
            # Check for Google account avatar
            avatar = await self._page.query_selector(self.SELECTORS["user_avatar"])
            if avatar:
                return True

            # Check URL for sign-in
            if "accounts.google.com" in self._page.url:
                return False

            # Look for sign in button
            sign_in = await self._page.query_selector(self.SELECTORS["sign_in_button"])
            if sign_in:
                return False

            return "gemini.google.com" in self._page.url

        except Exception as e:
            logger.warning(f"[Gemini] Auth check error: {e}")
            return False

    async def _submit_prompt(self, prompt: str) -> None:
        """Submit prompt to Gemini."""
        # Try main input selector
        input_field = await self._page.query_selector(self.SELECTORS["prompt_input"])

        if not input_field:
            input_field = await self._page.query_selector(self.SELECTORS["prompt_input_alt"])

        if not input_field:
            raise Exception("Could not find Gemini input field")

        # Clear and type prompt
        await input_field.click()
        await input_field.fill(prompt)

        await asyncio.sleep(0.5)

        # Click send or press Enter
        send_button = await self._page.query_selector(self.SELECTORS["send_button"])
        if send_button:
            await send_button.click()
        else:
            await self._page.keyboard.press("Enter")

        await asyncio.sleep(1)

    async def _wait_for_response(self, timeout: int) -> str:
        """Wait for Gemini's response to complete."""
        start_time = asyncio.get_event_loop().time()
        timeout_sec = timeout / 1000

        last_content = ""
        stable_count = 0

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_sec:
                raise TimeoutError(f"Response timeout after {timeout_sec}s")

            # Check if still loading
            loading = await self._page.query_selector(self.SELECTORS["streaming_indicator"])

            # Get current response
            content = await self._extract_response()

            if not loading:
                if content == last_content and content:
                    stable_count += 1
                    if stable_count >= 3:
                        return content
                else:
                    stable_count = 0
                    last_content = content

            await asyncio.sleep(0.5)

    async def _extract_response(self) -> str:
        """Extract response text from Gemini's interface."""
        try:
            # Find response containers
            responses = await self._page.query_selector_all(self.SELECTORS["response_text"])

            if not responses:
                responses = await self._page.query_selector_all(
                    self.SELECTORS["response_container"]
                )

            if not responses:
                return ""

            # Get the last response
            last_response = responses[-1]
            text = await last_response.inner_text()

            return text.strip()

        except Exception as e:
            logger.warning(f"[Gemini] Extract error: {e}")
            return ""

    async def _start_new_conversation(self) -> None:
        """Start a new Gemini conversation."""
        new_chat = await self._page.query_selector(self.SELECTORS["new_chat_button"])

        if new_chat:
            await new_chat.click()
        else:
            await self._page.goto(self.BASE_URL, timeout=self.PAGE_LOAD_TIMEOUT)

        await asyncio.sleep(2)

    async def verify_with_search(
        self,
        headline: str,
        source: str,
        tickers: list[str],
        category: str,
    ) -> dict:
        """
        Verify news using Gemini's Google Search grounding.

        Gemini Advanced has built-in search capability.
        """
        prompt = f"""TASK: Verify this financial news headline. Use Google Search to find corroborating sources.

HEADLINE: "{headline}"
ORIGINAL SOURCE: {source}
TICKERS: {", ".join(tickers) if tickers else "None"}
CATEGORY: {category}

VERIFICATION STEPS:
1. Search Google for this news story
2. Find official sources (company sites, SEC, Fed)
3. Cross-reference with major news outlets
4. Note any contradictions

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "status": "verified" or "partial" or "unverified" or "contradicted",
    "confidence": 0-10,
    "verified_facts": ["fact 1", "fact 2"],
    "what_happened": "factual summary",
    "sources_found": ["source 1", "source 2"],
    "contradictions": [],
    "verification_notes": "explanation"
}}

Return ONLY the JSON, no other text."""

        response = await self.send_prompt_with_retry(prompt)

        if not response.success:
            return {
                "status": "error",
                "confidence": 0,
                "error": response.error,
            }

        return self._parse_json_response(response.content)

    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from model response."""
        import json

        try:
            content = content.strip()

            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"[Gemini] JSON parse error: {e}")
            return {"error": f"JSON parse failed: {e}", "raw": content}
