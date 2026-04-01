"""
ChatGPT Browser Agent

Interacts with chat.openai.com via browser automation.
Uses your paid ChatGPT Plus subscription - no API costs.

Features:
- Persistent login state
- Handles streaming responses
- GPT-4 model selection
- Automatic retry on rate limits
"""

import asyncio
import logging

from news_pipeline.browser_agents.base import (
    BrowserModelSession,
    ModelType,
    SessionStatus,
)

logger = logging.getLogger(__name__)


class ChatGPTBrowserAgent(BrowserModelSession):
    """
    Browser automation for chat.openai.com.

    Uses ChatGPT Plus subscription via browser.
    """

    MODEL_TYPE = ModelType.CHATGPT
    BASE_URL = "https://chat.openai.com"
    RESPONSE_TIMEOUT = 120000  # 2 minutes

    # Selectors for ChatGPT interface
    SELECTORS = {
        # Input area
        "prompt_input": "#prompt-textarea",
        "send_button": 'button[data-testid="send-button"]',
        # Response area
        "response_container": 'div[data-message-author-role="assistant"]',
        "streaming_indicator": ".result-streaming",
        # Navigation
        "new_chat_button": 'a[href="/"]',
        "model_selector": 'button[aria-label*="Model"]',
        # Auth indicators
        "user_menu": 'button[aria-label*="Open user menu"]',
        "login_button": 'button:has-text("Log in")',
        # Rate limit
        "rate_limit_message": 'text="Too many requests"',
    }

    async def _check_authenticated(self) -> bool:
        """Check if user is logged into ChatGPT."""
        try:
            # Look for prompt input (indicates logged in and ready)
            prompt_input = await self._page.query_selector(self.SELECTORS["prompt_input"])
            if prompt_input:
                return True

            # Check URL for login page
            if "/auth/login" in self._page.url:
                return False

            # Look for login button
            login_btn = await self._page.query_selector(self.SELECTORS["login_button"])
            if login_btn:
                return False

            return "chat.openai.com" in self._page.url

        except Exception as e:
            logger.warning(f"[ChatGPT] Auth check error: {e}")
            return False

    async def _submit_prompt(self, prompt: str) -> None:
        """Submit prompt to ChatGPT."""
        # Find input field
        input_field = await self._page.wait_for_selector(
            self.SELECTORS["prompt_input"],
            timeout=10000,
        )

        # Clear and type prompt
        await input_field.click()
        await input_field.fill(prompt)

        await asyncio.sleep(0.5)

        # Click send button
        send_button = await self._page.wait_for_selector(
            self.SELECTORS["send_button"],
            timeout=5000,
        )
        await send_button.click()

        await asyncio.sleep(1)

    async def _wait_for_response(self, timeout: int) -> str:
        """Wait for ChatGPT's response to complete."""
        start_time = asyncio.get_event_loop().time()
        timeout_sec = timeout / 1000

        last_content = ""
        stable_count = 0

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_sec:
                raise TimeoutError(f"Response timeout after {timeout_sec}s")

            # Check for rate limiting
            rate_limit = await self._page.query_selector(self.SELECTORS["rate_limit_message"])
            if rate_limit:
                self.state.status = SessionStatus.RATE_LIMITED
                raise Exception("Rate limit reached")

            # Check if still streaming
            streaming = await self._page.query_selector(self.SELECTORS["streaming_indicator"])

            # Get current response
            content = await self._extract_response()

            if not streaming:
                if content == last_content and content:
                    stable_count += 1
                    if stable_count >= 3:
                        return content
                else:
                    stable_count = 0
                    last_content = content

            await asyncio.sleep(0.5)

    async def _extract_response(self) -> str:
        """Extract response text from ChatGPT's interface."""
        try:
            # Find all assistant messages
            responses = await self._page.query_selector_all(self.SELECTORS["response_container"])

            if not responses:
                return ""

            # Get the last response
            last_response = responses[-1]
            text = await last_response.inner_text()

            return text.strip()

        except Exception as e:
            logger.warning(f"[ChatGPT] Extract error: {e}")
            return ""

    async def _start_new_conversation(self) -> None:
        """Start a new ChatGPT conversation."""
        new_chat = await self._page.query_selector(self.SELECTORS["new_chat_button"])

        if new_chat:
            await new_chat.click()
        else:
            await self._page.goto(self.BASE_URL, timeout=self.PAGE_LOAD_TIMEOUT)

        await asyncio.sleep(2)

    async def format_story(
        self,
        verified_facts: list[str],
        what_happened: str,
        affected_assets: list[str],
        category: str,
    ) -> dict:
        """
        Format verified story into structured content.

        ChatGPT excels at clear, structured formatting.
        """
        facts = "\n".join(f"- {f}" for f in verified_facts)

        prompt = f"""TASK: Format this verified financial news into clean, structured content.

VERIFIED FACTS:
{facts}

SUMMARY: {what_happened}

AFFECTED ASSETS: {", ".join(affected_assets)}
CATEGORY: {category}

STYLE GUIDELINES:
- Write in active voice, present tense
- Be concise and factual
- Use financial terminology appropriately
- Headlines should be informative, not clickbait

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "title": "clear headline under 80 characters",
    "what_happened": "1-2 sentence factual summary",
    "bullet_points": [
        "key point 1",
        "key point 2",
        "key point 3"
    ],
    "affected_assets": ["TICKER1", "TICKER2"],
    "time_sensitivity": "urgent" or "normal" or "background"
}}

Return ONLY the JSON, no other text."""

        response = await self.send_prompt_with_retry(prompt)

        if not response.success:
            return {"error": response.error}

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
            logger.warning(f"[ChatGPT] JSON parse error: {e}")
            return {"error": f"JSON parse failed: {e}", "raw": content}
