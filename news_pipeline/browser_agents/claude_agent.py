"""
Claude Browser Agent

Interacts with claude.ai via browser automation.
Uses your paid Claude Pro subscription - no API costs.

Features:
- Persistent login state
- Handles streaming responses
- Supports web search (Claude's built-in feature)
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


class ClaudeBrowserAgent(BrowserModelSession):
    """
    Browser automation for claude.ai.

    Uses Claude Pro subscription via browser.
    Supports web search capability built into Claude.
    """

    MODEL_TYPE = ModelType.CLAUDE
    BASE_URL = "https://claude.ai/new"
    RESPONSE_TIMEOUT = 180000  # 3 minutes for complex queries with search

    # Selectors for claude.ai interface
    SELECTORS = {
        # Input area
        "prompt_input": 'div[contenteditable="true"].ProseMirror',
        "send_button": 'button[aria-label="Send Message"]',
        # Response area
        "response_container": "div[data-is-streaming]",
        "response_text": "div.font-claude-message",
        "streaming_indicator": '[data-is-streaming="true"]',
        # Navigation
        "new_chat_button": 'a[href="/new"]',
        "sidebar": 'nav[aria-label="Chat history"]',
        # Auth indicators
        "user_menu": 'button[data-testid="user-menu"]',
        "login_button": 'a[href="/login"]',
        # Rate limit
        "rate_limit_message": 'text="You\'ve reached"',
    }

    async def _check_authenticated(self) -> bool:
        """Check if user is logged into Claude."""
        try:
            # Look for user menu (indicates logged in)
            user_menu = await self._page.query_selector(self.SELECTORS["user_menu"])
            if user_menu:
                return True

            # Look for login button (indicates not logged in)
            login_btn = await self._page.query_selector(self.SELECTORS["login_button"])
            if login_btn:
                return False

            # Check URL for auth redirects
            if "login" in self._page.url.lower():
                return False

            # Default: assume authenticated if on claude.ai main page
            return "claude.ai" in self._page.url

        except Exception as e:
            logger.warning(f"[Claude] Auth check error: {e}")
            return False

    async def _submit_prompt(self, prompt: str) -> None:
        """Submit prompt to Claude."""
        # Find input field
        input_field = await self._page.wait_for_selector(
            self.SELECTORS["prompt_input"],
            timeout=10000,
        )

        # Clear existing content and type prompt
        await input_field.click()
        await self._page.keyboard.press("Control+a")
        await input_field.fill(prompt)

        # Small delay before sending
        await asyncio.sleep(0.5)

        # Click send or press Enter
        send_button = await self._page.query_selector(self.SELECTORS["send_button"])
        if send_button and await send_button.is_enabled():
            await send_button.click()
        else:
            await self._page.keyboard.press("Enter")

        # Wait for response to start
        await asyncio.sleep(1)

    async def _wait_for_response(self, timeout: int) -> str:
        """Wait for Claude's response to complete."""
        start_time = asyncio.get_event_loop().time()
        timeout_sec = timeout / 1000

        last_content = ""
        stable_count = 0
        required_stable = 3  # Require 3 checks with same content

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

            # Get current response content
            content = await self._extract_response()

            if not streaming:
                # Not streaming - check if content is stable
                if content == last_content and content:
                    stable_count += 1
                    if stable_count >= required_stable:
                        return content
                else:
                    stable_count = 0
                    last_content = content

            await asyncio.sleep(0.5)

    async def _extract_response(self) -> str:
        """Extract response text from Claude's interface."""
        try:
            # Find all response messages
            responses = await self._page.query_selector_all(self.SELECTORS["response_text"])

            if not responses:
                return ""

            # Get the last (most recent) response
            last_response = responses[-1]
            text = await last_response.inner_text()

            return text.strip()

        except Exception as e:
            logger.warning(f"[Claude] Extract error: {e}")
            return ""

    async def _start_new_conversation(self) -> None:
        """Start a new Claude conversation."""
        # Click new chat button or navigate to /new
        new_chat = await self._page.query_selector(self.SELECTORS["new_chat_button"])

        if new_chat:
            await new_chat.click()
        else:
            await self._page.goto(self.BASE_URL, timeout=self.PAGE_LOAD_TIMEOUT)

        await asyncio.sleep(2)

    async def send_with_search(self, prompt: str) -> str:
        """
        Send prompt that triggers Claude's web search.

        Claude Pro includes built-in web search capability.
        The prompt should naturally trigger search behavior.
        """
        # Prepend search instruction
        search_prompt = f"""Please search the web to answer this question with current information:

{prompt}

Include sources and verify facts from multiple sources."""

        response = await self.send_prompt_with_retry(search_prompt)

        if response.success:
            return response.content
        else:
            raise Exception(f"Search query failed: {response.error}")

    async def verify_news_story(
        self,
        headline: str,
        source: str,
        tickers: list[str],
        category: str,
    ) -> dict:
        """
        Verify a news story using Claude's web search.

        Returns structured verification result.
        """
        prompt = f"""TASK: Verify this financial news headline using web search.

HEADLINE: "{headline}"
SOURCE: {source}
RELATED TICKERS: {", ".join(tickers) if tickers else "None"}
CATEGORY: {category}

INSTRUCTIONS:
1. Search for this news story from multiple credible sources
2. Check official sources (company websites, SEC filings, Fed statements)
3. Identify any contradicting information
4. Rate your confidence in the story's accuracy

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "status": "verified" or "partial" or "unverified" or "contradicted",
    "confidence": 0-10,
    "verified_facts": ["fact 1", "fact 2"],
    "what_happened": "1-2 sentence factual summary",
    "sources_found": ["source 1", "source 2"],
    "contradictions": ["any conflicting info"],
    "verification_notes": "brief explanation"
}}

Return ONLY the JSON, no other text."""

        response = await self.send_prompt_with_retry(prompt)

        if not response.success:
            return {
                "status": "error",
                "confidence": 0,
                "error": response.error,
            }

        # Parse JSON from response
        return self._parse_json_response(response.content)

    async def add_editorial(
        self,
        title: str,
        what_happened: str,
        bullet_points: list[str],
        affected_assets: list[str],
        category: str,
    ) -> dict:
        """
        Add editorial "why it matters" analysis.

        Returns structured editorial content.
        """
        bullets = "\n".join(f"- {b}" for b in bullet_points)

        prompt = f"""TASK: Add editorial analysis to this verified financial news story.

TITLE: {title}

WHAT HAPPENED:
{what_happened}

KEY POINTS:
{bullets}

AFFECTED ASSETS: {", ".join(affected_assets)}
CATEGORY: {category}

INSTRUCTIONS:
1. Write a compelling "why it matters" section (2-3 sentences)
2. Explain market implications
3. Note what traders should watch for
4. Assign priority (1-10) based on market impact

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "title": "polished title (keep under 80 chars)",
    "what_happened": "refined summary",
    "why_it_matters": "2-3 sentences on market significance",
    "bullet_points": ["point 1", "point 2", "point 3"],
    "market_implications": "how markets might react",
    "trading_considerations": "what traders should watch",
    "is_breaking": true or false,
    "priority": 1-10,
    "tags": ["tag1", "tag2"]
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
            # Try to find JSON in the response
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

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"[Claude] JSON parse error: {e}")
            return {"error": f"JSON parse failed: {e}", "raw": content}
