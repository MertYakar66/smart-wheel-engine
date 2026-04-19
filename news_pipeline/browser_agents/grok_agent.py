"""
Grok Browser Agent — pull market intelligence from X via Grok.

Grok (grok.x.ai) has real-time access to X/Twitter posts and is the
only free-tier AI that can search the live firehose of market
sentiment. This agent automates the browser interaction to:

1. Navigate to grok.x.ai
2. Authenticate via existing X session cookies
3. Send a structured market-intelligence prompt
4. Extract the response (including cited X posts)
5. Return a structured ModelResponse for the news pipeline

Integration:
  This agent plugs into the existing browser-agent fallback chain in
  ``news_pipeline/orchestrator.py``. The orchestrator tries agents in
  priority order (Claude → ChatGPT → Grok → Gemini → Local); Grok is
  inserted between ChatGPT and Gemini because it uniquely provides
  live X post context that no other agent has.

Requirements:
  - An X/Twitter account (free tier works)
  - Playwright + Chromium installed
  - Existing browser session cookies for x.com (set up via --setup-auth)

Cost: $0. Grok is free for X users. No API key required.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from news_pipeline.browser_agents.types import ModelResponse, ModelType, SessionStatus

logger = logging.getLogger(__name__)

GROK_URL = "https://grok.x.ai/"
GROK_CHAT_URL = "https://grok.x.ai/"


class GrokBrowserAgent:
    """Browser-automated Grok agent for X/Twitter market intelligence.

    Follows the same interface as ClaudeBrowserAgent, ChatGPTBrowserAgent,
    and GeminiBrowserAgent so it can be swapped into the fallback chain.
    """

    model_type = ModelType.GROK

    def __init__(
        self,
        cookies_dir: str | Path = "~/.smart_wheel/cookies",
        headless: bool = True,
        timeout_ms: int = 30_000,
    ) -> None:
        self.cookies_dir = Path(cookies_dir).expanduser()
        self.cookies_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.status = SessionStatus.DISCONNECTED
        self._browser = None
        self._context = None
        self._page = None

    async def _ensure_browser(self):
        """Lazy-init Playwright browser with persistent context."""
        if self._browser is not None:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Grok agent requires playwright. "
                "Install: pip install playwright && playwright install chromium"
            )

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self.headless)
        cookie_file = self.cookies_dir / "grok_state.json"

        self._context = await self._browser.new_context(
            storage_state=str(cookie_file) if cookie_file.exists() else None,
            viewport={"width": 1280, "height": 800},
        )
        self._page = await self._context.new_page()
        self.status = SessionStatus.CONNECTING

    async def _save_cookies(self):
        """Persist session cookies for next run."""
        if self._context:
            cookie_file = self.cookies_dir / "grok_state.json"
            await self._context.storage_state(path=str(cookie_file))

    async def _is_authenticated(self) -> bool:
        """Check if we're logged into Grok (X session active)."""
        if self._page is None:
            return False
        try:
            await self._page.goto(GROK_URL, wait_until="domcontentloaded", timeout=self.timeout_ms)
            await self._page.wait_for_timeout(2000)
            # Check for auth indicators — Grok shows a chat input when logged in
            chat_input = await self._page.query_selector('textarea, [contenteditable="true"], input[type="text"]')
            return chat_input is not None
        except Exception as e:
            logger.debug("Grok auth check failed: %s", e)
            return False

    async def setup_auth(self) -> bool:
        """Interactive authentication setup.

        Opens a visible browser window for the user to log into X/Grok.
        Once authenticated, cookies are saved for future headless runs.
        """
        logger.info("Opening Grok for authentication — log in with your X account")
        # Force visible for auth setup
        old_headless = self.headless
        self.headless = False

        try:
            if self._browser:
                await self.close()
            await self._ensure_browser()
            await self._page.goto(GROK_URL, wait_until="domcontentloaded")

            print("\n" + "=" * 60)
            print("Grok Authentication Setup")
            print("=" * 60)
            print("1. Log in with your X/Twitter account in the browser window")
            print("2. Once you see the Grok chat interface, press Enter here")
            print("=" * 60)
            input("\nPress Enter after logging in...")

            await self._save_cookies()
            logger.info("Grok session cookies saved")
            self.status = SessionStatus.AUTHENTICATED
            return True

        except Exception as e:
            logger.error("Grok auth setup failed: %s", e)
            return False
        finally:
            self.headless = old_headless

    async def send_prompt(self, prompt: str) -> ModelResponse:
        """Send a prompt to Grok and extract the response.

        The prompt should be a structured market-intelligence query.
        Grok will search X posts in real-time and synthesize a response.
        """
        start = time.monotonic()
        try:
            await self._ensure_browser()

            if not await self._is_authenticated():
                return ModelResponse(
                    success=False,
                    content="",
                    model="grok",
                    error="not_authenticated — run with --setup-auth first",
                )

            self.status = SessionStatus.BUSY

            # Navigate to a fresh Grok chat
            await self._page.goto(GROK_URL, wait_until="domcontentloaded", timeout=self.timeout_ms)
            await self._page.wait_for_timeout(2000)

            # Find and fill the chat input
            input_sel = 'textarea, [contenteditable="true"], input[type="text"]'
            chat_input = await self._page.wait_for_selector(input_sel, timeout=self.timeout_ms)
            if chat_input is None:
                return ModelResponse(
                    success=False, content="", model="grok",
                    error="chat_input_not_found",
                )

            await chat_input.fill(prompt)
            await self._page.keyboard.press("Enter")

            # Wait for response — Grok streams, so wait for the response
            # container to stabilize (no new text for 3 seconds)
            await self._page.wait_for_timeout(5000)

            # Extract response text — Grok renders in markdown-like blocks
            # Try multiple selectors for the response container
            response_text = ""
            for selector in [
                '[data-testid="messageContent"]',
                ".message-content",
                "article",
                '[class*="response"]',
                '[class*="answer"]',
            ]:
                elements = await self._page.query_selector_all(selector)
                if elements:
                    texts = []
                    for el in elements:
                        t = await el.inner_text()
                        if t and len(t) > 50:
                            texts.append(t.strip())
                    if texts:
                        response_text = texts[-1]  # Last response block
                        break

            if not response_text:
                # Fallback: grab all visible text in main content area
                body = await self._page.inner_text("main") if await self._page.query_selector("main") else ""
                if body and len(body) > 100:
                    response_text = body[-3000:]  # Last 3000 chars

            await self._save_cookies()
            self.status = SessionStatus.READY

            elapsed_ms = int((time.monotonic() - start) * 1000)
            return ModelResponse(
                success=bool(response_text),
                content=response_text,
                model="grok",
                latency_ms=elapsed_ms,
                metadata={"source": "x.com/grok", "prompt_length": len(prompt)},
            )

        except Exception as e:
            self.status = SessionStatus.ERROR
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error("Grok prompt failed: %s", e)
            return ModelResponse(
                success=False,
                content="",
                model="grok",
                latency_ms=elapsed_ms,
                error=str(e)[:200],
            )

    async def close(self):
        """Clean up browser resources."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
            self._context = None
            self._page = None
        if hasattr(self, "_pw") and self._pw:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None
        self.status = SessionStatus.DISCONNECTED

    # ------------------------------------------------------------------
    # Market-intelligence prompt templates
    # ------------------------------------------------------------------
    @staticmethod
    def market_pulse_prompt(tickers: list[str] | None = None) -> str:
        """Generate a structured prompt for market-moving X posts.

        This is the standard prompt the orchestrator sends to Grok.
        Grok will search X in real-time and cite specific posts.
        """
        ticker_str = ", ".join(tickers[:10]) if tickers else "S&P 500 stocks"
        return (
            f"Search recent X posts from the last 24 hours about {ticker_str}. "
            f"Focus on:\n"
            f"1. Unusual options activity or flow (large block trades, sweeps)\n"
            f"2. Earnings announcements, guidance changes, or analyst upgrades/downgrades\n"
            f"3. Insider buying or selling\n"
            f"4. Macro events affecting these names (Fed, CPI, trade policy)\n"
            f"5. Unusual short interest or squeeze potential\n\n"
            f"For each finding, cite the specific X post (author and approximate time). "
            f"Rate each item's market impact as HIGH / MEDIUM / LOW. "
            f"Return the results as a numbered list, most impactful first."
        )

    @staticmethod
    def sentiment_prompt(ticker: str) -> str:
        """Single-ticker sentiment scan from X."""
        return (
            f"Search recent X posts about ${ticker} from the last 48 hours. "
            f"Summarize the overall sentiment (bullish/bearish/neutral) with "
            f"specific evidence from X posts. Include:\n"
            f"- Key bullish arguments being made\n"
            f"- Key bearish arguments\n"
            f"- Any upcoming catalysts mentioned\n"
            f"- Unusual trading activity discussed\n"
            f"Cite specific X posts where possible."
        )
