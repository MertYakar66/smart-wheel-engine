"""
Base Browser Model Session

Abstract base for interacting with AI models via browser.
Handles common patterns: navigation, authentication, prompt/response cycles.

Note: Requires playwright for browser automation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Import enums that don't need playwright
from news_pipeline.browser_agents.types import ModelType, SessionStatus

# Re-export for backwards compatibility
__all__ = ["ModelType", "SessionStatus", "BrowserModelSession", "SessionManager", "ModelResponse"]

# Lazy import playwright to allow importing enums without playwright installed
if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

logger = logging.getLogger(__name__)


def _get_playwright():
    """Get playwright module, raising helpful error if not installed."""
    try:
        from playwright.async_api import Browser, BrowserContext, Page, async_playwright
        return Browser, BrowserContext, Page, async_playwright
    except ImportError as e:
        raise ImportError(
            "Browser agents require playwright. Install with: pip install playwright && playwright install"
        ) from e


@dataclass
class SessionState:
    """Persistent session state."""

    model_type: ModelType
    status: SessionStatus = SessionStatus.DISCONNECTED
    last_activity: datetime | None = None
    message_count: int = 0
    error_count: int = 0
    last_error: str | None = None
    cookies_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type.value,
            "status": self.status.value,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


@dataclass
class ModelResponse:
    """Response from a model."""

    success: bool
    content: str = ""
    raw_html: str = ""
    error: str | None = None
    response_time_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class BrowserModelSession(ABC):
    """
    Abstract base class for browser-based model interaction.

    Manages:
    - Playwright browser instance
    - Authentication state
    - Prompt submission and response extraction
    - Rate limiting and retry logic
    - Session persistence
    """

    # Override in subclasses
    MODEL_TYPE: ModelType = ModelType.LOCAL
    BASE_URL: str = ""
    COOKIES_FILE: str = ""

    # Timing configuration
    PAGE_LOAD_TIMEOUT: int = 30000
    RESPONSE_TIMEOUT: int = 120000  # 2 minutes for long responses
    RETRY_DELAY: int = 5000
    MAX_RETRIES: int = 3

    def __init__(self, headless: bool = True, user_data_dir: str | None = None):
        """
        Initialize browser session.

        Args:
            headless: Run browser without GUI (set False for debugging)
            user_data_dir: Directory for persistent browser data
        """
        self.headless = headless
        self.user_data_dir = user_data_dir or str(
            Path.home() / ".news_pipeline" / "browser_data" / self.MODEL_TYPE.value
        )

        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

        self.state = SessionState(model_type=self.MODEL_TYPE)

    @property
    def is_ready(self) -> bool:
        """Check if session is ready to send prompts."""
        return self.state.status == SessionStatus.READY

    @property
    def is_authenticated(self) -> bool:
        """Check if session is authenticated."""
        return self.state.status in (SessionStatus.AUTHENTICATED, SessionStatus.READY)

    async def initialize(self) -> bool:
        """
        Initialize browser and load session.

        Returns:
            True if ready to use, False if authentication needed
        """
        logger.info(f"[{self.MODEL_TYPE.value}] Initializing browser session...")

        self.state.status = SessionStatus.CONNECTING

        try:
            # Ensure data directory exists
            Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)

            # Launch browser with persistent context
            _, _, _, async_playwright = _get_playwright()
            self._playwright = await async_playwright().start()
            self._context = await self._playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=self.headless,
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )

            self._page = (
                self._context.pages[0] if self._context.pages else await self._context.new_page()
            )

            # Navigate to model URL
            await self._page.goto(self.BASE_URL, timeout=self.PAGE_LOAD_TIMEOUT)
            await asyncio.sleep(2)  # Let page settle

            # Check authentication
            if await self._check_authenticated():
                self.state.status = SessionStatus.READY
                logger.info(f"[{self.MODEL_TYPE.value}] Session ready (authenticated)")
                return True
            else:
                self.state.status = SessionStatus.DISCONNECTED
                logger.warning(f"[{self.MODEL_TYPE.value}] Authentication required")
                return False

        except Exception as e:
            self.state.status = SessionStatus.ERROR
            self.state.last_error = str(e)
            logger.error(f"[{self.MODEL_TYPE.value}] Initialization failed: {e}")
            return False

    async def close(self) -> None:
        """Close browser session."""
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()

        self.state.status = SessionStatus.DISCONNECTED
        logger.info(f"[{self.MODEL_TYPE.value}] Session closed")

    async def send_prompt(
        self,
        prompt: str,
        wait_complete: bool = True,
        timeout: int | None = None,
    ) -> ModelResponse:
        """
        Send a prompt and get response.

        Args:
            prompt: The prompt text to send
            wait_complete: Wait for complete response
            timeout: Custom timeout in ms

        Returns:
            ModelResponse with content or error
        """
        if not self.is_ready:
            return ModelResponse(
                success=False,
                error=f"Session not ready: {self.state.status.value}",
            )

        self.state.status = SessionStatus.BUSY
        start_time = datetime.utcnow()

        try:
            # Submit prompt
            await self._submit_prompt(prompt)

            # Wait for response
            response_timeout = timeout or self.RESPONSE_TIMEOUT
            content = await self._wait_for_response(response_timeout)

            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            self.state.status = SessionStatus.READY
            self.state.last_activity = datetime.utcnow()
            self.state.message_count += 1

            logger.info(
                f"[{self.MODEL_TYPE.value}] Response received "
                f"({elapsed_ms}ms, {len(content)} chars)"
            )

            return ModelResponse(
                success=True,
                content=content,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            self.state.status = SessionStatus.ERROR
            self.state.error_count += 1
            self.state.last_error = str(e)

            logger.error(f"[{self.MODEL_TYPE.value}] Prompt failed: {e}")

            return ModelResponse(
                success=False,
                error=str(e),
            )

    async def send_prompt_with_retry(
        self,
        prompt: str,
        max_retries: int | None = None,
    ) -> ModelResponse:
        """
        Send prompt with automatic retry on failure.

        Args:
            prompt: The prompt text
            max_retries: Override default max retries

        Returns:
            ModelResponse from successful attempt or last failure
        """
        retries = max_retries or self.MAX_RETRIES
        last_response = None

        for attempt in range(retries + 1):
            response = await self.send_prompt(prompt)

            if response.success:
                return response

            last_response = response

            if attempt < retries:
                delay_sec = (self.RETRY_DELAY * (attempt + 1)) / 1000
                logger.warning(
                    f"[{self.MODEL_TYPE.value}] Retry {attempt + 1}/{retries} "
                    f"in {delay_sec}s: {response.error}"
                )
                await asyncio.sleep(delay_sec)

                # Try to recover session
                await self._recover_session()

        return last_response or ModelResponse(success=False, error="All retries failed")

    async def start_new_chat(self) -> bool:
        """Start a new chat/conversation."""
        try:
            await self._start_new_conversation()
            self.state.message_count = 0
            logger.info(f"[{self.MODEL_TYPE.value}] Started new conversation")
            return True
        except Exception as e:
            logger.error(f"[{self.MODEL_TYPE.value}] Failed to start new chat: {e}")
            return False

    # Abstract methods - implement in subclasses

    @abstractmethod
    async def _check_authenticated(self) -> bool:
        """Check if user is authenticated."""
        pass

    @abstractmethod
    async def _submit_prompt(self, prompt: str) -> None:
        """Submit prompt to the model."""
        pass

    @abstractmethod
    async def _wait_for_response(self, timeout: int) -> str:
        """Wait for and extract model response."""
        pass

    @abstractmethod
    async def _start_new_conversation(self) -> None:
        """Start a new conversation/chat."""
        pass

    async def _recover_session(self) -> None:
        """Attempt to recover from error state."""
        try:
            await self._page.reload()
            await asyncio.sleep(2)

            if await self._check_authenticated():
                self.state.status = SessionStatus.READY
            else:
                self.state.status = SessionStatus.DISCONNECTED

        except Exception as e:
            logger.error(f"[{self.MODEL_TYPE.value}] Recovery failed: {e}")


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


_session_manager = None


class SessionManager:
    """
    Manages multiple browser sessions across models.

    Provides:
    - Session pooling
    - Automatic failover between models
    - Persistence across runs
    """

    def __init__(self, state_dir: str | None = None):
        self.state_dir = Path(state_dir or Path.home() / ".news_pipeline" / "sessions")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.sessions: dict[ModelType, BrowserModelSession] = {}
        self._state_file = self.state_dir / "session_state.json"

    async def get_session(self, model_type: ModelType) -> BrowserModelSession | None:
        """Get or create a session for a model type."""
        if model_type in self.sessions:
            session = self.sessions[model_type]
            if session.is_ready:
                return session

        # Create new session
        session = self._create_session(model_type)
        if session:
            initialized = await session.initialize()
            if initialized:
                self.sessions[model_type] = session
                return session

        return None

    def _create_session(self, model_type: ModelType) -> BrowserModelSession | None:
        """Create appropriate session for model type."""
        from news_pipeline.browser_agents.chatgpt_agent import ChatGPTBrowserAgent
        from news_pipeline.browser_agents.claude_agent import ClaudeBrowserAgent
        from news_pipeline.browser_agents.gemini_agent import GeminiBrowserAgent

        session_map = {
            ModelType.CLAUDE: ClaudeBrowserAgent,
            ModelType.CHATGPT: ChatGPTBrowserAgent,
            ModelType.GEMINI: GeminiBrowserAgent,
        }

        session_class = session_map.get(model_type)
        if session_class:
            return session_class(headless=True)
        return None

    async def get_available_session(
        self,
        preferred: list[ModelType] | None = None,
    ) -> BrowserModelSession | None:
        """
        Get any available session, with preference order.

        Useful for fallback when primary model is unavailable.
        """
        order = preferred or [ModelType.CLAUDE, ModelType.CHATGPT, ModelType.GEMINI]

        for model_type in order:
            session = await self.get_session(model_type)
            if session and session.is_ready:
                return session

        return None

    async def close_all(self) -> None:
        """Close all sessions."""
        for session in self.sessions.values():
            await session.close()
        self.sessions.clear()

    def save_state(self) -> None:
        """Save session states to disk."""
        state = {
            model_type.value: session.state.to_dict()
            for model_type, session in self.sessions.items()
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> dict:
        """Load previous session states."""
        if self._state_file.exists():
            with open(self._state_file) as f:
                return json.load(f)
        return {}
