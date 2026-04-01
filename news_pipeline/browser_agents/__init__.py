"""
Browser-Based Model Agents

Interacts with Claude, ChatGPT, and Gemini through browser automation.
Uses paid subscriptions instead of APIs - zero marginal cost.

Each agent:
- Opens model's web interface
- Manages authentication state
- Sends prompts and extracts responses
- Handles rate limits and session recovery

Note: Requires playwright to be installed. Import will raise ImportError if missing.
"""

__all__ = [
    "BrowserModelSession",
    "ModelType",
    "ClaudeBrowserAgent",
    "ChatGPTBrowserAgent",
    "GeminiBrowserAgent",
]

# Lazy imports to defer playwright dependency check
_PLAYWRIGHT_ERROR = None

try:
    from news_pipeline.browser_agents.base import (
        BrowserModelSession,
        ModelType,
        SessionManager,
        SessionStatus,
    )
    from news_pipeline.browser_agents.chatgpt_agent import ChatGPTBrowserAgent
    from news_pipeline.browser_agents.claude_agent import ClaudeBrowserAgent
    from news_pipeline.browser_agents.gemini_agent import GeminiBrowserAgent
except ImportError as e:
    _PLAYWRIGHT_ERROR = e
    BrowserModelSession = None  # type: ignore
    ModelType = None  # type: ignore
    SessionManager = None  # type: ignore
    SessionStatus = None  # type: ignore
    ClaudeBrowserAgent = None  # type: ignore
    ChatGPTBrowserAgent = None  # type: ignore
    GeminiBrowserAgent = None  # type: ignore


def require_playwright():
    """Raise ImportError if playwright is not available."""
    if _PLAYWRIGHT_ERROR is not None:
        raise ImportError(
            "Browser agents require playwright. Install with: pip install playwright && playwright install"
        ) from _PLAYWRIGHT_ERROR
