"""
Browser-Based Model Agents

Interacts with Claude, ChatGPT, and Gemini through browser automation.
Uses paid subscriptions instead of APIs - zero marginal cost.

Each agent:
- Opens model's web interface
- Manages authentication state
- Sends prompts and extracts responses
- Handles rate limits and session recovery

Note: Browser session classes require playwright to be installed.
ModelType and SessionStatus enums can be imported without playwright.
"""

# Enums that don't require playwright
from news_pipeline.browser_agents.types import ModelResponse, ModelType, SessionStatus

__all__ = [
    "BrowserModelSession",
    "ModelType",
    "ModelResponse",
    "SessionStatus",
    "SessionManager",
    "ClaudeBrowserAgent",
    "ChatGPTBrowserAgent",
    "GeminiBrowserAgent",
    "require_playwright",
]


def require_playwright():
    """Raise ImportError if playwright is not available."""
    try:
        import playwright  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Browser agents require playwright. Install with: pip install playwright && playwright install"
        ) from e


def __getattr__(name: str):
    """Lazy import browser session classes that require playwright."""
    if name == "BrowserModelSession":
        from news_pipeline.browser_agents.base import BrowserModelSession
        return BrowserModelSession
    if name == "SessionManager":
        from news_pipeline.browser_agents.base import SessionManager
        return SessionManager
    if name == "ClaudeBrowserAgent":
        from news_pipeline.browser_agents.claude_agent import ClaudeBrowserAgent
        return ClaudeBrowserAgent
    if name == "ChatGPTBrowserAgent":
        from news_pipeline.browser_agents.chatgpt_agent import ChatGPTBrowserAgent
        return ChatGPTBrowserAgent
    if name == "GeminiBrowserAgent":
        from news_pipeline.browser_agents.gemini_agent import GeminiBrowserAgent
        return GeminiBrowserAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
