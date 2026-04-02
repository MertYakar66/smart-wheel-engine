"""
News Pipeline - Browser-Based Multi-Model System

Zero API cost architecture using paid subscriptions via browser automation.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    LOCAL ORCHESTRATOR                        │
    │         (Playwright browser control + state management)      │
    └─────────────────────────────────────────────────────────────┘
                │              │              │
       ┌────────┴────────┐    │    ┌─────────┴─────────┐
       ▼                 ▼    ▼    ▼                   ▼
    ┌──────┐         ┌──────────────────┐         ┌──────────┐
    │Scrape│         │ Browser Sessions │         │ Local DB │
    │ News │         │ Claude/GPT/Gemini│         │ + Publish│
    └──────┘         └──────────────────┘         └──────────┘

Pipeline Flow:
    1. SCRAPE: Fetch headlines from Bloomberg, Reuters, Fed, SEC
    2. PREPROCESS: Local LLM filters duplicates, categorizes
    3. VERIFY: Claude (browser) verifies with web search
    4. FORMAT: ChatGPT (browser) structures content
    5. EDITORIAL: Claude (browser) adds "why it matters"
    6. PUBLISH: Save to database, push to dashboard

Cost: $0 marginal (uses existing subscriptions)
"""

# Lazy imports for browser agents (require Playwright)
# Import these directly when needed: from news_pipeline.browser_agents import ...

from news_pipeline.models import (
    DiscoveryRequest,
    FinalizedStory,
    PipelineResult,
)

__version__ = "2.0.0"


def __getattr__(name: str):
    """Lazy import browser agents to avoid Playwright dependency at import time."""
    browser_exports = {
        "BrowserModelSession",
        "ClaudeBrowserAgent",
        "ChatGPTBrowserAgent",
        "GeminiBrowserAgent",
    }
    if name in browser_exports:
        from news_pipeline import browser_agents

        return getattr(browser_agents, name)
    if name == "NewsPipelineOrchestrator":
        from news_pipeline.orchestrator import NewsPipelineOrchestrator

        return NewsPipelineOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NewsPipelineOrchestrator",
    "BrowserModelSession",
    "ClaudeBrowserAgent",
    "ChatGPTBrowserAgent",
    "GeminiBrowserAgent",
    "DiscoveryRequest",
    "PipelineResult",
    "FinalizedStory",
]
