"""
Browser-Based Model Agents

Interacts with Claude, ChatGPT, and Gemini through browser automation.
Uses paid subscriptions instead of APIs - zero marginal cost.

Each agent:
- Opens model's web interface
- Manages authentication state
- Sends prompts and extracts responses
- Handles rate limits and session recovery
"""

from news_pipeline.browser_agents.base import BrowserModelSession, ModelType
from news_pipeline.browser_agents.chatgpt_agent import ChatGPTBrowserAgent
from news_pipeline.browser_agents.claude_agent import ClaudeBrowserAgent
from news_pipeline.browser_agents.gemini_agent import GeminiBrowserAgent

__all__ = [
    "BrowserModelSession",
    "ModelType",
    "ClaudeBrowserAgent",
    "ChatGPTBrowserAgent",
    "GeminiBrowserAgent",
]
