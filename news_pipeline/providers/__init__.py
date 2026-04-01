"""
AI Model Providers

Each provider wraps a specific AI model and implements a stage of the pipeline:

- GrokProvider: News discovery via real-time web search (xAI)
- GeminiProvider: Cross-source verification with grounding (Google)
- ChatGPTProvider: Clear, structured formatting (OpenAI)
- ClaudeProvider: Editorial polish and analysis (Anthropic)

Providers share a common interface for consistency and testability.
"""

from news_pipeline.providers.base import (
    BaseProvider,
    DiscoveryProvider,
    EditorialProvider,
    FormattingProvider,
    VerificationProvider,
)
from news_pipeline.providers.chatgpt import ChatGPTProvider
from news_pipeline.providers.claude import ClaudeProvider
from news_pipeline.providers.gemini import GeminiProvider
from news_pipeline.providers.grok import GrokProvider

__all__ = [
    # Base classes
    "BaseProvider",
    "DiscoveryProvider",
    "VerificationProvider",
    "FormattingProvider",
    "EditorialProvider",
    # Implementations
    "GrokProvider",
    "GeminiProvider",
    "ChatGPTProvider",
    "ClaudeProvider",
]
