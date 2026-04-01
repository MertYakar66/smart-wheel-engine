"""
AI Model Providers for News Pipeline

Each provider handles a specific stage of the pipeline:
- Grok: News discovery via web search
- Gemini: Cross-source verification with grounding
- ChatGPT: Story formatting and structuring
- Claude: Editorial polish and "why it matters"
"""

from .base import (
    BaseProvider,
    DiscoveryProvider,
    EditorialProvider,
    FormattingProvider,
    VerificationProvider,
)
from .chatgpt import ChatGPTProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .grok import GrokProvider

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
