"""
Smart Wheel News Pipeline

Enterprise-grade multi-model AI pipeline for financial news discovery,
verification, formatting, and publishing.

Architecture:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │    Grok     │───▶│   Gemini    │───▶│  ChatGPT    │───▶│   Claude    │
    │  Discovery  │    │ Verification│    │  Formatting │    │  Editorial  │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
           │                  │                  │                  │
           ▼                  ▼                  ▼                  ▼
      Candidates         Verified            Formatted          Finalized
       Stories            Facts              Content            Stories
                                                                    │
                                                                    ▼
                                                              ┌─────────────┐
                                                              │  Publisher  │
                                                              │   (Feed)    │
                                                              └─────────────┘

Each model contributes its unique strengths:
- Grok (xAI): Real-time web search, breaking news detection
- Gemini (Google): Grounded verification with search, fact-checking
- ChatGPT (OpenAI): Clear, structured formatting
- Claude (Anthropic): Editorial polish, "why it matters" analysis

Usage:
    from news_pipeline import NewsPipelineOrchestrator, DiscoveryRequest

    orchestrator = NewsPipelineOrchestrator()
    request = DiscoveryRequest(
        categories=["fed", "earnings", "oil"],
        time_window="overnight",
    )
    result = await orchestrator.run(request)
"""

from news_pipeline.config import PipelineConfig
from news_pipeline.models import (
    CandidateStory,
    DiscoveryRequest,
    FinalizedStory,
    FormattedStory,
    PipelineResult,
    PipelineStage,
    PublishedFeedItem,
    StoryCategory,
    VerificationResult,
    VerificationStatus,
)
from news_pipeline.orchestrator import NewsPipelineOrchestrator

__version__ = "1.0.0"
__all__ = [
    # Core
    "NewsPipelineOrchestrator",
    "PipelineConfig",
    # Enums
    "PipelineStage",
    "StoryCategory",
    "VerificationStatus",
    # Models
    "DiscoveryRequest",
    "CandidateStory",
    "VerificationResult",
    "FormattedStory",
    "FinalizedStory",
    "PublishedFeedItem",
    "PipelineResult",
]
