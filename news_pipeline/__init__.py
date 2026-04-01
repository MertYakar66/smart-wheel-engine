"""
Smart Wheel News Pipeline

Multi-model AI pipeline for financial news discovery, verification, and publishing.

Pipeline Flow:
    Grok (Discovery) → Gemini (Verification) → ChatGPT (Formatting) → Claude (Editorial)
"""

from .models.schema import (
    CandidateStory,
    DiscoveryRequest,
    FinalizedStory,
    FormattedStory,
    PipelineRun,
    PipelineStage,
    PublishedFeedItem,
    StoryCategory,
    VerificationEvidence,
    VerificationResult,
    VerificationStatus,
)
from .orchestrator import NewsPipelineOrchestrator

__all__ = [
    # Orchestrator
    "NewsPipelineOrchestrator",
    # Enums
    "PipelineStage",
    "StoryCategory",
    "VerificationStatus",
    # Models
    "DiscoveryRequest",
    "CandidateStory",
    "VerificationEvidence",
    "VerificationResult",
    "FormattedStory",
    "FinalizedStory",
    "PublishedFeedItem",
    "PipelineRun",
]
