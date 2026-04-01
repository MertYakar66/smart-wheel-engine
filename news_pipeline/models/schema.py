"""
News Pipeline Data Models

Defines the canonical data structures used throughout the multi-model pipeline:
- Discovery (Grok) -> Verification (Gemini) -> Formatting (ChatGPT) -> Editorial (Claude) -> Publish
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PipelineStage(Enum):
    """Pipeline processing stages."""

    DISCOVERED = "discovered"  # Found by Grok
    VERIFIED = "verified"  # Verified by Gemini
    FORMATTED = "formatted"  # Formatted by ChatGPT
    FINALIZED = "finalized"  # Finalized by Claude
    PUBLISHED = "published"  # Published to website
    DISCARDED = "discarded"  # Failed verification


class StoryCategory(Enum):
    """News story categories for targeting."""

    SP500_EVENTS = "sp500_events"
    OIL = "oil"
    GEOPOLITICS = "geopolitics"
    FED = "fed"
    INFLATION = "inflation"
    LABOR = "labor"
    EARNINGS = "earnings"
    MACRO = "macro"
    CRYPTO = "crypto"
    TECH = "tech"


class VerificationStatus(Enum):
    """Verification outcome status."""

    VERIFIED = "verified"
    PARTIAL = "partial"  # Some facts verified
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


@dataclass
class DiscoveryRequest:
    """Request to discover news stories."""

    run_id: str
    tickers: list[str]
    categories: list[StoryCategory]
    time_window: str  # e.g., "overnight", "last_6h", "today"
    max_stories: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tickers": self.tickers,
            "categories": [c.value for c in self.categories],
            "time_window": self.time_window,
            "max_stories": self.max_stories,
        }


@dataclass
class CandidateStory:
    """
    A raw story discovered by the discovery layer.

    This is the output from Grok's web search.
    """

    candidate_id: str
    run_id: str
    headline: str
    source_name: str
    source_url: str
    published_at: datetime | None
    discovered_at: datetime

    # Extracted metadata
    tickers: list[str] = field(default_factory=list)
    category_guess: StoryCategory | None = None
    snippet: str = ""
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    # Processing state
    stage: PipelineStage = PipelineStage.DISCOVERED

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "headline": self.headline,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "discovered_at": self.discovered_at.isoformat(),
            "tickers": self.tickers,
            "category_guess": self.category_guess.value if self.category_guess else None,
            "snippet": self.snippet,
            "stage": self.stage.value,
        }


@dataclass
class VerificationEvidence:
    """A piece of corroborating evidence for a story."""

    source_name: str
    source_url: str
    evidence_type: str  # "corroboration", "official", "contradiction"
    summary: str
    weight: float = 1.0  # How much this evidence contributes to confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_url": self.source_url,
            "evidence_type": self.evidence_type,
            "summary": self.summary,
            "weight": self.weight,
        }


@dataclass
class VerificationResult:
    """
    Result of verifying a candidate story.

    This is the output from Gemini's verification.
    """

    candidate_id: str
    status: VerificationStatus
    confidence: int  # 0-10 scale
    verified_facts: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    evidence: list[VerificationEvidence] = field(default_factory=list)
    verification_notes: str = ""
    verified_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "confidence": self.confidence,
            "verified_facts": self.verified_facts,
            "contradictions": self.contradictions,
            "evidence": [e.to_dict() for e in self.evidence],
            "verification_notes": self.verification_notes,
            "verified_at": self.verified_at.isoformat(),
        }


@dataclass
class FormattedStory:
    """
    A story formatted for the feed.

    This is the output from ChatGPT's formatting.
    """

    story_id: str
    candidate_id: str
    title: str
    what_happened: str
    bullet_points: list[str] = field(default_factory=list)
    affected_assets: list[str] = field(default_factory=list)
    category: StoryCategory | None = None
    confidence: int = 0
    formatted_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "candidate_id": self.candidate_id,
            "title": self.title,
            "what_happened": self.what_happened,
            "bullet_points": self.bullet_points,
            "affected_assets": self.affected_assets,
            "category": self.category.value if self.category else None,
            "confidence": self.confidence,
            "formatted_at": self.formatted_at.isoformat(),
        }


@dataclass
class FinalizedStory:
    """
    A finalized story ready for publishing.

    This is the output from Claude's editorial pass.
    """

    story_id: str
    title: str
    what_happened: str
    why_it_matters: str
    bullet_points: list[str] = field(default_factory=list)
    affected_assets: list[str] = field(default_factory=list)
    category: StoryCategory | None = None
    verification_confidence: int = 0
    finalized_at: datetime = field(default_factory=datetime.utcnow)

    # Source tracking
    source_urls: list[str] = field(default_factory=list)
    original_candidate_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "what_happened": self.what_happened,
            "why_it_matters": self.why_it_matters,
            "bullet_points": self.bullet_points,
            "affected_assets": self.affected_assets,
            "category": self.category.value if self.category else None,
            "verification_confidence": self.verification_confidence,
            "finalized_at": self.finalized_at.isoformat(),
            "source_urls": self.source_urls,
        }


@dataclass
class PublishedFeedItem:
    """
    A published feed item on the website.
    """

    feed_item_id: str
    story_id: str
    title: str
    what_happened: str
    why_it_matters: str
    bullet_points: list[str]
    affected_assets: list[str]
    category: str
    confidence: int
    published_at: datetime
    source_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feed_item_id": self.feed_item_id,
            "story_id": self.story_id,
            "title": self.title,
            "what_happened": self.what_happened,
            "why_it_matters": self.why_it_matters,
            "bullet_points": self.bullet_points,
            "affected_assets": self.affected_assets,
            "category": self.category,
            "confidence": self.confidence,
            "published_at": self.published_at.isoformat(),
            "source_urls": self.source_urls,
        }


@dataclass
class PipelineRun:
    """
    A complete pipeline run record.
    """

    run_id: str
    started_at: datetime
    ended_at: datetime | None = None
    run_type: str = "morning"  # morning, evening, manual

    # Request parameters
    tickers: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    time_window: str = "overnight"

    # Results
    candidates_found: int = 0
    stories_verified: int = 0
    stories_published: int = 0
    stories_discarded: int = 0

    # Errors
    errors: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"  # running, completed, failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "run_type": self.run_type,
            "tickers": self.tickers,
            "categories": self.categories,
            "time_window": self.time_window,
            "candidates_found": self.candidates_found,
            "stories_verified": self.stories_verified,
            "stories_published": self.stories_published,
            "stories_discarded": self.stories_discarded,
            "errors": self.errors,
            "status": self.status,
        }
