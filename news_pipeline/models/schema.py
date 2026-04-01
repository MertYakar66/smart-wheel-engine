"""
Pipeline Data Schema

Defines all data models used throughout the multi-model news pipeline.
Models are immutable dataclasses optimized for serialization and validation.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum


class PipelineStage(Enum):
    """Stages in the news pipeline."""

    DISCOVERY = "discovery"
    VERIFICATION = "verification"
    FORMATTING = "formatting"
    EDITORIAL = "editorial"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


class StoryCategory(Enum):
    """Supported news categories aligned with financial_news taxonomy."""

    # Market-moving categories
    FED = "fed"
    EARNINGS = "earnings"
    SP500_EVENTS = "sp500_events"
    OIL = "oil"
    GEOPOLITICS = "geopolitics"

    # Additional categories
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    FOREX = "forex"
    MACRO = "macro"
    TECH = "tech"
    ENERGY = "energy"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"

    # Meta
    BREAKING = "breaking"
    OTHER = "other"


class VerificationStatus(Enum):
    """Verification outcome status."""

    VERIFIED = "verified"  # Fully corroborated
    PARTIAL = "partial"  # Some facts verified
    UNVERIFIED = "unverified"  # Could not verify
    CONTRADICTED = "contradicted"  # Found conflicting info
    ERROR = "error"  # Verification failed


@dataclass(frozen=True)
class DiscoveryRequest:
    """
    Request parameters for news discovery.

    Immutable to ensure request integrity throughout the pipeline.
    """

    tickers: tuple[str, ...] = field(default_factory=tuple)
    categories: tuple[str, ...] = field(
        default_factory=lambda: ("fed", "earnings", "sp500_events", "oil", "geopolitics")
    )
    time_window: str = "overnight"
    max_results: int = 50
    priority_tickers: tuple[str, ...] = field(default_factory=tuple)
    exclude_sources: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Convert lists to tuples for immutability
        object.__setattr__(self, "tickers", tuple(self.tickers))
        object.__setattr__(self, "categories", tuple(self.categories))
        object.__setattr__(self, "priority_tickers", tuple(self.priority_tickers))
        object.__setattr__(self, "exclude_sources", tuple(self.exclude_sources))

    @property
    def request_id(self) -> str:
        """Generate unique request ID based on parameters."""
        content = f"{self.tickers}{self.categories}{self.time_window}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class CandidateStory:
    """
    A news story candidate discovered by Grok.

    Raw story data before verification.
    """

    story_id: str
    headline: str
    source_name: str
    source_url: str
    snippet: str
    discovered_at: datetime

    # Metadata
    tickers: list[str] = field(default_factory=list)
    category: str = "other"
    published_at: datetime | None = None
    relevance_score: float = 0.0

    # Source quality indicators
    source_type: str = "unknown"  # official, mainstream, social, unknown
    source_credibility: float = 0.5  # 0-1 scale

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["discovered_at"] = self.discovered_at.isoformat()
        if self.published_at:
            data["published_at"] = self.published_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateStory":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("discovered_at"), str):
            data["discovered_at"] = datetime.fromisoformat(data["discovered_at"])
        if isinstance(data.get("published_at"), str):
            data["published_at"] = datetime.fromisoformat(data["published_at"])
        return cls(**data)


@dataclass
class Evidence:
    """Evidence supporting or contradicting a story."""

    source_url: str
    source_name: str
    quote: str
    supports_claim: bool
    credibility: float = 0.5  # 0-1 scale
    retrieved_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VerificationResult:
    """
    Result of Gemini verification.

    Contains confidence score and supporting evidence.
    """

    story_id: str
    candidate: CandidateStory
    status: VerificationStatus
    confidence: int  # 0-10 scale

    # Verified content
    verified_facts: list[str] = field(default_factory=list)
    what_happened: str = ""

    # Evidence
    evidence: list[Evidence] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    # Metadata
    sources_checked: int = 0
    verification_notes: str = ""
    verified_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_publishable(self) -> bool:
        """Check if verification confidence is high enough."""
        return self.confidence >= 6 and self.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.PARTIAL,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "story_id": self.story_id,
            "candidate": self.candidate.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "verified_facts": self.verified_facts,
            "what_happened": self.what_happened,
            "contradictions": self.contradictions,
            "sources_checked": self.sources_checked,
            "verification_notes": self.verification_notes,
            "verified_at": self.verified_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationResult":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("candidate"), dict):
            data["candidate"] = CandidateStory.from_dict(data["candidate"])
        if isinstance(data.get("status"), str):
            data["status"] = VerificationStatus(data["status"])
        if isinstance(data.get("verified_at"), str):
            data["verified_at"] = datetime.fromisoformat(data["verified_at"])
        # Remove evidence field if present (not reconstructed)
        data.pop("evidence", None)
        return cls(**data)


@dataclass
class FormattedStory:
    """
    Story formatted by ChatGPT.

    Structured, clear content ready for editorial polish.
    """

    story_id: str
    title: str
    what_happened: str
    bullet_points: list[str]
    affected_assets: list[str]

    # Metadata
    category: str
    verification_confidence: int
    formatted_at: datetime = field(default_factory=datetime.utcnow)

    # Optional enrichments
    related_tickers: list[str] = field(default_factory=list)
    sector_impact: str | None = None
    time_sensitivity: str = "normal"  # urgent, normal, background

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "story_id": self.story_id,
            "title": self.title,
            "what_happened": self.what_happened,
            "bullet_points": self.bullet_points,
            "affected_assets": self.affected_assets,
            "category": self.category,
            "verification_confidence": self.verification_confidence,
            "formatted_at": self.formatted_at.isoformat(),
            "related_tickers": self.related_tickers,
            "sector_impact": self.sector_impact,
            "time_sensitivity": self.time_sensitivity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FormattedStory":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("formatted_at"), str):
            data["formatted_at"] = datetime.fromisoformat(data["formatted_at"])
        return cls(**data)


@dataclass
class FinalizedStory:
    """
    Story finalized by Claude.

    Complete story with editorial polish and "why it matters" analysis.
    """

    story_id: str
    title: str
    what_happened: str
    why_it_matters: str
    bullet_points: list[str]
    affected_assets: list[str]

    # Metadata
    category: str
    verification_confidence: int
    finalized_at: datetime = field(default_factory=datetime.utcnow)

    # Editorial additions
    market_implications: str | None = None
    trading_considerations: str | None = None
    related_stories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Quality flags
    is_breaking: bool = False
    priority: int = 5  # 1-10, higher = more important

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "story_id": self.story_id,
            "title": self.title,
            "what_happened": self.what_happened,
            "why_it_matters": self.why_it_matters,
            "bullet_points": self.bullet_points,
            "affected_assets": self.affected_assets,
            "category": self.category,
            "verification_confidence": self.verification_confidence,
            "finalized_at": self.finalized_at.isoformat(),
            "market_implications": self.market_implications,
            "trading_considerations": self.trading_considerations,
            "is_breaking": self.is_breaking,
            "priority": self.priority,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FinalizedStory":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("finalized_at"), str):
            data["finalized_at"] = datetime.fromisoformat(data["finalized_at"])
        # Handle related_stories if present
        data.pop("related_stories", None)
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PublishedFeedItem:
    """
    Published story in the website feed.

    Final format delivered to end users.
    """

    story_id: str
    title: str
    what_happened: str
    why_it_matters: str
    bullet_points: list[str]
    affected_assets: list[str]

    # Publishing metadata
    category: str
    verification_confidence: int
    published_at: datetime
    feed_url: str = ""

    # Engagement tracking
    view_count: int = 0
    share_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "story_id": self.story_id,
            "title": self.title,
            "what_happened": self.what_happened,
            "why_it_matters": self.why_it_matters,
            "bullet_points": self.bullet_points,
            "affected_assets": self.affected_assets,
            "category": self.category,
            "verification_confidence": self.verification_confidence,
            "published_at": self.published_at.isoformat(),
            "feed_url": self.feed_url,
        }


@dataclass
class PipelineResult:
    """
    Complete result of a pipeline run.

    Contains statistics and all processed stories.
    """

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None

    # Request
    request: DiscoveryRequest | None = None

    # Stage counts
    discovered_count: int = 0
    verified_count: int = 0
    formatted_count: int = 0
    finalized_count: int = 0
    published_count: int = 0

    # Results
    stories: list[FinalizedStory] = field(default_factory=list)
    published_items: list[PublishedFeedItem] = field(default_factory=list)

    # Errors
    errors: list[str] = field(default_factory=list)
    stage_errors: dict = field(default_factory=dict)

    # Status
    status: str = "running"  # running, completed, failed

    @property
    def duration_seconds(self) -> float:
        """Get pipeline run duration in seconds."""
        if not self.completed_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def success_rate(self) -> float:
        """Calculate success rate from discovery to publishing."""
        if self.discovered_count == 0:
            return 0.0
        return self.published_count / self.discovered_count

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "discovered_count": self.discovered_count,
            "verified_count": self.verified_count,
            "formatted_count": self.formatted_count,
            "finalized_count": self.finalized_count,
            "published_count": self.published_count,
            "success_rate": self.success_rate,
            "status": self.status,
            "errors": self.errors,
            "stories": [s.to_dict() for s in self.stories],
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)
